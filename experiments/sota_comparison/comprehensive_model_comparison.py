#!/usr/bin/env python3
"""
综合模型对比实验
===============================
科学对比多种不同模型架构的ECG分类性能:
1. 随机森林 (传统机器学习基线)
2. ResNet (深度卷积网络)
3. 标准PLRNN (无GTF)
4. GTF-PLRNN (固定alpha)
5. 自适应GTF-PLRNN (动态alpha)
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
import argparse
import random

class AdaptiveGTF:
    """自适应GTF机制"""
    
    def __init__(self, initial_alpha=0.7, gamma=0.9, update_step=20, method="variance"):
        self.alpha = initial_alpha
        self.gamma = gamma
        self.update_step = update_step
        self.method = method
        self.alpha_history = [initial_alpha]
    
    def update_alpha(self, Z, model, optimization_step, alpha_min=0.05, alpha_max=0.95):
        """基于隐藏状态统计的alpha自适应更新"""
        if optimization_step % self.update_step == 0:
            with torch.no_grad():
                if self.method == "variance":
                    z_var = torch.var(Z, dim=0).mean().item()
                    z_norm = torch.norm(Z, dim=1).mean().item()
                    complexity = z_var / (z_norm + 1e-8)
                    estimated_alpha = max(alpha_min, min(alpha_max, 1.0 - complexity))
                    
                elif self.method == "gradient":
                    z_grad_proxy = torch.std(Z, dim=1).mean().item()
                    estimated_alpha = max(alpha_min, min(alpha_max, 0.5 + 0.4 * np.tanh(z_grad_proxy - 1.0)))
                
                # 平滑更新策略
                if estimated_alpha > self.alpha:
                    self.alpha = min(alpha_max, self.gamma * self.alpha + (1 - self.gamma) * estimated_alpha)
                else:
                    self.alpha = max(alpha_min, self.gamma * self.alpha + (1 - self.gamma) * estimated_alpha)
                
                self.alpha_history.append(self.alpha)
        
        return self.alpha

class RandomForestModel:
    """随机森林基线模型"""
    
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        self.model = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        )
        self.name = "Random Forest"
        self.device = "cpu"
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        print(f"🌳 训练随机森林模型...")
        start_time = time.time()
        
        self.model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # 训练集性能
        train_pred = self.model.predict(X_train)
        train_pred_proba = self.model.predict_proba(X_train)
        train_f1 = f1_score(y_train, train_pred, average='macro', zero_division=0)
        
        results = {
            'train_f1_macro': float(train_f1),
            'training_time': train_time,
            'model_name': self.name
        }
        
        # 验证集性能
        if X_val is not None:
            val_pred = self.model.predict(X_val)
            val_f1 = f1_score(y_val, val_pred, average='macro', zero_division=0)
            results['val_f1_macro'] = float(val_f1)
            print(f"   训练F1: {train_f1:.4f}, 验证F1: {val_f1:.4f}")
        else:
            print(f"   训练F1: {train_f1:.4f}")
        
        print(f"   训练时间: {train_time:.1f}秒")
        return results
    
    def predict(self, X):
        return self.model.predict_proba(X)

class ResNetModel(nn.Module):
    """ResNet模型用于ECG特征分类"""
    
    def __init__(self, input_dim=4, num_classes=25, hidden_dims=[64, 128, 256]):
        super().__init__()
        
        self.name = "ResNet"
        
        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # ResNet块
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.res_blocks.append(
                self._make_res_block(hidden_dims[i], hidden_dims[i+1])
            )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[-1] // 2, num_classes)
        )
        
        self._init_weights()
    
    def _make_res_block(self, in_dim, out_dim):
        """创建残差块"""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 输入层
        x = self.input_layer(x)
        
        # 残差块
        for res_block in self.res_blocks:
            residual = x
            x = res_block(x)
            # 残差连接（维度匹配时）
            if x.size(1) == residual.size(1):
                x = x + residual
        
        return self.output_layer(x)

class StandardPLRNN(nn.Module):
    """标准PLRNN模型（无GTF）"""
    
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=25, dropout_rate=0.1):
        super().__init__()
        
        self.name = "Standard PLRNN"
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # PLRNN核心组件
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.h = nn.Parameter(torch.zeros(hidden_dim))
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.activation = nn.ReLU()
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 输入投影
        z = self.input_projection(x)
        
        # PLRNN动态系统（无GTF）
        linear_part = torch.matmul(z, self.A.t())
        nonlinear_part = self.activation(torch.matmul(z, self.W.t()) + self.h)
        z_next = linear_part + nonlinear_part
        
        return self.output_projection(z_next)

class GTF_PLRNN(nn.Module):
    """GTF增强的PLRNN模型"""
    
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=25, 
                 alpha=0.9, dropout_rate=0.1, model_type="fixed"):
        super().__init__()
        
        self.current_alpha = alpha
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.name = f"GTF-PLRNN ({model_type})"
        
        # 输入投影
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # PLRNN核心组件
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.h = nn.Parameter(torch.zeros(hidden_dim))
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.activation = nn.ReLU()
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def set_alpha(self, alpha):
        self.current_alpha = alpha
    
    def forward(self, x):
        # 输入投影
        z = self.input_projection(x)
        
        # PLRNN动态系统
        linear_part = torch.matmul(z, self.A.t())
        nonlinear_part = self.activation(torch.matmul(z, self.W.t()) + self.h)
        z_next = linear_part + nonlinear_part
        
        # GTF机制
        z_gtf = self.current_alpha * z_next + (1 - self.current_alpha) * z
        
        return self.output_projection(z_gtf)

def load_comparative_data(n_samples=50000, test_ratio=0.2, val_ratio=0.2, 
                         data_dir="/export/home/zzhou/Master-Thesis/full_processed_dataset",
                         random_seed=42):
    """加载用于模型对比的ECG数据"""
    print(f"📊 加载对比实验数据: {n_samples}样本 (seed={random_seed})")
    
    # 设置随机种子
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # 读取数据
    print("   读取原始数据...")
    train_data = pd.read_csv(f"{data_dir}/train_data.csv", nrows=min(n_samples*2, 100000))
    train_labels = pd.read_csv(f"{data_dir}/train_binary_labels.csv", nrows=min(n_samples*2, 100000))
    
    val_data = pd.read_csv(f"{data_dir}/val_data.csv", nrows=min(n_samples//2, 50000))
    val_labels = pd.read_csv(f"{data_dir}/val_binary_labels.csv", nrows=min(n_samples//2, 50000))
    
    # 特征和标签选择
    feature_cols = ['rr_interval', 'qrs_axis', 'p_axis', 't_axis']
    label_cols = [col for col in train_labels.columns 
                  if col not in ['subject_id', 'study_id', 'ecg_time']][:25]
    
    # 分层采样
    if len(train_data) > n_samples:
        y_temp = train_labels[label_cols].sum(axis=1)
        stratify_bins = pd.cut(y_temp, bins=5, labels=False)
        
        sample_indices = []
        for bin_val in range(5):
            bin_indices = np.where(stratify_bins == bin_val)[0]
            bin_sample_size = min(len(bin_indices), n_samples // 5)
            if bin_sample_size > 0:
                bin_samples = np.random.choice(bin_indices, bin_sample_size, replace=False)
                sample_indices.extend(bin_samples)
        
        sample_indices = np.array(sample_indices[:n_samples])
        train_data = train_data.iloc[sample_indices].reset_index(drop=True)
        train_labels = train_labels.iloc[sample_indices].reset_index(drop=True)
    
    # 提取特征和标签
    X_train = train_data[feature_cols].fillna(0).values.astype(np.float32)
    y_train = train_labels[label_cols].fillna(0).values.astype(np.float32)
    
    X_val = val_data[feature_cols].fillna(0).values.astype(np.float32)
    y_val = val_labels[label_cols].fillna(0).values.astype(np.float32)
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    print(f"✅ 对比数据加载完成:")
    print(f"   训练集: {X_train.shape} -> {y_train.shape}")
    print(f"   验证集: {X_val.shape} -> {y_val.shape}")
    print(f"   平均标签数: {y_train.sum(axis=1).mean():.2f}")
    
    return X_train, y_train, X_val, y_val, scaler

def create_dataloader(X, y, batch_size=64, shuffle=True, seed=42):
    """创建数据加载器"""
    if shuffle:
        torch.manual_seed(seed)
        
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

def train_neural_model(model, train_loader, val_loader, device, epochs=50, 
                      adaptive_gtf=None, verbose=True):
    """训练神经网络模型"""
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), 
        epochs=epochs, pct_start=0.3
    )
    
    best_f1 = 0
    results = []
    alpha_evolution = []
    
    if verbose:
        print(f"🚀 训练{model.name}...")
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # 训练
        model.train()
        total_loss = 0
        epoch_alpha = getattr(model, 'current_alpha', 0.0)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # 自适应alpha更新
            if adaptive_gtf is not None:
                with torch.no_grad():
                    z_intermediate = model.input_projection(data)
                
                optimization_step = epoch * len(train_loader) + batch_idx
                current_alpha = adaptive_gtf.update_alpha(z_intermediate, model, optimization_step)
                model.set_alpha(current_alpha)
                epoch_alpha = current_alpha
            
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            del data, target, output
        
        alpha_evolution.append(epoch_alpha)
        
        # 验证
        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                
                preds = torch.sigmoid(output).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(target.cpu().numpy())
                
                del data, target, output
        
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        pred_binary = (all_preds > 0.5).astype(int)
        f1_macro = f1_score(all_targets, pred_binary, average='macro', zero_division=0)
        
        if f1_macro > best_f1:
            best_f1 = f1_macro
        
        results.append({
            'epoch': epoch,
            'train_loss': float(total_loss / len(train_loader)),
            'val_loss': float(val_loss / len(val_loader)),
            'val_f1_macro': float(f1_macro),
            'alpha': float(epoch_alpha)
        })
        
        scheduler.step()
        
        if verbose and epoch % 10 == 0:
            print(f"   Epoch {epoch}: Val F1={f1_macro:.4f}, α={epoch_alpha:.4f}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    train_time = time.time() - start_time
    
    return {
        'model_name': model.name,
        'best_f1_macro': float(best_f1),
        'training_time': train_time,
        'alpha_evolution': alpha_evolution,
        'training_history': results
    }

def main():
    parser = argparse.ArgumentParser(description='综合模型对比实验')
    
    parser.add_argument('--model', type=str, 
                        choices=['random_forest', 'resnet', 'standard_plrnn', 
                                'gtf_fixed', 'gtf_adaptive'], 
                        required=True, help='模型类型')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--samples', type=int, default=50000, help='样本数量')
    parser.add_argument('--batch-size', type=int, default=64, help='批次大小')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"   🔬 综合模型对比实验: {args.model}")
    print(f"   GPU: {args.gpu} | Epochs: {args.epochs} | Samples: {args.samples}")
    print(f"   Seed: {args.seed}")
    print("=" * 80)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    # 加载数据
    X_train, y_train, X_val, y_val, scaler = load_comparative_data(
        n_samples=args.samples, random_seed=args.seed
    )
    
    start_time = time.time()
    
    if args.model == 'random_forest':
        # 随机森林模型
        model = RandomForestModel(n_estimators=100, max_depth=15, random_state=args.seed)
        results = model.fit(X_train, y_train, X_val, y_val)
        
    else:
        # 神经网络模型
        train_loader = create_dataloader(
            X_train, y_train, batch_size=args.batch_size, shuffle=True, seed=args.seed
        )
        val_loader = create_dataloader(
            X_val, y_val, batch_size=args.batch_size*2, shuffle=False, seed=args.seed
        )
        
        if args.model == 'resnet':
            model = ResNetModel(input_dim=4, num_classes=25, hidden_dims=[64, 128, 256])
            results = train_neural_model(model, train_loader, val_loader, device, args.epochs)
            
        elif args.model == 'standard_plrnn':
            model = StandardPLRNN(input_dim=4, hidden_dim=128, output_dim=25)
            results = train_neural_model(model, train_loader, val_loader, device, args.epochs)
            
        elif args.model == 'gtf_fixed':
            model = GTF_PLRNN(input_dim=4, hidden_dim=128, output_dim=25, 
                             alpha=0.7, model_type="Fixed α=0.7")
            results = train_neural_model(model, train_loader, val_loader, device, args.epochs)
            
        elif args.model == 'gtf_adaptive':
            model = GTF_PLRNN(input_dim=4, hidden_dim=128, output_dim=25, 
                             alpha=0.7, model_type="Adaptive")
            adaptive_gtf = AdaptiveGTF(initial_alpha=0.7, gamma=0.9, 
                                     update_step=20, method="variance")
            results = train_neural_model(model, train_loader, val_loader, device, 
                                       args.epochs, adaptive_gtf)
    
    total_time = time.time() - start_time
    
    print("\\n" + "=" * 80)
    print(f"   🏆 {results['model_name']}实验完成!")
    print("=" * 80)
    print(f"最佳F1: {results.get('best_f1_macro', results.get('val_f1_macro', 0)):.4f}")
    print(f"训练时间: {results.get('training_time', total_time)/60:.1f}分钟")
    
    # 保存结果
    result_dir = f"comparative_results_{args.model}_s{args.samples}_gpu{args.gpu}"
    os.makedirs(result_dir, exist_ok=True)
    
    final_results = {
        'experiment_type': 'comprehensive_model_comparison',
        'model_type': args.model,
        'configuration': {
            'samples': args.samples,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'seed': args.seed,
            'gpu': args.gpu
        },
        'results': results,
        'experiment_time': total_time
    }
    
    with open(f'{result_dir}/comparative_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"📁 结果保存到: {result_dir}/comparative_results.json")
    
    return final_results

if __name__ == "__main__":
    main()