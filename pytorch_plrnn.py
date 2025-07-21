#!/usr/bin/env python3
"""
PyTorch PLRNN ECG Classification System for Mac M4
==================================================

A PyTorch implementation of Piecewise Linear Recurrent Neural Networks (PLRNN) 
for ECG arrhythmia classification optimized for Mac M4 chips.

Author: Master Thesis Project
Date: 2025
"""

import os
import gc
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wfdb
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, welch
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端，适合Mac环境
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from pathlib import Path

# 禁用警告
warnings.filterwarnings('ignore')

# Mac M4优化配置
print("=== Mac M4优化配置 ===")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ 使用Metal Performance Shaders (MPS)")
else:
    device = torch.device("cpu") 
    print("⚠️  MPS不可用，使用CPU")

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ===================================================================
# 1. 配置参数
# ===================================================================

class Config:
    """项目配置"""
    # 数据路径
    LABELED_DATA_CSV = '/Users/zixiang/PycharmProjects/Master-Thesis/ecg_5_class_data.csv'
    ECG_BASE_PATH = '/Users/zixiang/ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/'
    
    # 模型参数
    SEQUENCE_LENGTH = 500
    NUM_CHANNELS = 12  
    NUM_FEATURES = 8
    HIDDEN_DIM = 64
    NUM_PIECES = 4
    
    # 训练参数
    BATCH_SIZE = 8  # 减小批次大小以适应Mac M4
    LEARNING_RATE = 0.001
    EPOCHS = 5  # 先进行短期测试
    
    # 数据集大小（测试阶段使用较小规模）
    TRAIN_SAMPLES = 200
    VAL_SAMPLES = 50
    TEST_SAMPLES = 50
    
    # 设备
    DEVICE = device

config = Config()

# ===================================================================
# 2. PLRNN核心实现
# ===================================================================

class PiecewiseLinearActivation(nn.Module):
    """分段线性激活函数"""
    
    def __init__(self, num_pieces=3):
        super().__init__()
        self.num_pieces = num_pieces
        
        # 初始化分段点和斜率
        self.breakpoints = nn.Parameter(torch.linspace(-2, 2, num_pieces-1))
        self.slopes = nn.Parameter(torch.ones(num_pieces))
        self.intercepts = nn.Parameter(torch.zeros(num_pieces))
    
    def forward(self, x):
        """前向传播"""
        # 确保断点有序
        sorted_breakpoints = torch.sort(self.breakpoints)[0]
        
        output = torch.zeros_like(x)
        
        # 第一段: x < breakpoint[0]
        mask1 = (x < sorted_breakpoints[0]).float()
        output += mask1 * (self.slopes[0] * x + self.intercepts[0])
        
        # 中间段
        for i in range(1, self.num_pieces - 1):
            mask = ((x >= sorted_breakpoints[i-1]) & (x < sorted_breakpoints[i])).float()
            output += mask * (self.slopes[i] * x + self.intercepts[i])
        
        # 最后一段: x >= breakpoint[-1]
        mask_last = (x >= sorted_breakpoints[-1]).float()
        output += mask_last * (self.slopes[-1] * x + self.intercepts[-1])
        
        return output


class PLRNNCell(nn.Module):
    """PLRNN单元"""
    
    def __init__(self, input_size, hidden_size, num_pieces=3):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 权重矩阵
        self.W_ih = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        
        # 分段线性激活
        self.activation = PiecewiseLinearActivation(num_pieces)
        
        # 初始化权重
        nn.init.xavier_uniform_(self.W_ih.weight)
        nn.init.orthogonal_(self.W_hh.weight)
    
    def forward(self, x, h_prev=None):
        """前向传播"""
        batch_size = x.size(0)
        
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # PLRNN计算: h_t = f(W_ih * x_t + W_hh * h_{t-1})
        linear_output = self.W_ih(x) + self.W_hh(h_prev)
        h_new = self.activation(linear_output)
        
        return h_new


class PLRNN(nn.Module):
    """PLRNN层"""
    
    def __init__(self, input_size, hidden_size, num_pieces=3, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 创建PLRNN单元
        self.cells = nn.ModuleList([
            PLRNNCell(input_size if i == 0 else hidden_size, hidden_size, num_pieces)
            for i in range(num_layers)
        ])
    
    def forward(self, x):
        """前向传播
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            output: (batch_size, hidden_size) - 最后时刻的隐状态
        """
        batch_size, seq_len, _ = x.size()
        
        # 初始化隐状态
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
             for _ in range(self.num_layers)]
        
        # 逐时间步处理
        for t in range(seq_len):
            input_t = x[:, t, :]
            
            for layer in range(self.num_layers):
                h[layer] = self.cells[layer](input_t, h[layer])
                input_t = h[layer]  # 下一层的输入
        
        return h[-1]  # 返回最后一层的最终隐状态


# ===================================================================
# 3. 数据处理与特征提取
# ===================================================================

def extract_medical_features(signal, fs=100):
    """提取医学特征（来自stats.py的功能）"""
    try:
        if signal is None or len(signal) == 0:
            return np.zeros(8)
        
        # 使用导联II进行分析
        lead_ii = signal[:, 1] if signal.shape[1] > 1 else signal[:, 0]
        features = []
        
        # 统计特征
        features.extend([
            np.mean(lead_ii),
            np.std(lead_ii), 
            skew(lead_ii),
            kurtosis(lead_ii)
        ])
        
        # 心率特征
        try:
            peaks, _ = find_peaks(lead_ii, height=np.std(lead_ii)*0.5, distance=fs//4)
            if len(peaks) >= 2:
                rr_intervals = np.diff(peaks) / fs
                heart_rate = 60 / np.mean(rr_intervals)
                hrv = np.std(rr_intervals) * 1000  # SDNN in ms
                rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2)) * 1000  # RMSSD in ms
                cv_rr = np.std(rr_intervals) / np.mean(rr_intervals)  # CV of RR intervals
                features.extend([heart_rate, hrv, rmssd, cv_rr])
            else:
                features.extend([70, 30, 25, 0.05])  # 默认值
        except:
            features.extend([70, 30, 25, 0.05])
        
        # 确保特征有效
        features = [float(f) if np.isfinite(f) else 0.0 for f in features]
        return np.array(features[:8])
    
    except Exception as e:
        print(f"特征提取错误: {e}")
        return np.zeros(8)


def preprocess_ecg_signal(raw_signal, target_length=500):
    """ECG信号预处理"""
    try:
        if raw_signal is None or len(raw_signal) == 0:
            return None
            
        signal = np.array(raw_signal, dtype=np.float64)
        
        # 检查信号格式
        if len(signal.shape) != 2 or signal.shape[1] != 12:
            print(f"信号格式错误: {signal.shape}, 期望: (N, 12)")
            return None
            
        # 检查数值有效性
        if not np.isfinite(signal).all():
            print("信号包含无效数值")
            return None
        
        # 重采样到目标长度
        if len(signal) != target_length:
            old_indices = np.linspace(0, len(signal)-1, len(signal))
            new_indices = np.linspace(0, len(signal)-1, target_length)
            
            resampled_signal = np.zeros((target_length, 12), dtype=np.float32)
            for i in range(12):
                if len(signal) > 1:
                    f = interp1d(old_indices, signal[:, i], kind='linear', fill_value='extrapolate')
                    resampled_signal[:, i] = f(new_indices)
                else:
                    resampled_signal[:, i] = signal[0, i]
        else:
            resampled_signal = signal.astype(np.float32)
        
        # 通道级别标准化
        for i in range(12):
            channel = resampled_signal[:, i]
            std_val = np.std(channel)
            if std_val > 1e-10:
                resampled_signal[:, i] = (channel - np.mean(channel)) / std_val
        
        # 异常值截断
        return np.clip(resampled_signal, -3, 3)
        
    except Exception as e:
        print(f"预处理错误: {e}")
        return None


def balanced_sampling(df, target_samples, random_state=42):
    """平衡采样以解决类别不平衡"""
    categories = df['ecg_category'].unique()
    samples_per_class = target_samples // len(categories)
    
    balanced_dfs = []
    for category in categories:
        category_df = df[df['ecg_category'] == category]
        if len(category_df) >= samples_per_class:
            sampled_df = category_df.sample(n=samples_per_class, random_state=random_state)
        else:
            # 重复采样
            sampled_df = category_df.sample(n=samples_per_class, replace=True, random_state=random_state)
        balanced_dfs.append(sampled_df)
    
    result_df = pd.concat(balanced_dfs, ignore_index=True)
    return result_df.sample(frac=1, random_state=random_state).reset_index(drop=True)


# ===================================================================
# 4. 数据集类
# ===================================================================

class ECGDataset(Dataset):
    """ECG数据集"""
    
    def __init__(self, dataframe, label_encoder, base_path=None, augment=False):
        self.df = dataframe.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.base_path = base_path or config.ECG_BASE_PATH
        self.augment = augment
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            
            # 构建完整路径
            relative_path = row['waveform_path']
            if relative_path.endswith('.hea'):
                relative_path = relative_path[:-4]  # 移除.hea扩展名
            elif relative_path.endswith('.dat'):
                relative_path = relative_path[:-4]  # 移除.dat扩展名
                
            full_path = os.path.join(self.base_path, relative_path)
            
            # 检查文件是否存在
            if not os.path.exists(full_path + '.dat'):
                print(f"文件不存在: {full_path}.dat")
                return self._get_default_sample()
            
            # 读取ECG记录
            record = wfdb.rdrecord(full_path)
            if record.p_signal is None:
                print(f"无法读取信号: {full_path}")
                return self._get_default_sample()
            
            # 预处理信号
            processed_signal = preprocess_ecg_signal(record.p_signal, config.SEQUENCE_LENGTH)
            if processed_signal is None:
                return self._get_default_sample()
            
            # 数据增强（仅训练时）
            if self.augment:
                processed_signal = self._augment_signal(processed_signal)
            
            # 提取医学特征
            medical_features = extract_medical_features(processed_signal)
            
            # 编码标签
            label = self.label_encoder.transform([row['ecg_category']])[0]
            
            return {
                'waveform': torch.FloatTensor(processed_signal),  # (500, 12)
                'features': torch.FloatTensor(medical_features),  # (8,)
                'label': torch.LongTensor([label])[0]
            }
            
        except Exception as e:
            print(f"加载数据时出错 (idx={idx}): {e}")
            return self._get_default_sample()
    
    def _get_default_sample(self):
        """返回默认样本（当数据加载失败时）"""
        return {
            'waveform': torch.zeros(config.SEQUENCE_LENGTH, config.NUM_CHANNELS),
            'features': torch.zeros(config.NUM_FEATURES),
            'label': torch.LongTensor([0])[0]
        }
    
    def _augment_signal(self, signal):
        """轻量级数据增强"""
        if np.random.random() < 0.3:
            # 添加高斯噪声
            noise = np.random.normal(0, 0.05, signal.shape)
            signal = signal + noise
        
        if np.random.random() < 0.2:
            # 幅度缩放
            scale = np.random.uniform(0.8, 1.2)
            signal = signal * scale
        
        return signal


# ===================================================================
# 5. PLRNN模型架构
# ===================================================================

class PLRNNECGClassifier(nn.Module):
    """基于PLRNN的ECG分类器"""
    
    def __init__(self, num_classes):
        super().__init__()
        
        # 多尺度CNN特征提取
        self.conv_layers = nn.ModuleList([
            self._make_conv_block(config.NUM_CHANNELS, 32, kernel_size=3),
            self._make_conv_block(config.NUM_CHANNELS, 32, kernel_size=5), 
            self._make_conv_block(config.NUM_CHANNELS, 32, kernel_size=7)
        ])
        
        # PLRNN层
        conv_output_size = 96  # 3 * 32
        self.plrnn1 = PLRNN(conv_output_size, config.HIDDEN_DIM, config.NUM_PIECES, num_layers=1)
        self.plrnn2 = PLRNN(config.HIDDEN_DIM, config.HIDDEN_DIM//2, config.NUM_PIECES-1, num_layers=1)
        
        # 波形分支
        self.waveform_branch = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM//2, 48),
            nn.ReLU(),
            nn.BatchNorm1d(48),
            nn.Dropout(0.3)
        )
        
        # 医学特征分支
        self.feature_branch = nn.Sequential(
            nn.Linear(config.NUM_FEATURES, 24),
            nn.ReLU(), 
            nn.BatchNorm1d(24),
            nn.Dropout(0.2)
        )
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(48 + 24, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def _make_conv_block(self, in_channels, out_channels, kernel_size):
        """创建卷积块"""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.MaxPool1d(2),
            nn.Dropout(0.1)
        )
    
    def forward(self, waveform, features):
        """前向传播
        Args:
            waveform: (batch_size, seq_len, num_channels)
            features: (batch_size, num_features)
        """
        batch_size = waveform.size(0)
        
        # 调整维度用于CNN: (batch_size, seq_len, channels) -> (batch_size, channels, seq_len)
        waveform = waveform.transpose(1, 2)
        
        # 多尺度CNN特征提取
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_out = conv_layer(waveform)  # (batch_size, 32, reduced_seq_len)
            conv_outputs.append(conv_out)
        
        # 拼接多尺度特征
        fused_conv = torch.cat(conv_outputs, dim=1)  # (batch_size, 96, reduced_seq_len)
        
        # 调整维度用于PLRNN: (batch_size, channels, seq_len) -> (batch_size, seq_len, channels)
        fused_conv = fused_conv.transpose(1, 2)
        
        # PLRNN层
        plrnn_out1 = self.plrnn1(fused_conv)
        # 为第二个PLRNN添加序列维度
        plrnn_out1_expanded = plrnn_out1.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        plrnn_out2 = self.plrnn2(plrnn_out1_expanded)
        
        # 波形分支
        waveform_features = self.waveform_branch(plrnn_out2)
        
        # 医学特征分支
        medical_features = self.feature_branch(features)
        
        # 特征融合
        fused_features = torch.cat([waveform_features, medical_features], dim=1)
        
        # 分类
        output = self.classifier(fused_features)
        
        return output


# ===================================================================
# 6. 训练与评估
# ===================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        waveform = batch['waveform'].to(device)
        features = batch['features'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(waveform, features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            waveform = batch['waveform'].to(device)
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(waveform, features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * correct / total
    return total_loss / len(dataloader), accuracy, all_preds, all_labels


# ===================================================================
# 7. 主训练流程
# ===================================================================

def main():
    """主函数"""
    print("=== PyTorch PLRNN ECG分类系统 ===")
    print(f"设备: {config.DEVICE}")
    
    # 1. 数据加载
    print("\n--- 数据加载 ---")
    try:
        full_df = pd.read_csv(config.LABELED_DATA_CSV, header=None,
                             names=['subject_id', 'waveform_path', 'ecg_category'])
        full_df.dropna(inplace=True)
        print(f"成功加载 {len(full_df)} 条记录")
        
        print("\n类别分布:")
        print(full_df['ecg_category'].value_counts())
        
    except FileNotFoundError:
        print(f"错误: 找不到标签文件 {config.LABELED_DATA_CSV}")
        return
    
    # 2. 数据切分  
    print("\n--- 数据切分 ---")
    all_subjects = full_df['subject_id'].unique()
    train_val_subjects, test_subjects = train_test_split(all_subjects, test_size=0.15, random_state=42)
    train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=0.15, random_state=42)
    
    train_df = full_df[full_df['subject_id'].isin(train_subjects)].reset_index(drop=True)
    val_df = full_df[full_df['subject_id'].isin(val_subjects)].reset_index(drop=True)
    test_df = full_df[full_df['subject_id'].isin(test_subjects)].reset_index(drop=True)
    
    # 平衡采样
    train_subset_df = balanced_sampling(train_df, config.TRAIN_SAMPLES, random_state=42)
    val_subset_df = balanced_sampling(val_df, config.VAL_SAMPLES, random_state=42)
    test_subset_df = balanced_sampling(test_df, config.TEST_SAMPLES, random_state=42)
    
    print(f"平衡采样后: 训练{len(train_subset_df)}, 验证{len(val_subset_df)}, 测试{len(test_subset_df)}")
    
    # 3. 标签编码
    label_encoder = LabelEncoder()
    all_labels = full_df['ecg_category'].unique()
    label_encoder.fit(all_labels)
    num_classes = len(all_labels)
    
    print(f"\n标签映射: {dict(zip(all_labels, label_encoder.transform(all_labels)))}")
    
    # 4. 创建数据集和数据加载器
    print("\n--- 创建数据集 ---")
    train_dataset = ECGDataset(train_subset_df, label_encoder, augment=True)
    val_dataset = ECGDataset(val_subset_df, label_encoder, augment=False)
    test_dataset = ECGDataset(test_subset_df, label_encoder, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                             shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                           shuffle=False, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, 
                            shuffle=False, num_workers=0, drop_last=False)
    
    # 5. 创建模型
    print("\n--- 创建PLRNN模型 ---")
    model = PLRNNECGClassifier(num_classes).to(config.DEVICE)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 6. 设置优化器和损失函数
    # 计算类别权重
    train_labels = [label_encoder.transform([cat])[0] for cat in train_subset_df['ecg_category']]
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_tensor = torch.FloatTensor(class_weights).to(config.DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # 7. 训练循环
    print(f"\n--- 开始训练 ({config.EPOCHS} epochs) ---")
    best_val_acc = 0
    train_history = {'loss': [], 'acc': []}
    val_history = {'loss': [], 'acc': []}
    
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        train_history['loss'].append(train_loss)
        train_history['acc'].append(train_acc)
        
        # 验证
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, config.DEVICE)
        val_history['loss'].append(val_loss)
        val_history['acc'].append(val_acc)
        
        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # 学习率调整
        scheduler.step(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'pytorch_plrnn_best_model.pth')
            print(f"✅ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
        
        # 垃圾回收（Mac M4优化）
        if epoch % 5 == 0:
            gc.collect()
    
    # 8. 测试评估
    print("\n--- 模型测试 ---")
    # 加载最佳模型
    model.load_state_dict(torch.load('pytorch_plrnn_best_model.pth'))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, config.DEVICE)
    
    print(f"测试结果: Loss={test_loss:.4f}, Accuracy={test_acc:.2f}%")
    
    # 详细分类报告
    print("\n分类报告:")
    print(classification_report(test_labels, test_preds, target_names=all_labels))
    
    print("\n混淆矩阵:")
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)
    
    # 9. 可视化结果
    plt.figure(figsize=(15, 5))
    
    # 训练曲线
    plt.subplot(1, 3, 1)
    plt.plot(train_history['acc'], label='Training Accuracy')
    plt.plot(val_history['acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_history['loss'], label='Training Loss')
    plt.plot(val_history['loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 混淆矩阵热力图
    plt.subplot(1, 3, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=all_labels, yticklabels=all_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('pytorch_plrnn_results.png', dpi=300, bbox_inches='tight')
    print("\n结果图保存为: pytorch_plrnn_results.png")
    
    # 10. 保存训练配置和结果
    results = {
        'config': {
            'sequence_length': config.SEQUENCE_LENGTH,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'epochs': config.EPOCHS,
            'device': str(config.DEVICE)
        },
        'results': {
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_loss': test_loss
        },
        'model_info': {
            'total_params': total_params,
            'trainable_params': trainable_params
        }
    }
    
    with open('pytorch_plrnn_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ 训练完成！")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"测试准确率: {test_acc:.2f}%")


if __name__ == "__main__":
    main()