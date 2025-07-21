#!/usr/bin/env python3
"""
PyTorch PLRNN Multi-Label ECG Classification System for Mac M4
============================================================

A PyTorch implementation of Piecewise Linear Recurrent Neural Networks (PLRNN) 
for ECG multi-label arrhythmia classification optimized for Mac M4 chips.
Based on scientific MIMIC-IV-ECG dataset with comprehensive diagnosis labels.

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
from sklearn.metrics import (
    classification_report, hamming_loss, accuracy_score,
    precision_recall_fscore_support, multilabel_confusion_matrix
)
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端，适合Mac环境
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from pathlib import Path
import ast

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
    MULTILABEL_DATA_CSV = '/Users/zixiang/PycharmProjects/Master-Thesis/mimic_ecg_multilabel_dataset.csv'
    BINARY_LABELS_CSV = '/Users/zixiang/PycharmProjects/Master-Thesis/mimic_ecg_binary_labels.csv'
    CONFIG_JSON = '/Users/zixiang/PycharmProjects/Master-Thesis/mimic_ecg_multilabel_dataset_config.json'
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
    EPOCHS = 3  # 快速测试
    
    # 数据集大小（快速测试阶段使用更小规模）
    TRAIN_SAMPLES = 100
    VAL_SAMPLES = 50
    TEST_SAMPLES = 50
    
    # 多标签分类阈值
    CLASSIFICATION_THRESHOLD = 0.5
    
    # 设备
    DEVICE = device

config = Config()

# ===================================================================
# 2. PLRNN核心实现 (复用原有代码)
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
# 3. 数据处理与特征提取 (复用原有代码)
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


# ===================================================================
# 4. 多标签数据集类
# ===================================================================

class MultiLabelECGDataset(Dataset):
    """多标签ECG数据集"""
    
    def __init__(self, multilabel_df, binary_df, label_columns, base_path=None, augment=False):
        self.multilabel_df = multilabel_df.reset_index(drop=True)
        self.binary_df = binary_df.reset_index(drop=True)
        self.label_columns = label_columns
        self.base_path = base_path or config.ECG_BASE_PATH
        self.augment = augment
        
        # 合并数据
        self.merged_df = pd.merge(self.multilabel_df, self.binary_df, 
                                 on=['subject_id', 'study_id', 'ecg_time'], 
                                 how='inner')
        print(f"成功合并数据: {len(self.merged_df)} 条记录")
        
    def __len__(self):
        return len(self.merged_df)
    
    def __getitem__(self, idx):
        try:
            row = self.merged_df.iloc[idx]
            
            # 根据MIMIC-IV-ECG实际结构构建路径
            subject_id = row['subject_id']
            study_id = row['study_id']
            
            # 获取subject_id的前4位数字
            subject_prefix = str(subject_id)[:4]
            
            # 构建正确的文件路径
            waveform_path = f"files/p{subject_prefix}/p{subject_id}/s{study_id}/{study_id}"
            full_path = os.path.join(self.base_path, waveform_path)
            
            signal_loaded = False
            processed_signal = None
            
            # 检查文件是否存在
            if os.path.exists(full_path + '.dat') and os.path.exists(full_path + '.hea'):
                try:
                    # 读取ECG记录
                    record = wfdb.rdrecord(full_path)
                    if record.p_signal is not None:
                        # 预处理信号
                        processed_signal = preprocess_ecg_signal(record.p_signal, config.SEQUENCE_LENGTH)
                        if processed_signal is not None:
                            signal_loaded = True
                except Exception as e:
                    pass
            
            if not signal_loaded or processed_signal is None:
                # print(f"无法加载信号 (idx={idx}, subject={subject_id}, study={study_id})")
                return self._get_default_sample()
            
            # 数据增强（仅训练时）
            if self.augment:
                processed_signal = self._augment_signal(processed_signal)
            
            # 提取医学特征
            medical_features = extract_medical_features(processed_signal)
            
            # 提取多标签
            multi_labels = torch.FloatTensor(row[self.label_columns].values.astype(float))
            
            return {
                'waveform': torch.FloatTensor(processed_signal),  # (500, 12)
                'features': torch.FloatTensor(medical_features),  # (8,)
                'labels': multi_labels  # (num_labels,) - 多标签二进制向量
            }
            
        except Exception as e:
            print(f"加载数据时出错 (idx={idx}): {e}")
            return self._get_default_sample()
    
    def _get_default_sample(self):
        """返回默认样本（当数据加载失败时）"""
        return {
            'waveform': torch.zeros(config.SEQUENCE_LENGTH, config.NUM_CHANNELS),
            'features': torch.zeros(config.NUM_FEATURES),
            'labels': torch.zeros(len(self.label_columns))
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
# 5. 多标签PLRNN模型架构
# ===================================================================

class MultiLabelPLRNNECGClassifier(nn.Module):
    """基于PLRNN的多标签ECG分类器"""
    
    def __init__(self, num_labels):
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
        
        # 多标签分类层 (使用sigmoid而非softmax)
        self.classifier = nn.Sequential(
            nn.Linear(48 + 24, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_labels)  # 输出维度为标签数量
            # 注意：不加sigmoid，因为BCEWithLogitsLoss内部包含sigmoid
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
        Returns:
            logits: (batch_size, num_labels) - 未经sigmoid的原始输出
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
        
        # 多标签分类 (输出logits)
        logits = self.classifier(fused_features)
        
        return logits


# ===================================================================
# 6. 多标签训练与评估
# ===================================================================

def train_epoch_multilabel(model, dataloader, criterion, optimizer, device):
    """训练一个epoch（多标签版本）"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        waveform = batch['waveform'].to(device)
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(waveform, features)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 计算预测结果
        probs = torch.sigmoid(logits)
        preds = (probs > config.CLASSIFICATION_THRESHOLD).float()
        
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    # 计算指标
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    hamming = hamming_loss(all_labels, all_preds)
    subset_acc = accuracy_score(all_labels, all_preds)
    
    return total_loss / len(dataloader), hamming, subset_acc


def evaluate_multilabel(model, dataloader, criterion, device):
    """评估模型（多标签版本）"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            waveform = batch['waveform'].to(device)
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(waveform, features)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            # 计算预测结果
            probs = torch.sigmoid(logits)
            preds = (probs > config.CLASSIFICATION_THRESHOLD).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # 计算指标
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    
    hamming = hamming_loss(all_labels, all_preds)
    subset_acc = accuracy_score(all_labels, all_preds)
    
    # 计算每个标签的F1分数
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0)
    
    macro_f1 = np.mean(f1)
    micro_f1 = precision_recall_fscore_support(
        all_labels, all_preds, average='micro', zero_division=0)[2]
    
    return total_loss / len(dataloader), hamming, subset_acc, macro_f1, micro_f1, all_preds, all_labels, all_probs


# ===================================================================
# 7. 主训练流程
# ===================================================================

def main():
    """主函数"""
    print("=== PyTorch PLRNN 多标签ECG分类系统 ===")
    print(f"设备: {config.DEVICE}")
    
    # 1. 加载多标签数据集
    print("\n--- 加载多标签数据集 ---")
    try:
        # 加载多标签数据
        multilabel_df = pd.read_csv(config.MULTILABEL_DATA_CSV)
        print(f"多标签数据集: {len(multilabel_df)} 条记录")
        
        # 加载二进制标签矩阵
        binary_df = pd.read_csv(config.BINARY_LABELS_CSV)
        print(f"二进制标签矩阵: {len(binary_df)} 条记录")
        
        # 获取标签列名
        label_columns = [col for col in binary_df.columns 
                        if col not in ['subject_id', 'study_id', 'ecg_time']]
        num_labels = len(label_columns)
        print(f"标签数量: {num_labels}")
        
        # 加载配置信息
        with open(config.CONFIG_JSON, 'r') as f:
            config_info = json.load(f)
        print(f"数据集统计: {config_info['dataset_stats']}")
        
    except Exception as e:
        print(f"错误: 无法加载数据集 {e}")
        return
    
    # 2. 数据切分
    print("\n--- 数据切分 ---")
    all_subjects = multilabel_df['subject_id'].unique()
    train_val_subjects, test_subjects = train_test_split(all_subjects, test_size=0.15, random_state=42)
    train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=0.15, random_state=42)
    
    train_multilabel = multilabel_df[multilabel_df['subject_id'].isin(train_subjects)].reset_index(drop=True)
    val_multilabel = multilabel_df[multilabel_df['subject_id'].isin(val_subjects)].reset_index(drop=True)  
    test_multilabel = multilabel_df[multilabel_df['subject_id'].isin(test_subjects)].reset_index(drop=True)
    
    train_binary = binary_df[binary_df['subject_id'].isin(train_subjects)].reset_index(drop=True)
    val_binary = binary_df[binary_df['subject_id'].isin(val_subjects)].reset_index(drop=True)
    test_binary = binary_df[binary_df['subject_id'].isin(test_subjects)].reset_index(drop=True)
    
    # 采样子集（测试阶段）
    train_multilabel = train_multilabel.head(config.TRAIN_SAMPLES)
    val_multilabel = val_multilabel.head(config.VAL_SAMPLES)
    test_multilabel = test_multilabel.head(config.TEST_SAMPLES)
    
    train_binary = train_binary[train_binary['subject_id'].isin(train_multilabel['subject_id'])]
    val_binary = val_binary[val_binary['subject_id'].isin(val_multilabel['subject_id'])]
    test_binary = test_binary[test_binary['subject_id'].isin(test_multilabel['subject_id'])]
    
    print(f"数据切分完成: 训练{len(train_multilabel)}, 验证{len(val_multilabel)}, 测试{len(test_multilabel)}")
    
    # 3. 创建数据集和数据加载器
    print("\n--- 创建多标签数据集 ---")
    train_dataset = MultiLabelECGDataset(train_multilabel, train_binary, label_columns, augment=True)
    val_dataset = MultiLabelECGDataset(val_multilabel, val_binary, label_columns, augment=False)
    test_dataset = MultiLabelECGDataset(test_multilabel, test_binary, label_columns, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                             shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                           shuffle=False, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, 
                            shuffle=False, num_workers=0, drop_last=False)
    
    # 4. 创建多标签模型
    print("\n--- 创建多标签PLRNN模型 ---")
    model = MultiLabelPLRNNECGClassifier(num_labels).to(config.DEVICE)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"多标签数量: {num_labels}")
    
    # 5. 设置优化器和损失函数（多标签版本）
    criterion = nn.BCEWithLogitsLoss()  # 多标签二分类损失
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)  # 监控loss而非accuracy
    
    # 6. 训练循环
    print(f"\n--- 开始多标签训练 ({config.EPOCHS} epochs) ---")
    best_val_loss = float('inf')
    train_history = {'loss': [], 'hamming': [], 'subset_acc': []}
    val_history = {'loss': [], 'hamming': [], 'subset_acc': [], 'macro_f1': [], 'micro_f1': []}
    
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        
        # 训练
        train_loss, train_hamming, train_subset_acc = train_epoch_multilabel(
            model, train_loader, criterion, optimizer, config.DEVICE)
        train_history['loss'].append(train_loss)
        train_history['hamming'].append(train_hamming)
        train_history['subset_acc'].append(train_subset_acc)
        
        # 验证
        val_loss, val_hamming, val_subset_acc, val_macro_f1, val_micro_f1, _, _, _ = evaluate_multilabel(
            model, val_loader, criterion, config.DEVICE)
        val_history['loss'].append(val_loss)
        val_history['hamming'].append(val_hamming)
        val_history['subset_acc'].append(val_subset_acc)
        val_history['macro_f1'].append(val_macro_f1)
        val_history['micro_f1'].append(val_micro_f1)
        
        print(f"训练 - Loss: {train_loss:.4f}, Hamming: {train_hamming:.4f}, Subset Acc: {train_subset_acc:.4f}")
        print(f"验证 - Loss: {val_loss:.4f}, Hamming: {val_hamming:.4f}, Subset Acc: {val_subset_acc:.4f}, Macro F1: {val_macro_f1:.4f}")
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'pytorch_plrnn_multilabel_best_model.pth')
            print(f"✅ 保存最佳模型 (验证损失: {val_loss:.4f})")
        
        # 垃圾回收（Mac M4优化）
        if epoch % 5 == 0:
            gc.collect()
    
    # 7. 测试评估
    print("\n--- 多标签模型测试 ---")
    # 加载最佳模型
    model.load_state_dict(torch.load('pytorch_plrnn_multilabel_best_model.pth'))
    test_loss, test_hamming, test_subset_acc, test_macro_f1, test_micro_f1, test_preds, test_labels, test_probs = evaluate_multilabel(
        model, test_loader, criterion, config.DEVICE)
    
    print(f"测试结果:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Hamming Loss: {test_hamming:.4f}")
    print(f"  Subset Accuracy: {test_subset_acc:.4f}")
    print(f"  Macro F1: {test_macro_f1:.4f}")  
    print(f"  Micro F1: {test_micro_f1:.4f}")
    
    # 详细分类报告
    print("\n--- 每个标签的详细指标 ---")
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, test_preds, average=None, zero_division=0)
    
    label_report = []
    for i, label in enumerate(label_columns):
        if support[i] > 0:  # 只显示有样本的标签
            label_report.append({
                'Label': label,
                'Precision': f'{precision[i]:.3f}',
                'Recall': f'{recall[i]:.3f}',
                'F1-Score': f'{f1[i]:.3f}',
                'Support': int(support[i])
            })
    
    # 按支持度排序并显示前15个
    label_report.sort(key=lambda x: x['Support'], reverse=True)
    print("\n前15个标签的性能指标:")
    for i, report in enumerate(label_report[:15]):
        print(f"{i+1:2d}. {report['Label']:30s} | P:{report['Precision']} R:{report['Recall']} F1:{report['F1-Score']} Sup:{report['Support']:3d}")
    
    # 8. 可视化结果
    plt.figure(figsize=(20, 12))
    
    # 训练曲线 - Loss
    plt.subplot(3, 4, 1)
    plt.plot(train_history['loss'], label='Training Loss')
    plt.plot(val_history['loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Hamming Loss
    plt.subplot(3, 4, 2)
    plt.plot(train_history['hamming'], label='Training Hamming Loss')
    plt.plot(val_history['hamming'], label='Validation Hamming Loss')
    plt.title('Hamming Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Hamming Loss')
    plt.legend()
    plt.grid(True)
    
    # Subset Accuracy
    plt.subplot(3, 4, 3)
    plt.plot(train_history['subset_acc'], label='Training Subset Acc')
    plt.plot(val_history['subset_acc'], label='Validation Subset Acc')
    plt.title('Subset Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # F1 Scores
    plt.subplot(3, 4, 4)
    plt.plot(val_history['macro_f1'], label='Macro F1')
    plt.plot(val_history['micro_f1'], label='Micro F1')
    plt.title('F1 Scores')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    # 标签分布 (前10个最频繁的标签)
    plt.subplot(3, 4, 5)
    top_labels = sorted(config_info['label_frequencies'].items(), key=lambda x: x[1], reverse=True)[:10]
    labels_names = [label[0].replace('_', '\n')[:15] for label in top_labels]
    labels_counts = [label[1] for label in top_labels]
    plt.bar(range(len(labels_names)), labels_counts, color='skyblue')
    plt.title('Top 10 Label Frequencies')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.xticks(range(len(labels_names)), labels_names, rotation=45, ha='right')
    plt.tight_layout()
    
    # 预测分布直方图
    plt.subplot(3, 4, 6)
    pred_counts = np.sum(test_preds, axis=0)
    true_counts = np.sum(test_labels, axis=0)
    
    x = np.arange(len(label_columns))
    plt.bar(x - 0.2, true_counts, 0.4, label='True', alpha=0.8)
    plt.bar(x + 0.2, pred_counts, 0.4, label='Predicted', alpha=0.8)
    plt.title('True vs Predicted Label Counts')
    plt.xlabel('Label Index')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 性能指标总结
    plt.subplot(3, 4, 7)
    metrics = ['Subset Acc', 'Macro F1', 'Micro F1', 'Hamming Loss']
    values = [test_subset_acc, test_macro_f1, test_micro_f1, test_hamming]
    colors = ['green', 'blue', 'orange', 'red']
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.title('Test Performance Summary')
    plt.ylabel('Score')
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    plt.ylim(0, max(values) * 1.2)
    
    # 多标签共现矩阵（前10个标签）
    plt.subplot(3, 4, 8)
    top_10_indices = [label_columns.index(label[0]) for label in top_labels if label[0] in label_columns][:10]
    if len(top_10_indices) >= 5:  # 确保有足够的标签
        cooccurrence_matrix = np.dot(test_labels[:, top_10_indices].T, test_labels[:, top_10_indices])
        top_10_names = [top_labels[i][0].replace('_', '\n')[:10] for i in range(len(top_10_indices))]
        sns.heatmap(cooccurrence_matrix, annot=True, fmt='.0f', cmap='Blues', 
                   xticklabels=top_10_names, yticklabels=top_10_names)
        plt.title('Label Co-occurrence Matrix')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
    
    # 额外的4个子图用于更多分析
    # 预测概率分布
    plt.subplot(3, 4, 9)
    plt.hist(test_probs.flatten(), bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=config.CLASSIFICATION_THRESHOLD, color='red', linestyle='--', 
               label=f'Threshold={config.CLASSIFICATION_THRESHOLD}')
    plt.title('Prediction Probability Distribution')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 每个样本的标签数量分布
    plt.subplot(3, 4, 10)
    true_label_counts = np.sum(test_labels, axis=1)
    pred_label_counts = np.sum(test_preds, axis=1)
    
    plt.hist(true_label_counts, bins=15, alpha=0.5, label='True', color='blue')
    plt.hist(pred_label_counts, bins=15, alpha=0.5, label='Predicted', color='red')
    plt.title('Labels per Sample Distribution')
    plt.xlabel('Number of Labels')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 精确率-召回率曲线（前5个主要标签）
    plt.subplot(3, 4, 11)
    from sklearn.metrics import precision_recall_curve, auc
    for i, (label_name, _) in enumerate(top_labels[:5]):
        if label_name in label_columns:
            label_idx = label_columns.index(label_name)
            precision_curve, recall_curve, _ = precision_recall_curve(
                test_labels[:, label_idx], test_probs[:, label_idx])
            auc_score = auc(recall_curve, precision_curve)
            plt.plot(recall_curve, precision_curve, 
                    label=f'{label_name[:15]}... (AUC={auc_score:.3f})')
    
    plt.title('Precision-Recall Curves (Top 5 Labels)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 训练过程中各指标的变化
    plt.subplot(3, 4, 12)
    epochs = range(1, len(val_history['loss']) + 1)
    plt.plot(epochs, val_history['loss'], 'b-', label='Val Loss')
    plt.plot(epochs, val_history['hamming'], 'r-', label='Val Hamming')
    plt.plot(epochs, val_history['subset_acc'], 'g-', label='Val Subset Acc')
    plt.plot(epochs, val_history['macro_f1'], 'm-', label='Val Macro F1')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pytorch_plrnn_multilabel_results.png', dpi=300, bbox_inches='tight')
    print("\n结果图保存为: pytorch_plrnn_multilabel_results.png")
    
    # 9. 保存训练配置和结果
    results = {
        'config': {
            'sequence_length': config.SEQUENCE_LENGTH,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'epochs': config.EPOCHS,
            'device': str(config.DEVICE),
            'classification_threshold': config.CLASSIFICATION_THRESHOLD
        },
        'results': {
            'test_loss': test_loss,
            'test_hamming_loss': test_hamming,
            'test_subset_accuracy': test_subset_acc,
            'test_macro_f1': test_macro_f1,
            'test_micro_f1': test_micro_f1,
            'best_val_loss': best_val_loss
        },
        'model_info': {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'num_labels': num_labels
        },
        'dataset_info': {
            'train_samples': len(train_multilabel),
            'val_samples': len(val_multilabel),
            'test_samples': len(test_multilabel),
            'label_columns': label_columns
        }
    }
    
    with open('pytorch_plrnn_multilabel_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ 多标签训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"测试Hamming损失: {test_hamming:.4f}")
    print(f"测试子集准确率: {test_subset_acc:.4f}")
    print(f"测试宏F1分数: {test_macro_f1:.4f}")
    print(f"测试微F1分数: {test_micro_f1:.4f}")


if __name__ == "__main__":
    main()