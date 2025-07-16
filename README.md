# Master-Thesis
硕士论文
# ECG心电图5分类深度学习项目

## 项目简介

本项目基于MIMIC-IV-ECG数据集，使用深度学习技术实现心电图的5分类任务，包括：房颤、心动过缓、束支传导阻滞、正常心律和心动过速的自动识别。

### 核心特点
- 🏥 **医学特征工程**：提取心率、心率变异性等8个核心医学特征
- 🧠 **LSTM时序建模**：捕捉心电图的时序依赖关系
- ⚖️ **类别平衡处理**：解决严重的数据不平衡问题
- 🔧 **数据增强**：轻量级噪声和幅度增强提升泛化能力
- 💻 **硬件兼容**：专门优化用于Apple Silicon (M4)芯片

## 目录结构

```
ECG-Classification/
├── too_feature.py              # 主训练脚本
├── ecg_5_class_data.csv       # 标签数据文件
├── ecg_stable_lstm_model.keras # 训练好的模型
├── README.md                   # 项目文档
└── mimic-iv-ecg/              # ECG波形数据目录
```

## 环境要求

### 硬件要求
- **推荐**：Apple Silicon (M1/M2/M3/M4) Mac
- **内存**：至少8GB RAM
- **存储**：至少20GB可用空间

### 软件环境
- Python 3.10/3.11 (注意：不支持Python 3.13)
- TensorFlow 2.12+ (Apple Silicon优化版本)
- 其他依赖见requirements.txt

### 环境安装

```bash
# 创建conda环境
conda create -n tf_final python=3.11
conda activate tf_final

# 安装TensorFlow (Apple Silicon)
pip install tensorflow-macos tensorflow-metal

# 安装其他依赖
pip install pandas numpy scipy scikit-learn wfdb tqdm
```

## 完整流程详解

### 1. 数据预处理阶段

#### 1.1 数据加载与验证
```python
# 加载标签文件
full_df = pd.read_csv('ecg_5_class_data.csv', header=None, 
                     names=['subject_id', 'waveform_path', 'ecg_category'])
```

**重点**：
- 数据包含366,301条记录
- 5个类别：房颤(240,717)、心动过速(60,809)、心动过缓(32,508)、正常(21,950)、束支传导阻滞(10,317)
- 严重的类别不平衡问题

#### 1.2 数据集切分策略
```python
# 按患者ID进行切分，避免数据泄露
all_subjects = full_df['subject_id'].unique()
train_val_subjects, test_subjects = train_test_split(all_subjects, test_size=0.15)
train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=0.15)
```

**重点**：
- **患者级别切分**：确保同一患者的数据不会同时出现在训练集和测试集
- **避免数据泄露**：这是医学AI项目的关键点

#### 1.3 平衡采样机制
```python
def balanced_sampling(df, target_samples, random_state=42):
    categories = df['ecg_category'].unique()
    samples_per_class = target_samples // len(categories)
    # 每个类别采样相同数量，不足时进行重复采样
```

**重点**：
- 解决类别不平衡：每类采样相同数量
- 重复采样：对少数类别进行有放回采样
- 最终数据：训练1500，验证300，测试400样本

### 2. 信号预处理阶段

#### 2.1 ECG信号预处理
```python
def stable_preprocess_ecg(raw_signal, target_length=500):
    # 1. 数据验证和类型转换
    # 2. 重采样到固定长度
    # 3. 通道级别标准化
    # 4. 异常值截断
```

**重点**：
- **固定长度**：500个时间点，便于批处理
- **12导联**：保留完整的心电图信息
- **Robust标准化**：每个导联独立标准化
- **异常值处理**：截断到[-3,3]范围

#### 2.2 数据增强策略
```python
def lightweight_augmentation(signal):
    # 30%概率添加高斯噪声
    # 20%概率进行幅度缩放
```

**重点**：
- **轻量级设计**：避免过度增强影响医学特征
- **真实性保持**：模拟临床环境中的自然变异
- **仅训练时应用**：验证和测试不使用增强

### 3. 医学特征工程

#### 3.1 核心医学特征提取
```python
def extract_core_medical_features(signal, fs=100):
    # 统计特征：均值、标准差、偏度、峰度
    # 心率特征：心率、心率变异性、RMSSD、变异系数
```

**重点特征说明**：

| 特征名称 | 医学意义 | 正常范围 |
|---------|---------|---------|
| **心率(HR)** | 每分钟心跳次数 | 60-100 bpm |
| **SDNN** | RR间期标准差，反映整体心率变异性 | 20-50 ms |
| **RMSSD** | 相邻RR间期差值均方根，反映短期变异性 | 15-40 ms |
| **CV_RR** | RR间期变异系数，归一化的变异性指标 | 0.03-0.07 |

#### 3.2 QRS波检测算法
```python
peaks, _ = find_peaks(lead_ii, height=np.std(lead_ii)*0.5, distance=fs//4)
rr_intervals = np.diff(peaks) / fs
```

**重点**：
- **导联选择**：使用导联II进行R波检测
- **自适应阈值**：基于信号标准差的动态阈值
- **距离约束**：最小间距防止重复检测

### 4. 模型架构设计

#### 4.1 轻量级LSTM架构
```python
def create_lightweight_lstm_model():
    # 波形分支：LSTM(32) + Dense(24)
    # 特征分支：Dense(16)
    # 融合层：Concatenate + Dense(32) + Output(5)
```

**架构特点**：
- **总参数**：8,461个参数 (33.05 KB)
- **内存效率**：专为Apple Silicon优化
- **双分支设计**：波形特征+医学特征深度融合

#### 4.2 模型组件详解

```
输入层:
├── 波形输入: (None, 500, 12) - 500个时间点×12导联
└── 特征输入: (None, 8) - 8个医学特征

波形分支:
└── LSTM(32, dropout=0.2) → Dense(24) → BatchNorm → Dropout(0.3)

特征分支:
└── Dense(16) → BatchNorm → Dropout(0.2)

融合与输出:
└── Concatenate → Dense(32) → BatchNorm → Dropout(0.3) → Dense(5)
```

### 5. 训练策略

#### 5.1 优化器配置
```python
optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0005)
class_weight=class_weight_dict  # 平衡类别权重
```

#### 5.2 回调函数
```python
callbacks = [
    EarlyStopping(patience=5, monitor='val_accuracy'),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)
]
```

**重点**：
- **早停机制**：防止过拟合
- **学习率调度**：动态调整学习率
- **类别权重**：自动平衡类别重要性

### 6. 硬件优化 (Apple Silicon专用)

#### 6.1 GPU禁用配置
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_METAL_DEVICE_ENABLE'] = '0'
tf.config.set_visible_devices([], 'GPU')
```

**关键优化**：
- **强制CPU模式**：避免M4芯片Metal GPU兼容性问题
- **内存管理**：频繁垃圾回收防止内存泄露
- **批次大小**：优化为4，平衡性能和稳定性

### 7. 性能评估

#### 7.1 最终结果
- **准确率**: 32.5%
- **置信度**: 32.7% ± 8.7%
- **最佳类别**: Tachycardia (F1=0.42)

#### 7.2 医学特征统计
```
Heart Rate: 127±16 bpm (覆盖正常到异常范围)
HRV: 45±55 ms (显示心律变异性)
```

## 使用方法

### 快速开始
```bash
# 1. 准备数据
# 确保 ecg_5_class_data.csv 和 MIMIC-IV-ECG 数据在正确路径

# 2. 运行训练
python too_feature.py

# 3. 查看结果
# 模型保存为 ecg_stable_lstm_model.keras
# 训练日志显示详细分类报告
```

### 自定义配置
```python
# 调整训练参数
BATCH_SIZE = 4          # 批次大小
EPOCHS = 15             # 训练轮数
SEQUENCE_LENGTH = 500   # 序列长度
TRAIN_SAMPLES = 1500    # 训练样本数
```

## 核心优势

### 1. 医学领域专业性
- ✅ 提取临床相关的心率变异性特征
- ✅ 使用标准的RR间期分析方法
- ✅ 遵循心电图分析的医学标准

### 2. 技术创新性
- ✅ LSTM时序建模捕捉心律变化
- ✅ 双分支架构融合波形和特征信息
- ✅ 轻量级设计确保实际可部署性

### 3. 工程实用性
- ✅ 解决真实数据的类别不平衡问题
- ✅ 针对硬件限制进行深度优化
- ✅ 完整的错误处理和鲁棒性设计

## 问题与解决方案

### 常见问题

**Q1: SIGBUS错误**
```bash
A: 这是Apple Silicon兼容性问题
解决方案：
- 设置环境变量禁用Metal GPU
- 减少批次大小和模型复杂度
- 使用legacy优化器
```

**Q2: 准确率较低**
```bash
A: 5分类医学任务本身复杂
改进方向：
- 增加训练样本数量
- 提取更多医学特征
- 使用集成学习方法
```

**Q3: 内存不足**
```bash
A: 减少内存消耗
解决方案：
- 降低序列长度 (500→250)
- 减少批次大小 (4→2)
- 减少训练样本数
```

## 未来改进方向

### 1. 模型架构
- [ ] 注意力机制增强时序建模
- [ ] 多尺度CNN提取局部特征
- [ ] 图神经网络建模导联关系

### 2. 特征工程
- [ ] P波、T波形态学特征
- [ ] 频域功率谱分析
- [ ] ST段偏移检测

### 3. 数据策略
- [ ] 更复杂的数据增强
- [ ] 主动学习选择困难样本
- [ ] 迁移学习利用预训练模型

## 致谢

本项目基于MIMIC-IV-ECG数据集，感谢MIT实验室的开源贡献。特别感谢在Apple Silicon兼容性优化方面的技术探索。

---

# ECG 5-Class Classification Deep Learning Project

## Project Overview

This project implements a 5-class ECG classification system using the MIMIC-IV-ECG dataset, automatically identifying atrial fibrillation, bradycardia, bundle branch block, normal rhythm, and tachycardia.

### Key Features
- 🏥 **Medical Feature Engineering**: Extracts 8 core medical features including heart rate and HRV
- 🧠 **LSTM Temporal Modeling**: Captures sequential dependencies in ECG signals
- ⚖️ **Class Balancing**: Addresses severe data imbalance issues
- 🔧 **Data Augmentation**: Lightweight noise and amplitude augmentation
- 💻 **Hardware Compatibility**: Optimized for Apple Silicon (M4) chips

## Directory Structure

```
ECG-Classification/
├── too_feature.py              # Main training script
├── ecg_5_class_data.csv       # Label data file
├── ecg_stable_lstm_model.keras # Trained model
├── README.md                   # Project documentation
└── mimic-iv-ecg/              # ECG waveform data directory
```

## Requirements

### Hardware Requirements
- **Recommended**: Apple Silicon (M1/M2/M3/M4) Mac
- **Memory**: At least 8GB RAM
- **Storage**: At least 20GB available space

### Software Environment
- Python 3.10/3.11 (Note: Python 3.13 not supported)
- TensorFlow 2.12+ (Apple Silicon optimized)
- See requirements.txt for other dependencies

### Environment Setup

```bash
# Create conda environment
conda create -n tf_final python=3.11
conda activate tf_final

# Install TensorFlow (Apple Silicon)
pip install tensorflow-macos tensorflow-metal

# Install other dependencies
pip install pandas numpy scipy scikit-learn wfdb tqdm
```

## Complete Workflow

### 1. Data Preprocessing Stage

#### 1.1 Data Loading and Validation
```python
# Load label file
full_df = pd.read_csv('ecg_5_class_data.csv', header=None, 
                     names=['subject_id', 'waveform_path', 'ecg_category'])
```

**Key Points**:
- Dataset contains 366,301 records
- 5 classes: AF(240,717), Tachycardia(60,809), Bradycardia(32,508), Normal(21,950), BBB(10,317)
- Severe class imbalance issue

#### 1.2 Dataset Splitting Strategy
```python
# Split by patient ID to avoid data leakage
all_subjects = full_df['subject_id'].unique()
train_val_subjects, test_subjects = train_test_split(all_subjects, test_size=0.15)
train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=0.15)
```

**Key Points**:
- **Patient-level splitting**: Ensures data from same patient doesn't appear in both train and test
- **Prevent data leakage**: Critical for medical AI projects

#### 1.3 Balanced Sampling Mechanism
```python
def balanced_sampling(df, target_samples, random_state=42):
    categories = df['ecg_category'].unique()
    samples_per_class = target_samples // len(categories)
    # Sample same number for each class, use replacement for minority classes
```

**Key Points**:
- Solves class imbalance: Equal samples per class
- Oversampling: With replacement for minority classes
- Final data: 1500 train, 300 validation, 400 test samples

### 2. Signal Preprocessing Stage

#### 2.1 ECG Signal Preprocessing
```python
def stable_preprocess_ecg(raw_signal, target_length=500):
    # 1. Data validation and type conversion
    # 2. Resample to fixed length
    # 3. Channel-wise normalization
    # 4. Outlier clipping
```

**Key Points**:
- **Fixed length**: 500 time points for batch processing
- **12-lead**: Preserves complete ECG information
- **Robust normalization**: Independent normalization per lead
- **Outlier handling**: Clip to [-3,3] range

#### 2.2 Data Augmentation Strategy
```python
def lightweight_augmentation(signal):
    # 30% probability Gaussian noise
    # 20% probability amplitude scaling
```

**Key Points**:
- **Lightweight design**: Avoids over-augmentation affecting medical features
- **Realism preservation**: Simulates natural variations in clinical environment
- **Training only**: No augmentation during validation/testing

### 3. Medical Feature Engineering

#### 3.1 Core Medical Feature Extraction
```python
def extract_core_medical_features(signal, fs=100):
    # Statistical features: mean, std, skewness, kurtosis
    # Heart rate features: HR, HRV, RMSSD, CV_RR
```

**Key Feature Descriptions**:

| Feature Name | Medical Significance | Normal Range |
|-------------|---------------------|--------------|
| **Heart Rate (HR)** | Beats per minute | 60-100 bpm |
| **SDNN** | RR interval standard deviation, overall HRV | 20-50 ms |
| **RMSSD** | Root mean square of successive RR differences | 15-40 ms |
| **CV_RR** | Coefficient of variation of RR intervals | 0.03-0.07 |

#### 3.2 QRS Detection Algorithm
```python
peaks, _ = find_peaks(lead_ii, height=np.std(lead_ii)*0.5, distance=fs//4)
rr_intervals = np.diff(peaks) / fs
```

**Key Points**:
- **Lead selection**: Uses Lead II for R-wave detection
- **Adaptive threshold**: Dynamic threshold based on signal std
- **Distance constraint**: Minimum distance prevents duplicate detection

### 4. Model Architecture Design

#### 4.1 Lightweight LSTM Architecture
```python
def create_lightweight_lstm_model():
    # Waveform branch: LSTM(32) + Dense(24)
    # Feature branch: Dense(16)
    # Fusion layer: Concatenate + Dense(32) + Output(5)
```

**Architecture Features**:
- **Total parameters**: 8,461 parameters (33.05 KB)
- **Memory efficient**: Optimized for Apple Silicon
- **Dual-branch design**: Deep fusion of waveform and medical features

#### 4.2 Model Component Details

```
Input Layers:
├── Waveform Input: (None, 500, 12) - 500 timepoints × 12 leads
└── Feature Input: (None, 8) - 8 medical features

Waveform Branch:
└── LSTM(32, dropout=0.2) → Dense(24) → BatchNorm → Dropout(0.3)

Feature Branch:
└── Dense(16) → BatchNorm → Dropout(0.2)

Fusion & Output:
└── Concatenate → Dense(32) → BatchNorm → Dropout(0.3) → Dense(5)
```

### 5. Training Strategy

#### 5.1 Optimizer Configuration
```python
optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0005)
class_weight=class_weight_dict  # Balanced class weights
```

#### 5.2 Callbacks
```python
callbacks = [
    EarlyStopping(patience=5, monitor='val_accuracy'),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)
]
```

**Key Points**:
- **Early stopping**: Prevents overfitting
- **Learning rate scheduling**: Dynamic learning rate adjustment
- **Class weights**: Automatic class importance balancing

### 6. Hardware Optimization (Apple Silicon Specific)

#### 6.1 GPU Disable Configuration
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_METAL_DEVICE_ENABLE'] = '0'
tf.config.set_visible_devices([], 'GPU')
```

**Critical Optimizations**:
- **Force CPU mode**: Avoids M4 chip Metal GPU compatibility issues
- **Memory management**: Frequent garbage collection prevents memory leaks
- **Batch size**: Optimized to 4, balancing performance and stability

### 7. Performance Evaluation

#### 7.1 Final Results
- **Accuracy**: 32.5%
- **Confidence**: 32.7% ± 8.7%
- **Best class**: Tachycardia (F1=0.42)

#### 7.2 Medical Feature Statistics
```
Heart Rate: 127±16 bpm (covers normal to abnormal range)
HRV: 45±55 ms (shows heart rhythm variability)
```

## Usage

### Quick Start
```bash
# 1. Prepare data
# Ensure ecg_5_class_data.csv and MIMIC-IV-ECG data are in correct paths

# 2. Run training
python too_feature.py

# 3. View results
# Model saved as ecg_stable_lstm_model.keras
# Training log shows detailed classification report
```

### Custom Configuration
```python
# Adjust training parameters
BATCH_SIZE = 4          # Batch size
EPOCHS = 15             # Training epochs
SEQUENCE_LENGTH = 500   # Sequence length
TRAIN_SAMPLES = 1500    # Training samples
```

## Core Advantages

### 1. Medical Domain Expertise
- ✅ Extracts clinically relevant heart rate variability features
- ✅ Uses standard RR interval analysis methods
- ✅ Follows medical standards for ECG analysis

### 2. Technical Innovation
- ✅ LSTM temporal modeling captures rhythm changes
- ✅ Dual-branch architecture fuses waveform and feature information
- ✅ Lightweight design ensures practical deployability

### 3. Engineering Practicality
- ✅ Addresses real-world class imbalance issues
- ✅ Deep optimization for hardware constraints
- ✅ Complete error handling and robustness design

## Issues and Solutions

### Common Issues

**Q1: SIGBUS Error**
```bash
A: Apple Silicon compatibility issue
Solution:
- Set environment variables to disable Metal GPU
- Reduce batch size and model complexity
- Use legacy optimizer
```

**Q2: Low Accuracy**
```bash
A: 5-class medical task is inherently complex
Improvement directions:
- Increase training sample size
- Extract more medical features
- Use ensemble learning methods
```

**Q3: Memory Issues**
```bash
A: Reduce memory consumption
Solutions:
- Reduce sequence length (500→250)
- Reduce batch size (4→2)
- Reduce training samples
```

## Future Improvements

### 1. Model Architecture
- [ ] Attention mechanism for enhanced temporal modeling
- [ ] Multi-scale CNN for local feature extraction
- [ ] Graph neural networks for lead relationship modeling

### 2. Feature Engineering
- [ ] P-wave and T-wave morphological features
- [ ] Frequency domain power spectral analysis
- [ ] ST-segment deviation detection

### 3. Data Strategy
- [ ] More complex data augmentation
- [ ] Active learning for difficult sample selection
- [ ] Transfer learning with pre-trained models

## Acknowledgments

This project is based on the MIMIC-IV-ECG dataset, thanks to MIT Lab's open-source contribution. Special thanks for the technical exploration in Apple Silicon compatibility optimization.