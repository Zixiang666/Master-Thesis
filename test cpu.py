# ==========================================================
# PyTorch版 LSTM ECG分类脚本 - 医学特征工程 + 轻量级架构
# ==========================================================
import os
import gc
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- PyTorch 设备选择 ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("--- 正在使用 NVIDIA CUDA GPU ---")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("--- 正在使用 Apple Metal (MPS) GPU ---")
else:
    device = torch.device("cpu")
    print("--- 切换到 CPU 模式运行 ---")

import pandas as pd
import numpy as np
import wfdb
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import math
import time  # 用于计时

# ===================================================================
# 配置参数 - 与原脚本保持一致
# ===================================================================
LABELED_DATA_CSV = 'ecg_5_class_data.csv'
ECG_BASE_PATH = '/Users/zixiang/ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/'

BATCH_SIZE = 4
EPOCHS = 15
TRAIN_SAMPLES = 1500
VAL_SAMPLES = 300
TEST_SAMPLES = 400
SEQUENCE_LENGTH = 500
SAMPLING_RATE = 100

# ===================================================================
# 阶段1: 数据加载、切分与平衡采样 (此部分与框架无关，保持不变)
# ===================================================================
print("--- 阶段1: 数据加载与平衡采样 ---")
# ... (从原脚本中复制所有数据加载和平衡采样的代码)
# 加载数据
try:
    full_df = pd.read_csv(LABELED_DATA_CSV, header=None,
                          names=['subject_id', 'waveform_path', 'ecg_category'])
    full_df.dropna(inplace=True)
    print(f"成功加载 {len(full_df)} 条记录")
except Exception as e:
    print(f"数据加载错误: {e}")
    sys.exit(1)

# 数据集切分
all_subjects = full_df['subject_id'].unique()
train_val_subjects, test_subjects = train_test_split(all_subjects, test_size=0.15, random_state=42)
train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=0.15, random_state=42)
train_df = full_df[full_df['subject_id'].isin(train_subjects)].reset_index(drop=True)
val_df = full_df[full_df['subject_id'].isin(val_subjects)].reset_index(drop=True)
test_df = full_df[full_df['subject_id'].isin(test_subjects)].reset_index(drop=True)


# 平衡采样函数
def balanced_sampling(df, target_samples, random_state=42):
    categories = df['ecg_category'].unique()
    samples_per_class = target_samples // len(categories)
    balanced_dfs = []
    for category in categories:
        category_df = df[df['ecg_category'] == category]
        if len(category_df) >= samples_per_class:
            sampled_df = category_df.sample(n=samples_per_class, random_state=random_state)
        else:
            sampled_df = category_df.sample(n=samples_per_class, replace=True, random_state=random_state)
        balanced_dfs.append(sampled_df)
    result_df = pd.concat(balanced_dfs, ignore_index=True)
    result_df = result_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return result_df


train_subset_df = balanced_sampling(train_df, TRAIN_SAMPLES, random_state=42)
val_subset_df = balanced_sampling(val_df, VAL_SAMPLES, random_state=42)
test_subset_df = balanced_sampling(test_df, TEST_SAMPLES, random_state=42)
print(
    f"\n平衡采样后的数据集:\n训练集: {len(train_subset_df)} | 验证集: {len(val_subset_df)} | 测试集: {len(test_subset_df)}")


# ===================================================================
# 辅助函数: 数据增强、特征提取、预处理 (与框架无关，保持不变)
# ===================================================================
def lightweight_augmentation(signal):
    # ... (从原脚本复制)
    try:
        if np.random.random() < 0.3:
            noise_level = np.random.uniform(0.01, 0.03)
            noise = np.random.normal(0, noise_level, signal.shape)
            signal = signal + noise
        if np.random.random() < 0.2:
            scale = np.random.uniform(0.9, 1.1)
            signal = signal * scale
        return np.clip(signal, -3, 3)
    except:
        return signal


def extract_core_medical_features(signal, fs=100):
    # ... (从原脚本复制)
    try:
        if signal is None or len(signal) == 0: return np.zeros(8)
        lead_ii = signal[:, 1] if signal.shape[1] > 1 else signal[:, 0]
        features = [np.mean(lead_ii), np.std(lead_ii), skew(lead_ii), kurtosis(lead_ii)]
        try:
            peaks, _ = find_peaks(lead_ii, height=np.std(lead_ii) * 0.5, distance=fs // 4)
            if len(peaks) >= 2:
                rr_intervals = np.diff(peaks) / fs
                features.extend([60 / np.mean(rr_intervals), np.std(rr_intervals) * 1000,
                                 np.sqrt(np.mean(np.diff(rr_intervals) ** 2)) * 1000,
                                 np.std(rr_intervals) / np.mean(rr_intervals)])
            else:
                features.extend([60, 0, 0, 0])
        except:
            features.extend([60, 0, 0, 0])
        features = features[:8];
        while len(features) < 8: features.append(0)
        return np.array([float(f) if np.isfinite(f) else 0.0 for f in features])
    except:
        return np.zeros(8)


def stable_preprocess_ecg(raw_signal, target_length=500):
    # ... (从原脚本复制)
    try:
        if raw_signal is None or len(raw_signal) == 0: return None
        signal = np.array(raw_signal, dtype=np.float64)
        if len(signal.shape) != 2 or signal.shape[1] != 12 or not np.isfinite(signal).all(): return None
        if len(signal) != target_length:
            old_indices = np.linspace(0, len(signal) - 1, len(signal))
            new_indices = np.linspace(0, len(signal) - 1, target_length)
            resampled_signal = np.zeros((target_length, 12), dtype=np.float32)
            for i in range(12):
                if len(signal) > 1:
                    f = interp1d(old_indices, signal[:, i], kind='linear', fill_value='extrapolate')
                    resampled_signal[:, i] = f(new_indices)
                else:
                    resampled_signal[:, i] = signal[0, i]
        else:
            resampled_signal = signal.astype(np.float32)
        for i in range(12):
            channel = resampled_signal[:, i]
            std_val = np.std(channel)
            if std_val > 1e-10:
                resampled_signal[:, i] = (channel - np.mean(channel)) / std_val
            else:
                resampled_signal[:, i] = 0
        resampled_signal = np.clip(resampled_signal, -3, 3)
        return resampled_signal if np.isfinite(resampled_signal).all() else None
    except:
        return None


# ===================================================================
# PyTorch 数据集类 (替代Keras Sequence)
# ===================================================================
class ECGDataset(Dataset):
    def __init__(self, df, label_map, augment=False):
        self.df = df
        self.label_map = label_map
        self.augment = augment
        print(f"PyTorch 数据集: {len(df)} 样本, 增强={augment}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        while True:  # 循环直到找到有效样本
            try:
                full_path = os.path.join(ECG_BASE_PATH, os.path.splitext(row['waveform_path'])[0])
                record = wfdb.rdrecord(full_path)

                if record.p_signal is None:
                    # 如果当前样本无效，尝试加载下一个
                    index = (index + 1) % len(self.df)
                    row = self.df.iloc[index]
                    continue

                processed_signal = stable_preprocess_ecg(record.p_signal, SEQUENCE_LENGTH)
                if processed_signal is None:
                    index = (index + 1) % len(self.df)
                    row = self.df.iloc[index]
                    continue

                if self.augment:
                    processed_signal = lightweight_augmentation(processed_signal)

                features = extract_core_medical_features(processed_signal, SAMPLING_RATE)

                # PyTorch需要 (Channels, Sequence) 或 (Sequence, Channels)
                # LSTM期望 (Sequence, Features)，所以我们保持 (SEQUENCE_LENGTH, 12)
                waveform_tensor = torch.tensor(processed_signal, dtype=torch.float32)
                features_tensor = torch.tensor(features, dtype=torch.float32)

                label = self.label_map[row['ecg_category']]
                label_tensor = torch.tensor(label, dtype=torch.long)  # CrossEntropyLoss需要long类型的标签索引

                return (waveform_tensor, features_tensor), label_tensor

            except Exception:
                # 出现任何异常，都尝试加载下一个样本
                index = (index + 1) % len(self.df)
                row = self.df.iloc[index]
                continue


# ===================================================================
# PyTorch 轻量级LSTM模型 (替代Keras Model)
# ===================================================================
class LightweightLSTMModel(nn.Module):
    def __init__(self, sequence_length, num_channels, num_features, num_classes):
        super(LightweightLSTMModel, self).__init__()

        # 波形分支
        # Keras的recurrent_dropout在PyTorch中没有直接对应，这里只用标准dropout
        self.lstm = nn.LSTM(input_size=num_channels, hidden_size=32, batch_first=True, dropout=0.2)
        self.waveform_dense = nn.Linear(32, 24)
        self.waveform_bn = nn.BatchNorm1d(24)
        self.waveform_dropout = nn.Dropout(0.3)

        # 特征分支
        self.feature_dense = nn.Linear(num_features, 16)
        self.feature_bn = nn.BatchNorm1d(16)
        self.feature_dropout = nn.Dropout(0.2)

        # 合并后的输出层
        self.concat_dense = nn.Linear(24 + 16, 32)
        self.concat_bn = nn.BatchNorm1d(32)
        self.concat_dropout = nn.Dropout(0.3)

        self.output_dense = nn.Linear(32, num_classes)

        self.relu = nn.ReLU()

    def forward(self, waveform_input, feature_input):
        # 波形分支
        # LSTM输出 (batch, seq_len, hidden_size), 我们取最后一个时间步的输出
        lstm_out, (h_n, c_n) = self.lstm(waveform_input)
        lstm_out_last = lstm_out[:, -1, :]

        x_wave = self.waveform_dense(lstm_out_last)
        x_wave = self.waveform_bn(x_wave)
        x_wave = self.relu(x_wave)
        x_wave = self.waveform_dropout(x_wave)

        # 特征分支
        x_feat = self.feature_dense(feature_input)
        x_feat = self.feature_bn(x_feat)
        x_feat = self.relu(x_feat)
        x_feat = self.feature_dropout(x_feat)

        # 合并
        concatenated = torch.cat((x_wave, x_feat), dim=1)

        # 输出层
        x = self.concat_dense(concatenated)
        x = self.concat_bn(x)
        x = self.relu(x)
        x = self.concat_dropout(x)

        # 输出原始Logits，nn.CrossEntropyLoss会处理softmax
        output = self.output_dense(x)

        return output


# 主训练流程
# ===================================================================
def main():
    print("\n--- 阶段2: PyTorch LSTM模型构建 ---")

    # 准备标签
    labels = sorted(full_df['ecg_category'].unique())
    label_map = {label: i for i, label in enumerate(labels)}
    num_classes = len(labels)
    # 计算类别权重
    train_labels = [label_map[cat] for cat in train_subset_df['ecg_category']]
    class_weights_np = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
    print(f"类别权重: {class_weights}")

    # 创建数据集和数据加载器
    train_dataset = ECGDataset(train_subset_df, label_map, augment=True)
    val_dataset = ECGDataset(val_subset_df, label_map, augment=False)
    test_dataset = ECGDataset(test_subset_df, label_map, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 创建模型
    model = LightweightLSTMModel(
        sequence_length=SEQUENCE_LENGTH,
        num_channels=12,
        num_features=8,
        num_classes=num_classes
    ).to(device)

    print("\nPyTorch LSTM模型架构:")
    print(model)

    # 定义损失函数、优化器和学习率调度器
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    # ===================================================================
    # 训练和评估
    # ===================================================================
    print(f"\n--- 阶段3: 开始 PyTorch LSTM训练 ({EPOCHS} epochs) ---")

    best_val_accuracy = 0
    patience_counter = 0
    patience = 5  # EarlyStopping的耐心值
    best_model_weights = None

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_train_loss = 0

        for i, (inputs, targets) in enumerate(train_loader):
            waveform_data, feature_data = inputs
            waveform_data, feature_data, targets = waveform_data.to(device), feature_data.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(waveform_data, feature_data)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            if i % 50 == 0:  # 每50个batch打印一次进度
                print(f"Epoch {epoch + 1}/{EPOCHS}, Batch {i + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # 验证
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                waveform_data, feature_data = inputs
                waveform_data, feature_data, targets = waveform_data.to(device), feature_data.to(device), targets.to(
                    device)

                outputs = model(waveform_data, feature_data)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val

        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch + 1}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Accuracy: {val_accuracy:.2f}% | "
              f"Duration: {epoch_duration:.2f}s")

        scheduler.step(avg_val_loss)  # 更新学习率

        # Early Stopping 逻辑
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            best_model_weights = model.state_dict()
            print(f"*** New best validation accuracy: {best_val_accuracy:.2f}%. Saving model state. ***")
        else:
            patience_counter += 1
            print(f"Validation accuracy did not improve. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("--- Early stopping triggered ---")
            break

        gc.collect()

    # 加载最佳模型权重
    if best_model_weights:
        model.load_state_dict(best_model_weights)
    # ===================================================================
    # 评估模型
    # ===================================================================
    print("\n--- 阶段4: PyTorch LSTM模型评估 ---")
    model.eval()
    y_true = []
    y_pred = []
    y_pred_proba = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            waveform_data, feature_data = inputs
            waveform_data, feature_data = waveform_data.to(device), feature_data.to(device)

            outputs = model(waveform_data, feature_data)

            # 计算概率（应用softmax）
            probabilities = torch.softmax(outputs, dim=1)
            max_probs, predicted = torch.max(probabilities, 1)

            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_pred_proba.extend(max_probs.cpu().numpy())

    print("\nPyTorch LSTM分类报告:")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

    print("\n混淆矩阵:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    print(f"\n预测置信度统计:")
    print(f"平均置信度: {np.mean(y_pred_proba):.4f}")
    print(f"置信度标准差: {np.std(y_pred_proba):.4f}")

    # 保存模型
    torch.save(model.state_dict(), 'ecg_stable_lstm_model.pth')
    print("\nPyTorch模型状态字典已保存为 ecg_stable_lstm_model.pth")

    print("\n=== PyTorch版 LSTM训练程序完成 ===")

    print("\n=== PyTorch版 LSTM训练程序完成 ===")


# ===================================================================
# 启动主程序 (This is the crucial part)
# ===================================================================
if __name__ == '__main__':
    # 这一行是必须的，它保护了你的主程序入口点
    main()




