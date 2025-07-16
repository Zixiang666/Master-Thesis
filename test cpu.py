# ==========================================================
# 稳定版LSTM ECG分类脚本 - 医学特征工程 + 轻量级架构
# ==========================================================
import os
import gc
import sys

# GPU禁用设置
print("--- 强制切换到CPU模式运行 ---")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_METAL_DEVICE_ENABLE'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

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
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import (Input, Dense, Concatenate, Dropout,
                                     LSTM, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ===================================================================
# 配置参数 - 为稳定性优化
# ===================================================================
LABELED_DATA_CSV = 'ecg_5_class_data.csv'
ECG_BASE_PATH = '/Users/zixiang/ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/'

# 保守的训练参数
BATCH_SIZE = 4  # 进一步减少
EPOCHS = 15
TRAIN_SAMPLES = 1500  # 减少样本数
VAL_SAMPLES = 300
TEST_SAMPLES = 400
SEQUENCE_LENGTH = 500  # 减少序列长度
SAMPLING_RATE = 100

print("--- 阶段1: 数据加载与平衡采样 ---")

# 加载数据
try:
    full_df = pd.read_csv(LABELED_DATA_CSV, header=None,
                          names=['subject_id', 'waveform_path', 'ecg_category'])
    full_df.dropna(inplace=True)
    print(f"成功加载 {len(full_df)} 条记录")
except Exception as e:
    print(f"数据加载错误: {e}")
    sys.exit(1)

# 检查类别分布
print("\n原始数据集类别分布:")
class_counts = full_df['ecg_category'].value_counts()
print(class_counts)

# 数据集切分
all_subjects = full_df['subject_id'].unique()
train_val_subjects, test_subjects = train_test_split(all_subjects, test_size=0.15, random_state=42)
train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=0.15, random_state=42)

train_df = full_df[full_df['subject_id'].isin(train_subjects)].reset_index(drop=True)
val_df = full_df[full_df['subject_id'].isin(val_subjects)].reset_index(drop=True)
test_df = full_df[full_df['subject_id'].isin(test_subjects)].reset_index(drop=True)


# 平衡采样
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

print(f"\n平衡采样后的数据集:")
print(f"训练集: {len(train_subset_df)} 样本")
print(f"验证集: {len(val_subset_df)} 样本")
print(f"测试集: {len(test_subset_df)} 样本")


# ===================================================================
# 轻量级数据增强
# ===================================================================
def lightweight_augmentation(signal):
    """轻量级数据增强"""
    try:
        if np.random.random() < 0.3:  # 降低增强概率
            # 只添加轻微噪声
            noise_level = np.random.uniform(0.01, 0.03)
            noise = np.random.normal(0, noise_level, signal.shape)
            signal = signal + noise

        if np.random.random() < 0.2:  # 降低增强概率
            # 轻微幅度缩放
            scale = np.random.uniform(0.9, 1.1)
            signal = signal * scale

        return np.clip(signal, -3, 3)
    except:
        return signal


# ===================================================================
# 核心医学特征提取（简化版）
# ===================================================================
def extract_core_medical_features(signal, fs=100):
    """提取核心医学特征（8个特征）"""
    try:
        if signal is None or len(signal) == 0:
            return np.zeros(8)

        # 使用导联II
        lead_ii = signal[:, 1] if signal.shape[1] > 1 else signal[:, 0]
        features = []

        # 1. 基本统计特征（4个）
        features.append(np.mean(lead_ii))
        features.append(np.std(lead_ii))
        features.append(skew(lead_ii))
        features.append(kurtosis(lead_ii))

        # 2. 心率相关特征（4个）
        try:
            peaks, _ = find_peaks(lead_ii, height=np.std(lead_ii) * 0.5, distance=fs // 4)

            if len(peaks) >= 2:
                rr_intervals = np.diff(peaks) / fs
                heart_rate = 60 / np.mean(rr_intervals)
                hrv = np.std(rr_intervals) * 1000  # SDNN
                rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2)) * 1000
                cv_rr = np.std(rr_intervals) / np.mean(rr_intervals)

                features.extend([heart_rate, hrv, rmssd, cv_rr])
            else:
                features.extend([60, 0, 0, 0])

        except:
            features.extend([60, 0, 0, 0])

        # 确保8个特征
        features = features[:8]
        while len(features) < 8:
            features.append(0)

        # 确保有限值
        features = [float(f) if np.isfinite(f) else 0.0 for f in features]
        return np.array(features)

    except:
        return np.zeros(8)


# ===================================================================
# 稳定的预处理
# ===================================================================
def stable_preprocess_ecg(raw_signal, target_length=500):
    """稳定的ECG预处理"""
    try:
        if raw_signal is None or len(raw_signal) == 0:
            return None

        signal = np.array(raw_signal, dtype=np.float64)

        if len(signal.shape) != 2 or signal.shape[1] != 12:
            return None
        if not np.isfinite(signal).all():
            return None

        # 简单重采样
        if len(signal) != target_length:
            old_indices = np.linspace(0, len(signal) - 1, len(signal))
            new_indices = np.linspace(0, len(signal) - 1, target_length)

            resampled_signal = np.zeros((target_length, 12), dtype=np.float32)
            for i in range(12):
                if len(signal) > 1:
                    f = interp1d(old_indices, signal[:, i], kind='linear',
                                 fill_value='extrapolate')
                    resampled_signal[:, i] = f(new_indices)
                else:
                    resampled_signal[:, i] = signal[0, i]
        else:
            resampled_signal = signal.astype(np.float32)

        # 简单标准化
        for i in range(12):
            channel = resampled_signal[:, i]
            std_val = np.std(channel)
            if std_val > 1e-10:
                resampled_signal[:, i] = (channel - np.mean(channel)) / std_val
            else:
                resampled_signal[:, i] = 0

        resampled_signal = np.clip(resampled_signal, -3, 3)

        if not np.isfinite(resampled_signal).all():
            return None

        return resampled_signal

    except:
        return None


# ===================================================================
# 稳定的数据生成器
# ===================================================================
class StableLSTMGenerator(Sequence):
    def __init__(self, df, batch_size, label_map, num_classes, shuffle=True, augment=False):
        self.df = df
        self.batch_size = batch_size
        self.label_map = label_map
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = self.df.index.tolist()

        if self.shuffle:
            np.random.shuffle(self.indexes)

        print(f"稳定数据生成器: 增强={augment}")

    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.loc[batch_indexes]

        batch_size = len(batch_df)
        X_waveform = np.zeros((batch_size, SEQUENCE_LENGTH, 12), dtype=np.float32)
        X_features = np.zeros((batch_size, 8), dtype=np.float32)  # 8个特征
        y = np.zeros((batch_size, self.num_classes), dtype=np.float32)

        valid_samples = 0

        for i, (idx, row) in enumerate(batch_df.iterrows()):
            try:
                full_path = os.path.join(ECG_BASE_PATH, os.path.splitext(row['waveform_path'])[0])

                if not os.path.exists(full_path + '.dat'):
                    continue

                record = wfdb.rdrecord(full_path)
                if record.p_signal is None:
                    continue

                processed_signal = stable_preprocess_ecg(record.p_signal, SEQUENCE_LENGTH)
                if processed_signal is None:
                    continue

                # 轻量级数据增强
                if self.augment:
                    processed_signal = lightweight_augmentation(processed_signal)

                X_waveform[i] = processed_signal

                # 提取医学特征
                features = extract_core_medical_features(processed_signal, SAMPLING_RATE)
                X_features[i] = features

                label_int = self.label_map[row['ecg_category']]
                y[i] = tf.keras.utils.to_categorical(label_int, num_classes=self.num_classes)

                valid_samples += 1

            except Exception:
                continue

        # 频繁内存清理
        if index % 10 == 0:
            gc.collect()

        return (X_waveform, X_features), y


# ===================================================================
# 轻量级LSTM模型
# ===================================================================
def create_lightweight_lstm_model(sequence_length, num_channels, num_features, num_classes):
    """创建轻量级LSTM模型"""

    # 输入层
    waveform_input = Input(shape=(sequence_length, num_channels), name='waveform_input')
    feature_input = Input(shape=(num_features,), name='feature_input')

    # 简化的波形分支
    # 只使用一个LSTM层
    lstm_out = LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(waveform_input)
    waveform_branch = Dense(24, activation='relu')(lstm_out)
    waveform_branch = BatchNormalization()(waveform_branch)
    waveform_branch = Dropout(0.3)(waveform_branch)

    # 简化的特征分支
    feature_branch = Dense(16, activation='relu')(feature_input)
    feature_branch = BatchNormalization()(feature_branch)
    feature_branch = Dropout(0.2)(feature_branch)

    # 合并
    concatenated = Concatenate()([waveform_branch, feature_branch])

    # 简化的输出层
    x = Dense(32, activation='relu')(concatenated)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[waveform_input, feature_input], outputs=output)

    # 保守的优化器设置
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ===================================================================
# 主训练流程
# ===================================================================
print("\n--- 阶段2: 稳定LSTM模型构建 ---")

# 准备标签
labels = sorted(full_df['ecg_category'].unique())
label_map = {label: i for i, label in enumerate(labels)}
num_classes = len(labels)

print(f"类别标签: {labels}")

# 计算类别权重
train_labels = [label_map[cat] for cat in train_subset_df['ecg_category']]
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print(f"类别权重: {class_weight_dict}")

# 创建数据生成器
train_generator = StableLSTMGenerator(
    train_subset_df, BATCH_SIZE, label_map, num_classes, shuffle=True, augment=True)
val_generator = StableLSTMGenerator(
    val_subset_df, BATCH_SIZE, label_map, num_classes, shuffle=False, augment=False)

# 创建轻量级LSTM模型
model = create_lightweight_lstm_model(
    sequence_length=SEQUENCE_LENGTH,
    num_channels=12,
    num_features=8,  # 8个核心医学特征
    num_classes=num_classes
)

print("\n轻量级LSTM模型架构:")
model.summary()

# 设置回调函数
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, monitor='val_loss')
]

# ===================================================================
# 训练和评估
# ===================================================================
print(f"\n--- 阶段3: 开始稳定LSTM训练 ({EPOCHS} epochs) ---")

try:
    # 训练模型
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    print("稳定LSTM训练完成!")

    # 评估模型
    print("\n--- 阶段4: 稳定LSTM模型评估 ---")

    test_generator = StableLSTMGenerator(
        test_subset_df, BATCH_SIZE, label_map, num_classes, shuffle=False, augment=False)

    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"测试集结果: 损失={test_loss:.4f}, 准确率={test_accuracy:.4f}")

    # 生成分类报告
    print("\n生成详细分类报告...")
    y_true = []
    y_pred = []
    y_pred_proba = []

    for i in range(len(test_generator)):
        (X_waveform, X_features), y_batch = test_generator[i]
        pred_batch = model.predict([X_waveform, X_features], verbose=0)

        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred.extend(np.argmax(pred_batch, axis=1))
        y_pred_proba.extend(np.max(pred_batch, axis=1))

    print("\n稳定LSTM分类报告:")
    print(classification_report(y_true, y_pred, target_names=labels))

    print("\n混淆矩阵:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # 预测置信度
    print(f"\n预测置信度统计:")
    print(f"平均置信度: {np.mean(y_pred_proba):.4f}")
    print(f"置信度标准差: {np.std(y_pred_proba):.4f}")

    # 特征重要性分析
    print(f"\n医学特征统计:")
    feature_names = ['Mean', 'Std', 'Skew', 'Kurt', 'HR', 'HRV', 'RMSSD', 'CV_RR']

    # 分析一个批次的特征
    test_batch = test_generator[0]
    sample_features = test_batch[0][1]  # 特征数据

    for i, name in enumerate(feature_names):
        feat_mean = np.mean(sample_features[:, i])
        feat_std = np.std(sample_features[:, i])
        print(f"{name}: {feat_mean:.3f} ± {feat_std:.3f}")

    # 保存模型
    model.save('ecg_stable_lstm_model.keras')
    print("稳定LSTM模型已保存为 ecg_stable_lstm_model.keras")

except Exception as e:
    print(f"训练或评估过程中出错: {e}")
    import traceback

    traceback.print_exc()

print("\n=== 稳定LSTM训练程序完成 ===")