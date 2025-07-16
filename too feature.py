# ==========================================================
# 增强版LSTM ECG分类脚本 - 医学特征工程 + 数据增强
# ==========================================================
import os
import gc
import sys

# GPU禁用设置（经过验证的稳定配置）
print("--- 强制切换到CPU模式运行 ---")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_METAL_DEVICE_ENABLE'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import pandas as pd
import numpy as np
import wfdb
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import math
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import (Input, Dense, Concatenate, Dropout,
                                     LSTM, Bidirectional, BatchNormalization,
                                     Conv1D, MaxPooling1D, GlobalMaxPooling1D)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ===================================================================
# 配置参数
# ===================================================================
LABELED_DATA_CSV = 'ecg_5_class_data.csv'
ECG_BASE_PATH = '/Users/zixiang/ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/'

# 训练参数
BATCH_SIZE = 8
EPOCHS = 20  # 增加训练轮数
TRAIN_SAMPLES = 2400
VAL_SAMPLES = 480
TEST_SAMPLES = 600
SEQUENCE_LENGTH = 1000
SAMPLING_RATE = 100  # 重采样后的采样率

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

# 数据集切分（按患者ID）
all_subjects = full_df['subject_id'].unique()
train_val_subjects, test_subjects = train_test_split(all_subjects, test_size=0.15, random_state=42)
train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=0.15, random_state=42)

train_df = full_df[full_df['subject_id'].isin(train_subjects)].reset_index(drop=True)
val_df = full_df[full_df['subject_id'].isin(val_subjects)].reset_index(drop=True)
test_df = full_df[full_df['subject_id'].isin(test_subjects)].reset_index(drop=True)


# 平衡采样函数
def balanced_sampling(df, target_samples, random_state=42):
    """对每个类别进行平衡采样"""
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


# 创建平衡的数据集
train_subset_df = balanced_sampling(train_df, TRAIN_SAMPLES, random_state=42)
val_subset_df = balanced_sampling(val_df, VAL_SAMPLES, random_state=42)
test_subset_df = balanced_sampling(test_df, TEST_SAMPLES, random_state=42)

print(f"\n平衡采样后的数据集:")
print(f"训练集: {len(train_subset_df)} 样本")
print(f"验证集: {len(val_subset_df)} 样本")
print(f"测试集: {len(test_subset_df)} 样本")


# ===================================================================
# 数据增强函数
# ===================================================================
def ecg_data_augmentation(signal, augment_type='random'):
    """ECG信号数据增强"""
    try:
        augmented_signal = signal.copy()

        if augment_type == 'noise' or augment_type == 'random':
            # 添加高斯噪声
            noise_level = np.random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_level, signal.shape)
            augmented_signal += noise

        if augment_type == 'amplitude' or augment_type == 'random':
            # 幅度缩放
            scale_factor = np.random.uniform(0.8, 1.2)
            augmented_signal *= scale_factor

        if augment_type == 'shift' or augment_type == 'random':
            # 基线漂移
            baseline_shift = np.random.uniform(-0.1, 0.1)
            augmented_signal += baseline_shift

        if augment_type == 'stretch' or augment_type == 'random':
            # 轻微的时间拉伸/压缩
            stretch_factor = np.random.uniform(0.95, 1.05)
            if stretch_factor != 1.0:
                old_indices = np.linspace(0, len(signal) - 1, len(signal))
                new_length = int(len(signal) * stretch_factor)
                new_indices = np.linspace(0, len(signal) - 1, new_length)

                stretched_signal = np.zeros((new_length, signal.shape[1]))
                for i in range(signal.shape[1]):
                    f = interp1d(old_indices, signal[:, i], kind='linear', fill_value='extrapolate')
                    stretched_signal[:, i] = f(new_indices)

                # 调整回原始长度
                if new_length > len(signal):
                    augmented_signal = stretched_signal[:len(signal)]
                else:
                    padding = np.zeros((len(signal) - new_length, signal.shape[1]))
                    augmented_signal = np.vstack([stretched_signal, padding])

        return np.clip(augmented_signal, -5, 5)
    except:
        return signal


# ===================================================================
# 医学特征工程
# ===================================================================
def extract_medical_features(signal, fs=100):
    """提取医学相关的ECG特征"""
    try:
        if signal is None or len(signal) == 0:
            return np.zeros(16)  # 返回16个特征

        # 使用导联II（通常是索引1）作为主要分析导联
        lead_ii = signal[:, 1] if signal.shape[1] > 1 else signal[:, 0]

        features = []

        # 1. 基本统计特征
        features.append(np.mean(lead_ii))
        features.append(np.std(lead_ii))
        features.append(skew(lead_ii))
        features.append(kurtosis(lead_ii))

        # 2. QRS复合波检测相关特征
        try:
            # 检测R波峰值
            peaks, properties = find_peaks(lead_ii, height=np.std(lead_ii), distance=fs // 3)

            if len(peaks) >= 2:
                # 心率相关特征
                rr_intervals = np.diff(peaks) / fs  # RR间期（秒）
                heart_rate = 60 / np.mean(rr_intervals)  # 心率
                features.append(heart_rate)

                # 心率变异性特征
                features.append(np.std(rr_intervals) * 1000)  # SDNN (ms)
                features.append(np.sqrt(np.mean(np.diff(rr_intervals) ** 2)) * 1000)  # RMSSD (ms)

                # RR间期的变异系数
                features.append(np.std(rr_intervals) / np.mean(rr_intervals))
            else:
                features.extend([60, 0, 0, 0])  # 默认值

        except:
            features.extend([60, 0, 0, 0])

        # 3. 频域特征
        try:
            # 计算功率谱
            from scipy.fft import fft, fftfreq
            fft_signal = fft(lead_ii)
            freqs = fftfreq(len(lead_ii), 1 / fs)
            power_spectrum = np.abs(fft_signal) ** 2

            # 不同频段的功率
            # 低频 (0.04-0.15 Hz)
            lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
            lf_power = np.sum(power_spectrum[lf_mask])

            # 高频 (0.15-0.4 Hz)
            hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
            hf_power = np.sum(power_spectrum[hf_mask])

            features.append(np.log(lf_power + 1e-10))
            features.append(np.log(hf_power + 1e-10))

            # LF/HF比率
            if hf_power > 0:
                features.append(lf_power / hf_power)
            else:
                features.append(0)

        except:
            features.extend([0, 0, 0])

        # 4. 形态学特征
        try:
            # 信号的最大值和最小值
            features.append(np.max(lead_ii))
            features.append(np.min(lead_ii))

            # 信号的动态范围
            features.append(np.max(lead_ii) - np.min(lead_ii))

            # 过零率
            zero_crossings = np.sum(np.diff(np.signbit(lead_ii)))
            features.append(zero_crossings / len(lead_ii))

        except:
            features.extend([0, 0, 0, 0])

        # 确保特征数量正确
        while len(features) < 16:
            features.append(0)

        features = features[:16]  # 确保正好16个特征

        # 确保所有特征都是有限值
        features = [float(f) if np.isfinite(f) else 0.0 for f in features]

        return np.array(features)

    except Exception as e:
        return np.zeros(16)


# ===================================================================
# 增强的预处理函数
# ===================================================================
def enhanced_preprocess_ecg(raw_signal, target_length=1000, fs_original=500, fs_target=100):
    """增强的ECG预处理函数"""
    try:
        if raw_signal is None or len(raw_signal) == 0:
            return None

        signal = np.array(raw_signal, dtype=np.float64)

        if len(signal.shape) != 2 or signal.shape[1] != 12:
            return None
        if not np.isfinite(signal).all():
            return None

        # 1. 滤波去噪（轻量级）
        try:
            # 去除基线漂移（高通滤波 0.5Hz）
            nyquist = fs_original / 2
            high_cutoff = 0.5 / nyquist
            b_high, a_high = butter(1, high_cutoff, btype='high')

            # 去除高频噪声（低通滤波 40Hz）
            low_cutoff = 40 / nyquist
            b_low, a_low = butter(1, low_cutoff, btype='low')

            filtered_signal = np.zeros_like(signal)
            for i in range(12):
                # 高通滤波
                filtered_signal[:, i] = filtfilt(b_high, a_high, signal[:, i])
                # 低通滤波
                filtered_signal[:, i] = filtfilt(b_low, a_low, filtered_signal[:, i])

        except:
            filtered_signal = signal

        # 2. 重采样
        if len(filtered_signal) != target_length:
            old_indices = np.linspace(0, len(filtered_signal) - 1, len(filtered_signal))
            new_indices = np.linspace(0, len(filtered_signal) - 1, target_length)

            resampled_signal = np.zeros((target_length, 12), dtype=np.float32)
            for i in range(12):
                if len(filtered_signal) > 1:
                    f = interp1d(old_indices, filtered_signal[:, i], kind='linear',
                                 fill_value='extrapolate')
                    resampled_signal[:, i] = f(new_indices)
                else:
                    resampled_signal[:, i] = filtered_signal[0, i]
        else:
            resampled_signal = filtered_signal.astype(np.float32)

        # 3. 标准化（每个导联独立）
        for i in range(12):
            channel = resampled_signal[:, i]
            # 使用robust标准化
            median = np.median(channel)
            mad = np.median(np.abs(channel - median))
            if mad > 1e-10:
                resampled_signal[:, i] = (channel - median) / (mad * 1.4826)
            else:
                resampled_signal[:, i] = 0

        # 4. 截断异常值
        resampled_signal = np.clip(resampled_signal, -3, 3)

        if not np.isfinite(resampled_signal).all():
            return None

        return resampled_signal

    except Exception as e:
        return None


# ===================================================================
# 增强的LSTM数据生成器
# ===================================================================
class EnhancedLSTMGenerator(Sequence):
    def __init__(self, df, batch_size, label_map, num_classes, shuffle=True, augment=False):
        self.df = df
        self.batch_size = batch_size
        self.label_map = label_map
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment = augment  # 是否进行数据增强
        self.indexes = self.df.index.tolist()

        if self.shuffle:
            np.random.shuffle(self.indexes)

        print(f"数据生成器初始化: 增强={augment}, 打乱={shuffle}")

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
        X_features = np.zeros((batch_size, 16), dtype=np.float32)  # 16个医学特征
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

                processed_signal = enhanced_preprocess_ecg(record.p_signal, SEQUENCE_LENGTH)
                if processed_signal is None:
                    continue

                # 数据增强（仅在训练时）
                if self.augment and np.random.random() < 0.5:
                    processed_signal = ecg_data_augmentation(processed_signal, 'random')

                X_waveform[i] = processed_signal

                # 提取医学特征
                features = extract_medical_features(processed_signal, SAMPLING_RATE)
                X_features[i] = features

                label_int = self.label_map[row['ecg_category']]
                y[i] = tf.keras.utils.to_categorical(label_int, num_classes=self.num_classes)

                valid_samples += 1

            except Exception:
                continue

        if index % 20 == 0:
            gc.collect()

        return (X_waveform, X_features), y


# ===================================================================
# 增强的LSTM模型架构
# ===================================================================
def create_enhanced_lstm_model(sequence_length, num_channels, num_features, num_classes):
    """创建增强的LSTM模型架构"""

    # 输入层
    waveform_input = Input(shape=(sequence_length, num_channels), name='waveform_input')
    feature_input = Input(shape=(num_features,), name='feature_input')

    # 波形分支：混合CNN-LSTM架构
    # 首先使用1D CNN提取局部特征
    x = Conv1D(32, kernel_size=5, activation='relu', padding='same')(waveform_input)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    # 然后使用双向LSTM学习时序依赖
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))(x)
    waveform_branch = Dense(48, activation='relu')(x)
    waveform_branch = BatchNormalization()(waveform_branch)
    waveform_branch = Dropout(0.3)(waveform_branch)

    # 特征分支：处理医学特征
    feature_branch = Dense(32, activation='relu')(feature_input)
    feature_branch = BatchNormalization()(feature_branch)
    feature_branch = Dropout(0.2)(feature_branch)

    feature_branch = Dense(16, activation='relu')(feature_branch)
    feature_branch = Dropout(0.2)(feature_branch)

    # 合并分支
    concatenated = Concatenate()([waveform_branch, feature_branch])

    # 输出层
    x = Dense(64, activation='relu')(concatenated)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)

    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[waveform_input, feature_input], outputs=output)

    # 使用适中的学习率
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),  # 稍微提高学习率
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ===================================================================
# 主训练流程
# ===================================================================
print("\n--- 阶段2: 增强LSTM模型构建 ---")

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

# 创建数据生成器（训练时启用数据增强）
train_generator = EnhancedLSTMGenerator(
    train_subset_df, BATCH_SIZE, label_map, num_classes, shuffle=True, augment=True)
val_generator = EnhancedLSTMGenerator(
    val_subset_df, BATCH_SIZE, label_map, num_classes, shuffle=False, augment=False)

# 创建增强的LSTM模型
model = create_enhanced_lstm_model(
    sequence_length=SEQUENCE_LENGTH,
    num_channels=12,
    num_features=16,  # 16个医学特征
    num_classes=num_classes
)

print("\n增强LSTM模型架构:")
model.summary()

# 设置回调函数
callbacks = [
    EarlyStopping(patience=6, restore_best_weights=True, monitor='val_accuracy'),
    ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6, monitor='val_loss')
]

# ===================================================================
# 训练和评估
# ===================================================================
print(f"\n--- 阶段3: 开始增强LSTM训练 ({EPOCHS} epochs) ---")

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

    print("增强LSTM训练完成!")

    # 评估模型
    print("\n--- 阶段4: 增强LSTM模型评估 ---")

    test_generator = EnhancedLSTMGenerator(
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

    print("\n增强LSTM分类报告:")
    print(classification_report(y_true, y_pred, target_names=labels))

    print("\n混淆矩阵:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # 分析预测置信度
    print(f"\n预测置信度统计:")
    print(f"平均置信度: {np.mean(y_pred_proba):.4f}")
    print(f"置信度标准差: {np.std(y_pred_proba):.4f}")

    # 保存模型
    model.save('ecg_enhanced_lstm_model.keras')
    print("增强LSTM模型已保存为 ecg_enhanced_lstm_model.keras")

    # 保存训练历史
    import pickle

    with open('enhanced_lstm_training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    print("训练历史已保存为 enhanced_lstm_training_history.pkl")

except Exception as e:
    print(f"训练或评估过程中出错: {e}")
    import traceback

    traceback.print_exc()

print("\n=== 增强LSTM训练程序完成 ===")