# ==========================================================
# PLRNN ECG Arrhythmia Classification on Large-Scale Clinical Data
# ==========================================================

import os
import gc
import sys
import numpy as np
import pandas as pd
import wfdb
import tensorflow as tf
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import math
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import (Input, Dense, Concatenate, Dropout,
                                     Conv1D, MaxPooling1D, BatchNormalization,
                                     MultiHeadAttention, Layer)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# GPU禁用设置 (Apple Silicon优化)
print("--- 配置运行环境 ---")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_METAL_DEVICE_ENABLE'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.config.set_visible_devices([], 'GPU')


# ===================================================================
# 1. PLRNN核心实现
# ===================================================================

class PiecewiseLinearActivation(Layer):
    """分段线性激活函数"""

    def __init__(self, num_pieces=3, **kwargs):
        super(PiecewiseLinearActivation, self).__init__(**kwargs)
        self.num_pieces = num_pieces

    def build(self, input_shape):
        # 初始化分段点和斜率
        self.breakpoints = self.add_weight(
            name='breakpoints',
            shape=(self.num_pieces - 1,),
            initializer='uniform',
            trainable=True
        )
        self.slopes = self.add_weight(
            name='slopes',
            shape=(self.num_pieces,),
            initializer='ones',
            trainable=True
        )
        self.intercepts = self.add_weight(
            name='intercepts',
            shape=(self.num_pieces,),
            initializer='zeros',
            trainable=True
        )
        super(PiecewiseLinearActivation, self).build(input_shape)

    def call(self, x):
        # 确保断点有序
        sorted_breakpoints = tf.sort(self.breakpoints)

        # 计算分段线性函数
        output = tf.zeros_like(x)

        # 第一段: x < breakpoint[0]
        mask1 = tf.cast(x < sorted_breakpoints[0], tf.float32)
        output += mask1 * (self.slopes[0] * x + self.intercepts[0])

        # 中间段
        for i in range(1, self.num_pieces - 1):
            mask = tf.cast(
                (x >= sorted_breakpoints[i - 1]) & (x < sorted_breakpoints[i]),
                tf.float32
            )
            output += mask * (self.slopes[i] * x + self.intercepts[i])

        # 最后一段: x >= breakpoint[-1]
        mask_last = tf.cast(x >= sorted_breakpoints[-1], tf.float32)
        output += mask_last * (self.slopes[-1] * x + self.intercepts[-1])

        return output


class PLRNNCell(Layer):
    """PLRNN单元"""

    def __init__(self, units, num_pieces=3, **kwargs):
        super(PLRNNCell, self).__init__(**kwargs)
        self.units = units
        self.num_pieces = num_pieces

    def build(self, input_shape):
        # 输入到隐藏状态的权重
        self.W_ih = self.add_weight(
            name='W_ih',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )

        # 隐藏状态到隐藏状态的权重
        self.W_hh = self.add_weight(
            name='W_hh',
            shape=(self.units, self.units),
            initializer='orthogonal',
            trainable=True
        )

        # 偏置
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )

        # 分段线性激活函数
        self.activation = PiecewiseLinearActivation(self.num_pieces)

        super(PLRNNCell, self).build(input_shape)

    def call(self, inputs, states):
        h_prev = states[0] if states else tf.zeros((tf.shape(inputs)[0], self.units))

        # PLRNN计算: h_t = f(W_ih * x_t + W_hh * h_{t-1} + b)
        linear_output = tf.matmul(inputs, self.W_ih) + tf.matmul(h_prev, self.W_hh) + self.bias
        h_new = self.activation(linear_output)

        return h_new, [h_new]


class PLRNN(Layer):
    """PLRNN层"""

    def __init__(self, units, num_pieces=3, return_sequences=False, **kwargs):
        super(PLRNN, self).__init__(**kwargs)
        self.units = units
        self.num_pieces = num_pieces
        self.return_sequences = return_sequences
        self.cell = PLRNNCell(units, num_pieces)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]

        # 初始化隐藏状态
        h = tf.zeros((batch_size, self.units))
        outputs = []

        # 循环处理序列
        for t in range(seq_length):
            h, _ = self.cell(inputs[:, t, :], [h])
            outputs.append(h)

        if self.return_sequences:
            return tf.stack(outputs, axis=1)
        else:
            return outputs[-1]


# ===================================================================
# 2. 数据处理
# ===================================================================

# 配置参数
LABELED_DATA_CSV = 'ecg_5_class_data.csv'
ECG_BASE_PATH = '/Users/zixiang/ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/'

BATCH_SIZE = 8
EPOCHS = 20
TRAIN_SAMPLES = 2000
VAL_SAMPLES = 400
TEST_SAMPLES = 500
SEQUENCE_LENGTH = 500


def extract_medical_features(signal, fs=100):
    """提取医学特征"""
    try:
        if signal is None or len(signal) == 0:
            return np.zeros(8)

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
            peaks, _ = find_peaks(lead_ii, height=np.std(lead_ii) * 0.5, distance=fs // 4)
            if len(peaks) >= 2:
                rr_intervals = np.diff(peaks) / fs
                heart_rate = 60 / np.mean(rr_intervals)
                hrv = np.std(rr_intervals) * 1000
                rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2)) * 1000
                cv_rr = np.std(rr_intervals) / np.mean(rr_intervals)
                features.extend([heart_rate, hrv, rmssd, cv_rr])
            else:
                features.extend([70, 30, 25, 0.05])
        except:
            features.extend([70, 30, 25, 0.05])

        features = [float(f) if np.isfinite(f) else 0.0 for f in features]
        return np.array(features[:8])
    except:
        return np.zeros(8)


def preprocess_ecg(raw_signal, target_length=500):
    """ECG预处理"""
    try:
        if raw_signal is None or len(raw_signal) == 0:
            return None

        signal = np.array(raw_signal, dtype=np.float64)

        if len(signal.shape) != 2 or signal.shape[1] != 12:
            return None
        if not np.isfinite(signal).all():
            return None

        # 重采样
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

        # 标准化
        for i in range(12):
            channel = resampled_signal[:, i]
            std_val = np.std(channel)
            if std_val > 1e-10:
                resampled_signal[:, i] = (channel - np.mean(channel)) / std_val

        return np.clip(resampled_signal, -3, 3)
    except:
        return None


def balanced_sampling(df, target_samples, random_state=42):
    """平衡采样"""
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
    return result_df.sample(frac=1, random_state=random_state).reset_index(drop=True)


class PLRNNECGGenerator(Sequence):
    """数据生成器"""

    def __init__(self, df, batch_size, label_map, num_classes, shuffle=True):
        self.df = df
        self.batch_size = batch_size
        self.label_map = label_map
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.indexes = self.df.index.tolist()

        if self.shuffle:
            np.random.shuffle(self.indexes)

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
        X_features = np.zeros((batch_size, 8), dtype=np.float32)
        y = np.zeros((batch_size, self.num_classes), dtype=np.float32)

        for i, (idx, row) in enumerate(batch_df.iterrows()):
            try:
                full_path = os.path.join(ECG_BASE_PATH, os.path.splitext(row['waveform_path'])[0])

                if not os.path.exists(full_path + '.dat'):
                    continue

                record = wfdb.rdrecord(full_path)
                if record.p_signal is None:
                    continue

                processed_signal = preprocess_ecg(record.p_signal, SEQUENCE_LENGTH)
                if processed_signal is None:
                    continue

                X_waveform[i] = processed_signal
                features = extract_medical_features(processed_signal)
                X_features[i] = features

                label_int = self.label_map[row['ecg_category']]
                y[i] = tf.keras.utils.to_categorical(label_int, num_classes=self.num_classes)

            except Exception:
                continue

        if index % 15 == 0:
            gc.collect()

        return (X_waveform, X_features), y


# ===================================================================
# 3. PLRNN模型架构
# ===================================================================

def create_plrnn_ecg_model(sequence_length, num_channels, num_features, num_classes):
    """创建PLRNN ECG分类模型"""

    # 输入层
    waveform_input = Input(shape=(sequence_length, num_channels), name='waveform_input')
    feature_input = Input(shape=(num_features,), name='feature_input')

    # 多尺度卷积特征提取
    conv_outputs = []
    for kernel_size in [3, 5, 7]:
        conv = Conv1D(32, kernel_size, activation='relu', padding='same')(waveform_input)
        conv = BatchNormalization()(conv)
        conv = MaxPooling1D(2)(conv)
        conv_outputs.append(conv)

    # 融合多尺度特征
    if len(conv_outputs) > 1:
        fused_conv = Concatenate(axis=-1)(conv_outputs)
    else:
        fused_conv = conv_outputs[0]

    # PLRNN层
    plrnn1 = PLRNN(units=64, num_pieces=4, return_sequences=True)(fused_conv)
    plrnn1 = BatchNormalization()(plrnn1)
    plrnn1 = Dropout(0.2)(plrnn1)

    plrnn2 = PLRNN(units=32, num_pieces=3, return_sequences=False)(plrnn1)

    # 波形分支输出
    waveform_branch = Dense(48, activation='relu')(plrnn2)
    waveform_branch = BatchNormalization()(waveform_branch)
    waveform_branch = Dropout(0.3)(waveform_branch)

    # 医学特征分支
    feature_branch = Dense(24, activation='relu')(feature_input)
    feature_branch = BatchNormalization()(feature_branch)
    feature_branch = Dropout(0.2)(feature_branch)

    # 特征融合
    concatenated = Concatenate()([waveform_branch, feature_branch])

    # 分类层
    x = Dense(64, activation='relu')(concatenated)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)

    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[waveform_input, feature_input], outputs=output, name='PLRNN_ECG_Model')

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ===================================================================
# 4. 主训练流程
# ===================================================================

def main():
    print("=== PLRNN ECG Arrhythmia Classification ===")

    # 1. 数据加载
    print("\n--- 数据加载 ---")
    try:
        full_df = pd.read_csv(LABELED_DATA_CSV, header=None,
                              names=['subject_id', 'waveform_path', 'ecg_category'])
        full_df.dropna(inplace=True)
        print(f"成功加载 {len(full_df)} 条记录")

        print("\n类别分布:")
        print(full_df['ecg_category'].value_counts())

    except FileNotFoundError:
        print(f"错误: 找不到标签文件 {LABELED_DATA_CSV}")
        print("请确保文件路径正确")
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
    train_subset_df = balanced_sampling(train_df, TRAIN_SAMPLES, random_state=42)
    val_subset_df = balanced_sampling(val_df, VAL_SAMPLES, random_state=42)
    test_subset_df = balanced_sampling(test_df, TEST_SAMPLES, random_state=42)

    print(f"平衡采样后: 训练{len(train_subset_df)}, 验证{len(val_subset_df)}, 测试{len(test_subset_df)}")

    # 3. 准备标签
    labels = sorted(full_df['ecg_category'].unique())
    label_map = {label: i for i, label in enumerate(labels)}
    num_classes = len(labels)

    print(f"\n标签映射: {label_map}")

    # 计算类别权重
    train_labels = [label_map[cat] for cat in train_subset_df['ecg_category']]
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    # 4. 创建数据生成器
    print("\n--- 创建数据生成器 ---")
    train_generator = PLRNNECGGenerator(train_subset_df, BATCH_SIZE, label_map, num_classes, shuffle=True)
    val_generator = PLRNNECGGenerator(val_subset_df, BATCH_SIZE, label_map, num_classes, shuffle=False)

    # 5. 创建PLRNN模型
    print("\n--- 创建PLRNN模型 ---")
    model = create_plrnn_ecg_model(
        sequence_length=SEQUENCE_LENGTH,
        num_channels=12,
        num_features=8,
        num_classes=num_classes
    )

    print("\nPLRNN模型架构:")
    model.summary()

    # 6. 设置回调函数
    callbacks = [
        EarlyStopping(patience=6, restore_best_weights=True, monitor='val_accuracy'),
        ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6, monitor='val_loss')
    ]

    # 7. 训练模型
    print(f"\n--- 开始PLRNN训练 ({EPOCHS} epochs) ---")
    try:
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=EPOCHS,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )

        print("PLRNN训练完成!")

    except Exception as e:
        print(f"训练过程中出错: {e}")
        return

    # 8. 模型评估
    print("\n--- PLRNN模型评估 ---")
    test_generator = PLRNNECGGenerator(test_subset_df, BATCH_SIZE, label_map, num_classes, shuffle=False)

    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"测试结果: 损失={test_loss:.4f}, 准确率={test_accuracy:.4f}")

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

    print("\nPLRNN分类报告:")
    print(classification_report(y_true, y_pred, target_names=labels))

    print("\n混淆矩阵:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    print(f"\n预测置信度: {np.mean(y_pred_proba):.4f} ± {np.std(y_pred_proba):.4f}")

    # 9. 保存模型
    model.save('plrnn_ecg_model.keras')
    print("PLRNN模型已保存为 plrnn_ecg_model.keras")

    # 10. 绘制训练曲线
    if 'history' in locals():
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('PLRNN Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('PLRNN Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig('plrnn_training_curves.png', dpi=300, bbox_inches='tight')
        print("训练曲线已保存为 plrnn_training_curves.png")

    print("\n=== PLRNN训练完成 ===")


if __name__ == "__main__":
    main()