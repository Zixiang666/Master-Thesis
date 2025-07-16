import numpy as np
import wfdb
from tensorflow.keras.utils import Sequence
import math


# (请确保您已经有了我们之前写的 preprocess_ecg_for_rnn_fft 函数)

def extract_simple_features(signal, fs=100):
    """一个简单的传统特征提取函数示例"""
    try:
        # 使用wfdb的qrs检测器
        qrs_inds = wfdb.processing.xqrs_detect(sig=signal[:, 0], fs=fs)

        # 如果检测到的R波太少，则返回默认值
        if len(qrs_inds) < 2:
            return [60, 0]  # 默认心率60，HRV为0

        # 计算R-R间期 (秒)
        rr_intervals = np.diff(qrs_inds) / fs

        # 计算平均心率
        heart_rate = 60 / np.mean(rr_intervals)

        # 计算HRV的一个指标: SDNN
        sdnn = np.std(rr_intervals) * 1000  # 转换到毫秒

        return [heart_rate, sdnn]
    except Exception:
        return [60, 0]  # 出错时返回默认值


class HybridECGGenerator(Sequence):
    def __init__(self, df, batch_size, label_map, num_classes):
        self.df = df
        self.batch_size = batch_size
        self.label_map = label_map
        self.num_classes = num_classes
        self.indexes = self.df.index.tolist()

    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.loc[batch_indexes]

        # 准备两个输入和一个输出
        # 输入1: 预处理后的波形 (1000, 12)
        X_waveform = np.zeros((len(batch_df), 1000, 12))
        # 输入2: 提取的传统特征 (2个特征)
        X_features = np.zeros((len(batch_df), 2))
        # 输出: 标签
        y = np.zeros((len(batch_df), self.num_classes))

        for i, (idx, row) in enumerate(batch_df.iterrows()):
            try:
                # 加载和预处理
                full_path = os.path.join(ECG_BASE_PATH, os.path.splitext(row['waveform_path'])[0])
                record = wfdb.rdrecord(full_path)
                processed_signal = preprocess_ecg_for_rnn_fft(record.p_signal)  # 使用我们之前的预处理函数

                # 填充波形数据
                if processed_signal is not None and processed_signal.shape == (1000, 12):
                    X_waveform[i] = processed_signal
                    # 提取传统特征
                    features = extract_simple_features(processed_signal)
                    X_features[i] = np.array(features)

                # 填充标签
                label_int = self.label_map[row['ecg_category']]
                y[i] = tf.keras.utils.to_categorical(label_int, num_classes=self.num_classes)
            except Exception:
                # 如果任何步骤出错，保持为0
                pass

        return [X_waveform, X_features], y