import pandas as pd
import numpy as np
import wfdb
import os
from scipy.signal import butter, filtfilt, resample, welch
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt


# --- 1. 配置部分 (可修改) ---
# 假设我们先用一小部分数据进行测试
LABELED_DATA_CSV = 'heart_rate_labeled_data.csv'  # 您的带标签的CSV文件
ECG_BASE_PATH = '/Users/zixiang/ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/'


# --- 2. 完整的预处理函数 (顺序修正版) ---
def preprocess_ecg_for_rnn_fft(raw_signal, original_fs=500, new_fs=100, clip_limit=3):
    """
    一个修正了处理顺序的完整ECG预处理流水线.
    """
    # --- 步骤1: 在原始500Hz信号上进行滤波 ---
    fs = original_fs  # 使用原始采样率进行滤波

    # 陷波滤波器去除60Hz工频干扰
    try:
        b_notch, a_notch = butter(2, [59 / (fs / 2), 61 / (fs / 2)], 'bandstop')
        notched_signal = filtfilt(b_notch, a_notch, raw_signal, axis=0)
    except Exception as e:
        print(f"陷波滤波时出错: {e}")
        notched_signal = raw_signal  # 如果出错，则跳过

    # 带通滤波器 (0.5Hz - 40Hz)
    try:
        nyquist = 0.5 * fs
        low = 0.5 / nyquist
        high = 40 / nyquist
        b_band, a_band = butter(1, [low, high], 'band')
        filtered_signal = filtfilt(b_band, a_band, notched_signal, axis=0)
    except Exception as e:
        print(f"带通滤波时出错: {e}")
        filtered_signal = notched_signal  # 如果出错，则使用上一步的结果

    # --- 步骤2: 对滤波后的信号进行降采样 ---
    num_samples_original = len(filtered_signal)
    num_samples_new = int(num_samples_original * new_fs / original_fs)
    resampled_signal = resample(filtered_signal, num_samples_new)

    # --- 步骤3: 削波 ---
    clipped_signal = np.clip(resampled_signal, -clip_limit, clip_limit)

    # --- 步骤4: Z-score标准化 ---
    mean = np.mean(clipped_signal, axis=0)
    std = np.std(clipped_signal, axis=0)
    normalized_signal = (clipped_signal - mean) / (std + 1e-9)

    return normalized_signal


# --- 3. 演示流程 ---
if __name__ == '__main__':
    # 加载数据子集进行演示
    df = pd.read_csv(LABELED_DATA_CSV)
    sample_ecg_info = df.sample(1).iloc[0]  # 随机抽取一条记录

    # 加载原始ECG
    record_relative_path = os.path.splitext(sample_ecg_info['record_name'])[0]
    full_record_path = os.path.join(ECG_BASE_PATH, record_relative_path)
    raw_record = wfdb.rdrecord(full_record_path)

    print(f"正在处理记录: {record_relative_path}")
    print(f"原始信号形状: {raw_record.p_signal.shape}")

    # 应用预处理流水线
    processed_signal = preprocess_ecg_for_rnn_fft(raw_record.p_signal)

    print(f"处理后信号形状: {processed_signal.shape}")
    print("这个 (1000, 12) 的矩阵可以直接作为RNN的输入。")

    # --- 4. 频域分析演示 ---
    print("\nPerforming frequency domain analysis...")
    lead_to_analyze = 0  # We will analyze the first lead as an example
    signal_lead = processed_signal[:, lead_to_analyze]

    # FFT 分析
    N = len(signal_lead)
    T = 1.0 / 100.0  # New sampling interval after downsampling to 100Hz
    yf = fft(signal_lead)
    xf = fftfreq(N, T)[:N // 2]

    # PSD 分析 (Welch方法)
    f, Pxx_den = welch(signal_lead, fs=100, nperseg=256)

    # --- 5. 可视化 (全英文版) ---
    plt.figure(figsize=(15, 12))

    # Original vs. Processed Signal
    plt.subplot(3, 1, 1)
    plt.title("Original vs. Preprocessed Signal (Lead I)")
    plt.plot(np.linspace(0, 10, 5000), raw_record.p_signal[:, lead_to_analyze], label='Original Signal (500Hz)',
             alpha=0.7)
    plt.plot(np.linspace(0, 10, 1000), signal_lead, label='Processed Signal (100Hz)', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (mV)')
    plt.legend()
    plt.grid(True)

    # FFT Magnitude Spectrum
    plt.subplot(3, 1, 2)
    plt.title("FFT Magnitude Spectrum")
    plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Power Spectral Density (PSD)
    plt.subplot(3, 1, 3)
    plt.title("Power Spectral Density (PSD)")
    plt.semilogy(f, Pxx_den)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (V^2/Hz)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()