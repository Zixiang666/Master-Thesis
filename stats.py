# %% [markdown]
# ## 1. 导入所需库
# 我们将导入处理数据所需的标准Python库，包括Numpy、Pandas、Scipy（用于信号处理）、Matplotlib/Seaborn（用于可视化）和PyWavelets（用于小波分析）。

# %%
# Import required python libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
import pywt  # PyWavelets for wavelet analysis
import statsmodels.graphics.tsaplots as tsaplots

# 设置绘图样式
sns.set_context('talk')
plt.style.use('seaborn-v0_8-whitegrid')
print("Libraries imported successfully.")

# %% [markdown]
# ## 2. 数据加载 (占位符)
# 在这里，您需要加载您的MIMIC-ECG数据。由于您已完成此步骤，您可以将您的数据加载代码放在这里。
# 为了使后续代码可运行，我们将创建一个模拟的ECG信号样本。
# **请注意:** ECG信号通常是一个二维数组，格式为 `(channels, samples)` 或 `(leads, samples)`。

# %%
# --- 用户需要替换为自己的数据加载逻辑 ---
# 示例:
# data = pd.read_csv("your_mimic_ecg_data.csv")
# ecg_signals = data.values.T  # 假设每列是一个时间点，每行是一个导联
# sampling_rate = 500 # 假设采样率为 500 Hz


# --- 为演示目的，创建模拟ECG数据 (已更新)---
sampling_rate = 250  # Hz
duration = 10  # seconds
n_samples = int(duration * sampling_rate)
time = np.linspace(0, duration, n_samples, endpoint=False)
t_pulse = np.linspace(-1, 1, n_samples) # Time vector for the pulse

# 创建一个模拟的ECG信号（例如，导联I） - 使用 gausspulse 替代 ricker
qrs_complex_I = signal.gausspulse(t_pulse, fc=5, bw=0.5) * 0.8
qrs_complex_II = signal.gausspulse(t_pulse, fc=5, bw=0.5) * 1.0

ecg_signal_lead_I = (0.1 * np.sin(2 * np.pi * 0.5 * time) +  # P wave
                 qrs_complex_I +   # QRS complex
                 0.2 * np.sin(2 * np.pi * 2 * time + np.pi/2) + # T wave
                 np.random.normal(0, 0.05, n_samples))    # Noise

ecg_signal_lead_II = (0.15 * np.sin(2 * np.pi * 0.5 * time) +
                  qrs_complex_II +
                  0.25 * np.sin(2 * np.pi * 2 * time + np.pi/2) +
                  np.random.normal(0, 0.05, n_samples))

# 组合成一个多导联信号数组 (2个导联, n_samples个样本)
mock_ecg_data = np.array([ecg_signal_lead_I, ecg_signal_lead_II])
lead_names = ['Lead I', 'Lead II']

print(f"Mock ECG data created with shape: {mock_ecg_data.shape}")
print(f"Sampling Rate: {sampling_rate} Hz")

# 绘制模拟信号
plt.figure(figsize=(15, 6))
plt.plot(time, mock_ecg_data[0], label='Simulated Lead I')
plt.title("Simulated ECG Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.legend()
plt.show()


# %% [markdown]
# ## 3. 核心分析函数
# 在这里，我们将定义所有需要的分析函数。这些函数根据您导师的要求分为几个部分。
#
# ### 3.1. 统计分析 (导师要求 #1)
# 这一部分包含计算描述性统计特征的函数。

# %%
def calculate_statistical_features(ecg_data):
    """
    计算ECG信号的描述性统计特征。

    参数:
        ecg_data (numpy.ndarray): 形状为 (n_leads, n_samples) 的ECG数据。

    返回:
        dict: 包含每个导联统计特征的字典。
    """
    features = {}

    # Activity (Variance)
    features['Variance'] = np.var(ecg_data, axis=1)

    # Hjorth Parameters (Activity, Mobility, Complexity)
    activity = features['Variance']
    diff1 = np.diff(ecg_data, axis=1)
    diff2 = np.diff(diff1, axis=1)

    mobility = np.sqrt(np.var(diff1, axis=1) / activity)
    complexity = np.sqrt(np.var(diff2, axis=1) / np.var(diff1, axis=1)) / mobility

    features['Hjorth_Activity'] = activity
    features['Hjorth_Mobility'] = mobility
    features['Hjorth_Complexity'] = complexity

    # Peak-to-Peak Amplitude
    features['Peak_to_Peak_Amplitude'] = np.ptp(ecg_data, axis=1)

    # Mean
    features['Mean'] = np.mean(ecg_data, axis=1)

    # Standard Deviation (STD)
    features['STD'] = np.std(ecg_data, axis=1)

    # Skewness
    features['Skewness'] = stats.skew(ecg_data, axis=1)

    # Kurtosis
    features['Kurtosis'] = stats.kurtosis(ecg_data, axis=1)

    # Mean Absolute Amplitude
    features['Mean_Amplitude'] = np.mean(np.abs(ecg_data), axis=1)

    # Median
    features['Median'] = np.median(ecg_data, axis=1)

    # Min & Max
    features['Min'] = np.min(ecg_data, axis=1)
    features['Max'] = np.max(ecg_data, axis=1)

    # Mode - Scipy's mode is more robust for continuous data
    mode_result = stats.mode(ecg_data, axis=1, keepdims=False)
    features['Mode'] = mode_result.mode

    return features


# %% [markdown]
# ### 3.2. 频率/功率分析 (导师要求 #2)
# 这一部分包括傅里叶变换(FFT)、短时傅里叶变换(STFT)、功率谱密度(PSD)和小波分析的函数。

# %%
def plot_fft(ecg_lead_data, fs, lead_name=""):
    """
    计算并绘制单导联ECG信号的FFT。
    """
    n = len(ecg_lead_data)
    yf = np.fft.fft(ecg_lead_data)
    xf = np.fft.fftfreq(n, 1 / fs)

    plt.figure(figsize=(12, 5))
    plt.plot(xf[:n // 2], 2.0 / n * np.abs(yf[0:n // 2]))
    plt.title(f'FFT of {lead_name}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()


def plot_stft(ecg_lead_data, fs, lead_name=""):
    """
    计算并绘制单导联ECG信号的STFT（谱图）。
    """
    f, t, Zxx = signal.stft(ecg_lead_data, fs, nperseg=256)
    plt.figure(figsize=(12, 5))
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title(f'STFT (Spectrogram) of {lead_name}')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.ylim(0, 40)  # 限制频率范围以便观察
    plt.colorbar(label='Magnitude')
    plt.show()


def plot_psd(ecg_lead_data, fs, lead_name=""):
    """
    计算并绘制单导联ECG信号的PSD。
    """
    f, Pxx = signal.welch(ecg_lead_data, fs, nperseg=1024)
    plt.figure(figsize=(12, 5))
    plt.semilogy(f, Pxx)
    plt.title(f'PSD (Welch\'s Method) of {lead_name}')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power/Frequency [V^2/Hz]')
    plt.xlim(0, 40)  # 限制频率范围以便观察
    plt.show()
    return f, Pxx


def plot_wavelet_transform(ecg_lead_data, lead_name=""):
    """
    执行并绘制连续小波变换（CWT）的结果（谱图）。
    """
    scales = np.arange(1, 128)
    wavelet = 'morl'  # Morlet小波常用于ECG
    coeffs, freqs = pywt.cwt(ecg_lead_data, scales, wavelet)

    plt.figure(figsize=(12, 5))
    plt.imshow(np.abs(coeffs), extent=[0, len(ecg_lead_data), 1, 128], cmap='viridis', aspect='auto',
               vmax=abs(coeffs).max(), vmin=-abs(coeffs).max())
    plt.title(f'Continuous Wavelet Transform of {lead_name}')
    plt.ylabel('Scale')
    plt.xlabel('Time (Samples)')
    plt.show()


# %% [markdown]
# ### 3.3. 自相关图 (导师要求 #3)

# %%
def plot_autocorrelation(ecg_lead_data, lead_name=""):
    """
    绘制单导联ECG信号的自相关图。
    """
    plt.figure(figsize=(12, 5))
    tsaplots.plot_acf(ecg_lead_data, lags=50, title=f'Autocorrelation of {lead_name}')
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation')
    plt.show()


# %% [markdown]
# ### 3.4. 平均ECG振幅与信噪比 (导师要求 #4 和 #5)

# %%
def plot_mean_ecg_amplitude(ecg_data, lead_names, time_vector):
    """
    计算并可视化所有ECG导联的平均振幅。
    """
    mean_ecg = np.mean(ecg_data, axis=0)

    plt.figure(figsize=(15, 6))
    for i, lead_name in enumerate(lead_names):
        plt.plot(time_vector, ecg_data[i], alpha=0.3, label=f'Raw {lead_name}')
    plt.plot(time_vector, mean_ecg, color='black', linewidth=2, label='Mean ECG Amplitude')
    plt.title('Mean ECG Amplitude Across All Leads')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (mV)')
    plt.legend()
    plt.show()


def calculate_snr(ecg_lead_data):
    """
    一个简单的信噪比（SNR）计算方法。
    SNR = Power(signal) / Power(noise)
    这里我们假设信号是原始数据，噪声是信号与其平滑版本之间的差异。
    """
    # 使用移动平均进行平滑
    window_size = 15
    smoothed_signal = np.convolve(ecg_lead_data, np.ones(window_size) / window_size, mode='same')

    noise = ecg_lead_data - smoothed_signal

    power_signal = np.mean(ecg_lead_data ** 2)
    power_noise = np.mean(noise ** 2)

    if power_noise == 0:
        return np.inf

    snr_ratio = power_signal / power_noise
    snr_db = 10 * np.log10(snr_ratio)

    return snr_db


# %% [markdown]
# ## 4. 执行分析与可视化
# 现在我们将使用上面定义的函数来处理我们的模拟ECG数据。

# %%
# --- 4.1 执行统计分析 ---
print("=" * 50)
print("Running Statistical Analysis...")
print("=" * 50)

# 使用模拟数据
statistical_results = calculate_statistical_features(mock_ecg_data)
df_stats = pd.DataFrame(statistical_results, index=lead_names)

# In the section "4.1 执行统计分析"
print("Descriptive Statistics:")
print(df_stats)


# --- 4.2 执行频率分析和绘图 ---
print("\n" + "=" * 50)
print("Running Frequency Analysis & Visualization...")
print("=" * 50)

# 我们只对第一个导联进行可视化，以避免过多输出
lead_to_plot_idx = 0
lead_data_sample = mock_ecg_data[lead_to_plot_idx]
lead_name_sample = lead_names[lead_to_plot_idx]

print(f"\n--- Visualizing for {lead_name_sample} ---")
plot_fft(lead_data_sample, sampling_rate, lead_name_sample)
plot_stft(lead_data_sample, sampling_rate, lead_name_sample)
_, _ = plot_psd(lead_data_sample, sampling_rate, lead_name_sample)
plot_wavelet_transform(lead_data_sample, lead_name_sample)

# --- 4.3 绘制自相关图 ---
print("\n" + "=" * 50)
print("Plotting Autocorrelation...")
print("=" * 50)
plot_autocorrelation(lead_data_sample, lead_name_sample)

# --- 4.4 可视化平均振幅并计算SNR ---
print("\n" + "=" * 50)
print("Visualizing Mean Amplitude and Calculating SNR...")
print("=" * 50)

plot_mean_ecg_amplitude(mock_ecg_data, lead_names, time)

snr_values = [calculate_snr(mock_ecg_data[i]) for i in range(mock_ecg_data.shape[0])]
print("\nSignal-to-Noise Ratio (SNR):")
for i, lead_name in enumerate(lead_names):
    print(f"  - {lead_name}: {snr_values[i]:.2f} dB")


# %% [markdown]
# ## 5. 整合为机器学习特征集
# 在实际应用中，您需要遍历所有ECG记录，为每个记录计算特征，然后将它们汇集成一个大的特征矩阵（DataFrame），用于后续的模型训练。

# %%
def process_all_ecg_records(list_of_ecg_records, sampling_rate):
    """
    处理ECG记录列表，提取特征，并返回一个DataFrame。

    参数:
        list_of_ecg_records (list): 每个元素是一个包含ECG数据的元组 (record_id, ecg_data_array)。
        sampling_rate (int): 采样率。

    返回:
        pandas.DataFrame: 包含所有记录特征的DataFrame。
    """
    all_features = []

    for record_id, ecg_data in list_of_ecg_records:
        # 1. 计算统计特征
        stats_features = calculate_statistical_features(ecg_data)

        # 2. 计算PSD并提取特征 (例如，不同频带的能量)
        # 这里只是一个例子，您可以提取更复杂的频域特征
        power_features = {}
        for i in range(ecg_data.shape[0]):  # 遍历每个导联
            f, Pxx = signal.welch(ecg_data[i], sampling_rate, nperseg=1024)
            # 示例：计算低频(0.5-15Hz)和高频(15-40Hz)的功率
            lf_power = np.trapezoid(Pxx[(f >= 0.5) & (f <= 15)], f[(f >= 0.5) & (f <= 15)])
            hf_power = np.trapezoid(Pxx[(f >= 15) & (f <= 40)], f[(f >= 15) & (f <= 40)])
            power_features[f'lead_{i}_lf_power'] = lf_power
            power_features[f'lead_{i}_hf_power'] = hf_power
            power_features[f'lead_{i}_lf_hf_ratio'] = lf_power / hf_power if hf_power != 0 else 0

        # 3. 计算SNR
        snr_features = {f'lead_{i}_snr_db': calculate_snr(ecg_data[i]) for i in range(ecg_data.shape[0])}

        # 4. 合并所有特征
        # 我们需要将多导联的特征展平，以便每行代表一个记录
        flat_stats = {f'lead_{i}_{key}': val[i] for key, val in stats_features.items() for i in range(len(val))}

        # 将当前记录的所有特征组合成一个字典
        record_features = {
            'record_id': record_id,
            **flat_stats,
            **power_features,
            **snr_features
        }
        all_features.append(record_features)

    return pd.DataFrame(all_features)


# --- 示例用法 ---
# 假设我们有一个包含多个记录的列表
# 在真实场景中，您会从文件中加载这些记录
mock_records_list = [
    ("record_001", mock_ecg_data),
    ("record_002", mock_ecg_data * 0.8 + np.random.normal(0, 0.02, mock_ecg_data.shape))  # 第二个记录，稍作修改
]

feature_dataframe = process_all_ecg_records(mock_records_list, sampling_rate)

print("\n" + "=" * 50)
print("Generated Feature DataFrame for Machine Learning")
print("=" * 50)
print(f"DataFrame shape: {feature_dataframe.shape}")
print("First 5 rows:")
print(feature_dataframe.head())
# 您可以将此DataFrame保存到CSV文件中
# feature_dataframe.to_csv("mimic_ecg_features.csv", index=False)
# print("\nFeature DataFrame saved to mimic_ecg_features.csv")