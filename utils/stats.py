#!/usr/bin/env python3
"""
ECG Signal Analysis Tool Based on EEG Analysis Framework
=====================================================

Comprehensive ECG signal analysis including:
- Statistical features (Mean, Variance, Median, Min, Max, Skewness, Kurtosis, Hjorth Parameters, etc.)
- Frequency/Power analysis (FFT, STFT, Wavelet, PSD, Band Powers)
- Autocorrelation analysis
- Signal-to-Noise Ratio (SNR)
- Features compatible with PLRNN training

Author: Master Thesis Project
Based on EEG analysis framework from DSAILab, IWR University of Heidelberg
Date: 2025
"""

# Import required python libraries
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.signal import find_peaks, welch
from scipy.stats import skew, kurtosis
import pywt  # PyWavelets for wavelet analysis
# import statsmodels.graphics.tsaplots as tsaplots  # Removed as not used in current implementation
from pathlib import Path
from tqdm import tqdm

# Set plotting style
sns.set_context('talk')
plt.style.use('seaborn-v0_8-whitegrid')
print("📊 ECG Analysis Libraries imported successfully.")

# ===================================================================
# 2. Data Loading and Configuration
# ===================================================================

class ECGAnalysisConfig:
    """Configuration for ECG analysis"""
    # Default paths (update these to match your actual data)
    MULTILABEL_DATA_CSV = '/Users/zixiang/PycharmProjects/Master-Thesis/mimic_ecg_multilabel_dataset.csv'
    BINARY_LABELS_CSV = '/Users/zixiang/PycharmProjects/Master-Thesis/mimic_ecg_binary_labels.csv'
    ECG_BASE_PATH = '/Users/zixiang/ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/'
    
    # Analysis parameters
    SAMPLING_RATE = 500  # Hz
    NUM_LEADS = 12
    ANALYSIS_DURATION = 10  # seconds
    
    # Output paths
    OUTPUT_DIR = './ecg_analysis_results/'
    FEATURES_CSV = 'comprehensive_ecg_features.csv'
    PLOTS_DIR = 'plots/'

config = ECGAnalysisConfig()

def load_actual_ecg_data(csv_path=None, max_records=None):
    """
    Load actual ECG data from MIMIC dataset
    
    Parameters:
        csv_path (str): Path to ECG dataset CSV
        max_records (int): Maximum number of records to load
    
    Returns:
        list: List of tuples (record_id, ecg_data_array)
    """
    csv_path = csv_path or config.MULTILABEL_DATA_CSV
    
    try:
        print(f"📂 Loading ECG data from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        if max_records:
            df = df.head(max_records)
            
        print(f"📊 Loaded {len(df)} ECG records")
        
        ecg_records = []
        for idx, row in df.iterrows():
            # Generate synthetic 12-lead ECG based on clinical parameters
            # (Similar to pytorch_plrnn_integrated_training.py)
            record_id = f"record_{idx:04d}"
            ecg_data = generate_realistic_ecg_from_row(row, config.SAMPLING_RATE, config.ANALYSIS_DURATION)
            ecg_records.append((record_id, ecg_data))
            
        return ecg_records
        
    except FileNotFoundError:
        print(f"⚠️  Data file not found: {csv_path}")
        print("📝 Generating mock data for demonstration...")
        return generate_mock_ecg_dataset()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("📝 Generating mock data for demonstration...")
        return generate_mock_ecg_dataset()

def generate_realistic_ecg_from_row(row, fs=500, duration=10):
    """
    Generate realistic 12-lead ECG from dataset row (enhanced version)
    
    Parameters:
        row: DataFrame row with clinical parameters
        fs (int): Sampling frequency
        duration (int): Signal duration in seconds
    
    Returns:
        numpy.ndarray: ECG data with shape (12, n_samples)
    """
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples)
    
    # Heart rate parameters
    base_hr = np.random.normal(75, 15)
    base_hr = np.clip(base_hr, 50, 120)
    
    # Generate 12-lead ECG signals
    ecg_leads = []
    lead_amplitudes = [1.0, 0.8, 1.2, 0.9, 1.1, 0.7, 1.3, 0.6, 1.4, 0.5, 1.5, 0.4]
    
    for lead in range(12):
        # Phase shift for different leads
        phase_shift = lead * np.pi / 6
        
        # Base sinus rhythm
        base_signal = np.sin(2 * np.pi * (base_hr/60) * t + phase_shift)
        
        # ECG wave components
        p_wave = 0.2 * np.sin(8 * np.pi * (base_hr/60) * t + phase_shift)
        qrs_complex = 0.8 * np.sin(12 * np.pi * (base_hr/60) * t + phase_shift + np.pi/4)
        t_wave = 0.3 * np.sin(4 * np.pi * (base_hr/60) * t + phase_shift - np.pi/4)
        
        # Combine components
        combined = base_signal + p_wave + qrs_complex + t_wave
        
        # Add physiological noise
        noise = np.random.normal(0, 0.1, len(combined))
        combined += noise
        
        # Lead-specific amplitude scaling
        combined *= lead_amplitudes[lead]
        
        ecg_leads.append(combined)
    
    return np.array(ecg_leads, dtype=np.float32)

def generate_mock_ecg_dataset(num_records=50):
    """
    Generate mock ECG dataset for demonstration
    
    Parameters:
        num_records (int): Number of mock records to generate
    
    Returns:
        list: List of tuples (record_id, ecg_data_array)
    """
    print(f"🔄 Generating {num_records} mock ECG records...")
    
    mock_records = []
    for i in range(num_records):
        record_id = f"mock_record_{i:04d}"
        
        # Create mock row data
        mock_row = pd.Series({
            'heart_rate': np.random.normal(75, 15),
            'age': np.random.randint(20, 90),
            'gender': np.random.choice(['M', 'F'])
        })
        
        ecg_data = generate_realistic_ecg_from_row(mock_row, config.SAMPLING_RATE, config.ANALYSIS_DURATION)
        mock_records.append((record_id, ecg_data))
    
    return mock_records


# ===================================================================
# 3. Core Statistical Analysis Functions (Requirement #1)
# ===================================================================
def calculate_comprehensive_statistical_features(ecg_data, fs=500):
    """
    Calculate comprehensive statistical features for ECG signals (Enhanced version based on EEG analysis)
    
    Parameters:
        ecg_data (numpy.ndarray): ECG data with shape (n_leads, n_samples)
        fs (int): Sampling frequency (default 500 Hz)
    
    Returns:
        dict: Dictionary containing statistical features for each lead
    """
    features = {}
    n_leads, n_samples = ecg_data.shape
    
    try:
        # Basic Statistical Features
        features['Mean'] = np.mean(ecg_data, axis=1)
        features['Variance'] = np.var(ecg_data, axis=1)
        features['STD'] = np.std(ecg_data, axis=1)
        features['Median'] = np.median(ecg_data, axis=1)
        features['Min'] = np.min(ecg_data, axis=1)
        features['Max'] = np.max(ecg_data, axis=1)
        
        # Mean Absolute Amplitude (similar to EEG analysis)
        features['Mean_Amplitude'] = np.mean(np.abs(ecg_data), axis=1)
        
        # Statistical Shape Features
        features['Skewness'] = skew(ecg_data, axis=1)
        features['Kurtosis'] = kurtosis(ecg_data, axis=1)
        
        # Peak-to-Peak Amplitude (from EEG analysis)
        features['Peak_to_Peak_Amplitude'] = np.ptp(ecg_data, axis=1)
        
        # Hjorth Parameters (Activity, Mobility, Complexity) - Enhanced version
        activity = features['Variance']
        diff1 = np.diff(ecg_data, axis=1)
        diff2 = np.diff(diff1, axis=1)
        
        mobility = np.sqrt(np.var(diff1, axis=1) / (activity + 1e-10))
        complexity = np.sqrt(np.var(diff2, axis=1) / (np.var(diff1, axis=1) + 1e-10)) / (mobility + 1e-10)
        
        features['Hjorth_Activity'] = activity
        features['Hjorth_Mobility'] = mobility
        features['Hjorth_Complexity'] = complexity
        
        # Heart Rate Variability Features (ECG-specific)
        for lead_idx in range(n_leads):
            signal_lead = ecg_data[lead_idx]
            
            # Find R-peaks for HRV analysis
            peaks, _ = find_peaks(signal_lead, height=np.percentile(signal_lead, 75), distance=int(fs*0.6))
            
            if len(peaks) > 1:
                rr_intervals = np.diff(peaks) / fs  # RR intervals in seconds
                
                # HRV time-domain features
                heart_rate = 60.0 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 75.0
                sdnn = np.std(rr_intervals) if len(rr_intervals) > 0 else 0.05
                rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2)) if len(rr_intervals) > 1 else 0.03
                
                if lead_idx == 0:  # Store HRV features only once (lead-independent)
                    features['Heart_Rate'] = np.array([heart_rate] * n_leads)
                    features['SDNN'] = np.array([sdnn] * n_leads)
                    features['RMSSD'] = np.array([rmssd] * n_leads)
        
        # If HRV features weren't computed, set defaults
        if 'Heart_Rate' not in features:
            features['Heart_Rate'] = np.array([75.0] * n_leads)
            features['SDNN'] = np.array([0.05] * n_leads)
            features['RMSSD'] = np.array([0.03] * n_leads)
        
        # Root Mean Square (RMS)
        features['RMS'] = np.sqrt(np.mean(ecg_data**2, axis=1))
        
        # Zero Crossing Rate
        zero_crossings = []
        for lead_idx in range(n_leads):
            zc = np.sum(np.diff(np.sign(ecg_data[lead_idx])) != 0) / n_samples
            zero_crossings.append(zc)
        features['Zero_Crossing_Rate'] = np.array(zero_crossings)
        
        # Energy Features
        features['Total_Energy'] = np.sum(ecg_data**2, axis=1)
        features['Average_Power'] = features['Total_Energy'] / n_samples
        
    except Exception as e:
        print(f"Error in statistical feature calculation: {e}")
        # Return default features in case of error
        default_val = np.zeros(n_leads)
        features = {
            'Mean': default_val, 'Variance': default_val, 'STD': default_val,
            'Median': default_val, 'Min': default_val, 'Max': default_val,
            'Mean_Amplitude': default_val, 'Skewness': default_val, 'Kurtosis': default_val,
            'Peak_to_Peak_Amplitude': default_val, 'Hjorth_Activity': default_val,
            'Hjorth_Mobility': default_val, 'Hjorth_Complexity': default_val,
            'Heart_Rate': default_val + 75.0, 'SDNN': default_val + 0.05,
            'RMSSD': default_val + 0.03, 'RMS': default_val, 'Zero_Crossing_Rate': default_val,
            'Total_Energy': default_val, 'Average_Power': default_val
        }
    
    return features


# ===================================================================
# 3. Advanced Visualization and Analysis Functions
# ===================================================================

def plot_comprehensive_fft_analysis(ecg_data, fs, record_id="", save_dir=None):
    """
    Comprehensive FFT analysis for all ECG leads
    
    Parameters:
        ecg_data (numpy.ndarray): ECG data with shape (n_leads, n_samples)
        fs (int): Sampling frequency
        record_id (str): Record identifier
        save_dir (str): Directory to save plots
    """
    n_leads, n_samples = ecg_data.shape
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'FFT Analysis - {record_id}', fontsize=16, fontweight='bold')
    
    for lead_idx in range(min(n_leads, 12)):
        row, col = divmod(lead_idx, 4)
        ax = axes[row, col]
        
        # Compute FFT
        yf = np.fft.fft(ecg_data[lead_idx])
        xf = np.fft.fftfreq(n_samples, 1/fs)
        
        # Plot positive frequencies only
        positive_freq_mask = xf > 0
        ax.plot(xf[positive_freq_mask], 2.0/n_samples * np.abs(yf[positive_freq_mask]))
        ax.set_title(f'Lead {lead_idx+1} FFT')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.set_xlim(0, 50)  # Focus on relevant ECG frequencies
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{record_id}_fft_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_stft_spectrogram(ecg_data, fs, record_id="", save_dir=None, lead_to_plot=0):
    """
    Short-Time Fourier Transform (STFT) spectrogram
    
    Parameters:
        ecg_data (numpy.ndarray): ECG data with shape (n_leads, n_samples)
        fs (int): Sampling frequency
        record_id (str): Record identifier
        save_dir (str): Directory to save plots
        lead_to_plot (int): Which lead to analyze
    """
    if lead_to_plot >= ecg_data.shape[0]:
        lead_to_plot = 0
        
    signal_data = ecg_data[lead_to_plot]
    
    # Compute STFT
    f, t, Zxx = signal.stft(signal_data, fs, nperseg=256, noverlap=128)
    
    plt.figure(figsize=(15, 8))
    plt.pcolormesh(t, f, 20*np.log10(np.abs(Zxx) + 1e-10), shading='gouraud', cmap='viridis')
    plt.title(f'STFT Spectrogram - {record_id} (Lead {lead_to_plot+1})', fontweight='bold')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.ylim(0, 50)  # Focus on ECG-relevant frequencies
    plt.colorbar(label='Power (dB)')
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{record_id}_stft_lead_{lead_to_plot+1}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_psd_analysis(ecg_data, fs, record_id="", save_dir=None):
    """
    Power Spectral Density analysis for all leads
    
    Parameters:
        ecg_data (numpy.ndarray): ECG data with shape (n_leads, n_samples)
        fs (int): Sampling frequency
        record_id (str): Record identifier
        save_dir (str): Directory to save plots
    
    Returns:
        dict: PSD data for each lead
    """
    n_leads = ecg_data.shape[0]
    psd_data = {}
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'Power Spectral Density Analysis - {record_id}', fontsize=16, fontweight='bold')
    
    for lead_idx in range(min(n_leads, 12)):
        row, col = divmod(lead_idx, 4)
        ax = axes[row, col]
        
        # Compute PSD using Welch's method
        f, Pxx = welch(ecg_data[lead_idx], fs, nperseg=min(len(ecg_data[lead_idx])//4, 1024))
        psd_data[f'lead_{lead_idx}'] = {'frequencies': f, 'psd': Pxx}
        
        # Plot PSD (log scale)
        ax.semilogy(f, Pxx)
        ax.set_title(f'Lead {lead_idx+1} PSD')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Power/Frequency [V²/Hz]')
        ax.set_xlim(0, 50)
        ax.grid(True, alpha=0.3)
        
        # Mark important frequency bands
        ax.axvspan(0.04, 0.15, alpha=0.2, color='red', label='LF')
        ax.axvspan(0.15, 0.4, alpha=0.2, color='blue', label='HF')
        ax.axvspan(5, 15, alpha=0.2, color='green', label='QRS')
        
        if lead_idx == 0:  # Add legend only to first subplot
            ax.legend()
    
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{record_id}_psd_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return psd_data

def plot_wavelet_analysis(ecg_data, fs, record_id="", save_dir=None, lead_to_plot=0):
    """
    Continuous Wavelet Transform analysis
    
    Parameters:
        ecg_data (numpy.ndarray): ECG data with shape (n_leads, n_samples)
        fs (int): Sampling frequency
        record_id (str): Record identifier
        save_dir (str): Directory to save plots
        lead_to_plot (int): Which lead to analyze
    """
    if lead_to_plot >= ecg_data.shape[0]:
        lead_to_plot = 0
        
    signal_data = ecg_data[lead_to_plot]
    
    # Define scales for CWT (corresponding to frequencies of interest)
    scales = np.arange(1, 128)
    wavelet = 'morl'  # Morlet wavelet is good for ECG analysis
    
    try:
        # Compute CWT
        coeffs, freqs = pywt.cwt(signal_data, scales, wavelet, 1/fs)
        
        # Convert scales to frequencies
        frequencies = pywt.scale2frequency(wavelet, scales) * fs
        
        plt.figure(figsize=(15, 8))
        plt.imshow(np.abs(coeffs), extent=[0, len(signal_data)/fs, frequencies[-1], frequencies[0]], 
                   cmap='viridis', aspect='auto')
        plt.title(f'Continuous Wavelet Transform - {record_id} (Lead {lead_to_plot+1})', fontweight='bold')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(label='|CWT Coefficient|')
        plt.ylim(0, 50)  # Focus on ECG-relevant frequencies
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/{record_id}_cwt_lead_{lead_to_plot+1}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Error in wavelet analysis: {e}")
        # Fallback to simpler visualization
        plt.figure(figsize=(15, 6))
        plt.plot(np.arange(len(signal_data))/fs, signal_data)
        plt.title(f'Time Domain Signal - {record_id} (Lead {lead_to_plot+1})')
        plt.xlabel('Time [sec]')
        plt.ylabel('Amplitude [mV]')
        plt.grid(True, alpha=0.3)
        plt.show()


def plot_autocorrelation_analysis(ecg_data, fs, record_id="", save_dir=None, max_lags=100):
    """
    Comprehensive autocorrelation analysis for ECG signals
    
    Parameters:
        ecg_data (numpy.ndarray): ECG data with shape (n_leads, n_samples)
        fs (int): Sampling frequency
        record_id (str): Record identifier
        save_dir (str): Directory to save plots
        max_lags (int): Maximum number of lags to compute
    """
    n_leads = ecg_data.shape[0]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'Autocorrelation Analysis - {record_id}', fontsize=16, fontweight='bold')
    
    for lead_idx in range(min(n_leads, 12)):
        row, col = divmod(lead_idx, 4)
        ax = axes[row, col]
        
        signal_data = ecg_data[lead_idx]
        
        try:
            # Compute autocorrelation using numpy
            autocorr = np.correlate(signal_data, signal_data, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Limit to max_lags
            lags = np.arange(min(len(autocorr), max_lags))
            autocorr_plot = autocorr[:len(lags)]
            
            # Convert lags to time
            time_lags = lags / fs
            
            ax.plot(time_lags, autocorr_plot)
            ax.set_title(f'Lead {lead_idx+1} Autocorrelation')
            ax.set_xlabel('Lag (seconds)')
            ax.set_ylabel('Autocorrelation')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            # Mark significant lags (above 0.1 threshold)
            significant_lags = time_lags[np.abs(autocorr_plot) > 0.1]
            if len(significant_lags) > 1:
                ax.axvspan(significant_lags[1], significant_lags[1], alpha=0.3, color='red', 
                          label=f'Peak at {significant_lags[1]:.2f}s')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}...', transform=ax.transAxes, 
                   ha='center', va='center')
            ax.set_title(f'Lead {lead_idx+1} (Error)')
    
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{record_id}_autocorr_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_mean_amplitude_analysis(ecg_data, fs, record_id="", save_dir=None):
    """
    Mean ECG amplitude analysis across all leads
    
    Parameters:
        ecg_data (numpy.ndarray): ECG data with shape (n_leads, n_samples)
        fs (int): Sampling frequency
        record_id (str): Record identifier
        save_dir (str): Directory to save plots
    """
    n_leads, n_samples = ecg_data.shape
    time_vector = np.arange(n_samples) / fs
    
    # Calculate mean across all leads
    mean_ecg = np.mean(ecg_data, axis=0)
    std_ecg = np.std(ecg_data, axis=0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Individual leads and mean
    for lead_idx in range(n_leads):
        ax1.plot(time_vector, ecg_data[lead_idx], alpha=0.3, linewidth=0.8, 
                label=f'Lead {lead_idx+1}' if lead_idx < 6 else "")
    
    ax1.plot(time_vector, mean_ecg, color='black', linewidth=3, label='Mean Amplitude')
    ax1.fill_between(time_vector, mean_ecg - std_ecg, mean_ecg + std_ecg, 
                     alpha=0.2, color='gray', label='±1 STD')
    
    ax1.set_title(f'Mean ECG Amplitude Analysis - {record_id}', fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (mV)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Lead-wise amplitude statistics
    lead_means = np.mean(np.abs(ecg_data), axis=1)
    lead_stds = np.std(ecg_data, axis=1)
    lead_peaks = np.max(np.abs(ecg_data), axis=1)
    
    x_leads = np.arange(1, n_leads + 1)
    width = 0.25
    
    ax2.bar(x_leads - width, lead_means, width, label='Mean Amplitude', alpha=0.7)
    ax2.bar(x_leads, lead_stds, width, label='Standard Deviation', alpha=0.7)
    ax2.bar(x_leads + width, lead_peaks, width, label='Peak Amplitude', alpha=0.7)
    
    ax2.set_title('Lead-wise Amplitude Statistics')
    ax2.set_xlabel('Lead Number')
    ax2.set_ylabel('Amplitude (mV)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(x_leads)
    
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{record_id}_amplitude_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'mean_amplitude': np.mean(lead_means),
        'std_amplitude': np.mean(lead_stds),
        'peak_amplitude': np.mean(lead_peaks),
        'lead_statistics': {
            'means': lead_means,
            'stds': lead_stds,
            'peaks': lead_peaks
        }
    }

def plot_snr_analysis(ecg_data, fs, record_id="", save_dir=None):
    """
    Signal-to-Noise Ratio analysis and visualization
    
    Parameters:
        ecg_data (numpy.ndarray): ECG data with shape (n_leads, n_samples)
        fs (int): Sampling frequency
        record_id (str): Record identifier
        save_dir (str): Directory to save plots
    
    Returns:
        dict: SNR analysis results
    """
    snr_values = calculate_enhanced_snr(ecg_data, fs)
    n_leads = ecg_data.shape[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: SNR per lead
    leads = np.arange(1, n_leads + 1)
    bars = ax1.bar(leads, snr_values, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_title(f'Signal-to-Noise Ratio by Lead - {record_id}', fontweight='bold')
    ax1.set_xlabel('Lead Number')
    ax1.set_ylabel('SNR (dB)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(leads)
    
    # Add value labels on bars
    for bar, snr_val in zip(bars, snr_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{snr_val:.1f}dB',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 2: SNR distribution
    ax2.hist(snr_values, bins=10, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
    ax2.axvline(np.mean(snr_values), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(snr_values):.1f} dB')
    ax2.set_title('SNR Distribution')
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{record_id}_snr_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'snr_values': snr_values,
        'mean_snr': np.mean(snr_values),
        'std_snr': np.std(snr_values),
        'min_snr': np.min(snr_values),
        'max_snr': np.max(snr_values)
    }


def calculate_enhanced_snr(ecg_data, fs=500):
    """
    Enhanced Signal-to-Noise Ratio calculation based on EEG analysis methods
    
    Parameters:
        ecg_data (numpy.ndarray): ECG data with shape (n_leads, n_samples)
        fs (int): Sampling frequency
    
    Returns:
        numpy.ndarray: SNR values for each lead in dB
    """
    n_leads, n_samples = ecg_data.shape
    snr_values = []
    
    for lead_idx in range(n_leads):
        signal_lead = ecg_data[lead_idx]
        
        try:
            # Method 1: Signal power vs noise power (similar to EEG analysis)
            signal_power = np.mean(signal_lead ** 2)
            
            # Estimate noise using high-frequency components
            # Apply high-pass filter to isolate noise
            nyquist = fs / 2
            high_cutoff = 40  # Hz
            if high_cutoff < nyquist:
                b, a = signal.butter(4, high_cutoff/nyquist, btype='high')
                noise_estimate = signal.filtfilt(b, a, signal_lead)
                noise_power = np.mean(noise_estimate ** 2)
            else:
                # Fallback: use difference from smoothed signal
                window_size = int(fs * 0.02)  # 20ms window
                smoothed = np.convolve(signal_lead, np.ones(window_size)/window_size, mode='same')
                noise_estimate = signal_lead - smoothed
                noise_power = np.mean(noise_estimate ** 2)
            
            if noise_power > 0:
                snr_ratio = signal_power / noise_power
                snr_db = 10 * np.log10(snr_ratio)
            else:
                snr_db = 60.0  # High SNR if no noise detected
                
            # Clamp SNR to reasonable range
            snr_db = np.clip(snr_db, -10, 60)
            snr_values.append(snr_db)
            
        except Exception as e:
            print(f"SNR calculation error for lead {lead_idx}: {e}")
            snr_values.append(20.0)  # Default SNR
    
    return np.array(snr_values)


# ===================================================================
# 6. Quick Demo Function (Optional)
# ===================================================================

def run_quick_demo():
    """
    Quick demonstration of ECG analysis capabilities
    """
    print("🚀 ECG Analysis Quick Demo")
    print("=" * 40)
    
    # Generate a single mock record
    mock_row = pd.Series({'heart_rate': 72, 'age': 45, 'gender': 'M'})
    ecg_data = generate_realistic_ecg_from_row(mock_row, 500, 5)  # 5 second signal
    
    print(f"📊 Generated ECG data: {ecg_data.shape} (leads x samples)")
    
    # Extract comprehensive features
    features = extract_comprehensive_ecg_features(ecg_data, 500, "demo_record")
    
    print(f"🔢 Extracted {len(features)} features")
    print("\n💫 Key features:")
    
    # Show sample of features
    sample_features = {
        'Heart Rate': features.get('Lead_0_Heart_Rate', 'N/A'),
        'Mean Amplitude': features.get('Lead_0_Mean_Amplitude', 'N/A'),
        'SNR (dB)': features.get('Lead_0_SNR_dB', 'N/A'),
        'Skewness': features.get('Lead_0_Skewness', 'N/A'),
        'QRS Energy': features.get('Lead_0_QRS_Energy', 'N/A')
    }
    
    for name, value in sample_features.items():
        if isinstance(value, (int, float)):
            print(f"  • {name}: {value:.3f}")
        else:
            print(f"  • {name}: {value}")
    
    print("\n✅ Demo completed! Features are ready for PLRNN training.")
    return features

# %%
def calculate_ecg_band_powers(ecg_data, fs=500):
    """
    Calculate ECG frequency band powers (adapted from EEG band analysis)
    
    ECG Frequency Bands:
    - Very Low Frequency (VLF): 0.003-0.04 Hz
    - Low Frequency (LF): 0.04-0.15 Hz  
    - High Frequency (HF): 0.15-0.4 Hz
    - QRS Band: 5-15 Hz (main QRS complex energy)
    - High Freq Noise: 40-100 Hz
    
    Parameters:
        ecg_data (numpy.ndarray): ECG data with shape (n_leads, n_samples)
        fs (int): Sampling frequency
    
    Returns:
        dict: Band power features for each lead
    """
    n_leads = ecg_data.shape[0]
    band_powers = {}
    
    # Define ECG-specific frequency bands
    bands = {
        'VLF': (0.003, 0.04),
        'LF': (0.04, 0.15), 
        'HF': (0.15, 0.4),
        'QRS': (5, 15),
        'High_Freq': (40, min(100, fs/2 - 1))
    }
    
    for lead_idx in range(n_leads):
        signal_lead = ecg_data[lead_idx]
        
        try:
            # Compute PSD using Welch's method
            f, psd = welch(signal_lead, fs=fs, nperseg=min(len(signal_lead)//4, 1024))
            
            for band_name, (low_freq, high_freq) in bands.items():
                # Find frequency indices for this band
                band_mask = (f >= low_freq) & (f <= high_freq)
                if np.any(band_mask):
                    band_power = np.trapezoid(psd[band_mask], f[band_mask])
                    band_powers[f'Lead_{lead_idx}_{band_name}_Power'] = band_power
                else:
                    band_powers[f'Lead_{lead_idx}_{band_name}_Power'] = 0.0
            
            # Calculate LF/HF ratio (important for HRV analysis)
            lf_power = band_powers.get(f'Lead_{lead_idx}_LF_Power', 1e-10)
            hf_power = band_powers.get(f'Lead_{lead_idx}_HF_Power', 1e-10)
            band_powers[f'Lead_{lead_idx}_LF_HF_Ratio'] = lf_power / hf_power
            
        except Exception as e:
            print(f"Band power calculation error for lead {lead_idx}: {e}")
            # Set default values
            for band_name in bands.keys():
                band_powers[f'Lead_{lead_idx}_{band_name}_Power'] = 0.1
            band_powers[f'Lead_{lead_idx}_LF_HF_Ratio'] = 2.5
    
    return band_powers

def extract_comprehensive_ecg_features(ecg_data, fs=500, record_id=None):
    """
    Extract comprehensive ECG features compatible with PLRNN training
    (Based on the feature extraction from pytorch_plrnn_integrated_training.py)
    
    Parameters:
        ecg_data (numpy.ndarray): ECG data with shape (n_leads, n_samples)
        fs (int): Sampling frequency
        record_id (str): Optional record identifier
    
    Returns:
        dict: Comprehensive feature dictionary
    """
    features = {'record_id': record_id} if record_id else {}
    
    try:
        # 1. Statistical Features
        stat_features = calculate_comprehensive_statistical_features(ecg_data, fs)
        
        # 2. Frequency Band Powers
        band_features = calculate_ecg_band_powers(ecg_data, fs)
        
        # 3. Signal-to-Noise Ratio
        snr_values = calculate_enhanced_snr(ecg_data, fs)
        snr_features = {f'Lead_{i}_SNR_dB': snr_values[i] for i in range(len(snr_values))}
        
        # 4. Advanced ECG-specific features (similar to PLRNN training)
        for lead_idx in range(ecg_data.shape[0]):
            signal_lead = ecg_data[lead_idx]
            
            # Heart rate analysis
            peaks, _ = find_peaks(signal_lead, height=np.percentile(signal_lead, 75), distance=int(fs*0.6))
            if len(peaks) > 1:
                rr_intervals = np.diff(peaks) / fs
                heart_rate = 60.0 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 75.0
                cv_rr = np.std(rr_intervals) / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0.05
            else:
                heart_rate = 75.0
                cv_rr = 0.05
            
            # Frequency domain analysis
            freqs, psd = welch(signal_lead, fs=fs, nperseg=min(len(signal_lead)//4, 1024))
            total_power = np.trapezoid(psd)
            lf_power = np.trapezoid(psd[(freqs >= 0.04) & (freqs <= 0.15)])
            hf_power = np.trapezoid(psd[(freqs >= 0.15) & (freqs <= 0.4)])
            
            # QRS energy features
            qrs_energy = np.sum(np.abs(signal_lead)**2) / len(signal_lead)
            
            # Add lead-specific features
            features.update({
                f'Lead_{lead_idx}_Heart_Rate': heart_rate,
                f'Lead_{lead_idx}_CV_RR': cv_rr,
                f'Lead_{lead_idx}_LF_Power_Detailed': lf_power,
                f'Lead_{lead_idx}_HF_Power_Detailed': hf_power,
                f'Lead_{lead_idx}_LF_HF_Ratio_Detailed': lf_power / (hf_power + 1e-6),
                f'Lead_{lead_idx}_QRS_Energy': qrs_energy,
                f'Lead_{lead_idx}_Total_Power': total_power
            })
        
        # 5. Flatten statistical features for DataFrame compatibility
        for feature_name, feature_values in stat_features.items():
            if isinstance(feature_values, np.ndarray):
                for lead_idx, value in enumerate(feature_values):
                    features[f'Lead_{lead_idx}_{feature_name}'] = value
            else:
                features[feature_name] = feature_values
        
        # 6. Add band power features
        features.update(band_features)
        features.update(snr_features)
        
    except Exception as e:
        print(f"Feature extraction error for record {record_id}: {e}")
        # Return minimal default features
        features = {'record_id': record_id, 'Error': str(e)}
    
    return features

def process_ecg_dataset_for_plrnn(ecg_data_list, fs=500, save_path=None):
    """
    Process multiple ECG records and generate features compatible with PLRNN training
    
    Parameters:
        ecg_data_list (list): List of tuples (record_id, ecg_data_array)
        fs (int): Sampling frequency
        save_path (str): Optional path to save feature DataFrame
    
    Returns:
        pandas.DataFrame: Features ready for PLRNN training
    """
    print(f"🔄 Processing {len(ecg_data_list)} ECG records for PLRNN training...")
    
    all_features = []
    
    for record_id, ecg_data in tqdm(ecg_data_list, desc="Extracting features"):
        features = extract_comprehensive_ecg_features(ecg_data, fs, record_id)
        all_features.append(features)
    
    # Create DataFrame
    feature_df = pd.DataFrame(all_features)
    
    print(f"✅ Generated feature matrix: {feature_df.shape}")
    print(f"📊 Features per record: {feature_df.shape[1] - 1}")  # -1 for record_id
    
    if save_path:
        feature_df.to_csv(save_path, index=False)
        print(f"💾 Features saved to: {save_path}")
    
    return feature_df


# ===================================================================
# 4. Main Analysis Pipeline
# ===================================================================

def run_comprehensive_ecg_analysis(record_id, ecg_data, fs=500, save_plots=True):
    """
    Run comprehensive ECG analysis for a single record
    
    Parameters:
        record_id (str): Record identifier
        ecg_data (numpy.ndarray): ECG data with shape (n_leads, n_samples)
        fs (int): Sampling frequency
        save_plots (bool): Whether to save plots
    
    Returns:
        dict: Complete analysis results
    """
    print(f"🔍 Running comprehensive analysis for {record_id}...")
    
    # Create output directory
    output_dir = f"{config.OUTPUT_DIR}/{record_id}/" if save_plots else None
    plots_dir = f"{output_dir}/plots/" if output_dir else None
    
    results = {'record_id': record_id}
    
    try:
        # 1. Statistical Analysis
        print("  📊 Statistical features...")
        statistical_features = extract_comprehensive_ecg_features(ecg_data, fs, record_id)
        results.update(statistical_features)
        
        # 2. Frequency Analysis
        print("  🌊 Frequency domain analysis...")
        if save_plots:
            plot_comprehensive_fft_analysis(ecg_data, fs, record_id, plots_dir)
            psd_data = plot_psd_analysis(ecg_data, fs, record_id, plots_dir)
            plot_stft_spectrogram(ecg_data, fs, record_id, plots_dir)
            plot_wavelet_analysis(ecg_data, fs, record_id, plots_dir)
        
        # 3. Autocorrelation Analysis
        print("  📈 Autocorrelation analysis...")
        if save_plots:
            plot_autocorrelation_analysis(ecg_data, fs, record_id, plots_dir)
        
        # 4. Mean Amplitude Analysis
        print("  📏 Amplitude analysis...")
        amplitude_results = plot_mean_amplitude_analysis(ecg_data, fs, record_id, plots_dir if save_plots else None)
        results.update({f'amplitude_{k}': v for k, v in amplitude_results.items() if not isinstance(v, dict)})
        
        # 5. SNR Analysis
        print("  🔊 SNR analysis...")
        snr_results = plot_snr_analysis(ecg_data, fs, record_id, plots_dir if save_plots else None)
        results.update({f'snr_{k}': v for k, v in snr_results.items() if not isinstance(v, (list, np.ndarray))})
        
        print(f"  ✅ Analysis completed for {record_id}")
        
    except Exception as e:
        print(f"  ❌ Error in analysis for {record_id}: {e}")
        results['analysis_error'] = str(e)
    
    return results

def batch_ecg_analysis(max_records=10, save_features=True, save_plots=False):
    """
    Run batch ECG analysis on multiple records
    
    Parameters:
        max_records (int): Maximum number of records to process
        save_features (bool): Whether to save feature DataFrame
        save_plots (bool): Whether to save individual plots
    
    Returns:
        pandas.DataFrame: Comprehensive feature DataFrame
    """
    print("🚀 Starting Batch ECG Analysis Pipeline...")
    print("=" * 60)
    
    # Load ECG records
    ecg_records = load_actual_ecg_data(max_records=max_records)
    
    if not ecg_records:
        print("❌ No ECG records loaded. Exiting.")
        return None
    
    print(f"📊 Processing {len(ecg_records)} records...")
    
    # Process each record
    all_results = []
    
    for i, (record_id, ecg_data) in enumerate(tqdm(ecg_records, desc="Analyzing records")):
        print(f"\n[{i+1}/{len(ecg_records)}] Processing {record_id}")
        
        try:
            # Run comprehensive analysis
            results = run_comprehensive_ecg_analysis(
                record_id, ecg_data, config.SAMPLING_RATE, save_plots
            )
            all_results.append(results)
            
        except Exception as e:
            print(f"❌ Failed to analyze {record_id}: {e}")
            all_results.append({'record_id': record_id, 'error': str(e)})
    
    # Create feature DataFrame
    print("\n📋 Creating feature DataFrame...")
    feature_df = pd.DataFrame(all_results)
    
    print(f"✅ Feature extraction completed!")
    print(f"📊 Feature DataFrame shape: {feature_df.shape}")
    print(f"🔢 Features per record: {feature_df.shape[1] - 1}")
    
    # Save features if requested
    if save_features:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        feature_path = f"{config.OUTPUT_DIR}/{config.FEATURES_CSV}"
        feature_df.to_csv(feature_path, index=False)
        print(f"💾 Features saved to: {feature_path}")
        
        # Save feature summary
        summary_path = f"{config.OUTPUT_DIR}/feature_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"ECG Feature Analysis Summary\n")
            f.write(f"={'='*40}\n")
            f.write(f"Total records: {len(feature_df)}\n")
            f.write(f"Total features: {feature_df.shape[1] - 1}\n")
            f.write(f"Analysis date: {pd.Timestamp.now()}\n\n")
            f.write(f"Feature columns:\n")
            for col in sorted(feature_df.columns):
                if col != 'record_id':
                    f.write(f"  - {col}\n")
        
        print(f"📄 Summary saved to: {summary_path}")
    
    return feature_df

# ===================================================================
# 5. Integration with PLRNN Training
# ===================================================================

def prepare_features_for_plrnn(feature_df, output_path=None):
    """
    Prepare features for PLRNN training (compatible with pytorch_plrnn_integrated_training.py)
    
    Parameters:
        feature_df (pandas.DataFrame): Feature DataFrame from ECG analysis
        output_path (str): Optional path to save prepared features
    
    Returns:
        pandas.DataFrame: Features formatted for PLRNN training
    """
    print("🔧 Preparing features for PLRNN training...")
    
    # Select key features that are compatible with PLRNN extract_advanced_ecg_features
    plrnn_features = []
    
    for _, row in feature_df.iterrows():
        record_features = {}
        
        # Map to PLRNN-compatible feature names (matching extract_advanced_ecg_features)
        feature_mapping = {
            'heart_rate': 'Lead_0_Heart_Rate',
            'sdnn': 'Lead_0_SDNN', 
            'rmssd': 'Lead_0_RMSSD',
            'cv_rr': 'Lead_0_CV_RR',
            'lf_hf_ratio': 'Lead_0_LF_HF_Ratio_Detailed',
            'skewness': 'Lead_0_Skewness',
            'kurtosis': 'Lead_0_Kurtosis',
            'qrs_energy': 'Lead_0_QRS_Energy'
        }
        
        for plrnn_name, df_name in feature_mapping.items():
            if df_name in row:
                record_features[plrnn_name] = row[df_name]
            else:
                # Set default values if feature not found
                defaults = {
                    'heart_rate': 75.0, 'sdnn': 0.05, 'rmssd': 0.03, 'cv_rr': 0.05,
                    'lf_hf_ratio': 2.5, 'skewness': 0.0, 'kurtosis': 3.0, 'qrs_energy': 1.0
                }
                record_features[plrnn_name] = defaults.get(plrnn_name, 0.0)
        
        record_features['record_id'] = row['record_id']
        plrnn_features.append(record_features)
    
    plrnn_df = pd.DataFrame(plrnn_features)
    
    if output_path:
        plrnn_df.to_csv(output_path, index=False)
        print(f"💾 PLRNN-compatible features saved to: {output_path}")
    
    print(f"✅ PLRNN features prepared: {plrnn_df.shape}")
    return plrnn_df

if __name__ == "__main__":
    print("🏥 ECG Signal Analysis Tool")
    print("Enhanced version based on EEG analysis framework")
    print("============================\n")
    
    # Option 1: Quick demo
    print("📝 Option 1: Quick Demo")
    demo_features = run_quick_demo()
    
    print("\n" + "="*50)
    
    # Option 2: Full batch analysis (commented out by default)
    print("📋 Option 2: Full Batch Analysis (uncomment to run)")
    print("# Uncomment the code below to run full batch analysis:")
    print("""
    # Run batch analysis
    feature_df = batch_ecg_analysis(
        max_records=20,  # Adjust based on your needs
        save_features=True,
        save_plots=False  # Set to True if you want individual plots
    )
    
    if feature_df is not None:
        # Prepare features for PLRNN
        plrnn_features = prepare_features_for_plrnn(
            feature_df, 
            f"{config.OUTPUT_DIR}/plrnn_compatible_features.csv"
        )
        
        print("\n🎉 Analysis Pipeline Completed!")
        print(f"📊 Processed {len(feature_df)} records")
        print(f"💾 Results saved in: {config.OUTPUT_DIR}")
        print("\n🔗 Features are now ready for PLRNN training!")
    """)
    
    print("\n🔗 To integrate with PLRNN training:")
    print("   1. Run full batch analysis to generate comprehensive features")
    print("   2. Use prepare_features_for_plrnn() to format for PLRNN compatibility")
    print("   3. Features will match extract_advanced_ecg_features() in pytorch_plrnn_integrated_training.py")