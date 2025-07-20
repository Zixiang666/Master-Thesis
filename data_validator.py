#!/usr/bin/env python3
"""
数据验证和统计分析脚本
=====================

用于验证ECG数据读取，修复路径问题，并进行统计分析
结合了原始stats.py的功能
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import wfdb
from scipy import signal, stats
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings('ignore')

# 设置绘图样式
plt.style.use('default')
sns.set_palette("husl")

class ECGDataValidator:
    """ECG数据验证器"""
    
    def __init__(self, csv_path, base_path):
        self.csv_path = csv_path
        self.base_path = base_path
        self.df = None
        self.valid_records = []
        self.invalid_records = []
        
    def load_csv(self):
        """加载CSV文件"""
        print("=== 加载CSV文件 ===")
        try:
            self.df = pd.read_csv(self.csv_path, header=None,
                                 names=['subject_id', 'waveform_path', 'ecg_category'])
            self.df.dropna(inplace=True)
            print(f"✅ 成功加载 {len(self.df)} 条记录")
            
            print("\n类别分布:")
            category_counts = self.df['ecg_category'].value_counts()
            print(category_counts)
            
            return True
        except Exception as e:
            print(f"❌ 加载CSV失败: {e}")
            return False
    
    def check_base_path(self):
        """检查基础路径"""
        print(f"\n=== 检查基础路径 ===")
        if os.path.exists(self.base_path):
            print(f"✅ 基础路径存在: {self.base_path}")
            
            # 列出几个子目录
            subdirs = [d for d in os.listdir(self.base_path) 
                      if os.path.isdir(os.path.join(self.base_path, d))][:5]
            print(f"子目录示例: {subdirs}")
            return True
        else:
            print(f"❌ 基础路径不存在: {self.base_path}")
            
            # 尝试寻找可能的路径
            possible_paths = [
                '/Users/zixiang/ecg/',
                '/Users/zixiang/Downloads/',
                '/Users/zixiang/Desktop/',
                os.path.expanduser('~/Downloads/'),
                os.path.expanduser('~/Desktop/'),
            ]
            
            print("尝试寻找可能的MIMIC数据路径...")
            for path in possible_paths:
                if os.path.exists(path):
                    mimic_dirs = [d for d in os.listdir(path) if 'mimic' in d.lower()]
                    if mimic_dirs:
                        print(f"发现可能的MIMIC目录: {path}")
                        for mimic_dir in mimic_dirs:
                            print(f"  - {os.path.join(path, mimic_dir)}")
            return False
    
    def validate_sample_records(self, sample_size=10):
        """验证样本记录"""
        print(f"\n=== 验证 {sample_size} 个样本记录 ===")
        
        if self.df is None:
            print("❌ 请先加载CSV文件")
            return
        
        sample_df = self.df.sample(n=min(sample_size, len(self.df)))
        
        for idx, (_, row) in enumerate(sample_df.iterrows()):
            print(f"\n--- 样本 {idx+1} ---")
            print(f"患者ID: {row['subject_id']}")
            print(f"类别: {row['ecg_category']}")
            print(f"原始路径: {row['waveform_path']}")
            
            # 处理路径
            relative_path = row['waveform_path']
            if relative_path.endswith('.hea'):
                relative_path = relative_path[:-4]
            elif relative_path.endswith('.dat'):
                relative_path = relative_path[:-4]
            
            full_path = os.path.join(self.base_path, relative_path)
            print(f"完整路径: {full_path}")
            
            # 检查文件存在性
            dat_file = full_path + '.dat'
            hea_file = full_path + '.hea'
            
            if os.path.exists(dat_file) and os.path.exists(hea_file):
                print("✅ 文件存在")
                
                try:
                    # 尝试读取记录
                    record = wfdb.rdrecord(full_path)
                    print(f"✅ 成功读取记录")
                    print(f"   信号形状: {record.p_signal.shape}")
                    print(f"   采样率: {record.fs}")
                    print(f"   导联数: {len(record.sig_name)}")
                    print(f"   导联名称: {record.sig_name}")
                    
                    self.valid_records.append({
                        'index': idx,
                        'path': full_path,
                        'shape': record.p_signal.shape,
                        'fs': record.fs,
                        'category': row['ecg_category']
                    })
                    
                except Exception as e:
                    print(f"❌ 读取记录失败: {e}")
                    self.invalid_records.append({
                        'index': idx,
                        'path': full_path,
                        'error': str(e)
                    })
            else:
                print("❌ 文件不存在")
                missing_files = []
                if not os.path.exists(dat_file):
                    missing_files.append('.dat')
                if not os.path.exists(hea_file):
                    missing_files.append('.hea')
                print(f"   缺失文件: {missing_files}")
                
                self.invalid_records.append({
                    'index': idx,
                    'path': full_path,
                    'error': f"Missing files: {missing_files}"
                })
    
    def analyze_valid_records(self):
        """分析有效记录"""
        if not self.valid_records:
            print("\n❌ 没有有效记录可分析")
            return
        
        print(f"\n=== 分析 {len(self.valid_records)} 个有效记录 ===")
        
        # 信号形状分析
        shapes = [record['shape'] for record in self.valid_records]
        fs_values = [record['fs'] for record in self.valid_records]
        
        print(f"信号形状分布:")
        unique_shapes = list(set(shapes))
        for shape in unique_shapes:
            count = shapes.count(shape)
            print(f"  {shape}: {count} 个记录")
        
        print(f"\n采样率分布:")
        unique_fs = list(set(fs_values))
        for fs in unique_fs:
            count = fs_values.count(fs)
            print(f"  {fs} Hz: {count} 个记录")
        
        # 类别分布
        categories = [record['category'] for record in self.valid_records]
        print(f"\n有效记录的类别分布:")
        unique_categories = list(set(categories))
        for category in unique_categories:
            count = categories.count(category)
            print(f"  {category}: {count} 个记录")
    
    def analyze_ecg_signals(self, num_samples=3):
        """分析ECG信号特征（来自stats.py）"""
        if not self.valid_records:
            print("\n❌ 没有有效记录可分析")
            return
        
        print(f"\n=== ECG信号统计分析 ===")
        
        analyzed_records = []
        for i, record_info in enumerate(self.valid_records[:num_samples]):
            try:
                print(f"\n--- 分析记录 {i+1}: {record_info['category']} ---")
                
                # 读取记录
                record = wfdb.rdrecord(record_info['path'])
                signal = record.p_signal
                
                # 基础统计
                print(f"信号形状: {signal.shape}")
                print(f"采样率: {record.fs} Hz")
                
                # 统计特征（来自stats.py）
                stats_features = self.calculate_statistical_features(signal.T)  # 转置为(leads, samples)
                
                print(f"统计特征:")
                for feature_name, values in stats_features.items():
                    print(f"  {feature_name}: {values}")
                
                # 心率分析（使用导联II）
                if signal.shape[1] >= 2:
                    lead_ii = signal[:, 1]  # 导联II
                    heart_rate_info = self.analyze_heart_rate(lead_ii, record.fs)
                    print(f"心率分析: {heart_rate_info}")
                
                analyzed_records.append({
                    'category': record_info['category'],
                    'stats': stats_features,
                    'heart_rate': heart_rate_info if signal.shape[1] >= 2 else None
                })
                
            except Exception as e:
                print(f"❌ 分析记录失败: {e}")
        
        return analyzed_records
    
    def calculate_statistical_features(self, ecg_data):
        """计算统计特征（来自stats.py）"""
        features = {}
        
        # Activity (Variance)
        features['Variance'] = np.var(ecg_data, axis=1)
        
        # Hjorth Parameters
        activity = features['Variance']
        diff1 = np.diff(ecg_data, axis=1)
        diff2 = np.diff(diff1, axis=1)
        
        mobility = np.sqrt(np.var(diff1, axis=1) / (activity + 1e-10))
        complexity = np.sqrt(np.var(diff2, axis=1) / (np.var(diff1, axis=1) + 1e-10)) / (mobility + 1e-10)
        
        features['Hjorth_Mobility'] = mobility
        features['Hjorth_Complexity'] = complexity
        
        # 基础统计
        features['Mean'] = np.mean(ecg_data, axis=1)
        features['STD'] = np.std(ecg_data, axis=1)
        features['Skewness'] = stats.skew(ecg_data, axis=1)
        features['Kurtosis'] = stats.kurtosis(ecg_data, axis=1)
        
        return features
    
    def analyze_heart_rate(self, lead_signal, fs):
        """心率分析"""
        try:
            # R波检测
            peaks, _ = find_peaks(lead_signal, height=np.std(lead_signal)*0.5, distance=fs//4)
            
            if len(peaks) >= 2:
                rr_intervals = np.diff(peaks) / fs
                heart_rate = 60 / np.mean(rr_intervals)
                hrv = np.std(rr_intervals) * 1000  # ms
                rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2)) * 1000  # ms
                
                return {
                    'heart_rate': round(heart_rate, 2),
                    'hrv_sdnn': round(hrv, 2),
                    'rmssd': round(rmssd, 2),
                    'num_beats': len(peaks)
                }
            else:
                return {'error': 'Insufficient R peaks detected'}
        except Exception as e:
            return {'error': str(e)}
    
    def create_summary_report(self):
        """创建总结报告"""
        print(f"\n{'='*50}")
        print("数据验证总结报告")
        print(f"{'='*50}")
        
        if self.df is not None:
            print(f"总记录数: {len(self.df)}")
            print(f"有效记录数: {len(self.valid_records)}")
            print(f"无效记录数: {len(self.invalid_records)}")
            print(f"有效率: {len(self.valid_records)/len(self.df)*100:.1f}%" if len(self.df) > 0 else "N/A")
        
        print(f"\n基础路径检查: {'✅ 通过' if os.path.exists(self.base_path) else '❌ 失败'}")
        
        if self.valid_records:
            print(f"\n✅ 数据读取正常，可以运行训练")
        else:
            print(f"\n❌ 数据读取失败，需要修复路径配置")
            print(f"请检查以下配置:")
            print(f"1. CSV文件路径: {self.csv_path}")
            print(f"2. MIMIC数据基础路径: {self.base_path}")
    
    def visualize_sample_signals(self, num_samples=2):
        """可视化样本信号"""
        if not self.valid_records:
            print("\n❌ 没有有效记录可可视化")
            return
        
        print(f"\n=== 可视化 {num_samples} 个样本信号 ===")
        
        fig, axes = plt.subplots(num_samples, 2, figsize=(15, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, record_info in enumerate(self.valid_records[:num_samples]):
            try:
                # 读取记录
                record = wfdb.rdrecord(record_info['path'])
                signal = record.p_signal
                time = np.arange(len(signal)) / record.fs
                
                # 绘制原始信号（导联I和II）
                axes[i, 0].plot(time, signal[:, 0], 'b-', alpha=0.8)
                axes[i, 0].set_title(f'{record_info["category"]} - Lead I')
                axes[i, 0].set_xlabel('Time (s)')
                axes[i, 0].set_ylabel('Amplitude (mV)')
                axes[i, 0].grid(True)
                
                if signal.shape[1] > 1:
                    axes[i, 1].plot(time, signal[:, 1], 'r-', alpha=0.8)
                    axes[i, 1].set_title(f'{record_info["category"]} - Lead II')
                    axes[i, 1].set_xlabel('Time (s)')
                    axes[i, 1].set_ylabel('Amplitude (mV)')
                    axes[i, 1].grid(True)
                
            except Exception as e:
                print(f"可视化记录 {i+1} 失败: {e}")
        
        plt.tight_layout()
        plt.savefig('data_validation_samples.png', dpi=300, bbox_inches='tight')
        print("样本信号图保存为: data_validation_samples.png")
        plt.show()


def main():
    """主函数"""
    print("=== ECG数据验证和分析工具 ===")
    
    # 配置路径
    csv_path = '/Users/zixiang/PycharmProjects/Master-Thesis/ecg_5_class_data.csv'
    base_path = '/Users/zixiang/ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/'
    
    # 创建验证器
    validator = ECGDataValidator(csv_path, base_path)
    
    # 1. 加载CSV
    if not validator.load_csv():
        return
    
    # 2. 检查基础路径
    if not validator.check_base_path():
        print("\n请修复基础路径配置后重新运行")
        return
    
    # 3. 验证样本记录
    validator.validate_sample_records(sample_size=20)
    
    # 4. 分析有效记录
    validator.analyze_valid_records()
    
    # 5. ECG信号统计分析
    analyzed_data = validator.analyze_ecg_signals(num_samples=5)
    
    # 6. 可视化样本
    validator.visualize_sample_signals(num_samples=2)
    
    # 7. 创建总结报告
    validator.create_summary_report()
    
    # 8. 建议下一步
    if validator.valid_records:
        print(f"\n{'='*50}")
        print("✅ 数据验证成功！")
        print("建议下一步:")
        print("1. 运行 PyTorch PLRNN 训练:")
        print("   python pytorch_plrnn.py")
        print("2. 调整配置参数以优化性能")
        print("3. 监控训练过程和结果")
    else:
        print(f"\n{'='*50}")
        print("❌ 数据验证失败！")
        print("需要修复以下问题:")
        print("1. 检查 MIMIC-IV-ECG 数据是否正确下载和解压")
        print("2. 更新 pytorch_plrnn.py 中的 ECG_BASE_PATH")
        print("3. 确保 CSV 文件路径正确")


if __name__ == "__main__":
    main()