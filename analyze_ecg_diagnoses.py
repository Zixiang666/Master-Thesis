#!/usr/bin/env python3
"""
分析MIMIC-IV-ECG数据集中的诊断信息
用于创建多标签分类数据集的数据探索
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

def load_data(file_path):
    """加载ECG诊断数据"""
    print("正在加载数据...")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"数据集大小: {df.shape}")
    return df

def analyze_report_columns(df):
    """分析报告列的基本信息"""
    report_cols = [col for col in df.columns if col.startswith('report_')]
    print(f"\n报告列数量: {len(report_cols)}")
    
    # 统计每个报告列的非空值数量
    report_stats = {}
    for col in report_cols:
        non_null_count = df[col].notna().sum()
        total_count = len(df)
        report_stats[col] = {
            'non_null': non_null_count,
            'percentage': (non_null_count / total_count) * 100
        }
    
    print("\n各报告列的非空统计:")
    for col, stats in report_stats.items():
        print(f"{col}: {stats['non_null']:,} ({stats['percentage']:.2f}%)")
    
    return report_cols, report_stats

def extract_all_diagnoses(df, report_cols):
    """提取所有诊断术语"""
    all_diagnoses = []
    
    for col in report_cols:
        diagnoses = df[col].dropna().tolist()
        all_diagnoses.extend(diagnoses)
    
    return all_diagnoses

def analyze_diagnosis_frequency(all_diagnoses):
    """分析诊断术语频率"""
    diagnosis_counter = Counter(all_diagnoses)
    print(f"\n独特诊断术语总数: {len(diagnosis_counter)}")
    print(f"诊断实例总数: {len(all_diagnoses)}")
    
    # 显示最常见的诊断
    print("\n最常见的诊断术语 (Top 30):")
    for diagnosis, count in diagnosis_counter.most_common(30):
        percentage = (count / len(all_diagnoses)) * 100
        print(f"{diagnosis}: {count:,} ({percentage:.2f}%)")
    
    return diagnosis_counter

def analyze_multilabel_patterns(df, report_cols):
    """分析多标签模式"""
    print("\n=== 多标签模式分析 ===")
    
    # 统计每条记录有多少个诊断标签
    label_counts = []
    diagnosis_combinations = []
    
    for idx, row in df.iterrows():
        labels = []
        for col in report_cols:
            if pd.notna(row[col]):
                labels.append(row[col])
        
        label_counts.append(len(labels))
        if len(labels) > 1:
            diagnosis_combinations.append(tuple(sorted(labels)))
    
    # 统计标签数量分布
    label_count_dist = Counter(label_counts)
    print("\n每条记录的诊断标签数量分布:")
    for count, freq in sorted(label_count_dist.items()):
        percentage = (freq / len(df)) * 100
        print(f"{count} 个标签: {freq:,} 条记录 ({percentage:.2f}%)")
    
    # 分析常见的诊断组合
    combo_counter = Counter(diagnosis_combinations)
    print(f"\n多标签组合总数: {len(combo_counter)}")
    print("\n最常见的诊断组合 (Top 20):")
    for combo, count in combo_counter.most_common(20):
        percentage = (count / len(diagnosis_combinations)) * 100
        print(f"{' + '.join(combo)}: {count} ({percentage:.2f}%)")
    
    return label_count_dist, combo_counter

def identify_cardiac_conditions(diagnosis_counter):
    """识别主要心脏疾病类别"""
    print("\n=== 心脏疾病类别识别 ===")
    
    # 定义心脏疾病关键词
    condition_patterns = {
        'Arrhythmia': ['arrhythmia', 'rhythm', 'tachycardia', 'bradycardia', 'fibrillation', 'flutter'],
        'Ischemia': ['ischemia', 'ischemic', 'infarction', 'myocardial', 'MI'],
        'Conduction': ['block', 'conduction', 'bundle', 'av block', 'sa block'],
        'Hypertrophy': ['hypertrophy', 'enlargement', 'lvh', 'rvh', 'lah', 'rah'],
        'Normal': ['normal', 'sinus rhythm'],
        'Abnormal_morphology': ['abnormal', 'abnormality', 'variant', 'deviation'],
        'ST_changes': ['st elevation', 'st depression', 'st changes', 't wave'],
        'QRS_abnormalities': ['qrs', 'low voltage', 'poor r wave progression']
    }
    
    categorized_diagnoses = defaultdict(list)
    
    for diagnosis, count in diagnosis_counter.items():
        diagnosis_lower = diagnosis.lower()
        for category, patterns in condition_patterns.items():
            if any(pattern in diagnosis_lower for pattern in patterns):
                categorized_diagnoses[category].append((diagnosis, count))
                break
        else:
            categorized_diagnoses['Other'].append((diagnosis, count))
    
    print("\n各类别诊断统计:")
    for category, diagnoses in categorized_diagnoses.items():
        total_count = sum(count for _, count in diagnoses)
        percentage = (total_count / len(all_diagnoses)) * 100
        print(f"\n{category}: {len(diagnoses)} 种诊断, {total_count:,} 次出现 ({percentage:.2f}%)")
        
        # 显示该类别最常见的诊断
        diagnoses_sorted = sorted(diagnoses, key=lambda x: x[1], reverse=True)[:10]
        for diagnosis, count in diagnoses_sorted:
            print(f"  - {diagnosis}: {count:,}")
    
    return categorized_diagnoses

def suggest_multilabel_strategy(diagnosis_counter, categorized_diagnoses, combo_counter):
    """提供多标签数据集创建策略建议"""
    print("\n" + "="*80)
    print("多标签分类数据集创建建议")
    print("="*80)
    
    # 建议1: 基于频率的标签选择
    print("\n1. 基于频率的标签选择策略:")
    print("   - 选择出现频率 >= 1000 的诊断作为主要标签")
    high_freq_diagnoses = [(d, c) for d, c in diagnosis_counter.most_common() if c >= 1000]
    print(f"   - 高频诊断数量: {len(high_freq_diagnoses)}")
    
    # 建议2: 基于类别的标签体系
    print("\n2. 基于疾病类别的多级标签体系:")
    for category, diagnoses in categorized_diagnoses.items():
        if category != 'Other':
            print(f"   - {category}: {len(diagnoses)} 种诊断")
    
    # 建议3: 多标签共现分析
    multilabel_records = sum(1 for combo in combo_counter if len(combo) > 1)
    total_records = len(combo_counter) + (len(all_diagnoses) - sum(combo_counter.values()))
    multilabel_percentage = (multilabel_records / total_records) * 100
    
    print(f"\n3. 多标签共现情况:")
    print(f"   - 多标签记录占比: {multilabel_percentage:.2f}%")
    print(f"   - 最常见的组合模式: {len([c for c in combo_counter.values() if c >= 50])} 种")
    
    # 建议4: 数据预处理策略
    print("\n4. 数据预处理建议:")
    print("   - 标准化诊断术语 (去除时间、程度修饰词)")
    print("   - 合并相似诊断 (如不同程度的心动过速)")
    print("   - 处理否定表述 (如 'No evidence of...')")
    print("   - 创建层次化标签结构")
    
    return high_freq_diagnoses

if __name__ == "__main__":
    # 数据文件路径
    data_file = "/Users/zixiang/ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/machine_measurements.csv"
    
    try:
        # 加载数据
        df = load_data(data_file)
        
        # 分析报告列
        report_cols, report_stats = analyze_report_columns(df)
        
        # 提取所有诊断
        all_diagnoses = extract_all_diagnoses(df, report_cols)
        
        # 分析诊断频率
        diagnosis_counter = analyze_diagnosis_frequency(all_diagnoses)
        
        # 分析多标签模式
        label_count_dist, combo_counter = analyze_multilabel_patterns(df, report_cols)
        
        # 识别心脏疾病类别
        categorized_diagnoses = identify_cardiac_conditions(diagnosis_counter)
        
        # 提供策略建议
        high_freq_diagnoses = suggest_multilabel_strategy(diagnosis_counter, categorized_diagnoses, combo_counter)
        
        print("\n" + "="*80)
        print("分析完成！")
        print("="*80)
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()