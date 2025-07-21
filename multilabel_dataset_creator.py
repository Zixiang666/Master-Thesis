#!/usr/bin/env python3
"""
MIMIC-IV-ECG 多标签分类数据集创建器
基于诊断分析结果，创建科学的多标签分类数据集
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict
import json

class ECGMultilabelDatasetCreator:
    def __init__(self, data_file):
        """初始化数据集创建器"""
        self.data_file = data_file
        self.df = None
        self.report_cols = []
        self.label_mapping = {}
        self.category_mapping = {}
        
    def load_data(self):
        """加载数据"""
        print("Loading ECG data...")
        self.df = pd.read_csv(self.data_file, low_memory=False)
        self.report_cols = [col for col in self.df.columns if col.startswith('report_')]
        print(f"Loaded {len(self.df)} records with {len(self.report_cols)} report columns")
        
    def create_label_taxonomy(self):
        """创建标签分类体系"""
        print("Creating label taxonomy...")
        
        # 定义主要疾病类别和对应的关键词模式
        self.category_mapping = {
            'NORMAL': {
                'patterns': ['normal ecg', 'sinus rhythm$', 'normal sinus rhythm'],
                'exclude': ['abnormal', 'irregular', 'variant']
            },
            'ARRHYTHMIA': {
                'patterns': ['tachycardia', 'bradycardia', 'fibrillation', 'flutter', 'arrhythmia', 
                           'pacemaker', 'junctional', 'ventricular rhythm', 'atrial rhythm'],
                'subcategories': {
                    'TACHYCARDIA': ['tachycardia'],
                    'BRADYCARDIA': ['bradycardia'],
                    'ATRIAL_FIBRILLATION': ['atrial fibrillation'],
                    'SUPRAVENTRICULAR': ['supraventricular', 'svt'],
                    'VENTRICULAR': ['ventricular tachycardia', 'v-tach', 'ventricular rhythm']
                }
            },
            'CONDUCTION_ABNORMALITY': {
                'patterns': ['block', 'conduction', 'bundle branch', 'fascicular', 'av block'],
                'subcategories': {
                    'BUNDLE_BRANCH_BLOCK': ['bundle branch block', 'rbbb', 'lbbb'],
                    'AV_BLOCK': ['av block', 'a-v block'],
                    'FASCICULAR_BLOCK': ['fascicular block']
                }
            },
            'MYOCARDIAL_INFARCTION': {
                'patterns': ['infarct', 'mi', 'stemi', 'nstemi', 'myocardial infarction'],
                'subcategories': {
                    'ANTERIOR_MI': ['anterior infarct', 'anterior mi'],
                    'INFERIOR_MI': ['inferior infarct', 'inferior mi'],
                    'LATERAL_MI': ['lateral infarct', 'lateral mi'],
                    'SEPTAL_MI': ['septal infarct', 'septal mi']
                }
            },
            'ISCHEMIA': {
                'patterns': ['ischemia', 'ischemic', 'st elevation', 'st depression'],
                'exclude': ['non-ischemic', 'no ischemia']
            },
            'HYPERTROPHY': {
                'patterns': ['hypertrophy', 'enlargement', 'lvh', 'rvh', 'lah', 'rah'],
                'subcategories': {
                    'LVH': ['left ventricular hypertrophy', 'lvh'],
                    'RVH': ['right ventricular hypertrophy', 'rvh'],
                    'ATRIAL_ENLARGEMENT': ['atrial enlargement', 'lah', 'rah']
                }
            },
            'REPOLARIZATION_ABNORMALITY': {
                'patterns': ['t wave', 'st changes', 'repolarization', 'qt interval'],
                'subcategories': {
                    'T_WAVE_ABNORMAL': ['t wave changes', 't wave abnormality'],
                    'ST_CHANGES': ['st changes', 'st elevation', 'st depression'],
                    'QT_PROLONGATION': ['prolonged qt', 'qt interval']
                }
            },
            'LOW_VOLTAGE': {
                'patterns': ['low voltage', 'low qrs voltage'],
                'subcategories': {
                    'PRECORDIAL_LOW_VOLTAGE': ['low qrs voltages in precordial leads'],
                    'LIMB_LOW_VOLTAGE': ['low qrs voltages in limb leads']
                }
            },
            'AXIS_DEVIATION': {
                'patterns': ['axis deviation', 'left axis', 'right axis', 'leftward axis'],
                'subcategories': {
                    'LEFT_AXIS_DEVIATION': ['left axis deviation', 'leftward axis'],
                    'RIGHT_AXIS_DEVIATION': ['right axis deviation']
                }
            },
            'BORDERLINE_ABNORMAL': {
                'patterns': ['borderline', 'possible', 'probable', 'consider'],
                'note': 'Uncertain findings that need clinical correlation'
            }
        }
        
    def standardize_diagnosis_text(self, text):
        """标准化诊断文本"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower().strip()
        
        # 移除标点符号和多余空格
        text = re.sub(r'[^\w\s-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 标准化常见缩写
        abbreviations = {
            'ecg': 'electrocardiogram',
            'ekg': 'electrocardiogram',
            'lvh': 'left ventricular hypertrophy',
            'rvh': 'right ventricular hypertrophy',
            'lah': 'left atrial hypertrophy',
            'rah': 'right atrial hypertrophy',
            'rbbb': 'right bundle branch block',
            'lbbb': 'left bundle branch block',
            'av': 'atrioventricular'
        }
        
        for abbr, full in abbreviations.items():
            text = re.sub(r'\b' + abbr + r'\b', full, text)
        
        return text
        
    def extract_labels_for_record(self, row):
        """为单条记录提取标签"""
        labels = set()
        raw_diagnoses = []
        
        # 收集所有非空的诊断
        for col in self.report_cols:
            if pd.notna(row[col]):
                diagnosis = self.standardize_diagnosis_text(row[col])
                if diagnosis:
                    raw_diagnoses.append(diagnosis)
        
        # 根据分类体系分配标签
        for category, config in self.category_mapping.items():
            category_found = False
            
            # 检查主要模式
            for diagnosis in raw_diagnoses:
                for pattern in config['patterns']:
                    if re.search(pattern, diagnosis):
                        # 检查排除条件
                        if 'exclude' in config:
                            exclude = any(re.search(excl, diagnosis) for excl in config['exclude'])
                            if exclude:
                                continue
                        
                        labels.add(category)
                        category_found = True
                        
                        # 检查子类别
                        if 'subcategories' in config:
                            for subcat, subpatterns in config['subcategories'].items():
                                for subpattern in subpatterns:
                                    if re.search(subpattern, diagnosis):
                                        labels.add(f"{category}_{subcat}")
                        break
                if category_found:
                    break
        
        return {
            'labels': list(labels),
            'raw_diagnoses': raw_diagnoses,
            'label_count': len(labels)
        }
    
    def create_multilabel_dataset(self, output_file=None, sample_size=None):
        """创建多标签数据集"""
        print("Creating multilabel dataset...")
        
        # 如果指定了样本大小，进行采样
        df_to_process = self.df
        if sample_size and sample_size < len(self.df):
            df_to_process = self.df.sample(n=sample_size, random_state=42)
            print(f"Sampling {sample_size} records from {len(self.df)} total records")
        
        results = []
        for idx, row in df_to_process.iterrows():
            result = self.extract_labels_for_record(row)
            
            record_data = {
                'subject_id': row['subject_id'],
                'study_id': row['study_id'],
                'ecg_time': row['ecg_time'],
                'labels': result['labels'],
                'label_count': result['label_count'],
                'raw_diagnoses': result['raw_diagnoses']
            }
            
            # 添加重要的生理参数
            for param in ['rr_interval', 'qrs_axis', 'p_axis', 't_axis']:
                if param in row.index:
                    record_data[param] = row[param]
            
            results.append(record_data)
        
        # 创建结果DataFrame
        multilabel_df = pd.DataFrame(results)
        
        # 统计信息
        print(f"\nMultilabel Dataset Statistics:")
        print(f"Total records: {len(multilabel_df)}")
        print(f"Records with labels: {(multilabel_df['label_count'] > 0).sum()}")
        print(f"Average labels per record: {multilabel_df['label_count'].mean():.2f}")
        print(f"Max labels per record: {multilabel_df['label_count'].max()}")
        
        # 标签频率统计
        all_labels = []
        for labels in multilabel_df['labels']:
            all_labels.extend(labels)
        
        label_freq = pd.Series(all_labels).value_counts()
        print(f"\nTop 20 Most Frequent Labels:")
        for label, freq in label_freq.head(20).items():
            percentage = (freq / len(multilabel_df)) * 100
            print(f"  {label}: {freq} ({percentage:.2f}%)")
        
        # 保存数据集
        if output_file:
            multilabel_df.to_csv(output_file, index=False)
            print(f"\nDataset saved to: {output_file}")
            
            # 保存标签映射和配置
            config_file = output_file.replace('.csv', '_config.json')
            config_data = {
                'category_mapping': self.category_mapping,
                'label_frequencies': {k: int(v) for k, v in label_freq.to_dict().items()},
                'dataset_stats': {
                    'total_records': int(len(multilabel_df)),
                    'labeled_records': int((multilabel_df['label_count'] > 0).sum()),
                    'avg_labels_per_record': float(multilabel_df['label_count'].mean()),
                    'max_labels_per_record': int(multilabel_df['label_count'].max())
                }
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            print(f"Configuration saved to: {config_file}")
        
        return multilabel_df, label_freq
    
    def create_binary_label_matrix(self, multilabel_df):
        """创建二进制标签矩阵用于机器学习"""
        print("Creating binary label matrix...")
        
        # 获取所有唯一标签
        all_labels = set()
        for labels in multilabel_df['labels']:
            all_labels.update(labels)
        
        all_labels = sorted(list(all_labels))
        
        # 创建二进制矩阵
        label_matrix = np.zeros((len(multilabel_df), len(all_labels)), dtype=int)
        
        for i, labels in enumerate(multilabel_df['labels']):
            for label in labels:
                label_idx = all_labels.index(label)
                label_matrix[i, label_idx] = 1
        
        # 创建标签矩阵DataFrame
        label_df = pd.DataFrame(label_matrix, columns=all_labels)
        
        # 合并基本信息和标签矩阵
        result_df = pd.concat([
            multilabel_df[['subject_id', 'study_id', 'ecg_time']].reset_index(drop=True),
            label_df
        ], axis=1)
        
        print(f"Binary label matrix shape: {label_matrix.shape}")
        print(f"Number of unique labels: {len(all_labels)}")
        
        return result_df, all_labels

def main():
    # 配置参数
    data_file = "/Users/zixiang/ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/machine_measurements.csv"
    output_file = "/Users/zixiang/PycharmProjects/Master-Thesis/mimic_ecg_multilabel_dataset.csv"
    binary_output_file = "/Users/zixiang/PycharmProjects/Master-Thesis/mimic_ecg_binary_labels.csv"
    
    # 创建数据集创建器
    creator = ECGMultilabelDatasetCreator(data_file)
    
    try:
        # 加载数据
        creator.load_data()
        
        # 创建标签分类体系
        creator.create_label_taxonomy()
        
        # 创建多标签数据集（使用样本以提高处理速度）
        multilabel_df, label_freq = creator.create_multilabel_dataset(
            output_file=output_file,
            sample_size=10000  # 处理1万条记录作为示例
        )
        
        # 创建二进制标签矩阵
        binary_df, all_labels = creator.create_binary_label_matrix(multilabel_df)
        binary_df.to_csv(binary_output_file, index=False)
        print(f"Binary label matrix saved to: {binary_output_file}")
        
        print("\n" + "="*80)
        print("MULTILABEL DATASET CREATION COMPLETED")
        print("="*80)
        print("\nFiles created:")
        print(f"1. Multilabel dataset: {output_file}")
        print(f"2. Binary label matrix: {binary_output_file}")
        print(f"3. Configuration file: {output_file.replace('.csv', '_config.json')}")
        
    except Exception as e:
        print(f"Error during dataset creation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()