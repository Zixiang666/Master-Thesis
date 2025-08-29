#!/usr/bin/env python3
"""
基于ECGFounder论文的MIMIC-ECG精细化标签系统
============================================
参考2410.04133v4.pdf中ECGFounder的150个标签分类体系
删除UNKNOWN样本，创建科学合理的心脏标签分类
"""

import pandas as pd
import numpy as np
import re
from collections import Counter

class RefinedECGLabelSystem:
    """基于ECGFounder论文的精细化ECG标签系统"""
    
    def __init__(self):
        # 基于ECGFounder论文Table S4的150个标签，选择适合MIMIC的核心标签
        # 优先选择临床重要且在MIMIC数据中常见的标签
        
        self.core_labels = {
            # 1. 正常节律 (Normal Rhythms)
            'NORMAL_SINUS_RHYTHM': {
                'patterns': [r'normal sinus rhythm', r'sinus rhythm$'],
                'exclude': [r'bradycardia', r'tachycardia', r'arrhythmia', r'abnormal']
            },
            'NORMAL_ECG': {
                'patterns': [r'normal ecg', r'normal electrocardiogram'],
                'exclude': [r'except', r'otherwise', r'abnormal']
            },
            
            # 2. 心律失常 - 心动过缓 (Bradyarrhythmias)  
            'SINUS_BRADYCARDIA': {
                'patterns': [r'sinus bradycardia']
            },
            'MARKED_SINUS_BRADYCARDIA': {
                'patterns': [r'marked sinus bradycardia', r'severe bradycardia']
            },
            
            # 3. 心律失常 - 心动过速 (Tachyarrhythmias)
            'SINUS_TACHYCARDIA': {
                'patterns': [r'sinus tachycardia']
            },
            'ATRIAL_FIBRILLATION': {
                'patterns': [r'atrial fibrillation', r'afib', r'a-fib']
            },
            'ATRIAL_FLUTTER': {
                'patterns': [r'atrial flutter']
            },
            'SUPRAVENTRICULAR_TACHYCARDIA': {
                'patterns': [r'supraventricular tachycardia', r'svt']
            },
            'VENTRICULAR_TACHYCARDIA': {
                'patterns': [r'ventricular tachycardia', r'v-tach', r'vtach']
            },
            
            # 4. 传导异常 (Conduction Abnormalities)
            'RIGHT_BUNDLE_BRANCH_BLOCK': {
                'patterns': [r'right bundle branch block', r'rbbb']
            },
            'LEFT_BUNDLE_BRANCH_BLOCK': {
                'patterns': [r'left bundle branch block', r'lbbb']
            },
            'INCOMPLETE_RIGHT_BUNDLE_BRANCH_BLOCK': {
                'patterns': [r'incomplete right bundle branch block', r'incomplete rbbb']
            },
            'LEFT_ANTERIOR_FASCICULAR_BLOCK': {
                'patterns': [r'left anterior fascicular block', r'lafb']
            },
            'AV_BLOCK_FIRST_DEGREE': {
                'patterns': [r'1st degree av block', r'first degree av block', r'prolonged pr']
            },
            
            # 5. 心肌梗死 (Myocardial Infarction)
            'ANTERIOR_INFARCT': {
                'patterns': [r'anterior infarct', r'anterior mi']
            },
            'INFERIOR_INFARCT': {
                'patterns': [r'inferior infarct', r'inferior mi']
            },
            'LATERAL_INFARCT': {
                'patterns': [r'lateral infarct', r'lateral mi']
            },
            'SEPTAL_INFARCT': {
                'patterns': [r'septal infarct', r'septal mi']
            },
            'ACUTE_MI': {
                'patterns': [r'acute mi', r'acute myocardial infarction', r'stemi', r'acute infarct']
            },
            
            # 6. 心室肥大 (Ventricular Hypertrophy)
            'LEFT_VENTRICULAR_HYPERTROPHY': {
                'patterns': [r'left ventricular hypertrophy', r'lvh']
            },
            'RIGHT_VENTRICULAR_HYPERTROPHY': {
                'patterns': [r'right ventricular hypertrophy', r'rvh']
            },
            
            # 7. 电轴偏移 (Axis Deviation)
            'LEFT_AXIS_DEVIATION': {
                'patterns': [r'left axis deviation', r'lad\b']
            },
            'RIGHT_AXIS_DEVIATION': {
                'patterns': [r'right axis deviation', r'rad\b']
            },
            
            # 8. 复极化异常 (Repolarization Abnormalities)
            'NONSPECIFIC_T_WAVE_ABNORMALITY': {
                'patterns': [r'nonspecific t wave abnormality', r't wave abnormality']
            },
            'NONSPECIFIC_ST_ABNORMALITY': {
                'patterns': [r'nonspecific st abnormality', r'st abnormality']
            },
            'ST_ELEVATION': {
                'patterns': [r'st elevation']
            },
            'ST_DEPRESSION': {
                'patterns': [r'st depression']
            },
            
            # 9. 早搏 (Premature Complexes)
            'PREMATURE_VENTRICULAR_COMPLEXES': {
                'patterns': [r'premature ventricular complex', r'pvc', r'ventricular ectopy']
            },
            'PREMATURE_ATRIAL_COMPLEXES': {
                'patterns': [r'premature atrial complex', r'pac', r'atrial ectopy']
            },
            
            # 10. 起搏器节律 (Paced Rhythms)
            'VENTRICULAR_PACED_RHYTHM': {
                'patterns': [r'ventricular.*paced', r'v.*paced']
            },
            'ATRIAL_PACED_RHYTHM': {
                'patterns': [r'atrial.*paced', r'a.*paced']
            },
            
            # 11. 低电压和其他 (Low Voltage & Others)
            'LOW_VOLTAGE_QRS': {
                'patterns': [r'low voltage', r'low qrs voltage']
            },
            'LEFT_ATRIAL_ENLARGEMENT': {
                'patterns': [r'left atrial enlargement', r'lae']
            },
            'RIGHT_ATRIAL_ENLARGEMENT': {
                'patterns': [r'right atrial enlargement', r'rae']
            }
        }
        
        # 计算总标签数
        self.total_labels = len(self.core_labels)
        print(f"📊 精细化标签系统包含 {self.total_labels} 个核心心脏疾病标签")
    
    def preprocess_diagnosis_text(self, text):
        """诊断文本预处理"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        text = text.lower().strip()
        text = re.sub(r'[^\w\s-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 医学术语标准化
        medical_abbrev = {
            'ecg': 'electrocardiogram',
            'mi': 'myocardial infarction',
            'av': 'atrioventricular',
            'lvh': 'left ventricular hypertrophy',
            'rvh': 'right ventricular hypertrophy',
            'rbbb': 'right bundle branch block',
            'lbbb': 'left bundle branch block',
            'afib': 'atrial fibrillation'
        }
        
        for abbrev, full_term in medical_abbrev.items():
            text = re.sub(r'\b' + abbrev + r'\b', full_term, text)
        
        return text
    
    def classify_diagnosis(self, diagnosis_text):
        """使用精细化规则进行诊断分类"""
        if not diagnosis_text:
            return []
        
        labels = []
        processed_text = self.preprocess_diagnosis_text(diagnosis_text)
        
        for label, rule_config in self.core_labels.items():
            # 检查匹配模式
            for pattern in rule_config['patterns']:
                if re.search(pattern, processed_text):
                    # 检查排除条件
                    if 'exclude' in rule_config:
                        excluded = any(re.search(excl, processed_text) 
                                     for excl in rule_config['exclude'])
                        if not excluded:
                            labels.append(label)
                            break
                    else:
                        labels.append(label)
                        break
        
        return labels
    
    def analyze_mimic_labels(self, mimic_path):
        """分析MIMIC数据集的标签分布"""
        print("🔄 分析MIMIC数据集标签分布...")
        
        df = pd.read_csv(mimic_path)
        
        all_diagnoses = []
        for i in range(18):  # report_0 to report_17
            col = f'report_{i}'
            if col in df.columns:
                diagnoses = df[col].dropna().tolist()
                all_diagnoses.extend(diagnoses)
        
        # 应用新的标签系统
        label_counts = Counter()
        processed_records = 0
        
        for diagnosis in all_diagnoses:
            labels = self.classify_diagnosis(diagnosis)
            for label in labels:
                label_counts[label] += 1
            processed_records += 1
            
            if processed_records % 10000 == 0:
                print(f"  已处理 {processed_records} 条诊断...")
        
        print(f"✅ 处理完成，共 {processed_records} 条诊断")
        print(f"📊 识别出的标签分布:")
        
        total_labels = sum(label_counts.values())
        for label, count in label_counts.most_common():
            percentage = count / total_labels * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
        
        return label_counts
    
    def generate_clean_dataset(self, mimic_path, output_path):
        """生成清理后的数据集（删除UNKNOWN样本）"""
        print("🔄 生成精细化清理数据集...")
        
        # 加载数据
        record_list = pd.read_csv(f"{mimic_path}/record_list.csv")
        note_links = pd.read_csv(f"{mimic_path}/waveform_note_links.csv")
        measurements = pd.read_csv(f"{mimic_path}/machine_measurements.csv")
        
        # 合并数据
        merged = record_list.merge(note_links, on='study_id', how='inner')
        merged = merged.merge(measurements, on='study_id', how='left')
        
        # 按患者去重
        patient_groups = merged.groupby('subject_id')
        clean_records = []
        
        for subject_id, group in patient_groups:
            latest_record = group.sort_values('study_id').iloc[-1]
            
            # 提取诊断文本
            diagnosis_text = ""
            for i in range(18):
                col = f'report_{i}'
                if col in latest_record and pd.notna(latest_record[col]):
                    diagnosis_text += str(latest_record[col]) + " "
            
            # 应用新标签系统
            labels = self.classify_diagnosis(diagnosis_text)
            
            # 只保留有标签的记录（删除UNKNOWN）
            if labels:
                # 创建独热编码
                label_vector = np.zeros(self.total_labels)
                label_names = list(self.core_labels.keys())
                
                for label in labels:
                    if label in label_names:
                        idx = label_names.index(label)
                        label_vector[idx] = 1
                
                clean_record = {
                    'subject_id': latest_record['subject_id'],
                    'study_id': latest_record['study_id'],
                    'labels': label_vector,
                    'label_names': labels,
                    'label_count': len(labels),
                    'diagnosis_text': diagnosis_text.strip(),
                    'ecg_file_path': f"{mimic_path}/files/p{str(latest_record['subject_id'])[:4]}/p{latest_record['subject_id']}/s{latest_record['study_id']}/{latest_record['study_id']}"
                }
                clean_records.append(clean_record)
        
        print(f"✅ 生成清理数据集: {len(clean_records)} 条有效记录")
        
        # 保存数据集
        clean_df = pd.DataFrame(clean_records)
        clean_df.to_csv(f"{output_path}/refined_ecg_dataset.csv", index=False)
        
        # 生成统计信息
        all_labels = []
        for record in clean_records:
            all_labels.extend(record['label_names'])
        
        stats = {
            'total_patients': len(clean_records),
            'total_labels': len(all_labels),
            'unique_labels': len(set(all_labels)),
            'label_distribution': dict(Counter(all_labels)),
            'avg_labels_per_patient': len(all_labels) / len(clean_records)
        }
        
        # 保存统计
        import json
        with open(f"{output_path}/refined_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        return clean_records, stats

def main():
    """主函数"""
    print("🚀 MIMIC-ECG精细化标签系统")
    print("基于ECGFounder论文(2410.04133v4.pdf)")
    print("=" * 60)
    
    # 初始化精细化标签系统
    label_system = RefinedECGLabelSystem()
    
    # 设置路径
    mimic_path = "/Users/zixiang/ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"
    output_path = "/Users/zixiang/PycharmProjects/Master-Thesis/refined_ecg_data"
    
    import os
    os.makedirs(output_path, exist_ok=True)
    
    # 分析MIMIC标签分布
    measurements_path = f"{mimic_path}/machine_measurements.csv"
    label_counts = label_system.analyze_mimic_labels(measurements_path)
    
    # 生成清理数据集
    clean_records, stats = label_system.generate_clean_dataset(mimic_path, output_path)
    
    print("\n🎉 精细化处理完成！")
    print(f"📊 最终数据集统计:")
    print(f"  总患者数: {stats['total_patients']}")
    print(f"  平均每患者标签数: {stats['avg_labels_per_patient']:.2f}")
    print(f"  标签类别数: {stats['unique_labels']}")
    
    print(f"\n📁 输出文件:")
    print(f"  数据集: {output_path}/refined_ecg_dataset.csv")
    print(f"  统计: {output_path}/refined_statistics.json")

if __name__ == "__main__":
    main()