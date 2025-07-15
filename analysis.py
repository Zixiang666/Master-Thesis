import os
import pandas as pd
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ===================================================================
# 1. 配置部分 (请根据您的真实路径进行修改)
# ===================================================================

# 包含标签和文件路径的CSV文件
LABELED_DATA_CSV = 'heart_rate_labeled_data.csv'

# MIMIC-IV-ECG 波形文件的根目录
ECG_BASE_PATH = '/Users/zixiang/ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/'

# 清洗后数据的输出目录
OUTPUT_DIR = 'processed_data/'


# ===================================================================
# 2. 预处理函数
# ===================================================================

def preprocess_ecg(signal, fs=500):
    """
    对ECG信号进行预处理: 带通滤波和标准化
    """
    # 设计一个0.5Hz到40Hz的带通滤波器，以去除基线漂移和高频噪声
    try:
        nyquist = 0.5 * fs
        low = 0.5 / nyquist
        high = 40 / nyquist
        b, a = butter(1, [low, high], btype='band')

        # 应用滤波器
        filtered_signal = filtfilt(b, a, signal, axis=0)

        # Z-score标准化
        mean = np.mean(filtered_signal, axis=0)
        std = np.std(filtered_signal, axis=0)
        # 加上一个很小的数(epsilon)防止除以零
        normalized_signal = (filtered_signal - mean) / (std + 1e-9)

        return normalized_signal
    except Exception as e:
        print(f"信号处理时出错: {e}")
        # 如果处理失败，返回原始信号或None
        return None


# ===================================================================
# 3. 主执行流程
# ===================================================================

def main():
    print("开始执行ECG数据预处理流程...")

    # --- 加载和切分数据 ---
    try:
        df = pd.read_csv(LABELED_DATA_CSV)
        df.dropna(subset=['heart_rate_category', 'record_name'], inplace=True)
        print(f"成功加载标签文件，共 {len(df)} 条有效记录。")
    except FileNotFoundError:
        print(f"错误: 标签文件 {LABELED_DATA_CSV} 未找到。请确保文件存在。")
        return

    # 按患者ID (subject_id) 切分数据集，防止数据泄露
    train_val_subjects, test_subjects = train_test_split(
        df['subject_id'].unique(), test_size=0.15, random_state=42
    )
    train_subjects, val_subjects = train_test_split(
        train_val_subjects, test_size=0.15, random_state=42
    )

    split_dfs = {
        'train': df[df['subject_id'].isin(train_subjects)],
        'val': df[df['subject_id'].isin(val_subjects)],
        'test': df[df['subject_id'].isin(test_subjects)]
    }

    print(
        f"数据集切分完毕: {len(split_dfs['train'])} 训练, {len(split_dfs['val'])} 验证, {len(split_dfs['test'])} 测试。")

    # --- 循环处理并保存数据 ---
    for split_name, split_df in split_dfs.items():

        # 创建输出子目录
        output_split_dir = os.path.join(OUTPUT_DIR, split_name)
        os.makedirs(output_split_dir, exist_ok=True)

        print(f"\n正在处理 {split_name} 数据集...")

        processed_records = []

        # 使用tqdm来显示进度条
        for index, row in tqdm(split_df.iterrows(), total=len(split_df)):
            try:
                # 构建完整的记录路径
                record_relative_path = os.path.splitext(row['record_name'])[0]
                full_record_path = os.path.join(ECG_BASE_PATH, record_relative_path)

                # 读取原始ECG记录
                record = wfdb.rdrecord(full_record_path)

                # 预处理ECG信号
                processed_signal = preprocess_ecg(record.p_signal, record.fs)

                if processed_signal is not None:
                    # 定义保存处理后数据的文件名
                    output_filename = f"{row['subject_id']}_{os.path.basename(record_relative_path)}.npy"
                    output_filepath = os.path.join(output_split_dir, output_filename)

                    # 保存为.npy文件，这是一种高效的Numpy数组存储格式
                    np.save(output_filepath, processed_signal)

                    # 记录处理成功的信息
                    processed_records.append({
                        'subject_id': row['subject_id'],
                        'record_name': row['record_name'],
                        'label': row['heart_rate_category'],
                        'processed_path': output_filepath
                    })

            except Exception as e:
                print(f"\n处理记录 {row['record_name']} 时出错: {e}")

        # 将处理好的记录信息保存为新的CSV，方便下一阶段调用
        processed_df = pd.DataFrame(processed_records)
        processed_df.to_csv(os.path.join(OUTPUT_DIR, f'{split_name}_processed.csv'), index=False)
        print(f"{split_name} 数据集处理完毕，相关信息已保存。")


if __name__ == '__main__':
    main()