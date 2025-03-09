from datasets import load_from_disk
import os

def save_dataset(dataset, save_path="processed_dataset"):
    dataset.save_to_disk(save_path)
    print(f"✅ Dataset 已保存到 {save_path}")

def load_processed_dataset(save_path="processed_dataset"):
    dataset = load_from_disk(save_path)
    print(f"✅ 重新加载已处理数据集: {save_path}")
    return dataset
