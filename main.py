import os
from datasets import load_dataset
from process_pipeline import process_dataset
from save_and_load import load_processed_dataset

# ✅ 1. 选择要使用的 processors
from processors.load_image import load_image
from processors.resize_image import resize_image
from processors.compute_feature import compute_feature
from processors.generate_caption import generate_caption
from processors.encode_image import encode_image

processors = [load_image, generate_caption, resize_image, encode_image]

save_path = "processed_dataset"

# ✅ 2. 先检查是否有已处理的数据集
if os.path.exists(save_path):
    dataset = load_processed_dataset(save_path)
    print("✅ 已加载已处理数据集，增量更新...")
else:
    dataset = load_dataset("json", data_files="dataset/complex_images.json", split="train")

# ✅ 3. 增量处理 `map()`，每 `5000` 张图片存储一次
dataset = process_dataset(dataset, processors, save_path, batch_size=3)

print("✅ 数据集准备完毕，可以上传到云端")
