import os
from datasets import load_dataset, Image
from process_pipeline import process_dataset
from upload_dataset import load_local_dataset, upload_to_huggingface

from processors.load_image import load_image
from processors.resize_image import resize_image
from processors.generate_caption import generate_caption
from processors.deepgaze_feature import deepgaze_process

# processors = [{"processor_function": load_image, "num_proc": 8},
#                 {"processor_function": generate_caption, "num_proc": 1},
#                 # {"processor_function": deepgaze_process, "num_proc": 1},
#                 {"processor_function": encode_image, "num_proc": 8}]              

dataset_path = "processed_dataset_playground"

# ✅ 读取 Hugging Face Token
HF_TOKEN = os.getenv("HF_TOKEN")  # 确保已设置
HF_USERNAME = "qiuweihao"  # ⚠️ 替换为你的 Hugging Face 用户名
HF_REPO_NAME = "test-dataset-5"  # ⚠️ 替换为你的数据集名称


os.makedirs(dataset_path, exist_ok=True)

dataset = load_dataset("json", data_files="dataset/complex_images.json", split="train")
# dataset = dataset.select(range(0,4))
# print(dataset[0])

# dataset = load_dataset("processed_dataset_playground")

print("✅running load_image")
dataset = dataset.map(
    lambda batch: load_image(batch, input_key="image_path", output_key="image_original"),
    batched=True,
    batch_size= 4,
    num_proc= 7,
    load_from_cache_file=True
)
# print(dataset[0])

print("✅running resize_image")
dataset = dataset.map(
    lambda batch: resize_image(batch, input_key="image_original", output_key="image_resized", max_dims=(1024, 768)),
    batched=True,
    batch_size= 4,
    num_proc= 7,
    load_from_cache_file=True
)
# print(dataset[0])

print("✅running generate_caption")
dataset = dataset.map(
    lambda batch: generate_caption(batch, input_key="image_resized", output_key="caption"),
    batched=True,
    batch_size= 1,
    num_proc= 1,
    load_from_cache_file=True
)
# print(dataset[0])

print("✅running deepgaze_feature")
dataset = dataset.map(
    lambda batch: deepgaze_process(batch, input_key="image_resized", output_key="deepgaze_feature", num_points=4, batch_random_size=1, total_iterations=10),
    batched=True,
    batch_size=1,
    num_proc= 1,
    load_from_cache_file=False
)
# print(dataset[0])

image_columns = ["image_original", "image_resized", "deepgaze_feature"]
for image_column in image_columns:
    if image_column not in dataset.column_names:
        continue
    print(f"✅ Cast {image_column} to Image")
    dataset = dataset.cast_column(image_column, Image())

upload_to_huggingface(dataset, HF_USERNAME, HF_REPO_NAME, HF_TOKEN)
