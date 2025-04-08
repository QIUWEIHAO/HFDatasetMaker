# import os
# from datasets import load_dataset, Image
# from process_pipeline import process_dataset
# from upload_dataset import load_local_dataset, upload_to_huggingface

# # ✅ 1. 选择要使用的 processors
# from processors.load_image import load_image
# from processors.resize_image import resize_image
# from processors.generate_caption import generate_caption
# from processors.encode_image import encode_image
# from processors.deepgaze_feature import deepgaze_process

# processors = [{"processor_function": load_image, "num_proc": 8},
#                 {"processor_function": generate_caption, "num_proc": 1},
#                 {"processor_function": deepgaze_process, "num_proc": 1},
#                 {"processor_function": encode_image, "num_proc": 8}]              

# dataset_path = "processed_dataset"

# # ✅ 读取 Hugging Face Token
# HF_TOKEN = os.getenv("HF_TOKEN")  # 确保已设置
# HF_USERNAME = "qiuweihao"  # ⚠️ 替换为你的 Hugging Face 用户名
# HF_REPO_NAME = "test-dataset-4"  # ⚠️ 替换为你的数据集名称


# if os.path.exists(dataset_path):
#     # remove the dataset_path
#     os.system(f"rm -rf {dataset_path}")
#     print("⚠️ 已存在已处理数据集，重新处理...")

# os.makedirs(dataset_path, exist_ok=True)

# dataset = load_dataset("json", data_files="dataset/complex_images.json", split="train")
# print(dataset)

# dataset = process_dataset(dataset, processors, batch_size=5, load_cache=False)

# dataset = dataset.cast_column("image", Image()) 
# dataset = dataset.cast_column("deepgaze_feature", Image()) 

# dataset.save_to_disk(dataset_path)

# print("✅ 数据集准备完毕，可以上传到云端")

# dataset_test = load_local_dataset(dataset_path)

# upload_to_huggingface(dataset_test, HF_USERNAME, HF_REPO_NAME, HF_TOKEN)


import os
from datasets import load_dataset, Image
from process_pipeline import process_dataset
from upload_dataset import load_local_dataset, upload_to_huggingface

from processors.load_image import load_image
from processors.resize_image import resize_image
from processors.generate_caption import generate_caption
from processors.encode_image import encode_image
from processors.deepgaze_feature import deepgaze_process, process_image_with_deepgaze_batch

dataset_path = "processed_dataset"

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = "qiuweihao"
HF_REPO_NAME = "test-dataset-4"

if os.path.exists(dataset_path):
    os.system(f"rm -rf {dataset_path}")
    print("⚠️ 已存在已处理数据集，重新处理...")

os.makedirs(dataset_path, exist_ok=True)

dataset = load_dataset("json", data_files="dataset/complex_images.json", split="train")
print(dataset)

# ✅ 跑前5张图，保存 npz heatmap
for idx, example in enumerate(dataset):
    if idx >= 5:
        break

    image_path = example["image"] if isinstance(example["image"], str) else example["image"]["path"]
    from PIL import Image
    image = Image.open(image_path).convert("RGB")

    process_image_with_deepgaze_batch(
        image=image,
        num_points=4,
        batch_size=1,
        total_iterations=10,
        original_image_path=image_path
    )

print("✅ 已保存 5 张图对应的 .npz 文件到 diff_output/npz_heatmaps")

# ✅ 正常 pipeline 跑 dataset
processors = [
    {"processor_function": load_image, "num_proc": 8},
    {"processor_function": generate_caption, "num_proc": 1},
    {"processor_function": deepgaze_process, "num_proc": 1},
    {"processor_function": encode_image, "num_proc": 8}
]

dataset = process_dataset(dataset, processors, batch_size=5, load_cache=False)
dataset = dataset.cast_column("image", Image())
dataset = dataset.cast_column("deepgaze_feature", Image())

dataset.save_to_disk(dataset_path)
print("✅ 数据集准备完毕，可以上传到云端")

dataset_test = load_local_dataset(dataset_path)
upload_to_huggingface(dataset_test, HF_USERNAME, HF_REPO_NAME, HF_TOKEN)
