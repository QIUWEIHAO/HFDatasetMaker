import os
import glob
import json
import argparse
import pyarrow.parquet as pq
import pyarrow as pa
from datasets import load_dataset, Dataset
from tqdm import tqdm

from processors.load_image import load_image
from processors.resize_image import resize_image
from processors.generate_caption import generate_caption
from processors.deepgaze_feature import deepgaze_process
from combine_parquet import combine_parquet
from upload_dataset import upload_to_huggingface

# **解析命令行参数**
parser = argparse.ArgumentParser(description="Process dataset into Parquet chunks.")
parser.add_argument("--combine", action="store_true", help="Combine Parquet chunks into a single dataset.")
parser.add_argument("--upload", action="store_true", help="Upload the dataset to Hugging Face.")
parser.add_argument("--resume", action="store_true", help="Resume processing from last completed chunk.")
parser.add_argument("--buffer_size", type=int, default=None, help="Batch size for processing data.")
parser.add_argument("--output_dir", type=str, default="parquet_chunks", help="Directory to store Parquet chunks.")
parser.add_argument("--dataset_path", type=str, default="dataset/complex_images.json", help="Path to the dataset file.")
parser.add_argument("--hf-token", type=str, help="Hugging Face API Token")
parser.add_argument("--repo-name", type=str, help="Hugging Face repository name")
args = parser.parse_args()

# **确保输出目录存在**
os.makedirs(args.output_dir, exist_ok=True)

# **检查 `meta.json` 是否存在**
meta_path = os.path.join(args.output_dir, "meta.json")
if args.resume and os.path.exists(meta_path):
    with open(meta_path, "r") as f:
        meta_data = json.load(f)
    buffer_size = meta_data["buffer_size"]
    dataset_size = meta_data["dataset_size"]
    print(f"🔄 读取已存的参数: buffer_size={buffer_size}, dataset_size={dataset_size}")
else:
    buffer_size = args.buffer_size if args.buffer_size else 4  # 默认 `4`
    dataset_size = None

# 1️⃣ **读取种子数据库**
dataset = load_dataset("json", data_files=args.dataset_path, split="train")

# **如果首次运行，计算数据集大小**
if not dataset_size:
    dataset_size = len(dataset)
    with open(meta_path, "w") as f:
        json.dump({"buffer_size": buffer_size, "dataset_size": dataset_size}, f)
    print(f"✅ 存储 `meta.json`，buffer_size={buffer_size}, dataset_size={dataset_size}")

# **获取已经处理过的 chunks**
existing_chunks = sorted(glob.glob(os.path.join(args.output_dir, "chunk_*.parquet")))
# print(existing_chunks[0].split("_")[-3])
processed_chunk_indices = {
    int(f.split("_")[-3]) for f in existing_chunks  # 解析 `start` 值
}

# **计算 resume 进度**
start_chunk_idx = max(processed_chunk_indices) + 1 if args.resume and processed_chunk_indices else 0

if args.resume and start_chunk_idx > 0:
    print(f"🔄 继续处理，从 chunk {start_chunk_idx} 开始...")

# 2️⃣ **处理 pipeline**
processors = [
    {"function": load_image, "input": "image_path", "output": "image_original", "params": {"num_proc": 4}},
    {"function": resize_image, "input": "image_original", "output": "image_resized", "params": {"num_proc": 4, "max_dims": (1024, 768)}},
    {"function": generate_caption, "input": "image_resized", "output": "caption", "params": {"num_proc": 1}},
    {"function": deepgaze_process, "input": "image_original", "output": "deepgaze_feature", "params": {"num_proc": 1, "num_points": 4, "batch_random_size": 1, "total_iterations": 10, "centerbias": "zeros"}},
]

# 3️⃣ **批量处理数据**
def process_and_save_buffer(batch_dataset, start, end, total_processed):
    """处理一个 buffer 并存储为 Parquet"""
    chunk_idx = start // buffer_size

    with tqdm(total=len(processors), desc=f"🛠 处理 Chunk {chunk_idx}", position=1, leave=False) as pbar:
        for processor in processors:
            func = processor["function"]
            input_key = processor["input"]
            output_key = processor["output"]
            params = processor["params"]

            batch_dataset = batch_dataset.map(
                lambda batch: func(batch, input_key=input_key, output_key=output_key, params=params),
                batched=True,
                batch_size=1,
                num_proc=params["num_proc"],
                load_from_cache_file=False,
                desc=f"🔄 {func.__name__} (num_proc={params['num_proc']})"
            )
            pbar.update(1)

    buffer = batch_dataset.to_list()
    table = pa.Table.from_pylist(buffer)

    output_file = os.path.join(args.output_dir, f"chunk_{start}_{end}_of_{dataset_size}.parquet")
    pq.write_table(table, output_file)

    total_processed += len(buffer)
    progress = (total_processed / dataset_size) * 100
    tqdm.write(f"✅ 存储 {len(buffer)} 条数据到 {output_file} - 进度: {progress:.2f}%")
    
    return total_processed

# **批量读取数据并处理**
total_processed = start_chunk_idx * buffer_size
with tqdm(total=dataset_size, desc="📊 处理数据集进度", position=0) as dataset_pbar:
    for start in range(start_chunk_idx * buffer_size, dataset_size, buffer_size):
        end = min(start + buffer_size, dataset_size)

        # **检查是否已处理**
        chunk_filename = f"chunk_{start}_{end}_of_{dataset_size}.parquet"
        if args.resume and os.path.exists(os.path.join(args.output_dir, chunk_filename)):
            continue  # **跳过已处理的 chunk**
        
        batch_dataset = dataset.select(range(start, end))  # **✅ 更快地获取数据**
        total_processed = process_and_save_buffer(batch_dataset, start, end, total_processed)
        dataset_pbar.update(end - start)

# **调用合并 & 上传**
if args.combine:
    dataset = combine_parquet(args.output_dir, output_dir=args.output_dir+"_combined", image_columns="image_resized,image_original,deepgaze_feature")


if args.upload:
    if not args.combine:
        dataset = combine_parquet(args.output_dir)
    HF_USERNAME = "qiuweihao"
    HF_REPO_NAME = args.repo_name if args.repo_name else args.output_dir
    HF_TOKEN = args.hf_token
    upload_to_huggingface(dataset, HF_USERNAME, HF_REPO_NAME, HF_TOKEN)
