import os
import shutil
from datasets import load_from_disk, concatenate_datasets, Dataset

def meta_processor(batch, processors):
    """ ✅ 批量执行所有 `processors`，避免重复 map() """
    for processor in processors:
        batch = processor(batch)
    return batch

def process_dataset(dataset, processors, save_path, num_proc=4, batch_size=5000):
    """ 
    ✅ 确保 image 只加载一次
    ✅ 每 `batch_size` 处理一次，增量存储
    ✅ 避免 `save_to_disk()` 覆盖原数据，改为 `concatenate_datasets()`
    """

    temp_save_path = save_path + "_tmp"  # ✅ 存到独立的 `processed_dataset_tmp/`

    # ✅ 1. 如果 `processed_dataset/` 存在，加载它（用于拼接）
    if os.path.exists(save_path):
        processed_dataset = load_from_disk(save_path)
        processed_image_paths = set(processed_dataset["image_path"])
        print(f"✅ 发现已处理数据集，已有 {len(processed_image_paths)} 张图片")
        
        # ✅ 2. 过滤掉已处理的图片，避免重复计算
        dataset = dataset.filter(lambda x: x["image_path"] not in processed_image_paths)
    else:
        processed_dataset = None  # 说明是第一次运行

    # ✅ 3. 计算 batch 数量
    num_batches = len(dataset) // batch_size + 1

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(dataset))
        batch = dataset.select(range(start, end))

        print(f"Processing batch {i+1}/{num_batches}...")

        batch = batch.map(
            lambda x: meta_processor(x, processors),
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc
        )

        # ✅ 4. 先存到临时数据集（拼接）
        if processed_dataset:
            processed_dataset = concatenate_datasets([processed_dataset, batch])  # ✅ 拼接新 batch
        else:
            processed_dataset = batch  # ✅ 第一次处理，直接赋值

        # ✅ 5. 存储到 `processed_dataset_tmp/`
        processed_dataset.save_to_disk(temp_save_path)
        print(f"✅ Batch {i+1} 存储完成: {temp_save_path}")

    print(f"✅ 处理完所有 batch，准备替换 `processed_dataset/`")

    # ✅ 6. 替换原 `processed_dataset/`
    if os.path.exists(save_path):
        shutil.rmtree(save_path)  # ✅ 删除旧 `processed_dataset/`
    shutil.move(temp_save_path, save_path)  # ✅ 替换为新数据集

    print(f"✅ 数据处理完成，最终数据存储到: {save_path}")
    return processed_dataset
