import os
import glob
import argparse
from datasets import Dataset, Image

def combine_parquet(chunk_dir, output_dir, image_columns):
    """合并 Parquet chunks 并保存"""
    parquet_files = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.parquet")))

    if not parquet_files:
        raise ValueError("❌ 没有找到 Parquet chunks，请先运行 `process_dataset.py`")

    dataset = Dataset.from_parquet(parquet_files)

    # **转换指定列为 Image 类型**
    if image_columns:
        image_columns = image_columns.split(",")  # 逗号分隔的列名
        for column in image_columns:
            if column in dataset.column_names:
                dataset = dataset.cast_column(column, Image())
                print(f"✅ 已将 `{column}` 转换为 Image() 类型")

    dataset.save_to_disk(output_dir)
    print(f"✅ 合并完成，共 {len(dataset)} 条数据，并已保存到 {output_dir}")
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Combine Parquet chunks into a single dataset.")
    parser.add_argument("--chunk-dir", type=str, required=True, help="Directory containing Parquet chunks")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the combined dataset")
    parser.add_argument("--cast-image-columns", type=str, default="", help="Comma-separated list of columns to cast to Image()")

    args = parser.parse_args()
    
    combine_parquet(args.chunk_dir, args.output_dir, args.cast_image_columns)

if __name__ == "__main__":
    main()
