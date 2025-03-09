from datasets import load_from_disk, Image
from huggingface_hub import HfApi
import os

# ✅ 1. 读取 Hugging Face Token
HF_TOKEN = os.getenv("HF_TOKEN")  # ⚠️ 确保你已设置环境变量
if HF_TOKEN is None:
    raise ValueError("⚠️ 需要 Hugging Face Token，请设置环境变量 `HF_TOKEN`")

api = HfApi(token=HF_TOKEN)

# ✅ 2. 本地加载数据集
dataset_path = "processed_dataset"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"⚠️ 数据集 {dataset_path} 不存在，请检查路径！")

print("✅ 加载本地数据集...")
dataset = load_from_disk(dataset_path)

# ✅ 3. 打印数据集基本信息
print(f"✅ 数据集加载成功！")
print(dataset)
print(dataset[0])  # 打印第一条数据

# ✅ 4. Hugging Face Hub 设置
HF_USERNAME = "qiuweihao"  # ⚠️ 替换为你的 Hugging Face 用户名
HF_DATASET_REPO = f"{HF_USERNAME}/test-dataset-2"  # ⚠️ 替换为你的数据集 Repo 名称

try:
    existing_datasets = api.list_datasets(author=HF_USERNAME)
    existing_repo_ids = set(repo.id for repo in existing_datasets if hasattr(repo, "id"))
except Exception as e:
    print(f"⚠️ 无法获取 Hugging Face 现有数据集: {e}")
    existing_repo_ids = set()  # 失败时假设没有数据集

# ✅ 6. 检查 Hugging Face Repo 是否存在，如果不存在则创建
if HF_DATASET_REPO not in existing_repo_ids:
    print("🚀 数据集不存在，正在创建...")
    api.create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset", exist_ok=True)
    print(f"✅ 数据集 {HF_DATASET_REPO} 创建成功！")
else:
    print(f"✅ 数据集 {HF_DATASET_REPO} 已存在，继续上传...")

# ✅ 7. 上传数据到 Hugging Face Hub
print("🚀 开始同步数据到 Hugging Face Hub...")
dataset = dataset.cast_column("image", Image()) 
dataset.push_to_hub(HF_DATASET_REPO, token=HF_TOKEN)
print(f"✅ 数据集已成功上传至: https://huggingface.co/datasets/{HF_DATASET_REPO}")
