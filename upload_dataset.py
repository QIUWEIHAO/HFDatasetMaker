from datasets import load_from_disk, Dataset, Image
from huggingface_hub import HfApi
import os

def load_local_dataset(dataset_path: str):
    """
    ✅ 加载本地数据集
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"⚠️ 数据集 `{dataset_path}` 不存在，请检查路径！")
    
    print("✅ 加载本地数据集...")
    dataset = load_from_disk(dataset_path)

    print(f"✅ 数据集加载成功！")
    print(dataset)
    print(dataset[0])  # 打印第一条数据
    
    return dataset

def upload_to_huggingface(dataset, hf_username: str, hf_repo_name: str, hf_token: str):
    """
    ✅ 上传 `dataset` 到 Hugging Face Hub
    """
    if hf_token is None:
        raise ValueError("⚠️ 需要 Hugging Face Token，请设置环境变量 `HF_TOKEN`")

    api = HfApi(token=hf_token)
    hf_dataset_repo = f"{hf_username}/{hf_repo_name}"

    # ✅ 检查 Hugging Face Repo 是否存在
    try:
        existing_datasets = api.list_datasets(author=hf_username)
        existing_repo_ids = set(repo.id for repo in existing_datasets if hasattr(repo, "id"))
    except Exception as e:
        print(f"⚠️ 无法获取 Hugging Face 现有数据集: {e}")
        existing_repo_ids = set()  # 假设没有数据集

    # ✅ 如果 repo 不存在，创建数据集
    if hf_dataset_repo not in existing_repo_ids:
        print(f"🚀 数据集 `{hf_dataset_repo}` 不存在，正在创建...")
        api.create_repo(repo_id=hf_dataset_repo, repo_type="dataset", exist_ok=True)
        print(f"✅ 数据集 `{hf_dataset_repo}` 创建成功！")
    else:
        print(f"✅ 数据集 `{hf_dataset_repo}` 已存在，继续上传...")

    # ✅ 上传数据到 Hugging Face Hub
    print("🚀 开始同步数据到 Hugging Face Hub...")
    dataset.push_to_hub(hf_dataset_repo, token=hf_token)
    print(f"✅ 数据集已成功上传至: https://huggingface.co/datasets/{hf_dataset_repo}")
