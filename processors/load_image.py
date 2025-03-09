from PIL import Image

def load_image(batch):
    """ ✅ 读取 `PIL.Image` 并存入 `Dataset`，但不 `encode` """
    batch["image"] = [Image.open(p).convert("RGB") for p in batch["image_path"]]
    return batch
