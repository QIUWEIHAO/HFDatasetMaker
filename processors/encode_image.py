from datasets import Image as HFImage
from PIL import Image

def encode_image(batch):
    """ ✅ 处理完成后，再 `encode` 成 `datasets.Image()` """
    batch["image"] = [HFImage().encode_example(img) for img in batch["image"]]
    return batch
