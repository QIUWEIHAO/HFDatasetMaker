from datasets import Image as HFImage
from PIL import Image

def encode_image(batch):
    """ ✅ 处理完成后，再 `encode` 成 `datasets.Image()` """
    batch["image"] = [HFImage().encode_example(Image.fromarray((img.numpy() * 255).astype("uint8").transpose(1, 2, 0))) for img in batch["image"]]
    return batch
