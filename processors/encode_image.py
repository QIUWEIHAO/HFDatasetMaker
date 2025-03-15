from datasets import Image as HFImage
from PIL import Image

def encode_image(batch, input_key, output_key, params=None):
    """ ✅ 处理完成后，再 `encode` 成 `datasets.Image()` """
    batch[output_key] = [HFImage().encode_example(img) for img in batch[input_key]]
    return batch
