from PIL import Image

def load_image(batch, input_key, output_key, params=None):
    batch[output_key] = [Image.open(p).convert("RGB") for p in batch[input_key]]
    return batch
