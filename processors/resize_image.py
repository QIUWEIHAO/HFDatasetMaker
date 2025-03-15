from PIL import Image

def resize_max(image, max_width, max_height):
    # 获取原始尺寸
    orig_width, orig_height = image.size
    
    # 计算缩放比例
    scale_w = max_width / orig_width
    scale_h = max_height / orig_height
    scale = min(scale_w, scale_h)  # 确保不会超过 1920x1080
    
    # 计算新尺寸，确保一边等于 1920 或 1080
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)

    # 使用高质量缩放
    return image.resize((new_width, new_height), Image.LANCZOS)

def resize_image(batch, input_key, output_key, params={"max_dims": [1920, 1080]}):
    max_dims = params["max_dims"]
    batch[output_key] = [resize_max(img, max_width=max_dims[0], max_height= max_dims[1]) for img in batch[input_key]]
    return batch
