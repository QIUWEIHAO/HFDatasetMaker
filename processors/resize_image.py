import torchvision.transforms as transforms

resize_resolution = 512
# ✅ 定义 `resize` 处理 pipeline
transform = transforms.Compose([
    transforms.Resize(resize_resolution, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Pad(
        padding=(0, (resize_resolution - int(resize_resolution * (300 / 700))) // 2, 0, 
                    (resize_resolution - int(resize_resolution * (300 / 700)) + 1) // 2),
        fill=(255, 0, 255)  # 用品紅色填充
    ),
    transforms.Resize((resize_resolution, resize_resolution), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])
def resize_image(batch):
    """ ✅ 使用 `torchvision.transforms` 进行 `resize` """
    batch["image"] = [transform(img) for img in batch["image"]]
    return batch
