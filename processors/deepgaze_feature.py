import os
import torch
import numpy as np
from PIL import Image as PILImage
from datasets import Image as HFImage

from datasets import Dataset
import deepgaze_pytorch
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(device)

resize_resolution = 512
# ✅ 定义 `resize` 处理 pipeline
transform = transforms.Compose([
    transforms.Resize(resize_resolution, interpolation=transforms.InterpolationMode.BILINEAR),
    # transforms.Pad(
    #     padding=(0, (resize_resolution - int(resize_resolution * (300 / 700))) // 2, 0, 
    #                 (resize_resolution - int(resize_resolution * (300 / 700)) + 1) // 2),
    #     fill=(255, 0, 255)  # 用品紅色填充
    # ),
    # transforms.Resize((resize_resolution, resize_resolution), interpolation=transforms.InterpolationMode.BILINEAR),
    # transforms.ToTensor(),
])

def process_image_with_deepgaze(image, num_points=4, num_iterations=10):
    image = transform(image)
    image_np = np.array(image, dtype=np.float32)  # ✅ 确保 `numpy` 数据类型正确
    print(image_np.shape)
    H, W = image_np.shape[:2]  # 取得图片高宽

    # 生成 num_iterations 个随机 fixation points
    points = np.random.randint(0, [W, H], size=(num_iterations, num_points, 2))
    fixation_history_x = points[:, :, 0].astype(np.float32)  # 形状: (num_iterations, num_points)
    fixation_history_y = points[:, :, 1].astype(np.float32)

    # 批量化 centerbias 和 image tensor
    centerbias = np.zeros((num_iterations, H, W), dtype=np.float32)  # (batch_size, H, W)
    image_batch = np.repeat(image_np[None, :, :, :], num_iterations, axis=0)  # 形状: (batch_size, H, W, C)

    # 转换为 PyTorch tensors 并移动到 `device`
    image_tensor = torch.tensor(image_batch.transpose(0, 3, 1, 2), dtype=torch.float32, device=device)  # (batch, C, H, W)
    centerbias_tensor = torch.tensor(centerbias, dtype=torch.float32, device=device)  # (batch, H, W)
    x_hist_tensor = torch.tensor(fixation_history_x, dtype=torch.float32, device=device)  # (batch, num_points)
    y_hist_tensor = torch.tensor(fixation_history_y, dtype=torch.float32, device=device)  # (batch, num_points)

    # 批量预测
    print("enter model")      
    print(image_tensor.shape)
    log_density_predictions = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)  # 形状: (batch, 1, H, W)
    print("exit model")      
    # 提取 heatmaps 并计算均值
    heatmaps = log_density_predictions.detach().cpu().numpy()[:, 0]  # 形状: (batch, H, W)
    print(heatmaps.shape)
    
    final_heatmap = np.mean(heatmaps, axis=0)  # 形状: (H, W)
    print(final_heatmap.shape)

    # 归一化 heatmap 到 [0, 255]
    final_heatmap = (255 * (final_heatmap - np.min(final_heatmap)) / (np.max(final_heatmap) - np.min(final_heatmap))).astype(np.uint8)
    return final_heatmap
# def process_image_with_deepgaze(image, num_points=4, num_iterations=10):
#     image_np = np.array(image, dtype=np.float32)  # ✅ 确保 `numpy` 数据类型正确
#     centerbias = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)
#     heatmaps = []
#     for i in range(num_iterations):
#         points = np.random.randint(0, [image.width, image.height], size=(num_points, 2))
#         fixation_history_x = points[:, 0].astype(np.float32)
#         fixation_history_y = points[:, 1].astype(np.float32)

#         centerbias = np.zeros((image.width, image.height), dtype=np.float32)
#         image_np = np.array(image, dtype=np.float32)

#         image_tensor = torch.tensor(image_np.transpose(2, 0, 1)[None, :, :, :], dtype=torch.float32, device=device)
#         centerbias_tensor = torch.tensor(centerbias[None, :, :], dtype=torch.float32, device=device)
#         x_hist_tensor = torch.tensor(fixation_history_x[None, :], dtype=torch.float32, device=device)
#         y_hist_tensor = torch.tensor(fixation_history_y[None, :], dtype=torch.float32, device=device)

#         log_density_prediction = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)
#         predicted_heatmap = log_density_prediction.detach().cpu().numpy()[0, 0]
#         heatmaps.append(predicted_heatmap)

#     final_heatmap = np.mean(heatmaps, axis=0)
#     #normalize
#     final_heatmap = (255 * (final_heatmap - np.min(final_heatmap)) / (np.max(final_heatmap) - np.min(final_heatmap))).astype(np.uint8)
#     return final_heatmap

def deepgaze_process(batch, num_threads=1, num_points=4, num_iterations=10):    
    processed_images = []
    for image in batch["image"]:  
        print("enter batch")      
        heatmap = process_image_with_deepgaze(image, num_points, num_iterations)
        processed_images.append(HFImage().encode_example(heatmap))  # 这里确保 heatmap 是 numpy.ndarray 类型

    # 直接返回 list[numpy.ndarray]
    batch["deepgaze_feature"] = processed_images  
    return batch

# def deepgaze_process(batch, num_threads=1, num_points=4, num_iterations=10):    
#     for image in batch["image"]:        
#         batch["deepgaze_feature"] = HFImage().encode_example(process_image_with_deepgaze(image, num_points, num_iterations))
#     return batch
