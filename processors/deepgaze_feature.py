import torch
from PIL import Image as PILImage
from datasets import Image as HFImage
import deepgaze_pytorch
import numpy as np
import matplotlib.cm as cm
from scipy.ndimage import zoom
from scipy.special import logsumexp
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
deepgaze_model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(device)

centerbias_template_original = np.load('deepgaze_pytorch/centerbias_mit1003.npy')
centerbias_template_zeros = np.zeros([centerbias_template_original.shape[0], centerbias_template_original.shape[1]])

def apply_colormap(image_array, colormap="jet"):
    """
    Converts a grayscale NumPy array into a colored image using a colormap.

    Args:
        image_array (np.ndarray): Input grayscale image (H, W) or (H, W, 1)
        colormap (str): Colormap name (e.g., "jet", "viridis", "magma")

    Returns:
        PIL.Image: Colormapped image
    """
    if len(image_array.shape) == 3 and image_array.shape[2] == 1:
        image_array = image_array.squeeze()  # Convert (H, W, 1) -> (H, W)
    
    # Normalize to [0,1] if not already
    normalized = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    
    # Apply the colormap
    colormap_function = cm.get_cmap(colormap)
    colored_image = colormap_function(normalized)  # (H, W, 4) -> RGBA
    
    # Convert to uint8 and remove the alpha channel
    colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)  # RGB format
    
    # Convert to PIL image
    return PILImage.fromarray(colored_image)

def normalize_image_array_8bit(image):
    normalized_image = (255 * (image - np.min(image)) / (np.max(image) - np.min(image))).astype(np.uint8)
    return normalized_image

def process_image_with_deepgaze_batch(model,centerbias_template, image, num_points, batch_size, total_iterations):
    image_np = np.array(image)  # ✅ 确保 `numpy` 数据类型正确
    print(image_np.shape)
    H, W = image_np.shape[:2]  # 取得图片高宽
    # rescale to match image size
    centerbias = zoom(centerbias_template, (H/centerbias_template.shape[0], W/centerbias_template.shape[1]), order=0, mode='nearest')
    # renormalize log density
    centerbias -= logsumexp(centerbias)
    centerbias_tensor = torch.tensor([centerbias], device=device)  # (batch, H, W)
        
    iteration_counter = 0
    batch_heatmaps = []
    while(iteration_counter < total_iterations):

        cur_batch_size = min(batch_size, total_iterations - iteration_counter)
        print(f"Iteration: {iteration_counter} to {iteration_counter + cur_batch_size} | Total Iterations: {total_iterations}")

        # 生成 num_iterations 个随机 fixation points
        points = np.random.randint(0, [W, H], size=(cur_batch_size, num_points, 2))
        fixation_history_x = points[:, :, 0] # 形状: (num_iterations, num_points)
        fixation_history_y = points[:, :, 1]

        # 批量化 centerbias 和 image tensor
        image_batch = np.repeat(image_np[None, :, :, :], cur_batch_size, axis=0)  # 形状: (batch_size, H, W, C)
        # 转换为 PyTorch tensors 并移动到 `device`
        image_tensor = torch.tensor(image_batch.transpose(0, 3, 1, 2), device=device)  # (batch, C, H, W)
        x_hist_tensor = torch.tensor(fixation_history_x, device=device)  # (batch, num_points)
        y_hist_tensor = torch.tensor(fixation_history_y, device=device)  # (batch, num_points)

        # 批量预测
        print("enter model")      
        print(image_tensor.shape)
        log_density_predictions = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)  # 形状: (batch, 1, H, W)
        print("exit model")      
        # 提取 heatmaps 并计算均值
        heatmaps = log_density_predictions.detach().cpu().numpy()[:, 0]  # 形状: (batch, H, W)
        print(heatmaps.shape)

        
        batch_heatmaps.append(heatmaps)
        iteration_counter += cur_batch_size
        
    batch_heatmaps = np.concatenate(batch_heatmaps, axis=0)  # (total_iterations, H, W)
    print(batch_heatmaps.shape)

    final_heatmap = np.mean(batch_heatmaps, axis=0)  # 形状: (H, W)
    print(final_heatmap.shape)

    # 归一化 heatmap 到 [0, 255]
    # final_heatmap = (255 * (final_heatmap - np.min(final_heatmap)) / (np.max(final_heatmap) - np.min(final_heatmap))).astype(np.uint8)
    return final_heatmap

def deepgaze_process(batch, input_key, output_key, params={"num_points": 4, "batch_random_size": 1, "total_iterations": 10, "centerbias": "zeros"}):
    result = []
    for image in batch[input_key]:  
        print("enter batch")
        centerbias_template = centerbias_template_zeros if params["centerbias"] == "zeros" else centerbias_template_original
        print("centerbias_template")
        print(centerbias_template)
        heatmap = process_image_with_deepgaze_batch(deepgaze_model, centerbias_template, image, params["num_points"], params["batch_random_size"], params["total_iterations"])
        print(type(heatmap))
        
        heatmap_image = apply_colormap(heatmap)
        print(type(heatmap_image))
        result.append(heatmap_image)
    batch[output_key] = result
    return batch


# # Shaw's original code
# def process_image_with_deepgaze(image, num_points, num_iterations):
#     image_np = np.array(image) 
#     image_height, image_width = image_np.shape[:2]
#     # rescale to match image size
#     centerbias = zoom(centerbias_template, (image_height/centerbias_template.shape[0], image_width/centerbias_template.shape[1]), order=0, mode='nearest')
#     # renormalize log density
#     centerbias -= logsumexp(centerbias)
    
#     centerbias = np.zeros([image_np.shape[0], image_np.shape[1]])
#     centerbias_tensor = torch.tensor([centerbias], device=device)  # (batch, H, W)
    
#     heatmaps= []
    
#     for _ in range(num_iterations):
#         # Generate random fixation points that cover the entire image
#         points = [(np.random.randint(0, image_width), np.random.randint(0, image_height)) for _ in range(num_points)]

#         # Extract x and y coordinates of the fixation points
#         fixation_history_x = np.array([p[0] for p in points])
#         fixation_history_y = np.array([p[1] for p in points])

#         # Convert the inputs to tensors
#         image_tensor = torch.tensor([image_np.transpose(2, 0, 1)]).float().to(device)
#         centerbias_tensor = torch.tensor([centerbias]).float().to(device)
#         x_hist_tensor = torch.tensor([fixation_history_x]).float().to(device)
#         y_hist_tensor = torch.tensor([fixation_history_y]).float().to(device)

#         # Generate the log density prediction
#         log_density_prediction = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)
#         predicted_heatmap = log_density_prediction.detach().cpu().numpy()[0, 0]
#         heatmaps.append(predicted_heatmap)

#     # Average the heatmaps to reduce the influence of individual fixation points
#     final_heatmap = np.mean(heatmaps, axis=0)
#     return final_heatmap