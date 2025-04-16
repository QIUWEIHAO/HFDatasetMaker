import os
import torch
import numpy as np
from PIL import Image as PILImage
import matplotlib.cm as cm
from scipy.ndimage import zoom
from scipy.special import logsumexp

import sys
sys.path.insert(0, "/Users/shaw/HFDatasetMaker")

from deepgaze_pytorch.deepgaze3 import DeepGazeIII

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepGazeIII(pretrained=True).to(device)

# Load both types of centerbias templates
centerbias_template_original = np.load("/Users/shaw/HFDatasetMaker/deepgaze_pytorch/centerbias_mit1003.npy")
centerbias_template_zeros = np.zeros_like(centerbias_template_original)

def apply_colormap(image_array, colormap="jet"):
    if len(image_array.shape) == 3 and image_array.shape[2] == 1:
        image_array = image_array.squeeze()
    normalized = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
    colormap_function = cm.get_cmap(colormap)
    colored_image = colormap_function(normalized)
    colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
    return PILImage.fromarray(colored_image)

def save_heatmaps_npz(batch_heatmaps, original_image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_basename = os.path.basename(original_image_path)
    image_id = os.path.splitext(image_basename)[0]
    save_path = os.path.join(output_dir, f"{image_id}_heatmaps.npz")
    np.savez_compressed(save_path, heatmaps=batch_heatmaps)
    print(f"✅ Saved raw heatmaps -> {save_path}")    

def process_image_with_deepgaze_batch(
    image, 
    num_points, 
    batch_size, 
    total_iterations, 
    centerbias_type="original", 
    original_image_path=None,
    npz_output_dir=None,
    feature_method="mean"
):
    print(f"📌 Using centerbias type: {centerbias_type}")
    
    if original_image_path is not None and npz_output_dir is not None:
        image_id = os.path.splitext(os.path.basename(original_image_path))[0]
        npz_path = os.path.join(npz_output_dir, f"{image_id}_heatmaps.npz")
        if os.path.exists(npz_path):
            print(f"📁 Found existing .npz → Loading: {npz_path}")
            heatmaps = np.load(npz_path)["heatmaps"]
        else:
            image_np = np.array(image)
            H, W = image_np.shape[:2]

            centerbias_template = (
                centerbias_template_original if centerbias_type == "original" else centerbias_template_zeros
            )
            centerbias = zoom(centerbias_template, (H / centerbias_template.shape[0], W / centerbias_template.shape[1]), order=0, mode='nearest')
            centerbias -= logsumexp(centerbias)
            centerbias_tensor = torch.tensor([centerbias], device=device)

            iteration_counter = 0
            batch_heatmaps = []

            while iteration_counter < total_iterations:
                cur_batch_size = min(batch_size, total_iterations - iteration_counter)
                points = np.random.randint(0, [W, H], size=(cur_batch_size, num_points, 2))
                fixation_history_x = points[:, :, 0]
                fixation_history_y = points[:, :, 1]

                image_batch = np.repeat(image_np[None, :, :, :], cur_batch_size, axis=0)
                image_tensor = torch.tensor(image_batch.transpose(0, 3, 1, 2), device=device)
                x_hist_tensor = torch.tensor(fixation_history_x, device=device)
                y_hist_tensor = torch.tensor(fixation_history_y, device=device)

                print(f"🔥 Iteration {iteration_counter} -> {iteration_counter + cur_batch_size}")
                log_density_predictions = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)
                heatmaps = log_density_predictions.detach().cpu().numpy()[:, 0]
                batch_heatmaps.append(heatmaps)
                iteration_counter += cur_batch_size

            heatmaps = np.concatenate(batch_heatmaps, axis=0)
            save_heatmaps_npz(heatmaps, original_image_path, npz_output_dir)
    else:
        raise ValueError("original_image_path and npz_output_dir must be provided.")

    if feature_method == "mean":
        final_heatmap = np.mean(heatmaps, axis=0)
    elif feature_method == "softmax_peak":
        softmaxed = np.exp(heatmaps - np.max(heatmaps, axis=(1, 2), keepdims=True))
        softmaxed /= np.sum(softmaxed, axis=(1, 2), keepdims=True)
        final_heatmap = np.max(softmaxed, axis=0)
    else:
        raise ValueError(f"Unsupported feature_method: {feature_method}")

    return final_heatmap

# for huggingface dataset
def deepgaze_process(
    batch, 
    input_key, 
    output_key, 
    num_points=4, 
    batch_random_size=1, 
    total_iterations=10, 
    centerbias_type="original",
    image_paths=None,
    feature_method="mean"
):    
    result = []
    for i, image in enumerate(batch[input_key]):  
        print(f"🔁 Processing image {i}")
        original_image_path = image_paths[i] if image_paths is not None else None
        heatmap = process_image_with_deepgaze_batch(
            image, 
            num_points, 
            batch_random_size, 
            total_iterations, 
            centerbias_type=centerbias_type, 
            original_image_path=original_image_path,
            npz_output_dir="diff_output/npz_heatmaps"  # default fallback for Hugging Face batch
        )
        heatmap_image = apply_colormap(heatmap)
        result.append(heatmap_image)

    batch[output_key] = result
    return batch
