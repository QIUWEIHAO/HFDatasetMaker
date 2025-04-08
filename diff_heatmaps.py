import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def load_heatmap(path):
    """Load a heatmap image and convert it to a normalized grayscale numpy array."""
    img = Image.open(path).convert('L')  # Convert to grayscale
    arr = np.array(img).astype(np.float32)
    arr /= 255.0  # Normalize to [0,1]
    return arr

def visualize_difference(map1, map2, output_path):
    """Visualize the absolute difference between two heatmaps."""
    diff = np.abs(map1 - map2)
    
    plt.imshow(diff, cmap='hot')
    plt.colorbar()
    plt.title("Heatmap Difference")
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✅ Saved difference heatmap -> {output_path}")

def main():
    # 替换成你的实际路径
    shaw_heatmap_path = "/Users/shaw/ScanpathAnalysis/validation_output/test_images/11_jones_2/final_feature_map.png"
    weihao_heatmap_path = "./test_outputs/centerbias_zeros_batch1/deepgaze_heatmap_0.png"
    output_diff_path = "./diff_output/heatmap_difference.png"

    os.makedirs(os.path.dirname(output_diff_path), exist_ok=True)

    heatmap1 = load_heatmap(shaw_heatmap_path)
    heatmap2 = load_heatmap(weihao_heatmap_path)

    # Resize heatmap2 to match heatmap1 if needed
    if heatmap1.shape != heatmap2.shape:
        heatmap2 = np.array(Image.fromarray((heatmap2 * 255).astype(np.uint8)).resize(heatmap1.shape[::-1], Image.BILINEAR)) / 255.0

    visualize_difference(heatmap1, heatmap2, output_diff_path)

if __name__ == "__main__":
    main()
