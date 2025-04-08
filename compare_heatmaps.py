import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 替换成你的实际路径
heatmap_path_1 = "/Users/shaw/ScanpathAnalysis/validation_output/test_images/original_image/final_feature_map.npy"
heatmap_path_2 = "/Users/shaw/HFDatasetMaker/test_outputs/weihao_raw_heatmap.npy"

# 2. 加载 heatmaps
h1 = np.load(heatmap_path_1)
h2 = np.load(heatmap_path_2)

print(f"Heatmap 1 shape: {h1.shape}")
print(f"Heatmap 2 shape: {h2.shape}")

# 3. 自动裁剪成相同 shape
min_h = min(h1.shape[0], h2.shape[0])
min_w = min(h1.shape[1], h2.shape[1])

h1_cropped = h1[:min_h, :min_w]
h2_cropped = h2[:min_h, :min_w]

print(f"After crop -> shape: {h1_cropped.shape}, {h2_cropped.shape}")

# 4. 计算差异
diff = np.abs(h1_cropped - h2_cropped)
print(f"Mean absolute diff: {diff.mean():.6f}")

# 5. 先保存单独的 Shaw & Weihao heatmaps
output_dir = "./comparison_outputs"
os.makedirs(output_dir, exist_ok=True)

plt.imshow(h1_cropped, cmap="jet")
plt.axis("off")
plt.savefig(f"{output_dir}/shaw_heatmap.jpg", bbox_inches="tight", pad_inches=0)
plt.close()

plt.imshow(h2_cropped, cmap="jet")
plt.axis("off")
plt.savefig(f"{output_dir}/weihao_heatmap.jpg", bbox_inches="tight", pad_inches=0)
plt.close()

print("✅ Saved shaw_heatmap.jpg & weihao_heatmap.jpg")

# 6. 保留原有对比可视化
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Shaw Heatmap")
plt.imshow(h1_cropped, cmap="jet")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Weihao Heatmap")
plt.imshow(h2_cropped, cmap="jet")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Absolute Difference")
plt.imshow(diff, cmap="hot")
plt.axis("off")

plt.tight_layout()
plt.show()
