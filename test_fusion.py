import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter

# 设置路径
npz_dir = "diff_output/npz_heatmaps"
output_dir = "diff_output/fusion_comparison"
diff_output_dir = os.path.join(output_dir, "diffs")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(diff_output_dir, exist_ok=True)

# 融合方法定义
def fusion_mean(heatmaps):
    return np.mean(heatmaps, axis=0)

def fusion_multiply_then_softmax(heatmaps):
    mul = np.prod(heatmaps, axis=0)
    exp = np.exp(mul - np.max(mul))
    return exp / np.sum(exp)

def fusion_softmax_then_multiply(heatmaps):
    softmaxed = np.exp(heatmaps - np.max(heatmaps, axis=(1, 2), keepdims=True))
    softmaxed /= np.sum(softmaxed, axis=(1, 2), keepdims=True)
    return np.prod(softmaxed, axis=0)

def fusion_softmax_then_multiply_then_softmax(heatmaps):
    # Step 1: 对每个 heatmap 做 softmax
    softmaxed = np.exp(heatmaps - np.max(heatmaps, axis=(1, 2), keepdims=True))
    softmaxed /= np.sum(softmaxed, axis=(1, 2), keepdims=True)  # shape: (10, H, W)

    # Step 2: multiply 所有 softmax 后的 heatmaps
    product = np.prod(softmaxed, axis=0)  # shape: (H, W)

    # Step 3: 再对乘积图做一次 softmax（全图归一化）
    final_exp = np.exp(product - np.max(product))
    final_softmax = final_exp / np.sum(final_exp)

    return final_softmax


def fusion_max(heatmaps):
    return np.max(heatmaps, axis=0)

def fusion_softor(heatmaps):
    softmaps = np.exp(heatmaps - np.max(heatmaps, axis=(1, 2), keepdims=True))
    softmaps /= np.sum(softmaps, axis=(1, 2), keepdims=True)
    complement = 1 - softmaps
    return 1 - np.prod(complement, axis=0)

def fusion_softmax_sum(heatmaps):
    """
    对每张 heatmap 做 softmax，然后直接逐像素相加。

    这会放大被多张图共同关注的区域，但也保留少量分布式关注。
    """
    # 对每张图做 softmax
    softmaxed = np.exp(heatmaps - np.max(heatmaps, axis=(1, 2), keepdims=True))
    softmaxed /= np.sum(softmaxed, axis=(1, 2), keepdims=True)

    # 直接逐像素加和（不做平均）
    return np.sum(softmaxed, axis=0)

def fusion_softmax_sum_then_softmax(heatmaps):
    """
    每张 heatmap 单独做 softmax → 加总 → 再做一次全局 softmax。
    强调所有 heatmap 的显著共同区域。
    """
    # 对每张图单独 softmax
    softmaxed = np.exp(heatmaps - np.max(heatmaps, axis=(1, 2), keepdims=True))
    softmaxed /= np.sum(softmaxed, axis=(1, 2), keepdims=True)

    summed = np.sum(softmaxed, axis=0)

    # 对加和结果再做一次 softmax
    exp = np.exp(summed - np.max(summed))
    return exp / np.sum(exp)


def fusion_weighted_softmax(heatmaps):
    weights = heatmaps.max(axis=(1, 2))
    print("📊 [weighted_softmax] Weights before normalization:", weights)
    weights = weights / weights.sum()
    print("📊 [weighted_softmax] Normalized weights:", weights)
    return np.sum(heatmaps * weights[:, None, None], axis=0)

def fusion_gaussian_smooth(heatmaps):
    smoothed = np.array([gaussian_filter(h, sigma=3) for h in heatmaps])
    return np.mean(smoothed, axis=0)

def fusion_local_max(heatmaps):
    return np.max(np.array([
        h * (h == gaussian_filter(h, sigma=1)) for h in heatmaps
    ]), axis=0)

def fusion_attention_like(heatmaps):
    weights = np.sum(heatmaps, axis=(1, 2))
    print("📊 [attention_like] Weights before normalization:", weights)
    weights /= np.sum(weights)
    print("📊 [attention_like] Normalized weights:", weights)
    return np.sum(heatmaps * weights[:, None, None], axis=0)

fusions = {
    "mean": fusion_mean,
    "multiply->softmax": fusion_multiply_then_softmax,
    "softmax->multiply": fusion_softmax_then_multiply,
    "softmax->multiply->softmax": fusion_softmax_then_multiply_then_softmax,
    "softor": fusion_softor,
    "fusion_softmax_sum": fusion_softmax_sum,
    "fusion_softmax_sum_then_softmax": fusion_softmax_sum_then_softmax,
    "weighted_softmax": fusion_weighted_softmax,
    # "gaussian_smooth": fusion_gaussian_smooth,
    "local_max": fusion_local_max,
    # "attention_like": fusion_attention_like,
}

def save_colormap(heatmap, save_path, cmap="jet"):
    norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    colored = cm.get_cmap(cmap)(norm)[..., :3]
    img = Image.fromarray((colored * 255).astype(np.uint8))
    img.save(save_path)

def save_difference_map(method_name, fused, mean_map, base_name):
    diff = np.abs(fused - mean_map)
    diff_path = os.path.join(diff_output_dir, f"{base_name}_diff_vs_mean_{method_name}.jpg")
    save_colormap(diff, diff_path, cmap="hot")
    print(f"📌 Saved difference map vs mean for {method_name} -> {diff_path}")

# 仅跑前三张图
sample_files = sorted(f for f in os.listdir(npz_dir) if f.endswith(".npz"))[:5]

for fname in sample_files:
    path = os.path.join(npz_dir, fname)
    data = np.load(path)["heatmaps"]  # shape [10, H, W]
    base = os.path.splitext(fname)[0]

    # 输出每个 heatmap 的统计信息
    print(f"\n📂 Processing {base}")
    for i, h in enumerate(data):
        print(f" - Heatmap {i}: mean={h.mean():.4f}, max={h.max():.4f}, std={h.std():.4f}")

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    mean_map = fusion_mean(data)

    for i, (method_name, fusion_func) in enumerate(fusions.items()):
        fused = fusion_func(data)
        out_path = os.path.join(output_dir, f"{base}_{method_name}.jpg")
        save_colormap(fused, out_path)

        norm = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)
        colored = cm.get_cmap("jet")(norm)[..., :3]
        axes[i].imshow(colored)
        axes[i].set_title(method_name)
        axes[i].axis("off")

        # 输出差异图（mean vs 当前方法）
        if method_name != "mean":
            save_difference_map(method_name, fused, mean_map, base)

    plt.tight_layout()
    grid_path = os.path.join(output_dir, f"{base}_grid.jpg")
    plt.savefig(grid_path, dpi=200)
    plt.close()
    print(f"✅ Saved comparison grid: {grid_path}")
