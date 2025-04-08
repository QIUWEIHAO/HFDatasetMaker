import os
import numpy as np
from PIL import Image
from processors.deepgaze_feature import process_image_with_deepgaze_batch, apply_colormap, save_heatmaps_npz

# === Config ===
image_dir = "dataset"  # ✅ 请根据你的实际路径修改
image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))])[:5]
output_dir = "diff_output/test_results"
os.makedirs(output_dir, exist_ok=True)

# === Process ===
for filename in image_filenames:
    image_path = os.path.join(image_dir, filename)
    print(f"▶ Processing {filename}...")

    try:
        image = Image.open(image_path).convert("RGB")
        final_heatmap = process_image_with_deepgaze_batch(
            image,
            num_points=4,
            batch_size=1,
            total_iterations=10,
            original_image_path=image_path
        )

        # Save mean heatmap
        heatmap_image = apply_colormap(final_heatmap)
        jpg_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_heatmap.jpg")
        heatmap_image.save(jpg_path)
        print(f"✅ Saved JPG heatmap to {jpg_path}")

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")


print("✅✅ All done!")
