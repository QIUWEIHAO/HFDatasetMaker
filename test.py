# import os
# # from processors.load_image import load_image
# # from processors.deepgaze_feature import deepgaze_process
# import sys
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "processors"))

# from load_image import load_image
# from resize_image import resize_image
# from deepgaze_feature import deepgaze_process

# from PIL import Image

# def main():
#     image_path = "/Users/shaw/Downloads/GeorgeCurationResizedUnpadded/Complex/C-art-3.jpg"

#     batch = {
#         "image_path": [image_path]
#     }

#     batch = load_image(batch, input_key="image_path", output_key="image_pil")
#         ### 👇 Step 1: 读取 fixation 点 ###
#     fix_x_list = []
#     fix_y_list = []
#     fix_dir = "/Users/shaw/ScanpathAnalysis/validation_output/fixations"

#     for i in range(10):  # 假设你跑了10个iteration
#         fix_x_list.append(np.load(os.path.join(fix_dir, f"fixation_x_iter{i}.npy")))
#         fix_y_list.append(np.load(os.path.join(fix_dir, f"fixation_y_iter{i}.npy")))

#     external_fixations = {
#         "x": np.stack(fix_x_list, axis=0),  # shape = (10, num_points)
#         "y": np.stack(fix_y_list, axis=0)
#     }

#     ### Step 2: 继续跑 deepgaze ###
#     params = {
#         "num_points": 4,
#         "batch_random_size": 1,
#         "total_iterations": 10,
#         "centerbias": "zeros"
#     }

#     batch = deepgaze_process(batch,
#                              input_key="image_pil",
#                              output_key="heatmap_pil",
#                              params=params,
#                              external_fixations=external_fixations)

#     ### Step 3: 保存图片 ###
#     output_dir = "./test_outputs"
#     save_heatmaps(batch["heatmap_pil"], output_dir)

#     print("All done with fixed fixation points!")

#     # batch = resize_image(batch, input_key="image_pil", output_key="image_pil", params={"max_dims": [1920, 1080]})
# #1
#     # params = {
#     #     "num_points": 4,
#     #     "batch_random_size": 1,
#     #     "total_iterations": 10,
#     #     "centerbias": "zeros"  # or "original"
#     # }

#     # batch = deepgaze_process(batch, input_key="image_pil", output_key="heatmap_pil", params=params)

#     # output_dir = "./test_outputs"
#     # os.makedirs(output_dir, exist_ok=True)

#     # for idx, heatmap_image in enumerate(batch["heatmap_pil"]):
#     #     out_path = os.path.join(output_dir, f"deepgaze_heatmap_{idx}.png")
#     #     heatmap_image.save(out_path)
#     #     print(f"Saved colormapped heatmap -> {out_path}")

# #     # Common parameters for DeepGaze
# #     base_params = {
# #         "num_points": 4,           # how many random fixations per iteration
# #         "batch_random_size": 1,    # run one iteration at a time in the batch
# #         "total_iterations": 10     # total random-fixation iterations
# #     }
# #2
# #     # We'll do two runs:
# #     #  1) centerbias = "zeros"
# #     #  2) centerbias = "original" (which uses centerbias_mit1003.npy)
# #     # and store results in separate directories.

# #     ### Run 1: centerbias = zeros
# #     run_params_zeros = {**base_params, "centerbias": "zeros"}
# #     batch_zeros = deepgaze_process(batch.copy(),
# #                                    input_key="image_pil",
# #                                    output_key="heatmap_pil",
# #                                    params=run_params_zeros)

# #     # Save outputs to a subdirectory
# #     output_dir_zeros = "./test_outputs/zeros"
# #     save_heatmaps(batch_zeros["heatmap_pil"], output_dir_zeros)

# #     ### Run 2: centerbias = original
# #     run_params_original = {**base_params, "centerbias": "original"}
# #     batch_original = deepgaze_process(batch.copy(),
# #                                       input_key="image_pil",
# #                                       output_key="heatmap_pil",
# #                                       params=run_params_original)

# #     # Save outputs to a subdirectory
# #     output_dir_original = "./test_outputs/original"
# #     save_heatmaps(batch_original["heatmap_pil"], output_dir_original)

# #     print("All runs complete!")

# # def save_heatmaps(heatmap_list, output_dir):
# #     """
# #     Saves a list of PIL images (colormapped heatmaps) to a directory.
# #     Each item is saved as a .png file with an index-based name.
# #     """
# #     os.makedirs(output_dir, exist_ok=True)
# #     for idx, heatmap_image in enumerate(heatmap_list):
# #         out_path = os.path.join(output_dir, f"deepgaze_heatmap_{idx}.png")
# #         heatmap_image.save(out_path)
# #         print(f"Saved colormapped heatmap -> {out_path}")    

# #3
#     centerbias_options = ["zeros", "original"]  # or just ["zeros"] if you only want to test zero bias
#     batch_size_options = [1, 4]                 # 1 => single-iteration style, 4 => more parallel

#     # We'll store common parameters here
#     base_params = {
#         "num_points": 4,
#         "total_iterations": 10
#         # We'll fill "batch_random_size" and "centerbias" in the loops
#     }

#     # For each combination of centerbias + batch_size, run deepgaze_process
#     for cb in centerbias_options:
#         for bs in batch_size_options:
#             # create a copy of base_params
#             run_params = dict(base_params)
#             run_params["centerbias"] = cb
#             run_params["batch_random_size"] = bs

#             # we create an output subdirectory that encodes our param choice
#             # e.g. test_outputs/centerbias_zeros_batch1/ or test_outputs/centerbias_original_batch4/
#             output_dir = f"./test_outputs/centerbias_{cb}_batch{bs}"

#             print(f"\n=== Running centerbias={cb}, batch_random_size={bs} ===")
#             # We pass a *copy* of batch, so we don't modify the original
#             result_batch = deepgaze_process(batch.copy(),
#                                             input_key="image_pil",
#                                             output_key="heatmap_pil",
#                                             params=run_params)

#             # This result_batch["heatmap_pil"] is a list of PIL images (one per item).
#             # We only have one item in 'batch', so there will be one final image.
#             heatmap_list = result_batch["heatmap_pil"]

#             save_heatmaps(heatmap_list, output_dir)

#     print("All runs complete!")

# def save_heatmaps(heatmap_list, output_dir):
#     """
#     Saves a list of PIL images (colormapped heatmaps) to a directory.
#     Each item is saved as a .png file with an index-based name.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     for idx, heatmap_image in enumerate(heatmap_list):
#         out_path = os.path.join(output_dir, f"deepgaze_heatmap_{idx}.png")
#         heatmap_image.save(out_path)
#         print(f"Saved colormapped heatmap -> {out_path}")


# if __name__ == "__main__":
#     main()

import os
import sys
import glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "processors"))

from load_image import load_image
from resize_image import resize_image
from deepgaze_feature import deepgaze_process
from PIL import Image
import matplotlib.pyplot as plt

def main():
    image_path = "/Users/shaw/Downloads/GeorgeCurationResizedUnpadded/Complex/C-art-1.jpg"

    batch = {
        "image_path": [image_path]
    }

    batch = load_image(batch, input_key="image_path", output_key="image_pil")

    # Step 1: Directly load fixations.npy
    fix_npy_path = "/Users/shaw/ScanpathAnalysis/validation_output/test_images/original_image/fixations.npy"
    fixation_data = np.load(fix_npy_path, allow_pickle=True)

    # Debug print to check file content
    print("Fixation file content:", fixation_data)
    print("Type:", type(fixation_data))

    if isinstance(fixation_data, np.ndarray) and fixation_data.ndim == 3 and fixation_data.shape[2] == 2:
        external_fixations = {
            "x": fixation_data[:, :, 0],
            "y": fixation_data[:, :, 1]
        }
    elif isinstance(fixation_data, dict):
        external_fixations = {
            "x": np.array(fixation_data["x"]),
            "y": np.array(fixation_data["y"])
        }
    elif isinstance(fixation_data, (list, tuple)) and len(fixation_data) == 2:
        external_fixations = {
            "x": np.array(fixation_data[0]),
            "y": np.array(fixation_data[1])
        }
    else:
        raise ValueError("Unsupported fixation file format!")

    # Step 2: Run deepgaze_process with shared fixations
    params = {
        "num_points": 4,
        "batch_random_size": 1,
        "total_iterations": 10,
        "centerbias": "zeros"
    }

    result_batch = deepgaze_process(batch.copy(),
                                    input_key="image_pil",
                                    output_key="heatmap_pil",
                                    params=params,
                                    external_fixations=external_fixations)

    heatmap_list = result_batch["heatmap_pil"]
    raw_heatmaps = result_batch["raw_heatmaps"]

    save_heatmaps(heatmap_list, "./test_outputs")

    # Optional: visualize raw heatmap directly (e.g., diff with Shaw’s)
    np.save("./test_outputs/weihao_raw_heatmap.npy", raw_heatmaps[0])
    print("Weihao's raw heatmap saved!")

def save_heatmaps(heatmap_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for idx, heatmap_image in enumerate(heatmap_list):
        out_path = os.path.join(output_dir, f"deepgaze_heatmap_{idx}.png")
        heatmap_image.save(out_path)
        print(f"Saved colormapped heatmap -> {out_path}")

if __name__ == "__main__":
    main()
