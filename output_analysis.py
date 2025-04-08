import matplotlib.pyplot as plt
import os

def compare_heatmaps():
    # 1. List the image paths you want to compare
    #    (Adjust the paths/titles to match your actual file locations)
    image_paths = [
        "./test_outputs/centerbias_original_batch1/deepgaze_heatmap_0.png",
        "./test_outputs/centerbias_original_batch4/deepgaze_heatmap_0.png",
        "./test_outputs/centerbias_zeros_batch1/deepgaze_heatmap_0.png",
        "./test_outputs/centerbias_zeros_batch4/deepgaze_heatmap_0.png"
    ]

    # 2. Provide a corresponding title for each subplot
    titles = [
        "Original + batch=1",
        "Original + batch=4",
        "Zeros + batch=1",
        "Zeros + batch=4"
    ]

    # 3. Create a figure and multiple subplots 
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    # 4. Loop over each path/title and display the image
    for ax, img_path, title in zip(axes.ravel(), image_paths, titles):
        # Read the PNG file as an image array
        image = plt.imread(img_path)
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")  # Hide axis ticks/labels

    # 5. Adjust spacing and show
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_heatmaps()
