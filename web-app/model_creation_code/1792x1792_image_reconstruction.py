import os
from PIL import Image
import numpy as np
from scipy.ndimage import label
import math
import time
from multiprocessing import Pool

def process_subfolder(subfolder_name, base_dir, output_dir_1792, original_width, original_height, patch_size):
    subfolder_path = os.path.join(base_dir, subfolder_name)
    if not os.path.isdir(subfolder_path):
        return  # Skip files, process directories only

    print(f"Processing image: {subfolder_name}")
    output_subdir_1792 = os.path.join(output_dir_1792, subfolder_name)
    os.makedirs(output_subdir_1792, exist_ok=True)

    x_tiles = math.ceil(original_width / patch_size)
    y_tiles = math.ceil(original_height / patch_size)
    mask = np.zeros((x_tiles, y_tiles), dtype=np.int32)

    for patch_filename in sorted(os.listdir(subfolder_path)):
        if patch_filename.endswith('.png'):
            parts = patch_filename.split('_')
            i = int(parts[1])
            j = int(parts[2].split('.')[0])
            mask[i, j] = 1

    structure = np.ones((3, 3), dtype=int)
    labeled_mask, num_features = label(mask, structure=structure)
    component_sizes = np.bincount(labeled_mask.flatten())[1:]
    if len(component_sizes) == 0:
        print(f"No components found in {subfolder_name}. Skipping.")
        return

    largest_component_size = component_sizes.max()
    threshold = 0.3 * largest_component_size
    valid_labels = [label+1 for label, size in enumerate(component_sizes) if size >= threshold]

    for valid_label in valid_labels:
        indices = np.argwhere(labeled_mask == valid_label)
        if indices.size == 0:
            continue
        i_coords, j_coords = indices[:, 0], indices[:, 1]
        min_i, max_i, min_j, max_j = i_coords.min(), i_coords.max(), j_coords.min(), j_coords.max()
        width = (max_i - min_i + 1) * patch_size
        height = (max_j - min_j + 1) * patch_size
        island_image = Image.new('RGB', (width, height), (255, 255, 255))

        for (i, j) in zip(i_coords, j_coords):
            x_coord = (i - min_i) * patch_size
            y_coord = (j - min_j) * patch_size
            patch_path = os.path.join(subfolder_path, f"patch_{i}_{j}.png")
            if not os.path.exists(patch_path):
                continue
            patch = Image.open(patch_path).convert('RGBA')
            island_image.paste(patch, (x_coord, y_coord), patch)

        max_dimension = 1792
        scaling_factor = min(max_dimension / island_image.width, max_dimension / island_image.height)
        new_width, new_height = int(island_image.width * scaling_factor), int(island_image.height * scaling_factor)
        island_image_resized = island_image.resize((new_width, new_height), Image.ANTIALIAS)

        final_image = Image.new('RGB', (1792, 1792), (255, 255, 255))
        paste_x, paste_y = (1792 - new_width) // 2, (1792 - new_height) // 2
        final_image.paste(island_image_resized, (paste_x, paste_y))
        output_path_1792 = os.path.join(output_subdir_1792, f"island_{valid_label}.png")
        final_image.save(output_path_1792)

    print(f"Finished processing {subfolder_name}")

def reconstruct_islands_in_directory(base_dir):
    start_time = time.time()
    original_width = 22315
    original_height = 51186
    patch_size = 224

    output_dir_1792 = '1792x1792_batch7'
    os.makedirs(output_dir_1792, exist_ok=True)

    subfolder_names = [subfolder for subfolder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, subfolder))]
    args = [(subfolder, base_dir, output_dir_1792, original_width, original_height, patch_size) for subfolder in subfolder_names]

    with Pool(processes=64) as pool:  # Explicitly set to use 64 cores
        pool.starmap(process_subfolder, args)

    total_execution_time = time.time() - start_time
    print(f"All images processed successfully. Total Execution Time: {total_execution_time:.2f} seconds")

if __name__ == "__main__":
    base_directory = "/home/d.uriartediaz/vabfmc/data/working/d.uriartediaz/francokrepel/project-root/data/patches/AN_Batch_04.29.22_BCC2020"
    reconstruct_islands_in_directory(base_directory)
