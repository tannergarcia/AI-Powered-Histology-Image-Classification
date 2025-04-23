import os
import sys
from PIL import Image
import numpy as np
from scipy.ndimage import label
import math
import time

def reconstruct_islands(base_patch_dir):
    start_time = time.time()

    # Set the parameters
    original_width = 22315  # Width of the original image
    original_height = 51186  # Height of the original image
    patch_size = 224  # Size of each patch (assuming square patches)
    patch_folder = base_patch_dir  # Folder where patches are saved

    # Get the name of the base directory to use in output paths
    base_dir_name = os.path.basename(os.path.normpath(base_patch_dir))

    # Print job start statement
    print(f"Job Started for image: {base_dir_name}")

    # Create output directories for the two sizes
    output_dir_1024 = os.path.join('1024x1024', base_dir_name)
    output_dir_256 = os.path.join('256x256', base_dir_name)
    os.makedirs(output_dir_1024, exist_ok=True)
    os.makedirs(output_dir_256, exist_ok=True)

    # Calculate the number of tiles in x and y directions
    x_tiles = math.ceil(original_width / patch_size)
    y_tiles = math.ceil(original_height / patch_size)

    # Create a mask of patches
    mask = np.zeros((x_tiles, y_tiles), dtype=np.int32)

    # Iterate over the patches in the folder and build the mask
    for patch_filename in sorted(os.listdir(patch_folder)):
        if patch_filename.endswith('.png'):
            # Extract the (i, j) tile indices from the filename
            parts = patch_filename.split('_')
            i = int(parts[1])
            j = int(parts[2].split('.')[0])

            # Set mask at the position
            mask[i, j] = 1

    # Perform connected components analysis
    structure = np.ones((3, 3), dtype=int)  # Defines connectivity
    labeled_mask, num_features = label(mask, structure=structure)

    # Compute the sizes of each connected component
    component_sizes = np.bincount(labeled_mask.flatten())

    # Ignore background label 0
    component_sizes = component_sizes[1:]  # Now component_sizes[i] corresponds to label i+1

    # Find the largest component
    largest_component_size = component_sizes.max()
    largest_component_label = np.argmax(component_sizes) + 1  # Since we skipped background

    # Threshold for valid components (30% of largest component)
    threshold = 0.3 * largest_component_size

    # Identify labels of valid components
    valid_labels = [label+1 for label, size in enumerate(component_sizes) if size >= threshold]

    # For each valid island (connected component)
    for valid_label in valid_labels:
        # Get the indices of patches belonging to this island
        indices = np.argwhere(labeled_mask == valid_label)
        if indices.size == 0:
            continue

        i_coords = indices[:, 0]
        j_coords = indices[:, 1]

        # Find the bounding box
        min_i = i_coords.min()
        max_i = i_coords.max()
        min_j = j_coords.min()
        max_j = j_coords.max()

        # Calculate the size of the island image
        width = (max_i - min_i + 1) * patch_size
        height = (max_j - min_j + 1) * patch_size

        # Create an empty image canvas for the island with white background
        island_image = Image.new('RGB', (width, height), (255, 255, 255))

        # Paste the patches into the island image
        for (i, j) in zip(i_coords, j_coords):
            x_coord = (i - min_i) * patch_size
            y_coord = (j - min_j) * patch_size

            # Load the patch image and convert to 'RGBA'
            patch_path = os.path.join(patch_folder, f"patch_{i}_{j}.png")
            patch = Image.open(patch_path).convert('RGBA')

            # Paste the patch into the island image using the alpha channel as mask
            island_image.paste(patch, (x_coord, y_coord), patch)

        # Resize the island image to fit within 1024x1024 pixels
        max_dimension = 1024
        island_width, island_height = island_image.size
        scaling_factor = min(max_dimension / island_width, max_dimension / island_height)

        # Scale the island image
        new_width = int(island_width * scaling_factor)
        new_height = int(island_height * scaling_factor)
        island_image_resized = island_image.resize((new_width, new_height), Image.ANTIALIAS)

        # Pad the image to 1024x1024 pixels with a white background
        final_image = Image.new('RGB', (1024, 1024), (255, 255, 255))
        paste_x = (1024 - new_width) // 2
        paste_y = (1024 - new_height) // 2
        final_image.paste(island_image_resized, (paste_x, paste_y))

        # Save the 1024x1024 image
        output_filename = f"island_{valid_label}.png"
        output_path_1024 = os.path.join(output_dir_1024, output_filename)
        final_image.save(output_path_1024)

        # Resize the final image to 256x256 pixels
        final_image_256 = final_image.resize((256, 256), Image.ANTIALIAS)
        output_path_256 = os.path.join(output_dir_256, output_filename)
        final_image_256.save(output_path_256)

    # Print job completion statement with execution time
    execution_time = time.time() - start_time
    print(f"Image {base_dir_name} converted successfully. Execution Time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    # Check if the base_patch_dir is provided
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <base_patch_dir>")
        sys.exit(1)

    base_patch_dir = sys.argv[1]
    reconstruct_islands(base_patch_dir)
