import os
import cv2
import openslide
from PIL import Image
import numpy as np
import math


# Load the slide
slide_path = 'raw/AN_Batch_02.09.22_2015_BCC/slide-2022-02-09T12-26-27-R5-S1.mrxs'
slide = openslide.OpenSlide(slide_path)

# Check slide properties
print("Level Dimensions:", slide.level_dimensions)
print("Level Downsamples:", slide.level_downsamples)

# Set magnification and calculate appropriate level
magnification_factor = 4  # for 10x from 40x
level = slide.get_best_level_for_downsample(magnification_factor)

print("Selected Level:", level)

# Get dimensions at the desired level
level_dimensions = slide.level_dimensions[level]
width, height = level_dimensions

# Define patch size and calculate number of tiles
patch_size = 224
x_tiles = math.ceil(width / patch_size)
y_tiles = math.ceil(height / patch_size)

# Create directory to save patches
output_dir = 'extracted_patches'
os.makedirs(output_dir, exist_ok=True)

# Define function to filter tissue-containing patches based on color thresholds
def is_tissue(patch, tissue_threshold=0.2):
    # Convert patch to HSV
    patch_np = np.array(patch)
    hsv_image = cv2.cvtColor(patch_np, cv2.COLOR_RGB2HSV)

    # Define color ranges in HSV
    lower_purple = np.array([110, 30, 30])
    upper_purple = np.array([170, 255, 255])
    lower_pink = np.array([140, 20, 50])
    upper_pink = np.array([180, 255, 255])
    lower_blue = np.array([90, 30, 30])
    upper_blue = np.array([130, 255, 255])
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for each color range
    mask_purple = cv2.inRange(hsv_image, lower_purple, upper_purple)
    mask_pink = cv2.inRange(hsv_image, lower_pink, upper_pink)
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    
    # Combine all the masks into a single mask
    combined_mask = cv2.bitwise_or(mask_purple, mask_pink)
    combined_mask = cv2.bitwise_or(combined_mask, mask_blue)
    combined_mask = cv2.bitwise_or(combined_mask, mask_red1)
    combined_mask = cv2.bitwise_or(combined_mask, mask_red2)

    # Calculate the proportion of tissue pixels
    tissue_pixels = np.sum(combined_mask > 0)
    total_pixels = combined_mask.size
    tissue_ratio = tissue_pixels / total_pixels

    # Check if the tissue ratio is above the threshold
    return tissue_ratio > tissue_threshold

# Extract and save tissue-containing patches
for i in range(x_tiles):
    for j in range(y_tiles):
        x = i * patch_size
        y = j * patch_size

        # Ensure we don't go beyond the image boundaries
        if x + patch_size > width:
            x = width - patch_size
        if y + patch_size > height:
            y = height - patch_size

        # Read the region at the specified level
        patch = slide.read_region((x * magnification_factor, y * magnification_factor), level, (patch_size, patch_size))
        patch = patch.convert('RGB')  # Convert from RGBA to RGB

        # Filter patches based on tissue content using color filtering
        if is_tissue(patch):
            # Generate a filename based on the coordinates
            filename = f"patch_{i}_{j}.png"
            patch.save(os.path.join(output_dir, filename))
            print(f"Saved {filename} at coordinates: ({x}, {y})")

print("All tissue patches have been saved.")
