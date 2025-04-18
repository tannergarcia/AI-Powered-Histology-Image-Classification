import os
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import label

def extract_islands_from_png(image_path, output_dir, max_dimension=1792):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image {image_path}")
        return
    
    # Convert to RGB (cv2 reads images in BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to HSV
    hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    
    # Define color ranges in HSV (same as original code)
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
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Label connected components
    structure = np.ones((3,3), dtype=int)
    labeled_mask, num_features = label(combined_mask, structure=structure)
    
    # Get sizes of each connected component
    component_sizes = np.bincount(labeled_mask.flatten())[1:]  # Exclude background (label 0)
    if len(component_sizes) == 0:
        print(f"No tissue regions found in {image_path}")
        return
    
    # Find the largest component size
    largest_component_size = component_sizes.max()
    
    # Set threshold as 0.3 times the largest component size (same as original code)
    threshold = 0.3 * largest_component_size
    
    # Get labels of components that meet the threshold
    valid_labels = [label_idx + 1 for label_idx, size in enumerate(component_sizes) if size >= threshold]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for label_idx in valid_labels:
        # Create mask for this label
        component_mask = (labeled_mask == label_idx).astype(np.uint8) * 255
        
        # Find bounding box
        coords = cv2.findNonZero(component_mask)
        x, y, w, h = cv2.boundingRect(coords)
        
        # Extract the region from the original image
        island_image = image_rgb[y:y+h, x:x+w]
        
        # Also extract the mask region to check for empty areas
        mask_region = component_mask[y:y+h, x:x+w]
        
        # Use the mask to ensure non-tissue regions are white instead of black
        island_image_masked = island_image.copy()
        island_image_masked[mask_region == 0] = [255, 255, 255]  # Set non-tissue to white
        island_image = island_image_masked
        
        # Resize to fit into max_dimension x max_dimension canvas
        scaling_factor = min(max_dimension / island_image.shape[1], max_dimension / island_image.shape[0])
        new_width = int(island_image.shape[1] * scaling_factor)
        new_height = int(island_image.shape[0] * scaling_factor)
        island_image_resized = cv2.resize(island_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create a white canvas
        final_image = np.ones((max_dimension, max_dimension, 3), dtype=np.uint8) * 255
        
        # Paste the island image onto the center of the canvas
        paste_x = (max_dimension - new_width) // 2
        paste_y = (max_dimension - new_height) // 2
        final_image[paste_y:paste_y+new_height, paste_x:paste_x+new_width] = island_image_resized
        
        # Save the final image
        output_path = os.path.join(output_dir, f"island_{label_idx}.png")
        final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, final_image_bgr)
        
    print(f"Finished processing {image_path}. {len(valid_labels)} islands saved.")

if __name__ == "__main__":
    image_path = '/Users/darianuriarte/Downloads/output_image88.png'  # Replace with your PNG image path
    output_directory = 'output_islands2'    # Replace with your desired output directory
    extract_islands_from_png(image_path, output_directory)
