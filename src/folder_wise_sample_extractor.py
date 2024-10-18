import cv2
import os
import numpy as np
from PIL import Image

# Function to replace black pixels with white
def convert_black_to_white(image):
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Identify black pixels (threshold based on RGB values)
    black_pixels = np.all(img_array == [0, 0, 0], axis=-1)
    
    # Set black pixels to white
    img_array[black_pixels] = [255, 255, 255]
    
    # Convert back to image
    return Image.fromarray(img_array)

# Function to extract tissue pieces as separate images
def extract_tissues(image_path, output_folder, min_size=10000, min_aspect_ratio=0.5):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Thresholding to make sure the background is white (black becomes white, foreground stays the same)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours (external only to avoid nested contours)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    tissue_count = 1
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    for contour in contours:
        # Filter out small contours based on area (size threshold)
        area = cv2.contourArea(contour)
        if area < min_size:
            continue  # Skip small contours
        
        # Get the bounding box around the tissue piece
        x, y, w, h = cv2.boundingRect(contour)
        
#         # Filter based on the aspect ratio (to avoid shapes like 'C')
#         aspect_ratio = w / float(h)
#         if aspect_ratio < min_aspect_ratio or aspect_ratio > 1 / min_aspect_ratio:
#             continue  # Skip contours with non-rectangular aspect ratios (too long or too narrow)
        
        # Extract tissue piece from the original image
        tissue_piece = image[y:y+h, x:x+w]
        
        # Save the extracted tissue piece as an image
        output_path = f"{output_folder}/{base_filename}_{tissue_count}.png"
        cv2.imwrite(output_path, tissue_piece)
        
        print(f"Saved: {output_path}")
        tissue_count += 1
        
# Function to process all images in a folder
def process_folder(input_folder, output_folder):
    # Iterate over all .png files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            
            # Create a corresponding output folder for each input image
            base_filename = os.path.splitext(filename)[0]
            image_output_folder = os.path.join(output_folder, base_filename)
            
            # Create the folder if it doesn't exist
            if not os.path.exists(image_output_folder):
                os.makedirs(image_output_folder)
            
            # Open the input image using PIL
            input_image = Image.open(image_path)
            
            # Step 1: Convert black pixels to white
            processed_image = convert_black_to_white(input_image)
            
            # Step 2: Save the processed image (black-to-white) before extracting tissues
            processed_image_path = os.path.join(image_output_folder, f"{base_filename}_processed.png")
            processed_image.save(processed_image_path)
            
            # Step 3: Extract individual tissue pieces and save them in the respective folder
            extract_tissues(processed_image_path, image_output_folder)

# Example usage
input_folder = 'Images/SCC_Images'  # Folder with input images
output_folder = 'Extracted_Samples/SCC_Samples'  # Folder to save extracted samples

# Process all images in the input folder
process_folder(input_folder, output_folder)
