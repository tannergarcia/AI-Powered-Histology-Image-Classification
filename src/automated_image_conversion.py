import openslide
import os
import sys


if len(sys.argv) < 3:
  print("Usage: python automated_image_conversion.py {INPUT_DIR} {OUTPUT_DIR}")
  sys.exit(1)

# Directory containing the .mrxs files
dir_path = sys.argv[1]

# Directory to output pngs
out_path = sys.argv[2]

# Output file for logging failures
log_file = 'conversion_failures.txt'

# Open the log file in write mode
with open(log_file, 'w') as log:
    # Iterate through all the files in the directory
    for filename in os.listdir(dir_path):
        # Check if the file has a .mrxs extension
        if filename.endswith(".mrxs"):
            slide_path = os.path.join(dir_path, filename)
            try:
                print(f"Processing {filename}...")

                # Open the slide
                slide = openslide.OpenSlide(slide_path)

                # Set the desired level for extraction (0 is the highest resolution)
                level = 3

                # Get the dimensions of the image at the selected level
                width, height = slide.level_dimensions[level]

                # Extract the image at the desired level
                image = slide.read_region((0, 0), level, (width, height))

                # Convert to RGB (remove alpha channel)
                image = image.convert("RGB")

                # Save the image as PNG
                output_image_path = f"{out_path}{filename.replace('.mrxs', '')}.png"
                image.save(output_image_path)

                print(f"Successfully converted {filename} to {output_image_path}")
            except Exception as e:
                # If an error occurs, log it and continue
                error_message = f"Failed to convert {filename}: {str(e)}\n"
                log.write(error_message)
                print(error_message)
