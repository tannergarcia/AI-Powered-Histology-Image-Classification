import openslide

# Path to your .mrxs file
slide_path = 'slide-2022-02-09T12-26-27-R5-S1.mrxs'

# Open the slide
slide = openslide.OpenSlide(slide_path)

# Set the desired level for extraction (0 is the highest resolution)
level = 5

# Get the dimensions of the image at the selected level
width, height = slide.level_dimensions[level]

# Extract the image at the desired level
image = slide.read_region((0, 0), level, (width, height))

# Convert to RGB (remove alpha channel)
image = image.convert("RGB")

# Save the image as PNG
image.save("output_image.png")

print("Conversion complete!")
