import openslide

# Path to your .mrxs file
slide_path = '/blue/vabfmc/data/working/tannergarcia/DermHisto/data/BCC/AN_Batch_02.09.22_2015_BCC/slide-2022-02-09T12-30-52-R5-S3.mrxs'

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
image.save("output_image.png")

print("Conversion complete!")
