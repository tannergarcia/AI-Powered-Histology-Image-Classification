import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import shutil
import os
from PIL import Image
from classification_models.tfkeras import Classifiers
# Enable mixed precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')


image_dir = '/blue/vabfmc/data/working/tannergarcia/DermHisto/data/SCC/1792x1792_fixed_size/'
metadata_file = '/blue/vabfmc/data/working/tannergarcia/DermHisto/data/SCC/Csv_parsed.csv'
output_dir = '/blue/vabfmc/data/working/tannergarcia/DermHisto/data/SCC/organized_data/'
os.makedirs(output_dir, exist_ok=True)

# Step 1: Load and process metadata
metadata = pd.read_csv(metadata_file)
metadata['label'] = metadata['label'].map({'Clear': 0, 'Present': 1})  # Map labels to 0 and 1

# Group by StudyID to split data
grouped = metadata.groupby('StudyID #')

# Count images per slide
slide_counts = {}
for slide_id in metadata['slide_id'].unique():
    slide_folder = os.path.join(image_dir, slide_id)
    if os.path.exists(slide_folder):
        slide_counts[slide_id] = len([f for f in os.listdir(slide_folder) if f.endswith('.png')])
    else:
        slide_counts[slide_id] = 0

print(f"Tissue images per slide: {slide_counts}")
print(f"Total tissue images found: {sum(slide_counts.values())}")

# Convert the keys to a list
train_study_ids, val_study_ids = train_test_split(list(grouped.groups.keys()), test_size=0.2, random_state=42)

# Create organized directories for training and validation sets
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

def organize_images(group_ids, dest_dir):
    total_copied = 0  # Track copied images
    for study_id in group_ids:
        group = grouped.get_group(study_id)
        for _, row in group.iterrows():
            slide_id = row['slide_id']
            label = row['label']
            label_dir = 'Clear' if label == 0 else 'Present'
            source_dir = os.path.join(image_dir, slide_id)
            dest_label_dir = os.path.join(dest_dir, label_dir)
            os.makedirs(dest_label_dir, exist_ok=True)
            if os.path.exists(source_dir):
                for file in os.listdir(source_dir):
                    if file.endswith('.png'):
                        # Append slide_id to filename to prevent overwriting
                        new_file_name = f"{slide_id}_{file}"
                        shutil.copy(
                            os.path.join(source_dir, file),
                            os.path.join(dest_label_dir, new_file_name)
                        )
                        total_copied += 1
    print(f"Total images copied to {dest_dir}: {total_copied}")

# Organize images
organize_images(train_study_ids, train_dir)
organize_images(val_study_ids, val_dir)


def check_images_in_directory(directory):
    bad_files = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.lower().endswith('.png'):
                filepath = os.path.join(root, name)
                try:
                    with Image.open(filepath) as img:
                        img.verify()  # Verify that it's an image
                except (IOError, SyntaxError, Image.DecompressionBombError) as e:
                    print(f"Bad file detected: {filepath} - {e}")
                    bad_files.append(filepath)
    return bad_files

# Check training directory
print("Checking training images...")
bad_train_files = check_images_in_directory(train_dir)

# Check validation directory
print("Checking validation images...")
bad_val_files = check_images_in_directory(val_dir)


ResNet34, preprocess_input = Classifiers.get('resnet34')

# Step 3: Load the Pre-trained ResNet-34 model without the top layers
base_model = ResNet34(weights='imagenet', include_top=False, input_shape=(1792, 1792, 3))

# Freeze the base model layers initially
base_model.trainable = False

# Step 5: Build the model
inputs = layers.Input(shape=(1792, 1792, 3))
x = base_model(inputs, training=False)

# Capture the output of the last convolutional layer
last_conv_layer_output = x  # This is the output we need for Grad-CAM

# Multi-scale feature detection
x1 = layers.Conv2D(
    filters=128, kernel_size=(3, 3), activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(0.01), padding='same'
)(x)

x2 = layers.Conv2D(
    filters=128, kernel_size=(5, 5), activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(0.01), padding='same'
)(x)

x = layers.Concatenate()([x1, x2])

# Add dropout for regularization
x = layers.SpatialDropout2D(0.2)(x)

# Downsample with strided convolution
x = layers.Conv2D(
    filters=256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same'
)(x)

# Flatten spatial features using Global Average Pooling
x = layers.GlobalAveragePooling2D()(x)

# Fully connected layers for classification
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.5)(x)  # Dropout for fully connected layer
outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)

# Build the complete model
model = models.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

# Step 6: Set up callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('new_SCC.h5', save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
]

# Step 2: Use ImageDataGenerator for data loading
image_size = (1792, 1792)
batch_size = 4  # Adjusted batch size to fit GPU memory

# Define ImageDataGenerators with appropriate preprocessing
# Use the appropriate preprocessing function for ResNet34
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Create Generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1,
    workers=64,  # Number of CPU cores to use for data loading
    use_multiprocessing=True  # Enable multiprocessing for data loading
)


# Step 8: Fine-tune the model
# Unfreeze the base model
base_model.trainable = True

# Recompile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Continue training
history_fine_tune = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1,
    workers=64,  # Number of CPU cores to use for data loading
    use_multiprocessing=True  # Enable multiprocessing for data loading
)

model.save('after_finetuning_new_SCC_model.h5')

original_model = model  # Backup the in-memory model
# Step 9: Plot training history
import matplotlib.pyplot as plt

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history_fine_tune.history['accuracy'], label='Fine-tune Train Accuracy')
plt.plot(history_fine_tune.history['val_accuracy'], label='Fine-tune Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig('training_history.png')
plt.close()