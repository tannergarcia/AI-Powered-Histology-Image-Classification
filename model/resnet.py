import pandas as pd
import os
import numpy as np
import subprocess
import math
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, ZeroPadding2D, Conv2D, BatchNormalization,
                                     Activation, MaxPooling2D, Add, Flatten, Dense, Dropout)
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC, Precision, Recall

# Ensure TensorFlow version is compatible (2.x)
print("TensorFlow Version:", tf.__version__)

# ========================
# Dynamic Batch Size Adjustment
# ========================

def get_available_gpu_memory():
    try:
        # Execute nvidia-smi command to get memory details
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        # Parse the result
        memory_free = [int(x) for x in result.strip().split('\n')]
        return memory_free
    except Exception as e:
        print("Error querying GPU memory:", e)
        # Default to a safe batch size if querying fails
        return [0]

def determine_batch_size(memory_free, memory_per_image=1000):
    """
    Determine batch size based on available GPU memory.
    Args:
        memory_free (list): List of free memory (MB) per GPU.
        memory_per_image (int): Estimated memory usage per image (MB).
    Returns:
        batch_size (int): Suitable batch size per GPU.
    """
    # Define a buffer to prevent memory over-allocation
    buffer = 500  # MB
    batch_size = []
    for mem in memory_free:
        max_images = (mem - buffer) // memory_per_image
        # Ensure at least a batch size of 1
        batch_size.append(max(1, max_images))
    return batch_size

# Get available GPU memory
memory_free = get_available_gpu_memory()
print("Available GPU Memory (MB):", memory_free)

# Determine batch size per GPU
batch_size_per_gpu = determine_batch_size(memory_free)
print("Batch Size per GPU:", batch_size_per_gpu)

# Total batch size is sum of batch sizes per GPU
total_batch_size = 16

# ========================
# Data Preparation
# ========================

# Paths to your CSV and image directories
bcc_labels_path = '/blue/vabfmc/data/working/tannergarcia/DermHisto/data/SCC/Csv_parsed.csv'
slides_dir = '/blue/vabfmc/data/working/tannergarcia/DermHisto/data/SCC/1792x1792_fixed_size/'

# Load labels
bcc_labels_df = pd.read_csv(bcc_labels_path)

# Process labels
bcc_labels_df = bcc_labels_df[['slide_id', 'StudyID #', 'label']]
bcc_labels_df['binary_label'] = bcc_labels_df['label'].apply(lambda x: 1 if x == "Present" else 0)

# Initialize lists to store matched file paths and labels
image_paths = []
labels = []
study_ids = []

# Match image paths with labels
for _, row in bcc_labels_df.iterrows():
    slide_id = row['slide_id']
    study_id = row['StudyID #']
    label = row['binary_label']
    
    folder_path = os.path.join(slides_dir, slide_id)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".png"):
                image_paths.append(os.path.join(folder_path, file_name))
                labels.append(label)
                study_ids.append(study_id)
    else:
        print(f"Warning: Directory not found for slide_id: {slide_id}")

# Create DataFrame
data_df = pd.DataFrame({
    'image_path': image_paths,
    'binary_label': labels,
    'study_id': study_ids
})

# Ensure DataFrame is not empty
if data_df.empty:
    raise ValueError("No matching images were found. Please verify the directory paths and slide IDs.")
else:
    # Display the first few rows of data_df
    print("Complete DataFrame:")
    print(data_df.head())

    # Split data
    train_df, test_df = train_test_split(data_df, test_size=0.2, stratify=data_df['binary_label'], random_state=42)

    # Display the first few rows of the training and test data
    print("\nTraining Data:")
    print(train_df.head())
    print("\nTesting Data:")
    print(test_df.head())

    # Get the number of images for each label in the overall, training, and testing datasets
    print("\nOverall distribution (total count):")
    print(data_df['binary_label'].value_counts())
    
    print("\nTraining distribution (total count):")
    print("No Cancer (0):", train_df['binary_label'].value_counts().get(0, 0))
    print("Cancer (1):", train_df['binary_label'].value_counts().get(1, 0))
    
    print("\nTesting distribution (total count):")
    print("No Cancer (0):", test_df['binary_label'].value_counts().get(0, 0))
    print("Cancer (1):", test_df['binary_label'].value_counts().get(1, 0))
    
    # Get the total number of images in each set
    print("\nTotal number of images:")
    print("Overall:", len(data_df))
    print("Training:", len(train_df))
    print("Testing:", len(test_df))


# Make explicit copies to avoid SettingWithCopyWarning
train_df = train_df.copy()
test_df = test_df.copy()

# Convert binary labels to strings
train_df['binary_label'] = train_df['binary_label'].astype(str)
test_df['binary_label'] = test_df['binary_label'].astype(str)

# Function to create generators
def create_generator(datagen, df, batch_size):
    return datagen.flow_from_dataframe(
        dataframe=df,
        x_col='image_path',
        y_col='binary_label',
        target_size=(1792, 1792),
        color_mode='rgb',
        class_mode='binary',
        batch_size=batch_size,
        shuffle=True
    )

# Define ImageDataGenerators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)


test_datagen = ImageDataGenerator(rescale=1./255)

# Create Generators
train_generator = create_generator(train_datagen, train_df, batch_size=total_batch_size)
test_generator = create_generator(test_datagen, test_df, batch_size=total_batch_size)


# Check the first batch from the train generator
train_batch = next(train_generator)
test_batch = next(test_generator)

# Print details of the train batch
print("Train Generator - First Batch")
print("Batch shape (images):", train_batch[0].shape)
print("Batch shape (labels):", train_batch[1].shape)
print("Sample labels from train batch:", train_batch[1][:10])  # Display first 10 labels

# Print details of the test batch
print("\nTest Generator - First Batch")
print("Batch shape (images):", test_batch[0].shape)
print("Batch shape (labels):", test_batch[1].shape)
print("Sample labels from test batch:", test_batch[1][:10])  # Display first 10 labels

# ========================
# Model Definition
# ========================

# Set mixed precision policy
mixed_precision.set_global_policy('mixed_float16')


# Define res_identity and res_conv functions
def res_identity(x, filters):
    x_skip = x
    f1, f2 = filters

    # First block
    x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second block
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Third block
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)

    # Add input (skip connection)
    x = Add()([x, x_skip])
    x = Activation('relu')(x)

    return x

def res_conv(x, s, filters):
    x_skip = x
    f1, f2 = filters

    # First block
    x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid',
               kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second block
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Third block
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)

    # Shortcut path
    x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid',
                    kernel_regularizer=l2(0.001))(x_skip)
    x_skip = BatchNormalization()(x_skip)

    # Add input (skip connection)
    x = Add()([x, x_skip])
    x = Activation('relu')(x)

    return x

def resnet_1792_head(input_shape):
    input_im = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(3, 3))(input_im)

    x = Conv2D(8, kernel_size=(7, 7), strides=(2, 2), kernel_regularizer=l2(0.001))(x)  # Output size 896
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)  # Output size 448

    x = res_conv(x, s=1, filters=(8, 32))
    x = res_identity(x, filters=(8, 32))
    x = res_identity(x, filters=(8, 32))

    x = res_conv(x, s=2, filters=(16, 64))  # Output Size 224
    x = res_identity(x, filters=(16, 64))
    x = res_identity(x, filters=(16, 64))
    x = res_identity(x, filters=(16, 64))

    x = res_conv(x, s=2, filters=(32, 128))  # Output size 112
    x = res_identity(x, filters=(32, 128))
    x = res_identity(x, filters=(32, 128))
    x = res_identity(x, filters=(32, 128))

    x = Conv2D(64, kernel_size=(1, 1), strides=(2, 2), padding='valid',
               kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    model = Model(inputs=input_im, outputs=x, name='HiResnet50')

    return model

def HiResNet(size, weights, classes):
    if size == 1792:
        input_shape = (1792, 1792, 3)
        hi_res_head = resnet_1792_head(input_shape)
    else:
        raise ValueError('Invalid size for Hi-ResNet head.')

    if weights == "Res50":
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    elif weights == "None":
        base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    else:
        raise ValueError('weights should be either: "Res50" or "None"')

    truncated_model = Model(inputs=base_model.layers[7].input, outputs=base_model.layers[-1].output)
    final_model = truncated_model(hi_res_head.output)
    model = Model(inputs=hi_res_head.input, outputs=final_model, name='HiResnet')

    # Add custom top layers
    head_model = MaxPooling2D(pool_size=(4, 4))(model.output)
    head_model = Flatten(name='flatten')(head_model)
    head_model = Dense(1024, activation='relu')(head_model)
    head_model = Dropout(0.1)(head_model)
    head_model = Dense(512, activation='relu')(head_model)
    head_model = Dropout(0.1)(head_model)
    head_model = Dense(classes, activation='sigmoid', dtype='float32')(head_model)  # Ensure output is float32

    return Model(model.input, head_model)

# ========================
# Model Compilation and Training
# ========================

# Define the distribution strategy
strategy = tf.distribute.MirroredStrategy()
print('Number of devices:', strategy.num_replicas_in_sync)

# Build and compile the model within the strategy scope
with strategy.scope():
    model = HiResNet(size=1792, weights="Res50", classes=1)

    # Choose an optimizer
    optimizer = Adam(learning_rate=1e-5)

    # Compile the model
    model.compile(
    loss="binary_crossentropy",
    optimizer=optimizer,
    metrics=['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')]
    )

# Summary of the model
model.summary()

# Define callbacks
checkpoint = ModelCheckpoint(
    "HiResNet_best.h5",
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_weights_only=False
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    mode='min',
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1,
    mode='min',
    min_lr=1e-7
)

callbacks = [checkpoint, early_stopping, reduce_lr]


# Calculate steps per epoch
train_steps = math.ceil(train_generator.n / train_generator.batch_size)
test_steps = math.ceil(test_generator.n / test_generator.batch_size)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    validation_data=test_generator,
    validation_steps=test_steps,
    epochs=50,
    callbacks=callbacks,
    workers=1,
    use_multiprocessing=False
)

# ========================
# Model Evaluation- NO DISPLAY
# ========================

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the best model
model.load_weights("HiResNet_best.h5")

# Evaluate on the test set
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Generate predictions
test_generator.reset()
Y_pred = model.predict(test_generator, steps=test_steps)
y_pred = (Y_pred > 0.5).astype(int).flatten()
y_true = test_generator.classes[:len(y_pred)]

# Classification report
print(classification_report(y_true, y_pred, target_names=['Non-Cancerous', 'Cancerous']))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Cancerous', 'Cancerous'],
            yticklabels=['Non-Cancerous', 'Cancerous'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png")

# Plot training history
def plot_history(history):
    # Plot accuracy
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot AUC
    plt.subplot(2, 2, 3)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('AUC over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    # Plot Precision
    plt.subplot(2, 2, 4)
    plt.plot(history.history['precision'], label='Train Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.title('Precision over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()

    # Plot Recall
    plt.figure(figsize=(7, 5))
    plt.plot(history.history['recall'], label='Train Recall')
    plt.plot(history.history['val_recall'], label='Validation Recall')
    plt.title('Recall over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()

    plt.tight_layout()
    plt.savefig("history.png")

plot_history(history)
