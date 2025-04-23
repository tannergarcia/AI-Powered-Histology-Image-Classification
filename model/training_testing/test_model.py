import os
import pandas as pd
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import SpatialDropout2D

# Paths
MODEL_PATH = '/blue/vabfmc/data/working/tannergarcia/DermHisto/AI-Powered-Histology-Image-Classification/model/trained/4-17_lower_val/after_finetuning_new_SCC_model.h5'
val_dir = '/blue/vabfmc/data/working/tannergarcia/DermHisto/data/SCC/organized_data/val/'


model = load_model(MODEL_PATH, custom_objects={"SpatialDropout2D": SpatialDropout2D})

# Data generator for test images
image_size = (1792, 1792)
batch_size = 4

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Step 1: Make predictions on the validation set
val_generator.reset()  # Ensure generator is at the start
predictions = model.predict(val_generator)
predicted_classes = (predictions > 0.5).astype("int32")  # Convert probabilities to binary labels

# Step 2: Get true labels from the generator
true_classes = val_generator.classes

# Step 3: Calculate the confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Step 4: Extract TP, TN, FP, FN from the confusion matrix
TN, FP, FN, TP = conf_matrix.ravel()  # Assuming a binary classification problem

# Display the results
print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")

# Optional: Print a classification report for additional metrics
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=val_generator.class_indices.keys()))

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(true_classes, predictions)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.close()


# Compute Precision-Recall curve and Average Precision
precision, recall, _ = precision_recall_curve(true_classes, predictions)
avg_precision = average_precision_score(true_classes, predictions)

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall Curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig('precision_recall_curve.png')
plt.close()