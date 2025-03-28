from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
from werkzeug.utils import secure_filename
import uuid
from image_proccessing import extract_islands_from_png  # Import the island extraction function
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.models import Model
import cv2
import json

# Initialize Flask app
app = Flask(__name__)
# Allow CORS for specific origins
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}}, supports_credentials=True)

def get_red_zone_coordinates(heatmap, img_width, img_height, threshold=0.5):
    # Resize heatmap to original image size
    heatmap_resized = cv2.resize(heatmap, (img_width, img_height))
    # Normalize heatmap to [0,255]
    heatmap_rescaled = np.uint8(255 * heatmap_resized)
    # Threshold the heatmap
    _, thresh = cv2.threshold(heatmap_rescaled, int(threshold * 255), 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get bounding boxes of contours
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    return bounding_boxes



# Load the model
MODEL_PATH = 'model/CAM_extended_BCC_model.h5'
model = load_model(MODEL_PATH, custom_objects={"SpatialDropout2D": SpatialDropout2D})

last_conv_layer_name = 'conv2d_2'
last_conv_layer = model.get_layer(last_conv_layer_name)
model_with_conv_outputs = Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])

# Directory to save uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Directory to save processed images
PROCESSED_FOLDER = 'processed'
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def make_gradcam_heatmap(img_array, model_with_conv_outputs, pred_index=None):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = model_with_conv_outputs(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        print("Gradients are None. Check the computation graph.")
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    conv_outputs = tf.cast(conv_outputs, tf.float32)
    pooled_grads = tf.cast(pooled_grads, tf.float32)

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())

    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap_rescaled = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_rescaled, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

    # Save the overlay image
    cv2.imwrite(cam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

# Preprocessing function
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(1792, 1792))  # Resize to match model input size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess using ResNet preprocessing
    return img_array

@app.route('/upload', methods=['POST'])
@cross_origin(origins=['http://localhost:3000'])
def upload_and_process():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file
    filename = secure_filename(file.filename)
    unique_id = str(uuid.uuid4())
    filename = f"{unique_id}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Process the image to extract islands
    output_subdir = os.path.join(PROCESSED_FOLDER, unique_id)
    os.makedirs(output_subdir, exist_ok=True)
    extract_islands_from_png(file_path, output_subdir)

    # Get list of processed images
    island_filenames = [f for f in os.listdir(output_subdir) if f.endswith('.png')]
    island_urls = [f"/processed/{unique_id}/{fname}" for fname in island_filenames]

    # Clean up the uploaded file if needed
    # os.remove(file_path)

    return jsonify({'islands': island_urls, 'id': unique_id})

@app.route('/processed/<unique_id>/<filename>')
@cross_origin(origins=['http://localhost:3000'])
def serve_processed_image(unique_id, filename):
    return send_from_directory(os.path.join(PROCESSED_FOLDER, unique_id), filename)

@app.route('/predict', methods=['POST'])
@cross_origin(origins=['http://localhost:3000'])
def predict():
    data = request.json
    image_url = data.get('image_url')
    if not image_url:
        return jsonify({'error': 'No image URL provided'}), 400

    # Extract the unique_id and filename from the image_url
    parts = image_url.strip('/').split('/')
    if len(parts) != 3:
        return jsonify({'error': 'Invalid image URL'}), 400
    _, unique_id, filename = parts
    image_path = os.path.join(PROCESSED_FOLDER, unique_id, filename)

    # Preprocess the image
    img_array = preprocess_image(image_path)

    # Make prediction
    prediction = model.predict(img_array)

    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_array, model_with_conv_outputs)

    # Get original image dimensions
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]

    # Process heatmap to find red zones
    red_zone_coords = get_red_zone_coordinates(heatmap, img_width, img_height)

    # Save the heatmap overlaid image
    heatmap_filename = f"heatmap_{filename}"
    heatmap_subdir = os.path.join(PROCESSED_FOLDER, unique_id)
    heatmap_path = os.path.join(heatmap_subdir, heatmap_filename)

    # Save the overlaid image
    if heatmap is not None:
        save_and_display_gradcam(image_path, heatmap, cam_path=heatmap_path)
        heatmap_url = f"/processed/{unique_id}/{heatmap_filename}"
    else:
        heatmap_url = None

    # Prepare response
    result = {
        'prediction': float(prediction[0][0]),
        'label': 'Present' if prediction[0][0] > 0.5 else 'Clear',
        'heatmap_url': heatmap_url,
        'red_zone_coords': red_zone_coords,
        'image_width': img_width,
        'image_height': img_height
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
