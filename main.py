from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
import logging
import os

# Set up Flask app and CORS
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

# Set maximum allowed size to 2 MB (example)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="snake_species_classifier.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/')
def home():
    return "Welcome to the Snake Detection App!"

@app.route('/predict', methods=['POST'])
def predict():
    if request.content_length > app.config['MAX_CONTENT_LENGTH']:
        logging.error("File too large")
        return jsonify({"error": "File size exceeds limit"}), 400

    file = request.files.get('file')
    if not file:
        logging.error("No file part in the request")
        return jsonify({"error": "No file received"}), 400

    if not file.content_type.startswith("image/"):
        logging.error("Unsupported file type")
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        logging.debug(f"File received: {file.filename}, content_type={file.content_type}")
        # Open and preprocess image
        image = Image.open(BytesIO(file.read())).convert("RGB")
        image = image.resize((128, 128))  # Resize to match model input
        image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], image_array)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        predictions = interpreter.get_tensor(output_details[0]['index'])
        logging.debug(f"Predictions: {predictions}")
        predicted_class = np.argmax(predictions)

        # Map predictions to class labels
        venomous_map = {
            "banded racer": "Non-Venomous",
            "checkered keelback": "Non-Venomous",
            "common rat snake": "Non-Venomous",
            "Common Sand Boa": "Non-Venomous",
            "Common Trinket": "Non-Venomous",
            "Indian Rock Python": "Non-Venomous",
            "Green Tree Vine": "Non-Venomous",
            "common krait": "Venomous",
            "king cobra": "Venomous",
            "Monocled Cobra": "Venomous",
            "Russell's Viper": "Venomous",
            "Saw-scaled Viper": "Venomous",
            "Spectacled Cobra": "Venomous",
        }
        class_labels = list(venomous_map.keys())
        predicted_species = class_labels[predicted_class]
        venom_status = venomous_map[predicted_species]

        response_data = {'species': predicted_species, 'status': venom_status}
        logging.debug(f"Response Data: {response_data}")
        return jsonify(response_data), 200

    except Exception as e:
        logging.error(f"Error processing the image: {e}")
        return jsonify({"error": "Failed to process the image"}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
