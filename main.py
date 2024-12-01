from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from io import BytesIO
from flask_cors import CORS
import logging
import numpy as np
import os

# Set up Flask app and CORS
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

# Load the model
model = load_model("snake_species_classifier.h5")

@app.route('/')
def home():
    return "Welcome to the Snake Detection App!"

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        logging.error("No file part in the request")
        return jsonify({"error": "No file received"}), 400
    
    logging.debug(f"File received: {file.filename}, content_type={file.content_type}")
    image = load_img(BytesIO(file.read()), target_size=(128, 128))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    logging.debug(f"Predictions: {predictions}")
    predicted_class = np.argmax(predictions)

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

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
