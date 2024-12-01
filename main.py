from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from io import BytesIO
from flask_cors import CORS
import logging
logging.basicConfig(level=logging.DEBUG)


import numpy as np

app = Flask(__name__)
model = load_model("snake_species_classifier.h5")
@app.route('/')
def home():
    return "Welcome to the Snake Detection App!"

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return jsonify({"error": "No file received"}), 400
    print(f"File received: {file.filename}")
    image = load_img(BytesIO(file.read()), target_size=(128, 128))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
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
    class_labels = [
        "banded racer",
        "checkered keelback",
        "common rat snake",
        "Common Sand Boa",
        "Common Trinket",
        "Indian Rock Python",
        "Green Tree Vine",
        "common krait",
        "king cobra",
        "Monocled Cobra",
        "Russell's Viper",
        "Saw-scaled Viper",
        "Spectacled Cobra"
    ]
    predicted_species = class_labels[predicted_class]
    venom_status = venomous_map[predicted_species]
    logging.debug(f"species: {predicted_species}, status:{venom_status}")
    if predicted_species and venom_status:
        response = jsonify({'species': predicted_species, 'status': venom_status})
        response.headers['Content-Type'] = 'application/json'
        logging.debug(f"Response: {response.get_json()}")
        return response
    else:
        response = jsonify({'species': 'unknown species', 'status': 'NA'})
        response.headers['Content-Type'] = 'application/json'
        logging.debug(f"Response: {response.get_json()}")
        return response

if __name__ == "__main__":
    app.run(debug=True)
    CORS(app)
import os
port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port)
