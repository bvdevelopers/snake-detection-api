from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from io import BytesIO

import numpy as np

app = Flask(__name__)
model = load_model("snake_species_classifier.h5")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = load_img(BytesIO(file.read()), target_size=(128, 128))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
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

    return jsonify({'species': predicted_species})

if __name__ == "__main__":
    app.run(debug=True)
