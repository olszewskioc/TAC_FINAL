from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Carregar o modelo
MODEL_PATH = "models/flower_model"
model = tf.keras.models.load_model(MODEL_PATH)

# Classes de flores
CLASS_NAMES = ['Daisy', 'Dandelion', 'Roses', 'Sunflowers', 'Tulips']

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file:
        # Salvar e processar a imagem
        file_path = os.path.join("temp", file.filename)
        file.save(file_path)

        img = load_img(file_path, target_size=(180, 180))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Fazer a previsão
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions)

        # Limpar arquivos temporários
        os.remove(file_path)

        return jsonify({
            "class": predicted_class,
            "confidence": float(confidence)
        })

    return jsonify({"error": "Invalid file"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
