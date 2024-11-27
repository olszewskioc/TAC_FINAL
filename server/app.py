from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow.keras.activations as activations
import os

app = Flask(__name__)

# Carregar o modelo TFLite
interpreter = tf.lite.Interpreter(model_path="models/model.tflite")
print(interpreter.get_signature_list())
classify_lite = interpreter.get_signature_runner('serving_default')
# interpreter.allocate_tensors()

# Função para realizar a inferência com o modelo TFLite
def predict(input_image):
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()

    # # Garantir que o formato da imagem esteja correto
    # input_image = np.array(input_image, dtype=np.float32)
    # interpreter.set_tensor(input_details[0]['index'], input_image)
    # interpreter.invoke()
    # prediction = interpreter.get_tensor(output_details[0]['index'])
    predictions_lite = classify_lite(keras_tensor_15=input_image)['output_0']
    
    return predictions_lite

# Classes de flores
CLASS_NAMES = ['Daisy', 'Dandelion', 'Roses', 'Sunflowers', 'Tulips']

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file:
        # Salvar e processar a imagem
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)  # Criar pasta temporária, se não existir
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)

        try:
            # Preprocessar a imagem
            img = tf.keras.utils.load_img(file_path, target_size=(180, 180))
            img_array = tf.keras.utils.img_to_array(img)  # Normalizar os pixels
            img_array = tf.expand_dims(img_array, 0)  # Adicionar dimensão batch

            predictions = predict(img_array)
            score = tf.nn.softmax(predictions)
            # probabilities = activations.softmax(predictions).numpy()[0]

            # Determinar a classe com maior probabilidade
            predicted_class = CLASS_NAMES[np.argmax(score)]
            confidence = np.max(score) * 100

            # Retornar o resultado
            return jsonify({
                "class": predicted_class,
                "confidence": float(confidence),
                "predictions": predictions.tolist()
            })
        finally:
            # Limpar arquivos temporários
            os.remove(file_path)

    return jsonify({"error": "Invalid file"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
