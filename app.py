import numpy as np
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime
import tflite_runtime.interpreter as tflite
from utils import preprocess_image, image_to_base64
import logging


app = Flask(__name__)

cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

gunicorn_error_logger = logging.getLogger('gunicorn.error')
app.logger.handlers.extend(gunicorn_error_logger.handlers)
app.logger.setLevel(logging.INFO)
app.logger.info("Starting application")

MODEL_FILENAME = "mnist_digit_classifier.tflite"

# Load the TFLite model
interpreter = tflite.Interpreter(model_path=MODEL_FILENAME)

def predictFromModel(image_file):
    """
    Predicts the digit from the image file.
    """

    # Allocate tensors
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor to the processed image
    interpreter.set_tensor(input_details[0]['index'], image_file)

    # Invoke interpreter to run inference
    interpreter.invoke()

    # Get output tensor and make prediction
    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_digit = np.argmax(prediction)

    return predicted_digit, prediction[0][predicted_digit] * 100


@app.route("/predict", methods=["POST"])
def predict():
    app.logger.info("Received request to '/predict'")

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Preprocess image
        processed_image = preprocess_image(file)

        predicted_digit, confidence = predictFromModel(processed_image)

        return jsonify({
            "digit": int(predicted_digit),
            "confidence": float(confidence),
            "processed_image": image_to_base64(processed_image)
        })

    except Exception as e:
        app.logger.error("Error in '/predict': " + str(e))
        return jsonify({"error": "Error predicting digit"}), 500


@app.route("/", methods=["GET"])
def index_html():
    app.logger.info("Received request to '/'")
    return send_from_directory("", "static/index.html")


@app.route("/style.css", methods=["GET"])
def style_css():
    app.logger.info("Received request to '/style.css'")
    return send_from_directory("", "static/style.css")


@app.route("/index.js", methods=["GET"])
def index_js():
    app.logger.info("Received request to '/index.js'")
    return send_from_directory("", "static/index.js")


@app.route("/github.svg", methods=["GET"])
def github_svg():
    app.logger.info("Received request to '/github.svg'")
    return send_from_directory("", "static/github.svg")


if __name__ == "__main__":
    if os.getenv("FLASK_ENV") == "development":
        app.run(debug=True, host="0.0.0.0", port=8081)
    else:
        app.logger.info("Running in production mode...")
