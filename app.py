import numpy as np
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
from datetime import datetime
from utils import preprocess_image


app = Flask(__name__)

cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Save the uploaded image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join("/home/daniel/Programming/mlcc-exercises_en/", filename)
        file.save(filepath)

        # Preprocess the image
        processed_image = preprocess_image(file)

        # Make the prediction using the loaded model
        prediction = model.predict(processed_image)
        predicted_digit = np.argmax(prediction)

        return jsonify({"digit": int(predicted_digit)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index_html():
    return send_from_directory("static", "static/index.html")


@app.route("/style.css", methods=["GET"])
def style_css():
    return send_from_directory("", "static/style.css")


@app.route("/index.js", methods=["GET"])
def index_js():
    return send_from_directory("", "static/index.js")


if __name__ == "__main__":
    if os.getenv("FLASK_ENV") == "development":
        app.run(debug=True, port=8000)
    else:
        print("Running in production mode...")
