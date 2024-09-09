import os
import tensorflow as tf


MODEL_FILENAME = "mnist_digit_classifier.h5"


def main():
    if not os.path.isfile(MODEL_FILENAME):
        print(f"Model with filename {MODEL_FILENAME} does not exist.")
        return

    # Load model from file
    model = tf.keras.models.load_model(MODEL_FILENAME)

    # Convert model to tensorflow lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the tensorflow lite model
    with open("mnist_digit_classifier.tflite", "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    main()
