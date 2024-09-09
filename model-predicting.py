import numpy as np
import tflite_runtime.interpreter as tflite
import matplotlib.pyplot as plt
from PIL import Image
from utils import preprocess_image


MODEL_FILENAME = "mnist_digit_classifier.tflite"

IMAGE_PATH = "images/test2.png"


def main():
    # Load the TFLite model
    interpreter = tflite.Interpreter(model_path=MODEL_FILENAME)

    # Allocate tensors
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess image
    processed_image = preprocess_image(IMAGE_PATH)

    # Display image
    plt.imshow(processed_image[0], cmap='gray')
    plt.title('Processed Image')
    plt.show()

    # Set input tensor to the processed image
    interpreter.set_tensor(input_details[0]['index'], processed_image)

    # Invoke interpreter to run inference
    interpreter.invoke()

    # Get output tensor and make prediction
    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_digit = np.argmax(prediction)

    print(f"Predicted Digit: {predicted_digit}")


if __name__ == "__main__":
    main()
