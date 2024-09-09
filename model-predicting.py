import numpy as np
import tflite_runtime.interpreter as tflite
import matplotlib.pyplot as plt
from PIL import Image


MODEL_FILENAME = "mnist_digit_classifier.tflite"

IMAGE_PATH = "images/test2.png"


def preprocess_image(img_path):
    """
    """

    # Get image
    img = Image.open(img_path).convert('L')

    # Resize the image to 28x28 pixels (because this is the size of the images
    # inthe MNIST training data).
    img = img.resize((28, 28))

    # Put image in np array
    img_array = np.array(img)

    # Reshape for the model
    img_array = img_array.reshape(1, 28, 28, 1)

    # Convert numbers to float
    img_array = img_array.astype(np.float32) / 255.0

    return img_array


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
