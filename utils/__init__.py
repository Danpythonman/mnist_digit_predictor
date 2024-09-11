import numpy as np
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt


def preprocess_image(img_path):
    """
    """

    # Get image and convert to grayscale
    img = Image.open(img_path).convert('L')

    # Resize the image to 28x28 pixels (because this is the size of the images
    # in the MNIST training data).
    img = img.resize((28, 28))

    # Put image in np array
    img_array = np.array(img)

    # Reshape for the model
    img_array = img_array.reshape(1, 28, 28, 1)

    # Convert numbers to float
    img_array = img_array.astype(np.float32) / 255.0

    return img_array


def image_to_base64(img_array):
    """
    """

    # Plot the image (img_array is (1, 28, 28, 1), we take img_array[0])
    fig, ax = plt.subplots()
    # Reshape back to 28x28 for plotting
    ax.imshow(img_array[0].reshape(28, 28), cmap="gray")
    ax.set_title("Processed Image with Pixels")

    # Remove axis for a cleaner look
    # ax.axis("off")

    # Save the plot to a buffer in memory
    buffer = io.BytesIO()
    plt.savefig(buffer, format="PNG", bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # Convert the buffer contents (image) to a base64 string
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return img_str
