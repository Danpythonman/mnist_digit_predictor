import base64
import io
import typing

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def preprocess_image(image: Image.Image) -> np.ndarray:
    '''
    Preprocesses a PIL image into a NumPy array suitable as input to the
    MNIST neural network.
    '''
    n = np.array(image, dtype=np.float32)

    n = n / 255.0

    x_mean, x_std = (0.13066114485263824, 0.30810731649398804)

    n = (n - x_mean) / x_std

    n = n.reshape(1, 1, 28, 28)

    return n


def image_to_base64(img_array: np.ndarray) -> str:
    '''
    Converts an image as a NumPy array into a base64 string.
    '''

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


def softmax(x: np.ndarray, axis: typing.Optional[int] = None) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)
