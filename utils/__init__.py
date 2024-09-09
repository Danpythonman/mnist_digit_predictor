import numpy as np
from PIL import Image

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
