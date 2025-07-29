import base64
import io
import typing

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter
import numpy as np
from PIL import Image
from pydantic import BaseModel

from mnist.api.utils import preprocess_image, image_to_base64, softmax
from mnist.api.inference import ONNXInference


class ImageRequest(BaseModel):
    image: str


def run_inference(image_bytes: bytes) -> dict:
    '''
    Runs the full inference pipeline on the image data.

    This consists of reading the image bytes into a PIL image, preprocessing the
    image, and running the model on the image to generate the prediction.

    Args:
        image_bytes: The bytes representing the image of the digit.

    Returns:
        A dictionary containing the predicted digit, its confidence, and the
        preprocessed image in base64 format.
    '''

    image = Image.open(io.BytesIO(image_bytes)).convert('L')

    image = image.resize((28, 28))

    n = preprocess_image(image)

    inference = ONNXInference()
    p = inference.predict(n)
    p = typing.cast(typing.List[np.ndarray], p)

    prediction = np.argmax(p[0], axis=1)[0]
    confidence = softmax(p[0], axis=1)[0][prediction] * 100
    return {
        'digit': int(prediction),
        'confidence': float(confidence),
        'processed_image': image_to_base64(n)
    }


app = FastAPI(
    title='MNIST Digit Predictor',
    version='0.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

router = APIRouter()


@router.post('/predict')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    return run_inference(contents)


@router.post('/base64/predict')
async def predict(image_request: ImageRequest):
    image_bytes = base64.b64decode(image_request.image)

    return run_inference(image_bytes)


app.include_router(router)
