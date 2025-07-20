import io
import typing

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter
import numpy as np
from PIL import Image

from mnist.api.utils import preprocess_image, image_to_base64, softmax
from mnist.api.inference import ONNXInference


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

    image = Image.open(io.BytesIO(contents)).convert('L')

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

app.include_router(router)
