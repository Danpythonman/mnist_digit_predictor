import base64
from  contextlib import contextmanager
import io
import json
import logging
from pathlib import Path
import time
import typing

import numpy as np
import onnxruntime as ort
from PIL import Image


logger = logging.getLogger()
logger.setLevel(logging.INFO)


@contextmanager
def timed(logger: logging.Logger, action: str):
    logger.info(f'Starting {action}...')
    t0 = time.time()
    yield
    t1 = time.time()
    logger.info(f'{action} took {(t1 - t0)*1000:.4f}ms')


def preprocess_image(image: Image.Image) -> np.ndarray:
    '''
    Preprocesses a PIL image into a NumPy array suitable as input to the
    MNIST neural network.
    '''

    n = np.array(image, dtype=np.float32)

    n = n / 255.0

    x_mean, x_std = (0.13066114485263824, 0.30810731649398804)

    n = (n - x_mean) / x_std

    n = n.reshape(1, 1, 64, 64)

    return n


def softmax(x: np.ndarray, axis: typing.Optional[int] = None) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def handler(event, context):
    logger.info(f'Starting execution for request id: {context.aws_request_id}')
    t0 = time.time()

    model_path = Path('model.onnx')
    if not model_path.exists():
        logger.error('Model path does not exist')
        return {'statusCode': 500, 'body': 'Model not found'}

    if 'body' not in event:
        logger.error('Event does not have a body')
        return {'statusCode': 400, 'body': 'Event does not have a body'}

    with timed(logger, 'Parsing request body'):
        body = json.loads(event['body'])

    if 'image' not in body:
        logger.error('Body does not have an image property')
        return {'statusCode': 400, 'body': 'Body does not have an image property'}

    image_base64 = body['image']

    logger.info(f'First 8 characters of base64 image: {str(image_base64)[:8]}')

    with timed(logger, 'Decoding base64 image'):
        image_bytes = base64.b64decode(image_base64)

    with timed(logger, 'Constructing ONNX inference session'):
        session = ort.InferenceSession(str(model_path))

    with timed(logger, 'Opening image'):
        image = Image.open(io.BytesIO(image_bytes)).convert('L')

    with timed(logger, 'Preprocessing image'):
        image = image.resize((64, 64))
        n = preprocess_image(image)

    with timed(logger, 'Running inference'):
        outputs = session.run(None, {'input': n})

    prediction = np.argmax(outputs[0], axis=1)[0]
    confidence = softmax(outputs[0], axis=1)[0][prediction] * 100

    t1 = time.time()
    logger.info(f'Request took {(t1-t1)*1000:.4f}ms. Returning prediction={int(prediction)}, confidence={float(confidence)}.')

    return {
        'statusCode': 200,
        'body': json.dumps({
            'digit': int(prediction),
            'confidence': float(confidence),
            'processed_image': ''
        })
    }
