import os

import onnxruntime as ort

from mnist.api.singleton import singleton


@singleton
class ONNXInference:

    session: ort.InferenceSession

    def __init__(self):
        model_path = os.environ.get('MODEL_PATH', False)
        if not model_path:
            raise Exception('MODEL_PATH environment variable not set')
        self.session = ort.InferenceSession(str(model_path))

    def predict(self, inputs):
        outputs = self.session.run(None, {'input': inputs})
        return outputs

