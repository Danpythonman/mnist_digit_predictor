FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install .

COPY model.onnx .

ENV IGNORE_INITIAL_MODULENOTFOUND=true
ENV MODEL_PATH=/app/model.onnx

EXPOSE 8000

CMD ["uvicorn", "src.mnist.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
