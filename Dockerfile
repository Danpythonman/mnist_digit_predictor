# Use the official Python 3.9 image
FROM python:3.9

# Set FLASK_ENV environment variable
ENV FLASK_ENV=production

# Change working directory
WORKDIR /usr/src/app

# Copy the TFLite model to the container
COPY mnist_digit_classifier.tflite .

# Fail if the TFLite model does not exist
RUN test -f mnist_digit_classifier.tflite || (echo "TFLite model not found" && exit 1)

# Copy Flask server code to the container
COPY app.py .

# Copy utils to the container
COPY utils utils

# Copy static files to container
COPY static static

# Copy requirements.txt to the container
COPY ./requirements-predicting.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-predicting.txt

# Expose port 8000
EXPOSE 8000

# Run the Flask server with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
