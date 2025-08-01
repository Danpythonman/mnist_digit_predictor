MNIST Digit Prediction Backend on AWS Lambda
============================================

To save costs, I'm hosting the backend (i.e. model inference) on AWS Lambda. This "backend" only runs when it receives a request and simply performs preprocessing and inference.

The Python script [lambda_function.py](./lambda_function.py) contains the entirety of the custom code to achieve this functionality. The [Dockerfile](./Dockerfile) builds the Lambda function as a Docker image with all the dependencies (Python libraries from [requirements.txt](./requirements.txt) and the ONNX model) included.

Building
--------

The Bash script [build.sh](./build.sh) builds and deploys the image to AWS. Here is how to build it manually:

1. First, ensure you are in this directory with the [Dockerfile](./Dockerfile).

2. Build the image with Docker Buildx.

    ```bash
    docker buildx build --platform linux/amd64 --provenance=false -t "<image-name>:<image-tag>" .
    ```

3. Run the Docker container locally.

    ```bash
    docker run --platform linux/amd64 -p 9000:8080 <image-name>:<image-tag>
    ```

4. From here you can send requests to the container.

    ```bash
    curl -XPOST http://localhost:9000/2015-03-31/functions/function/invocations -d {"body": {"image": "<base64 encoded image of handwritten digit>"}}
    ```

    Note that you can technically send requests from the browser here, but CORS is not enabled automatically, which will cause requests to fail.

Deploying
---------

The Bash script [build.sh](./build.sh) builds and deploys the image to AWS. Here is how to deploy it manually:

1. Authenticate Docker with AWS credentials.

    ```bash
    aws ecr get-login-password --region "<region>" \
        | docker login --username AWS --password-stdin "<account id>.dkr.ecr.<region>.amazonaws.com"
    ```

2. Ensure the Docker image is built. See [Building section](#building) for more information.

3. Tag the image for AWS ECR.

    ```bash
    docker tag "<image-name>:<image-tag>" "<account id>.dkr.ecr.<region>.amazonaws.com/<ecr repo name>:latest"
    ```

4. Push the Docker image to AWS ECR.

    ```bash
    docker push "<account id>.dkr.ecr.<region>.amazonaws.com/<ecr repo name>:latest"
    ```

5. Update the Lambda function to use the new image.

    ```bash
    aws lambda update-function-code \
        --function-name "<lambda function name>" \
        --image-uri "<account id>.dkr.ecr.<region>.amazonaws.com/<ecr repo name>:latest" \
        --region "<region>"
    ```
