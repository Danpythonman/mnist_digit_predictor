#!/bin/bash

# This script builds the Docker image for the AWS Lambda function, pushes it to
# AWS ECR, and updates the AWS Lambda function to use the new image.
#
# This script is based on the instructions in the following documentation:
# https://docs.aws.amazon.com/lambda/latest/dg/python-image.html#python-image-instructions

# Exit immediately on error
set -e

# Usage
if [ "$#" -ne 4 ]; then
    echo 'Error: incorrect usage'
    echo ''
    echo "Usage: $0 <DOCKER_TAG> <AWS_REGION> <AWS_ACCOUNT_ID> <LAMBDA_FUNCTION_NAME>"
    exit 1
fi

DOCKER_TAG="$1"
AWS_REGION="$2"
AWS_ACCOUNT_ID="$3"
LAMBDA_FUNCTION_NAME="$4"

REPO_NAME="mnist-digit-predictor"
LOCAL_IMAGE_NAME="mnist-lambda-image:${DOCKER_TAG}"
ECR_IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:latest"

# Authenticate with ECR
aws ecr get-login-password --region "${AWS_REGION}" \
    | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Build the Docker image
docker buildx build --platform linux/amd64 --provenance=false -t "${LOCAL_IMAGE_NAME}" .

# Tag the image for ECR
docker tag "${LOCAL_IMAGE_NAME}" "${ECR_IMAGE_URI}"

# Push the image to ECR
docker push "${ECR_IMAGE_URI}"

# Update Lambda function to use the new image
aws lambda update-function-code \
  --function-name "${LAMBDA_FUNCTION_NAME}" \
  --image-uri "${ECR_IMAGE_URI}" \
  --region "${AWS_REGION}"
