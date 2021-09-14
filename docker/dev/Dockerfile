# This dockerfile is used for development.

# Use TF 2.5 as base image with GPU support
FROM tensorflow/tensorflow:2.5.1-gpu

# Set working directory to deep-co-training
WORKDIR /deep-co-training
# Add the requirements.txt to the working directory
ADD ./requirements.txt .

# View contents while building dockerfile
RUN ls -a

# Install dependencies
RUN pip install -r requirements.txt