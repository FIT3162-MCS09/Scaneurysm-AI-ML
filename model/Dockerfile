FROM tensorflow/tensorflow:2.8.0-gpu

WORKDIR /opt/ml/model

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference code
COPY inference.py /opt/program/inference.py

# Set up the program in the image
COPY train.py /opt/program/train.py

# Set up entry point for SageMaker
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up for SageMaker
RUN chmod +x /opt/program/train.py

# SageMaker expects a train script
WORKDIR /opt/program
ENTRYPOINT ["python", "/opt/program/train.py"]