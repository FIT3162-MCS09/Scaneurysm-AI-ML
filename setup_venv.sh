#!/bin/bash

# Create a virtual environment
python -m venv sagemaker-env

# Activate the virtual environment
# On Windows use: .\sagemaker-env\Scripts\activate
source sagemaker-env/bin/activate

# Install required packages
pip install boto3 sagemaker torch numpy pillow jupyter

# Install VS Code Jupyter extension
code --install-extension ms-toolsai.jupyter

# Create a new Jupyter kernel
python3 -m ipykernel install --user --name sagemaker-env --display-name "SageMaker Environment"