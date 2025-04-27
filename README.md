# Scaneurysm AI - Brain CT Aneurysm Detection

An AI-powered system for detecting brain aneurysms in CT scan images using PyTorch and AWS SageMaker.

## Overview

This project implements a deep learning model that analyzes brain CT scans to detect potential aneurysms. It uses a VGG16-based architecture and can be deployed as a serverless endpoint on AWS SageMaker.

## Prerequisites

- Python 3.9+
- AWS Account with SageMaker access
- Docker (for local testing)
- Required Python packages (see `requirements.txt`)

## Setup

1. Create and activate virtual environment:
```sh
./setup_venv.sh
```

2. Install dependencies:
```sh
pip install -r requirements.txt
```

3. Setup AWS infrastructure:
```sh
./setup_aws_infrastructure.sh
```

## Project Structure

```
.
├── my_model/
│   ├── model.pth              # Your trained model weights
│   └── code/
│       ├── inference.py       # Model inference code
│       └── requirements.txt   # Inference dependencies
├── test-local.ipynb          # Local testing notebook
├── final.ipynb               # Deployment notebook
├── setup_aws_infrastructure.sh
├── setup_venv.sh
├── requirements.txt
└── Dockerfile
```

## Model Details

- Architecture: Modified VGG16
- Input: Brain CT scan images (224x224 RGB)
- Output: Binary classification (Aneurysm/Non-aneurysm)
- Framework: PyTorch 2.1.0

## Deployment

1. Package your trained model:
```sh
# Ensure your model.pth is in the my_model/ directory
tar -czf model.tar.gz my_model/
```

2. Open `final.ipynb` in Jupyter and follow the deployment steps.

3. The deployment will create a serverless SageMaker endpoint.

## Testing

1. Local testing:
   - Use `test-local.ipynb` to verify model functionality

2. Endpoint testing:
   - Use the provided test functions in `final.ipynb`
   - Test with sample CT scan images via URLs

## API Usage

```python
import boto3
import json

def invoke_endpoint(endpoint_name, image_url):
    runtime = boto3.client('sagemaker-runtime')
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps({"url": image_url})
    )
    return json.loads(response['Body'].read().decode())
```

## Environment Variables

Required AWS environment variables:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- VGG16 architecture from torchvision
- AWS SageMaker team for serverless inference capabilities

## Important Note

Remember to bring your own trained model (`model.pth`). This repository does not include pretrained weights.