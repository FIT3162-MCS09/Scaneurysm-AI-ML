# SHAP ResNet-50 SageMaker Endpoint

This project deploys a SHAP explainer for ResNet-50 as a SageMaker endpoint, provisioned through GitHub Actions and accessible from a Django backend.

## Directory Structure


## Cost Optimization

This deployment uses the following strategies to minimize costs:

1. **Instance Type**: Uses `ml.t2.medium` instances which are among the lowest-cost SageMaker instances.
2. **On-demand Provisioning**: The endpoint is created and destroyed as needed via GitHub Actions.
3. **Single Instance**: Default configuration uses only one instance.

## Deployment

To deploy the SageMaker endpoint:

1. Store your AWS credentials as GitHub secrets:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_SAGEMAKER_ROLE_ARN` (IAM role with SageMaker permissions)

2. Run the GitHub Actions workflow manually from the Actions tab.

3. After deployment, note the endpoint name from the workflow outputs.

## Django Integration

Add this to your Django application to call the endpoint:

```python
import boto3
import base64
from PIL import Image
import io
import json

def call_sagemaker_endpoint(image_path, endpoint_name, region_name='us-east-1'):
    # Load and prepare the image
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    # Encode image data as base64
    base64_encoded = base64.b64encode(image_data).decode('utf-8')
    
    # Create payload
    payload = json.dumps({'image': base64_encoded})
    
    # Call SageMaker endpoint
    runtime = boto3.client('sagemaker-runtime', region_name=region_name)
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=payload
    )
    
    # Parse response
    result = json.loads(response['Body'].read().decode())
    return result
```