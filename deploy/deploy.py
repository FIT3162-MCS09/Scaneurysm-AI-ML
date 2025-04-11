import argparse
import boto3
import json
import os
import time
import logging
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_model(sagemaker_client, model_name, role, image_uri, model_bucket, model_key):
    """
    Create a SageMaker model
    """
    try:
        model_response = sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': image_uri,
                'ModelDataUrl': f's3://{model_bucket}/{model_key}',
                'Environment': {
                    'S3_BUCKET': model_bucket,
                }
            },
            ExecutionRoleArn=role
        )
        logger.info(f"Model created: {model_response['ModelArn']}")
        return model_response['ModelArn']
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException' and 'Cannot create already existing model' in str(e):
            logger.info(f"Model {model_name} already exists.")
            return sagemaker_client.describe_model(ModelName=model_name)['ModelArn']
        else:
            raise

def create_endpoint_config(sagemaker_client, config_name, model_name, instance_type, instance_count=1):
    """
    Create a SageMaker endpoint configuration
    """
    try:
        endpoint_config_response = sagemaker_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    'VariantName': 'default',
                    'ModelName': model_name,
                    'InstanceType': instance_type,
                    'InitialInstanceCount': instance_count,
                    # Use Serverless Inference for cost optimization
                    # Alternatively, use scheduled auto-scaling for least cost
                }
            ]
        )
        logger.info(f"Endpoint config created: {endpoint_config_response['EndpointConfigArn']}")
        return endpoint_config_response['EndpointConfigArn']
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException' and 'Cannot create already existing endpoint configuration' in str(e):
            logger.info(f"Endpoint config {config_name} already exists.")
            return sagemaker_client.describe_endpoint_config(EndpointConfigName=config_name)['EndpointConfigArn']
        else:
            raise

def create_endpoint(sagemaker_client, endpoint_name, config_name):
    """
    Create or update a SageMaker endpoint
    """
    try:
        # Check if endpoint already exists
        try:
            endpoint_response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            logger.info(f"Endpoint {endpoint_name} already exists, updating...")
            
            # Update the endpoint
            sagemaker_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name
            )
        except ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                logger.info(f"Creating new endpoint {endpoint_name}...")
                endpoint_response = sagemaker_client.create_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=config_name
                )
            else:
                raise
                
        logger.info(f"Waiting for endpoint {endpoint_name} to be ready...")
        waiter = sagemaker_client.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=endpoint_name)
        
        endpoint = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        logger.info(f"Endpoint is ready: {endpoint['EndpointArn']}")
        return endpoint['EndpointArn']
    except Exception as e:
        logger.error(f"Error creating/updating endpoint: {e}")
        raise

def prepare_model_tar_gz(s3_client, bucket_name, model_key='model.tar.gz'):
    """
    Prepare a dummy model.tar.gz for deployment
    (SageMaker requires a model artifact even though we're using a pre-trained model)
    """
    import tarfile
    import tempfile
    
    # Create a temporary directory to store our model files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create class names file
        url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        try:
            import shap
            with open(shap.datasets.cache(url)) as file:
                class_names_data = json.load(file)
                
            # Save the class names locally
            with open(os.path.join(tmpdir, 'imagenet_class_index.json'), 'w') as f:
                json.dump(class_names_data, f)
                
            # Create model.tar.gz
            tar_path = os.path.join(tmpdir, 'model.tar.gz')
            with tarfile.open(tar_path, 'w:gz') as tar:
                tar.add(os.path.join(tmpdir, 'imagenet_class_index.json'), 
                        arcname='imagenet_class_index.json')
                
            # Upload to S3
            s3_client.upload_file(tar_path, bucket_name, model_key)
            logger.info(f"Uploaded model artifacts to s3://{bucket_name}/{model_key}")
            return model_key
        except Exception as e:
            logger.error(f"Error preparing model artifacts: {e}")
            raise

def main(args):
    """
    Main function to deploy the model to SageMaker
    """
    region = args.region
    
    # Initialize boto3 clients
    session = boto3.Session(region_name=region)
    sagemaker_client = session.client('sagemaker')
    s3_client = session.client('s3')
    
    # Confirm bucket exists or create it
    try:
        s3_client.head_bucket(Bucket=args.bucket)
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            logger.info(f"Bucket {args.bucket} does not exist. Creating...")
            if region == 'us-east-1':
                s3_client.create_bucket(Bucket=args.bucket)
            else:
                s3_client.create_bucket(
                    Bucket=args.bucket,
                    CreateBucketConfiguration={'LocationConstraint': region}
                )
        else:
            raise
    
    # Prepare model artifacts
    model_key = prepare_model_tar_gz(s3_client, args.bucket)
    
    # Get execution role or default to SageMaker-provided role
    if args.role:
        role = args.role
    else:
        # Get default role - this will only work if running from a SageMaker notebook instance
        try:
            import sagemaker
            role = sagemaker.get_execution_role()
        except Exception:
            raise ValueError("No execution role specified. Please provide an execution role ARN.")
    
    # Get appropriate container URI for TensorFlow
    # Note: In production, you should use specific region mapping
    image_uri = f"{args.account}.dkr.ecr.{region}.amazonaws.com/{args.repository}:latest"
    if args.image_uri:
        image_uri = args.image_uri
    
    # Create model
    model_name = args.model_name
    create_model(sagemaker_client, model_name, role, image_uri, args.bucket, model_key)
    
    # Create endpoint config
    config_name = f"{model_name}-config"
    create_endpoint_config(sagemaker_client, config_name, model_name, args.instance_type, args.instance_count)
    
    # Create or update endpoint
    endpoint_name = args.endpoint_name if args.endpoint_name else f"{model_name}-endpoint"
    endpoint_arn = create_endpoint(sagemaker_client, endpoint_name, config_name)
    
    # Output endpoint info
    logger.info(f"Endpoint ARN: {endpoint_arn}")
    endpoint_info = {
        'endpoint_name': endpoint_name,
        'endpoint_arn': endpoint_arn,
        'region': region
    }
    
    # Save endpoint info to file for GitHub Actions
    with open('endpoint_info.json', 'w') as f:
        json.dump(endpoint_info, f)
    
    return endpoint_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy a model to SageMaker')
    parser.add_argument('--region', type=str, default='us-east-1', help='AWS region')
    parser.add_argument('--bucket', type=str, required=True, help='S3 bucket for model artifacts')
    parser.add_argument('--model-name', type=str, default='shap-resnet50', help='Name of the model')
    parser.add_argument('--endpoint-name', type=str, help='Name of the endpoint (default: model-name-endpoint)')
    parser.add_argument('--role', type=str, help='SageMaker execution role ARN')
    parser.add_argument('--instance-type', type=str, default='ml.t2.medium', 
                        help='Instance type for the endpoint')
    parser.add_argument('--instance-count', type=int, default=1, 
                        help='Number of instances for the endpoint')
    parser.add_argument('--image-uri', type=str, help='Custom image URI')
    parser.add_argument('--account', type=str, help='AWS account ID for ECR')
    parser.add_argument('--repository', type=str, default='tensorflow-inference', 
                        help='ECR repository name')
    
    args = parser.parse_args()
    main(args)