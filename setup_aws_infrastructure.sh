#!/bin/bash

# Set variables
AWS_REGION="ap-southeast-1"
BUCKET_NAME="brain-ct-async-inference-krooldonutz"
SUCCESS_TOPIC_NAME="brain-ct-success-krooldonutz"
ERROR_TOPIC_NAME="brain-ct-error-krooldonutz"
TIMESTAMP="2025-04-27-0427"

# Create S3 bucket
echo "Creating S3 bucket..."
aws s3api create-bucket \
    --bucket "$BUCKET_NAME" \
    --region "$AWS_REGION" \
    --create-bucket-configuration LocationConstraint="$AWS_REGION"

# Enable versioning on the bucket (recommended for production)
aws s3api put-bucket-versioning \
    --bucket "$BUCKET_NAME" \
    --versioning-configuration Status=Enabled

# Create SNS topics
echo "Creating SNS topics..."
SUCCESS_TOPIC_ARN=$(aws sns create-topic \
    --name "$SUCCESS_TOPIC_NAME" \
    --region "$AWS_REGION" \
    --output text \
    --query 'TopicArn')

ERROR_TOPIC_ARN=$(aws sns create-topic \
    --name "$ERROR_TOPIC_NAME" \
    --region "$AWS_REGION" \
    --output text \
    --query 'TopicArn')

# Output the ARNs
echo "Created resources:"
echo "S3 Bucket: $BUCKET_NAME"
echo "Success Topic ARN: $SUCCESS_TOPIC_ARN"
echo "Error Topic ARN: $ERROR_TOPIC_ARN"

# Create bucket policy
cat > bucket-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowSageMakerToWriteOutput",
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::${BUCKET_NAME}",
                "arn:aws:s3:::${BUCKET_NAME}/*"
            ]
        }
    ]
}
EOF

# Apply bucket policy
echo "Applying bucket policy..."
aws s3api put-bucket-policy \
    --bucket "$BUCKET_NAME" \
    --policy file://bucket-policy.json

# Clean up temporary files
rm bucket-policy.json

echo "Setup complete! Please update your Python code with these ARNs"