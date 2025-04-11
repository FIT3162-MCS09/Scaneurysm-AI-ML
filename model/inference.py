import os
import json
import boto3
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import shap
from PIL import Image
import io

# Global variables
model = None
explainer = None
class_names = None

def model_fn(model_dir):
    """
    Load the model for inference
    """
    global model, explainer, class_names
    
    print("Loading model...")
    # Load pre-trained model
    model = ResNet50(weights="imagenet")
    
    # Download class names from S3 if not already available
    s3 = boto3.client('s3')
    try:
        # Check if we have the class names file locally
        if os.path.exists(os.path.join(model_dir, 'imagenet_class_index.json')):
            with open(os.path.join(model_dir, 'imagenet_class_index.json')) as file:
                class_names = [v[1] for v in json.load(file).values()]
        else:
            # Download from your S3 bucket
            bucket = os.environ.get('S3_BUCKET', 'your-bucket-name')
            s3.download_file(bucket, 'imagenet_class_index.json', 
                            os.path.join(model_dir, 'imagenet_class_index.json'))
            with open(os.path.join(model_dir, 'imagenet_class_index.json')) as file:
                class_names = [v[1] for v in json.load(file).values()]
    except:
        # Fallback to direct download
        url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        with open(shap.datasets.cache(url)) as file:
            class_names = [v[1] for v in json.load(file).values()]
    
    # Function to get model output
    def f(x):
        tmp = x.copy()
        preprocess_input(tmp)
        return model(tmp)
    
    # Create a sample image shape for the masker
    sample_shape = (1, 224, 224, 3)  # Default ImageNet shape
    
    # Define masker
    masker = shap.maskers.Image("inpaint_telea", sample_shape)
    
    # Create explainer
    explainer = shap.Explainer(f, masker, output_names=class_names)
    
    return model

def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        # Decode base64 image if needed
        if 'image' in input_data and isinstance(input_data['image'], str):
            import base64
            img_data = base64.b64decode(input_data['image'])
            image = Image.open(io.BytesIO(img_data))
            image = image.resize((224, 224))
            image_array = np.array(image)
            # Add batch dimension if needed
            if len(image_array.shape) == 3:
                image_array = np.expand_dims(image_array, axis=0)
            return image_array
        return np.array(input_data['instances'])
    else:
        # Handle binary data (like raw image)
        image = Image.open(io.BytesIO(request_body))
        image = image.resize((224, 224))
        image_array = np.array(image)
        # Add batch dimension if needed
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        return image_array

def predict_fn(input_data, model):
    """
    Generate explanation for the input data
    """
    global explainer, class_names
    
    # Make prediction
    tmp = input_data.copy()
    preprocess_input(tmp)
    predictions = model(tmp).numpy()
    top_classes = np.argsort(predictions[0])[-4:][::-1]
    
    # Create explanation
    max_evals = 100  # Lower for faster but less accurate explanations
    batch_size = 50
    shap_values = explainer(input_data, max_evals=max_evals, batch_size=batch_size, 
                            outputs=shap.Explanation.argsort.flip[:4])
    
    # Extract the SHAP values
    result = {
        'predictions': [{
            'class': class_names[idx], 
            'score': float(predictions[0][idx])
        } for idx in top_classes],
        'shap_values': shap_values.values.tolist(),
        'base_values': shap_values.base_values.tolist() if hasattr(shap_values, 'base_values') else None,
    }
    
    return result

def output_fn(prediction, content_type):
    """
    Serialize the prediction output
    """
    if content_type == 'application/json':
        return json.dumps(prediction), 'application/json'
    else:
        return json.dumps(prediction), 'application/json'