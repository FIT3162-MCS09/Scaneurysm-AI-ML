import os
import json
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
import requests
from PIL import Image
from datetime import datetime

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for param in self.vgg16.features.parameters():
            param.requires_grad = False
        in_feats = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(in_feats, 2)
    
    def forward(self, x):
        x = self.vgg16(x)
        return x

def model_fn(model_dir):
    """Load the PyTorch model with enhanced security features"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNNModel()
        
        # Load model using torch.load directly
        checkpoint = torch.load(
            os.path.join(model_dir, 'model.pth'),
            map_location=device
        )
        
        # Get state dict
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict',
                        checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint
            
        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model.to(device)
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def input_fn(request_body, content_type="application/json"):
    """Transform input data with enhanced validation."""
    print(f"Processing input at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"User: krooldonutz")
    
    try:
        if content_type != "application/json":
            raise ValueError(f"Unsupported content type: {content_type}")
            
        # Parse input
        try:
            input_data = json.loads(request_body)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
            
        url = input_data.get("url")
        if not url:
            raise ValueError("No URL provided in the request")
            
        # Download image with timeout and retry
        try:
            from requests.adapters import HTTPAdapter
            from urllib3.util import Retry
            
            session = requests.Session()
            retries = Retry(total=3, backoff_factor=1,
                          status_forcelist=[408, 429, 500, 502, 503, 504])
            session.mount('http://', HTTPAdapter(max_retries=retries))
            session.mount('https://', HTTPAdapter(max_retries=retries))
            
            response = session.get(url, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to download image: {str(e)}")
            
        # Process image
        try:
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
                
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Validate image dimensions
            if image.shape[0] < 10 or image.shape[1] < 10:
                raise ValueError("Image dimensions too small")
                
            # Normalize
            image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-7)
            image = Image.fromarray((image * 255).astype(np.uint8))
            
            # Transform
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            image_tensor = transform(image)
            return image_tensor.unsqueeze(0)
            
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")
            
    except Exception as e:
        print(f"❌ Error in input processing: {str(e)}")
        raise

def predict_fn(input_data, model):
    """Make prediction with enhanced error handling."""
    print(f"Making prediction at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"User: krooldonutz")
    
    try:
        # Move to same device as model
        device = next(model.parameters()).device
        input_data = input_data.to(device)
        
        # Make prediction
        with torch.no_grad():
            try:
                output = model(input_data)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                
                # Validate output
                if torch.isnan(probabilities).any():
                    raise ValueError("Model produced NaN values")
                if not torch.allclose(probabilities.sum(dim=1), torch.ones(probabilities.size(0)).to(device)):
                    raise ValueError("Invalid probability distribution")
                    
                return probabilities
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                raise
                
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        raise

def output_fn(prediction, accept="application/json"):
    """Format prediction output with validation."""
    print(f"Formatting output at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"User: krooldonutz")
    
    try:
        if accept != "application/json":
            raise ValueError(f"Unsupported accept type: {accept}")
            
        # Validate prediction
        if not isinstance(prediction, torch.Tensor):
            raise ValueError("Invalid prediction type")
            
        predictions = prediction.cpu().numpy().tolist()[0]
        if len(predictions) != 2:
            raise ValueError("Invalid prediction shape")
            
        # Create response
        response = {
            "prediction": "Aneurysm" if predictions[1] > predictions[0] else "Non-aneurysm",
            "confidence": float(max(predictions)),
            "probabilities": {
                "non_aneurysm": float(predictions[0]),
                "aneurysm": float(predictions[1])
            },
            "metadata": {
                "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                "user": "krooldonutz",
                "pytorch_version": torch.__version__
            }
        }
        
        # Validate response
        if not all(isinstance(v, float) for v in [
            response["confidence"],
            response["probabilities"]["non_aneurysm"],
            response["probabilities"]["aneurysm"]
        ]):
            raise ValueError("Invalid probability values")
            
        return json.dumps(response)
        
    except Exception as e:
        print(f"❌ Error formatting output: {str(e)}")
        raise
