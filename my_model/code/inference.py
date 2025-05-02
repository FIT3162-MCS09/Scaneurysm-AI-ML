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
        print(f"Loading model from directory: {model_dir}")
        
        # Initialize model
        model = CNNModel()
        
        # Load checkpoint
        model_path = os.path.join(model_dir, 'model.pth')
        print(f"Loading checkpoint from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        # Load using torch.load
        checkpoint = torch.load(model_path, map_location=device)
        print(f"Checkpoint loaded. Keys available: {checkpoint.keys()}")
        
        # Get state dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            raise ValueError("Checkpoint is not a dictionary")
            
        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        print("State dict loaded successfully")
        
        # Set to evaluation mode
        model.eval()
        model = model.to(device)
        print(f"Model loaded successfully and moved to {device}")
        
        return model
        
    except Exception as e:
        print(f"Error in model_fn: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error details: {e.__dict__}")
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
    """Make prediction with enhanced error handling and SHAP support."""
    print(f"Making prediction at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    try:
        device = next(model.parameters()).device
        input_data = input_data.to(device)
        
        with torch.no_grad():
            output = model(input_data)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            
            # Calculate SHAP values
            import shap
            background = torch.zeros((1, 3, 224, 224)).to(device)
            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(input_data)
            
            return {
                'probabilities': probabilities,
                'shap_values': shap_values
            }
            
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        raise

def output_fn(prediction, accept="application/json"):
    """Format prediction output with SHAP values."""
    try:
        if isinstance(prediction, dict):
            probabilities = prediction['probabilities'].cpu().numpy().tolist()[0]
            shap_data = [sv.tolist() for sv in prediction['shap_values']]
        else:
            probabilities = prediction.cpu().numpy().tolist()[0]
            shap_data = None
        
        response = {
            "prediction": "Aneurysm" if probabilities[1] > probabilities[0] else "Non-aneurysm",
            "confidence": float(max(probabilities)),
            "probabilities": {
                "non_aneurysm": float(probabilities[0]),
                "aneurysm": float(probabilities[1])
            },
            "metadata": {
                "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                "user": "krooldonutz",
                "pytorch_version": torch.__version__
            }
        }
        
        if shap_data:
            response["shap_values"] = shap_data
            
        return json.dumps(response)
        
    except Exception as e:
        print(f"❌ Error formatting output: {str(e)}")
        raise
