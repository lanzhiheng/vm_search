"""Feature extraction module for images using pre-trained models."""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
from typing import Union, List
from pathlib import Path


class FeatureExtractor:
    """Extract visual features from images using pre-trained deep learning models."""
    
    def __init__(self, model_name: str = "resnet50", device: str = None):
        """
        Initialize the feature extractor.
        
        Args:
            model_name: Name of the pre-trained model to use (default: resnet50)
            device: Device to run the model on ('cuda' or 'cpu'). Auto-detect if None.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # Load pre-trained model
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get model-specific transforms
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
        
        print(f"Loaded model: {model_name} on {self.device}")
        
    def extract_from_image(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        Extract features from a single image.
        
        Args:
            image: Path to image file or PIL Image object
            
        Returns:
            Feature vector as numpy array
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a path or PIL Image object")
        
        # Transform and extract features
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(img_tensor)
        
        # Convert to numpy and normalize
        features = features.cpu().numpy().flatten()
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    
    def extract_from_batch(self, images: List[Union[str, Path, Image.Image]]) -> np.ndarray:
        """
        Extract features from multiple images in batch.
        
        Args:
            images: List of image paths or PIL Image objects
            
        Returns:
            Feature matrix as numpy array (n_images, feature_dim)
        """
        features_list = []
        
        for image in images:
            features = self.extract_from_image(image)
            features_list.append(features)
        
        return np.vstack(features_list)
    
    def get_feature_dim(self) -> int:
        """Get the dimensionality of the feature vectors."""
        # Create a dummy image to get feature dimension
        dummy_img = Image.new('RGB', (224, 224))
        features = self.extract_from_image(dummy_img)
        return features.shape[0]
