"""CLIP-based feature extraction module for images."""

import torch
from transformers import CLIPConfig, CLIPProcessor, CLIPModel
from transformers.utils import cached_file, SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from PIL import Image
import numpy as np
from typing import Union, List
from pathlib import Path


def _load_clip_state_dict(model_name: str) -> dict:
    """Load checkpoint state dict and drop position_ids to avoid UNEXPECTED keys report."""
    resolved = cached_file(
        model_name, SAFE_WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False
    )
    if resolved is None:
        resolved = cached_file(model_name, WEIGHTS_NAME)
    if resolved is None:
        raise OSError(f"No weights file found for {model_name}")

    if resolved.endswith(".safetensors"):
        from safetensors.torch import load_file as safe_load_file
        state_dict = safe_load_file(resolved)
    else:
        state_dict = torch.load(resolved, map_location="cpu", weights_only=True)

    # Drop position_ids from checkpoint; model creates them as non-persistent buffers
    state_dict = {k: v for k, v in state_dict.items() if "position_ids" not in k}
    return state_dict


class CLIPExtractor:
    """Extract visual features from images using CLIP (ViT-Large-Patch14)."""
    
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: str = None):
        """
        Initialize the CLIP feature extractor.
        
        Args:
            model_name: CLIP model name from Hugging Face (default: openai/clip-vit-large-patch14)
            device: Device to run the model on ('cuda' or 'cpu'). Auto-detect if None.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        print(f"Loading CLIP model: {model_name}...")
        config = CLIPConfig.from_pretrained(model_name)
        self.model = CLIPModel(config)
        state_dict = _load_clip_state_dict(model_name)
        self.model.load_state_dict(state_dict, strict=False)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"CLIP model loaded on {self.device}")
        
    def extract_from_image(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        Extract features from a single image using CLIP.
        
        Args:
            image: Path to image file or PIL Image object
            
        Returns:
            Feature vector as numpy array (768-dimensional for ViT-Large)
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a path or PIL Image object")
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        
        # Get the tensor from the output (CLIP returns BaseModelOutputWithPooling)
        if hasattr(outputs, 'pooler_output'):
            image_features = outputs.pooler_output
        else:
            image_features = outputs
        
        # Convert to numpy and normalize
        features = image_features.cpu().numpy().flatten()
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    
    def extract_from_text(self, text: str) -> np.ndarray:
        """
        Extract features from a text string using CLIP text encoder.
        
        Args:
            text: Input text to encode
            
        Returns:
            Feature vector as numpy array (768-dimensional for ViT-Large)
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        if hasattr(outputs, 'pooler_output'):
            text_features = outputs.pooler_output
        else:
            text_features = outputs
        features = text_features.cpu().numpy().flatten()
        features = features / (np.linalg.norm(features) + 1e-8)
        return features
    
    def extract_from_batch(self, images: List[Union[str, Path, Image.Image]]) -> np.ndarray:
        """
        Extract features from multiple images in batch.
        
        Args:
            images: List of image paths or PIL Image objects
            
        Returns:
            Feature matrix as numpy array (n_images, 768)
        """
        # Load all images
        pil_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_images.append(Image.open(img).convert('RGB'))
            elif isinstance(img, Image.Image):
                pil_images.append(img)
            else:
                raise ValueError("Image must be a path or PIL Image object")
        
        # Process batch
        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        
        # Get the tensor from the output (CLIP returns BaseModelOutputWithPooling)
        if hasattr(outputs, 'pooler_output'):
            image_features = outputs.pooler_output
        else:
            image_features = outputs
        
        # Convert to numpy and normalize
        features = image_features.cpu().numpy()
        # Normalize each feature vector
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (norms + 1e-8)
        
        return features
    
    def get_feature_dim(self) -> int:
        """Get the dimensionality of the feature vectors."""
        # CLIP ViT-Large produces 768-dimensional embeddings
        return 768
