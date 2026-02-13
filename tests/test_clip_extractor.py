"""Tests for CLIP feature extractor module."""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clip_extractor import CLIPExtractor


def test_clip_extractor_initialization():
    """Test CLIP extractor can be initialized."""
    extractor = CLIPExtractor(model_name="openai/clip-vit-large-patch14")
    assert extractor is not None
    assert extractor.model is not None
    assert extractor.processor is not None


def test_extract_from_image():
    """Test feature extraction from a single image."""
    extractor = CLIPExtractor(model_name="openai/clip-vit-large-patch14")
    
    # Create a dummy image
    dummy_img = Image.new('RGB', (224, 224), color='red')
    
    # Extract features
    features = extractor.extract_from_image(dummy_img)
    
    # Check output
    assert isinstance(features, np.ndarray)
    assert len(features.shape) == 1  # Should be 1D array
    assert features.shape[0] == 768  # CLIP-Large dimension


def test_feature_normalization():
    """Test that features are normalized."""
    extractor = CLIPExtractor(model_name="openai/clip-vit-large-patch14")
    
    # Create a dummy image
    dummy_img = Image.new('RGB', (224, 224), color='blue')
    
    # Extract features
    features = extractor.extract_from_image(dummy_img)
    
    # Check normalization (L2 norm should be close to 1)
    norm = np.linalg.norm(features)
    assert abs(norm - 1.0) < 0.01


def test_get_feature_dim():
    """Test getting feature dimension."""
    extractor = CLIPExtractor(model_name="openai/clip-vit-large-patch14")
    dim = extractor.get_feature_dim()
    
    assert isinstance(dim, int)
    assert dim == 768  # CLIP ViT-Large dimension


def test_extract_from_batch():
    """Test batch feature extraction."""
    extractor = CLIPExtractor(model_name="openai/clip-vit-large-patch14")
    
    # Create multiple dummy images
    images = [
        Image.new('RGB', (224, 224), color='red'),
        Image.new('RGB', (224, 224), color='green'),
        Image.new('RGB', (224, 224), color='blue'),
    ]
    
    # Extract features in batch
    features = extractor.extract_from_batch(images)
    
    # Check output
    assert isinstance(features, np.ndarray)
    assert features.shape == (3, 768)  # 3 images, 768 dimensions
    
    # Check that each vector is normalized
    norms = np.linalg.norm(features, axis=1)
    assert np.all(np.abs(norms - 1.0) < 0.01)
