"""Tests for feature extractor module."""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_extractor import FeatureExtractor


def test_feature_extractor_initialization():
    """Test feature extractor can be initialized."""
    extractor = FeatureExtractor(model_name="resnet50")
    assert extractor is not None
    assert extractor.model is not None


def test_extract_from_image():
    """Test feature extraction from a single image."""
    extractor = FeatureExtractor(model_name="resnet50")
    
    # Create a dummy image
    dummy_img = Image.new('RGB', (224, 224), color='red')
    
    # Extract features
    features = extractor.extract_from_image(dummy_img)
    
    # Check output
    assert isinstance(features, np.ndarray)
    assert len(features.shape) == 1  # Should be 1D array
    assert features.shape[0] > 0  # Should have features


def test_feature_normalization():
    """Test that features are normalized."""
    extractor = FeatureExtractor(model_name="resnet50")
    
    # Create a dummy image
    dummy_img = Image.new('RGB', (224, 224), color='blue')
    
    # Extract features
    features = extractor.extract_from_image(dummy_img)
    
    # Check normalization (L2 norm should be close to 1)
    norm = np.linalg.norm(features)
    assert abs(norm - 1.0) < 0.01


def test_get_feature_dim():
    """Test getting feature dimension."""
    extractor = FeatureExtractor(model_name="resnet50")
    dim = extractor.get_feature_dim()
    
    assert isinstance(dim, int)
    assert dim > 0
