"""Utility functions for the image search engine."""

import os
from pathlib import Path
from typing import List
import cv2
import numpy as np
from PIL import Image


# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}


def is_image_file(filepath: Path) -> bool:
    """
    Check if a file is an image based on its extension.
    
    Args:
        filepath: Path to the file
        
    Returns:
        True if file is an image
    """
    return filepath.suffix.lower() in IMAGE_EXTENSIONS


def load_image_paths(directory: Path, recursive: bool = True) -> List[Path]:
    """
    Load all image file paths from a directory.
    
    Args:
        directory: Directory to search for images
        recursive: Whether to search subdirectories
        
    Returns:
        List of image file paths
    """
    directory = Path(directory)
    image_paths = []
    
    if recursive:
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(directory.rglob(f"*{ext}"))
            image_paths.extend(directory.rglob(f"*{ext.upper()}"))
    else:
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(directory.glob(f"*{ext}"))
            image_paths.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(image_paths)


def load_image(image_path: Path, target_size: tuple = None) -> np.ndarray:
    """
    Load an image from disk.
    
    Args:
        image_path: Path to the image
        target_size: Target size (width, height) to resize to
        
    Returns:
        Image as numpy array
    """
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if target_size:
        img = cv2.resize(img, target_size)
    
    return img


def save_image(image: np.ndarray, save_path: Path):
    """
    Save an image to disk.
    
    Args:
        image: Image as numpy array (RGB)
        save_path: Path to save the image
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), image)


def create_grid(images: List[np.ndarray], grid_size: tuple = None) -> np.ndarray:
    """
    Create a grid of images for visualization.
    
    Args:
        images: List of images as numpy arrays
        grid_size: (rows, cols) for the grid. Auto-calculated if None.
        
    Returns:
        Grid image as numpy array
    """
    if not images:
        return np.array([])
    
    n_images = len(images)
    
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_size
    
    # Get image dimensions (assuming all same size)
    h, w = images[0].shape[:2]
    channels = images[0].shape[2] if len(images[0].shape) == 3 else 1
    
    # Create grid
    if channels == 1:
        grid = np.zeros((rows * h, cols * w), dtype=np.uint8)
    else:
        grid = np.zeros((rows * h, cols * w, channels), dtype=np.uint8)
    
    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break
        
        i = idx // cols
        j = idx % cols
        grid[i*h:(i+1)*h, j*w:(j+1)*w] = img
    
    return grid
