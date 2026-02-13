"""Main image search engine implementation."""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Union, Optional
from tqdm import tqdm
import cv2

from .feature_extractor import FeatureExtractor
from .utils import load_image_paths, is_image_file


class ImageSearchEngine:
    """Image search engine using feature extraction and similarity search."""
    
    def __init__(
        self,
        model_name: str = "resnet50",
        index_type: str = "l2",
        device: str = None
    ):
        """
        Initialize the image search engine.
        
        Args:
            model_name: Pre-trained model name for feature extraction
            index_type: Type of similarity metric ('l2' or 'cosine')
            device: Device to run feature extraction on
        """
        self.feature_extractor = FeatureExtractor(model_name=model_name, device=device)
        self.index_type = index_type
        self.index = None
        self.image_paths = []
        self.feature_dim = self.feature_extractor.get_feature_dim()
        
    def build_index(
        self,
        image_dir: Union[str, Path],
        batch_size: int = 32,
        save_path: Optional[str] = None
    ):
        """
        Build search index from images in a directory.
        
        Args:
            image_dir: Directory containing images to index
            batch_size: Number of images to process at once
            save_path: Path to save the index (optional)
        """
        image_dir = Path(image_dir)
        
        # Find all image files
        print(f"Scanning directory: {image_dir}")
        self.image_paths = load_image_paths(image_dir)
        print(f"Found {len(self.image_paths)} images")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        # Extract features
        print("Extracting features...")
        features_list = []
        
        for i in tqdm(range(0, len(self.image_paths), batch_size)):
            batch_paths = self.image_paths[i:i + batch_size]
            try:
                batch_features = self.feature_extractor.extract_from_batch(batch_paths)
                features_list.append(batch_features)
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue
        
        # Combine all features
        features = np.vstack(features_list).astype('float32')
        
        # Build FAISS index
        print("Building search index...")
        if self.index_type == "cosine":
            # Normalize for cosine similarity
            faiss.normalize_L2(features)
            self.index = faiss.IndexFlatIP(self.feature_dim)
        else:  # l2
            self.index = faiss.IndexFlatL2(self.feature_dim)
        
        self.index.add(features)
        print(f"Index built with {self.index.ntotal} images")
        
        # Save index if path provided
        if save_path:
            self.save(save_path)
    
    def search(
        self,
        query_image: Union[str, Path],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for similar images.
        
        Args:
            query_image: Path to query image
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with 'path', 'score', and 'rank' keys
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Extract query features
        query_features = self.feature_extractor.extract_from_image(query_image)
        query_features = query_features.reshape(1, -1).astype('float32')
        
        # Normalize if using cosine similarity
        if self.index_type == "cosine":
            faiss.normalize_L2(query_features)
        
        # Search
        distances, indices = self.index.search(query_features, top_k)
        
        # Format results
        results = []
        for rank, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            results.append({
                'rank': rank + 1,
                'path': str(self.image_paths[idx]),
                'score': float(distance),
                'index': int(idx)
            })
        
        return results
    
    def save(self, save_path: str):
        """
        Save the index and metadata to disk.
        
        Args:
            save_path: Path to save the index
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = save_path / "index.faiss"
        faiss.write_index(self.index, str(index_file))
        
        # Save metadata
        metadata = {
            'image_paths': self.image_paths,
            'model_name': self.feature_extractor.model_name,
            'index_type': self.index_type,
            'feature_dim': self.feature_dim
        }
        metadata_file = save_path / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Index saved to {save_path}")
    
    def load(self, load_path: str):
        """
        Load a saved index from disk.
        
        Args:
            load_path: Path to load the index from
        """
        load_path = Path(load_path)
        
        # Load FAISS index
        index_file = load_path / "index.faiss"
        self.index = faiss.read_index(str(index_file))
        
        # Load metadata
        metadata_file = load_path / "metadata.pkl"
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        self.image_paths = metadata['image_paths']
        self.index_type = metadata['index_type']
        self.feature_dim = metadata['feature_dim']
        
        print(f"Index loaded from {load_path}")
        print(f"Loaded {len(self.image_paths)} images")
