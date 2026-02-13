"""Main image search engine implementation using CLIP and Milvus."""

import os
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Union, Optional
from tqdm import tqdm

from .clip_extractor import CLIPExtractor
from .milvus_adapter import MilvusAdapter
from .utils import load_image_paths


class ImageSearchEngine:
    """Image search engine using CLIP embeddings and Milvus vector database."""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        collection_name: str = "image_search",
        db_path: str = "./data/index/milvus_lite.db",
        device: str = None
    ):
        """
        Initialize the image search engine.
        
        Args:
            model_name: CLIP model name (default: openai/clip-vit-large-patch14)
            collection_name: Name for the Milvus collection
            db_path: Path for Milvus Lite database storage
            device: Device to run feature extraction on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.collection_name = collection_name
        self.db_path = db_path
        
        # Initialize CLIP feature extractor
        self.feature_extractor = CLIPExtractor(model_name=model_name, device=device)
        self.feature_dim = self.feature_extractor.get_feature_dim()
        
        # Initialize Milvus adapter
        self.milvus = MilvusAdapter(
            collection_name=collection_name,
            db_path=db_path,
            embedding_dim=self.feature_dim
        )
        
        self.image_paths = []
        
    def build_index(
        self,
        image_dir: Union[str, Path],
        batch_size: int = 32,
        save_path: Optional[str] = None,
        drop_existing: bool = False
    ):
        """
        Build search index from images in a directory.
        
        Args:
            image_dir: Directory containing images to index
            batch_size: Number of images to process at once
            save_path: Path to save metadata (Milvus persists automatically)
            drop_existing: If True, drop existing collection before building
        """
        image_dir = Path(image_dir)
        
        # Find all image files
        print(f"Scanning directory: {image_dir}")
        self.image_paths = load_image_paths(image_dir)
        print(f"Found {len(self.image_paths)} images")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        # Create Milvus collection
        self.milvus.create_collection(drop_existing=drop_existing)
        
        # Extract features and collect metadata
        print("Extracting CLIP features and collecting metadata...")
        
        for i in tqdm(range(0, len(self.image_paths), batch_size)):
            batch_paths = self.image_paths[i:i + batch_size]
            
            try:
                # Extract features
                batch_features = self.feature_extractor.extract_from_batch(batch_paths)
                
                # Collect metadata for each image
                batch_metadata = []
                for img_path in batch_paths:
                    stat = os.stat(img_path)
                    metadata = {
                        'image_path': str(img_path.absolute()),
                        'filename': img_path.name,
                        'file_size': stat.st_size,
                        'created_time': int(stat.st_ctime),
                        'modified_time': int(stat.st_mtime),
                    }
                    batch_metadata.append(metadata)
                
                # Insert into Milvus
                self.milvus.insert_images(batch_features, batch_metadata)
                
            except Exception as e:
                print(f"Error processing batch starting at {i}: {e}")
                continue
        
        print(f"Index built successfully with {len(self.image_paths)} images")
        
        # Save metadata if path provided
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
            List of dictionaries with metadata and similarity scores
            Keys: rank, id, score, image_path, filename, file_size, 
                  created_time, modified_time
        """
        # Extract query features
        query_features = self.feature_extractor.extract_from_image(query_image)
        
        # Search in Milvus
        results = self.milvus.search(
            query_embedding=query_features,
            top_k=top_k
        )
        
        # Rename image_path to path for backward compatibility
        for result in results:
            result['path'] = result.pop('image_path')
        
        return results
    
    def save(self, save_path: str):
        """
        Save metadata to disk (Milvus data persists automatically).
        
        Args:
            save_path: Path to save metadata
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration metadata
        metadata = {
            'model_name': self.model_name,
            'collection_name': self.collection_name,
            'db_path': self.db_path,
            'feature_dim': self.feature_dim,
            'num_images': len(self.image_paths)
        }
        
        metadata_file = save_path / "engine_metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Metadata saved to {save_path}")
        print(f"Milvus data persisted automatically to {self.db_path}")
    
    def load(self, load_path: str):
        """
        Load metadata from disk (Milvus connects to existing database).
        
        Args:
            load_path: Path to load metadata from
        """
        load_path = Path(load_path)
        
        # Load configuration metadata
        metadata_file = load_path / "engine_metadata.pkl"
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        self.model_name = metadata['model_name']
        self.collection_name = metadata['collection_name']
        self.db_path = metadata['db_path']
        self.feature_dim = metadata['feature_dim']
        
        # Reconnect to Milvus collection
        self.milvus = MilvusAdapter(
            collection_name=self.collection_name,
            db_path=self.db_path,
            embedding_dim=self.feature_dim
        )
        self.milvus.create_collection(drop_existing=False)
        
        stats = self.milvus.get_collection_stats()
        print(f"Engine loaded from {load_path}")
        print(f"Connected to collection with {stats.get('num_entities', 0)} images")
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the search engine.
        
        Returns:
            Dictionary with engine statistics
        """
        stats = self.milvus.get_collection_stats()
        stats['model_name'] = self.model_name
        stats['feature_dim'] = self.feature_dim
        return stats
