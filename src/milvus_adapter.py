"""Milvus database adapter for vector similarity search with metadata."""

from pymilvus import MilvusClient, DataType
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path


class MilvusAdapter:
    """Adapter for Milvus Lite database operations with image metadata."""
    
    def __init__(
        self,
        collection_name: str = "image_search",
        db_path: str = "./data/index/milvus_lite.db",
        embedding_dim: int = 768
    ):
        """
        Initialize Milvus adapter using MilvusClient API.
        
        Args:
            collection_name: Name of the Milvus collection
            db_path: Path to Milvus Lite database file
            embedding_dim: Dimension of embedding vectors (768 for CLIP-Large)
        """
        self.collection_name = collection_name
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        
        # Ensure db path parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to Milvus Lite using MilvusClient (newer API)
        print(f"Connecting to Milvus Lite at {db_path}...")
        self.client = MilvusClient(db_path)
        print("Connected to Milvus Lite")
        
    def create_collection(self, drop_existing: bool = False):
        """
        Create a new collection with schema for image embeddings and metadata.
        
        Args:
            drop_existing: If True, drop existing collection before creating new one
        """
        # Drop existing collection if requested
        if drop_existing and self.client.has_collection(self.collection_name):
            print(f"Dropping existing collection: {self.collection_name}")
            self.client.drop_collection(self.collection_name)
        
        # Check if collection already exists
        if self.client.has_collection(self.collection_name):
            print(f"Collection '{self.collection_name}' already exists")
            return
        
        # Define schema with MilvusClient API
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_field=False
        )
        
        # Add fields
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="image_path", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="filename", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="file_size", datatype=DataType.INT64)
        schema.add_field(field_name="created_time", datatype=DataType.INT64)
        schema.add_field(field_name="modified_time", datatype=DataType.INT64)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        
        # Create index parameters
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128}
        )
        
        # Create collection
        print(f"Creating collection: {self.collection_name}")
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
        print(f"Collection '{self.collection_name}' created successfully")
    
    def insert_images(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict]
    ) -> List[int]:
        """
        Insert image embeddings with metadata into collection.
        
        Args:
            embeddings: Array of embeddings (n_images, embedding_dim)
            metadata: List of metadata dicts with keys:
                      image_path, filename, file_size, created_time, modified_time
                      
        Returns:
            List of inserted IDs
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        # Prepare data for insertion (MilvusClient expects list of dicts)
        data = []
        for i, meta in enumerate(metadata):
            record = {
                "image_path": meta["image_path"],
                "filename": meta["filename"],
                "file_size": meta["file_size"],
                "created_time": meta["created_time"],
                "modified_time": meta["modified_time"],
                "embedding": embeddings[i].tolist(),
            }
            data.append(record)
        
        # Insert data
        result = self.client.insert(
            collection_name=self.collection_name,
            data=data
        )
        
        print(f"Inserted {len(metadata)} images into collection")
        return result.get("ids", [])
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Search for similar images using vector similarity.
        
        Args:
            query_embedding: Query embedding vector (embedding_dim,)
            top_k: Number of top results to return
            output_fields: Fields to include in results (None = all fields)
            
        Returns:
            List of result dictionaries with metadata and scores
        """
        # Default output fields
        if output_fields is None:
            output_fields = [
                "image_path",
                "filename",
                "file_size",
                "created_time",
                "modified_time"
            ]
        
        # Perform search using MilvusClient
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding.tolist()],
            limit=top_k,
            output_fields=output_fields,
        )
        
        # Format results
        formatted_results = []
        for i, hit in enumerate(results[0]):
            result = {
                "rank": i + 1,
                "id": hit.get("id"),
                "score": hit.get("distance"),  # MilvusClient uses "distance" instead of "score"
            }
            # Add all output fields
            for field in output_fields:
                result[field] = hit.get(field)
            formatted_results.append(result)
        
        return formatted_results
    
    def delete_collection(self):
        """Delete the collection."""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' deleted")
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        if not self.client.has_collection(self.collection_name):
            return {"exists": False}
        
        stats = self.client.get_collection_stats(self.collection_name)
        
        return {
            "exists": True,
            "name": self.collection_name,
            "num_entities": stats.get("row_count", 0),
        }
    
    def disconnect(self):
        """Disconnect from Milvus."""
        self.client.close()
        print("Disconnected from Milvus")
