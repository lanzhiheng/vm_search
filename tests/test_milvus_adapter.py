"""Tests for Milvus adapter module."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.milvus_adapter import MilvusAdapter


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_milvus.db"
    yield str(db_path)
    # Cleanup
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def milvus_adapter(temp_db_path):
    """Create a Milvus adapter for testing."""
    adapter = MilvusAdapter(
        collection_name="test_collection",
        db_path=temp_db_path,
        embedding_dim=768
    )
    yield adapter
    # Cleanup
    adapter.delete_collection()
    adapter.disconnect()


def test_milvus_adapter_initialization(temp_db_path):
    """Test Milvus adapter can be initialized."""
    adapter = MilvusAdapter(
        collection_name="test_init",
        db_path=temp_db_path,
        embedding_dim=768
    )
    assert adapter is not None
    assert adapter.collection_name == "test_init"
    assert adapter.embedding_dim == 768
    adapter.disconnect()


def test_create_collection(milvus_adapter):
    """Test collection creation."""
    milvus_adapter.create_collection(drop_existing=True)
    assert milvus_adapter.collection is not None
    
    stats = milvus_adapter.get_collection_stats()
    assert stats['exists'] is True
    assert stats['name'] == "test_collection"


def test_insert_and_search(milvus_adapter):
    """Test data insertion and search."""
    # Create collection
    milvus_adapter.create_collection(drop_existing=True)
    
    # Create test data
    embeddings = np.random.rand(5, 768).astype('float32')
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    metadata = [
        {
            'image_path': f'/path/to/image{i}.jpg',
            'filename': f'image{i}.jpg',
            'file_size': 1000 * (i + 1),
            'created_time': 1634567890 + i,
            'modified_time': 1634567890 + i,
        }
        for i in range(5)
    ]
    
    # Insert data
    ids = milvus_adapter.insert_images(embeddings, metadata)
    assert len(ids) == 5
    
    # Search with first embedding
    query = embeddings[0]
    results = milvus_adapter.search(query, top_k=3)
    
    assert len(results) == 3
    assert results[0]['rank'] == 1
    assert 'image_path' in results[0]
    assert 'filename' in results[0]
    assert 'file_size' in results[0]
    assert 'score' in results[0]
    
    # First result should be the query itself (highest similarity)
    assert results[0]['filename'] == 'image0.jpg'


def test_collection_stats(milvus_adapter):
    """Test getting collection statistics."""
    milvus_adapter.create_collection(drop_existing=True)
    
    # Insert test data
    embeddings = np.random.rand(10, 768).astype('float32')
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    metadata = [
        {
            'image_path': f'/path/to/image{i}.jpg',
            'filename': f'image{i}.jpg',
            'file_size': 1000,
            'created_time': 1634567890,
            'modified_time': 1634567890,
        }
        for i in range(10)
    ]
    
    milvus_adapter.insert_images(embeddings, metadata)
    
    # Get stats
    stats = milvus_adapter.get_collection_stats()
    assert stats['exists'] is True
    assert stats['num_entities'] == 10


def test_delete_collection(milvus_adapter):
    """Test collection deletion."""
    milvus_adapter.create_collection(drop_existing=True)
    assert milvus_adapter.collection is not None
    
    milvus_adapter.delete_collection()
    assert milvus_adapter.collection is None
    
    stats = milvus_adapter.get_collection_stats()
    assert stats['exists'] is False
