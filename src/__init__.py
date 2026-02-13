"""VM Search - Image Searching Engine"""

__version__ = "0.2.0"

from .clip_extractor import CLIPExtractor
from .milvus_adapter import MilvusAdapter
from .search_engine import ImageSearchEngine

# Keep old FeatureExtractor for backward compatibility
try:
    from .feature_extractor import FeatureExtractor
    __all__ = ["CLIPExtractor", "MilvusAdapter", "ImageSearchEngine", "FeatureExtractor"]
except ImportError:
    __all__ = ["CLIPExtractor", "MilvusAdapter", "ImageSearchEngine"]
