# VM Search - Image Searching Engine

A Python-based image searching engine powered by CLIP and Milvus for semantic image similarity search.

## Features

- **CLIP-based feature extraction** using OpenAI's ViT-Large-Patch14 model for superior semantic understanding
- **Milvus Lite vector database** for efficient similarity search (no separate server required)
- **Rich metadata storage** including file paths, sizes, and timestamps
- **Support for multiple image formats** (JPG, PNG, BMP, GIF, TIFF, WebP)
- **Easy-to-use API** for indexing and searching
- **Scalable architecture** ready for production use

## Setup

### Create Conda Environment

```bash
# Create the environment from the yml file
conda env create -f environment.yml

# Activate the environment
conda activate vm_search
```

Alternatively, if you prefer pip:

```bash
# Create a new conda environment
conda create -n vm_search python=3.11 -y
conda activate vm_search

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
vm_search/
├── data/                  # Directory for image datasets
│   ├── raw/              # Raw images to index
│   └── index/            # Generated index files
├── src/                  # Source code
│   ├── __init__.py
│   ├── clip_extractor.py     # CLIP feature extraction
│   ├── milvus_adapter.py     # Milvus database operations
│   ├── search_engine.py      # Search engine implementation
│   └── utils.py              # Utility functions
├── notebooks/            # Jupyter notebooks for experiments
├── tests/                # Unit tests
├── examples/             # Example scripts
├── environment.yml       # Conda environment file
├── requirements.txt      # Pip requirements
└── README.md            # This file
```

## Quick Start

```python
from src import ImageSearchEngine

# Initialize the search engine with CLIP and Milvus
engine = ImageSearchEngine(
    model_name="openai/clip-vit-large-patch14",
    collection_name="image_search",
    db_path="./data/index/milvus_lite.db"
)

# Build index from images in a directory
engine.build_index('data/raw/', batch_size=16)

# Search for similar images
results = engine.search('query_image.jpg', top_k=5)

# Display results with metadata
for result in results:
    print(f"Rank {result['rank']}: {result['filename']}")
    print(f"  Score: {result['score']:.4f}")
    print(f"  Size: {result['file_size'] / (1024*1024):.2f} MB")
    print(f"  Path: {result['path']}")
```

## Key Technologies

- **CLIP (Contrastive Language-Image Pre-training)**: State-of-the-art vision model from OpenAI
  - Model: `openai/clip-vit-large-patch14`
  - 768-dimensional embeddings
  - Superior semantic understanding compared to traditional CNN models

- **Milvus Lite**: Embedded vector database
  - No separate server installation required
  - Automatic data persistence
  - Production-ready performance
  - Cosine similarity search

## Usage Examples

See the `examples/` directory for detailed usage examples.

## Development

Run tests:
```bash
pytest tests/
```

Format code:
```bash
black src/ tests/
```

Lint code:
```bash
flake8 src/ tests/
```

## License

MIT License
