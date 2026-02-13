# VM Search - Image Searching Engine

A Python-based image searching engine that uses deep learning and similarity search to find visually similar images.

## Features

- Image feature extraction using pre-trained models
- Efficient similarity search with FAISS
- Support for multiple image formats
- Easy-to-use API for indexing and searching

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
│   ├── feature_extractor.py  # Feature extraction logic
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
from src.search_engine import ImageSearchEngine

# Initialize the search engine
engine = ImageSearchEngine()

# Index images from a directory
engine.build_index('data/raw/')

# Search for similar images
results = engine.search('query_image.jpg', top_k=5)

# Display results
for result in results:
    print(f"Image: {result['path']}, Score: {result['score']}")
```

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
