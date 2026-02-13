# Quick Start Guide

## Step 1: Create the Conda Environment

Run this command in your terminal:

```bash
conda env create -f environment.yml
```

This will create a new conda environment named `vm_search` with all required dependencies.

## Step 2: Activate the Environment

```bash
conda activate vm_search
```

## Step 3: Add Your Images

Place the images you want to search through in the `data/raw/` directory:

```bash
# Example: copy your images
cp /path/to/your/images/* data/raw/
```

## Step 4: Run the Basic Example

```bash
python examples/basic_usage.py
```

This will:
1. Load the CLIP model (openai/clip-vit-large-patch14)
2. Extract 768-dimensional embeddings from all images in `data/raw/`
3. Store embeddings and metadata in Milvus Lite database
4. Perform a test search and display results with metadata

## Step 5: Try the Visualization Example

```bash
python examples/search_with_visualization.py
```

This will show you a visual comparison of the query image and the top similar images found.

## Using in Your Own Code

```python
from src import ImageSearchEngine

# Initialize with CLIP and Milvus
engine = ImageSearchEngine(
    model_name="openai/clip-vit-large-patch14",
    collection_name="my_images",
    db_path="./data/index/milvus_lite.db"
)

# Build index (first time)
engine.build_index('data/raw/', batch_size=16, save_path='data/index')

# Or load existing engine
# engine.load('data/index')

# Search for similar images
results = engine.search('path/to/query/image.jpg', top_k=10)

# Process results with rich metadata
for result in results:
    print(f"Rank {result['rank']}: {result['filename']}")
    print(f"  Score: {result['score']:.4f}")
    print(f"  Size: {result['file_size'] / (1024*1024):.2f} MB")
    print(f"  Modified: {result['modified_time']}")
```

## Advanced Configuration

### Use Different CLIP Model

```python
# Use base model (faster, less accurate)
engine = ImageSearchEngine(model_name="openai/clip-vit-base-patch32")

# Use large model (default, slower, more accurate)
engine = ImageSearchEngine(model_name="openai/clip-vit-large-patch14")
```

### Customize Database Settings

```python
# Custom collection name and database path
engine = ImageSearchEngine(
    collection_name="product_images",
    db_path="./my_custom_db/milvus.db"
)
```

### Rebuild Index

```python
# Drop existing collection and rebuild from scratch
engine.build_index('data/raw/', drop_existing=True)
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

## Troubleshooting

### CUDA/GPU Issues

If you have GPU but it's not detected:

```bash
# Install PyTorch with CUDA support
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Memory Issues

If you run out of memory with large image collections:

- Reduce batch size: `engine.build_index(image_dir, batch_size=8)`
- Use smaller CLIP model: `ImageSearchEngine(model_name="openai/clip-vit-base-patch32")`
- Process images incrementally (build index in multiple runs)

### Milvus Database

Milvus Lite runs embedded (no separate server needed) and automatically persists data to:
- Default location: `./data/index/milvus_lite.db`
- Data is preserved between runs
- To reset: delete the database file or use `drop_existing=True`

### No Images Found

Make sure your images are in supported formats (jpg, png, bmp, gif, tiff, webp) and located in the `data/raw/` directory.
