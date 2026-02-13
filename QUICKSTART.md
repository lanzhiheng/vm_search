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
1. Build an index of all images in `data/raw/`
2. Save the index to `data/index/`
3. Perform a test search using one of your images

## Step 5: Try the Visualization Example

```bash
python examples/search_with_visualization.py
```

This will show you a visual comparison of the query image and the top similar images found.

## Using in Your Own Code

```python
from src import ImageSearchEngine

# Initialize
engine = ImageSearchEngine()

# Build index (first time)
engine.build_index('data/raw/', save_path='data/index')

# Or load existing index
# engine.load('data/index')

# Search for similar images
results = engine.search('path/to/query/image.jpg', top_k=10)

# Process results
for result in results:
    print(f"{result['rank']}: {result['path']} (score: {result['score']:.4f})")
```

## Advanced Configuration

### Use Different Model

```python
# Try different pre-trained models for better accuracy
engine = ImageSearchEngine(model_name="efficientnet_b0")
# or "vit_base_patch16_224", "convnext_tiny", etc.
```

### Use Different Similarity Metric

```python
# L2 distance (default)
engine = ImageSearchEngine(index_type="l2")

# Cosine similarity (recommended)
engine = ImageSearchEngine(index_type="cosine")
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
- Use a smaller model: `ImageSearchEngine(model_name="resnet18")`

### No Images Found

Make sure your images are in supported formats (jpg, png, bmp, gif, tiff, webp) and located in the `data/raw/` directory.
