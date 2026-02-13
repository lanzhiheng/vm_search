"""Script to build Milvus index from images in data/raw directory."""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import ImageSearchEngine
from src.utils import load_image_paths


def main():
    """Build the image search index."""
    
    print("="*60)
    print("VM Search - Building Image Index with CLIP and Milvus")
    print("="*60)
    
    # Initialize the search engine
    print("\n[1/4] Initializing CLIP model and Milvus connection...")
    engine = ImageSearchEngine(
        model_name="openai/clip-vit-large-patch14",
        collection_name="image_search",
        db_path="./data/index/milvus_lite.db"
    )
    print("✓ Engine initialized")
    
    # Check for images
    image_dir = Path("data/raw")
    print(f"\n[2/4] Scanning for images in {image_dir}...")
    
    if not image_dir.exists():
        print(f"Error: Directory {image_dir} does not exist!")
        return
    
    image_paths = load_image_paths(image_dir)
    if len(image_paths) == 0:
        print(f"Error: No images found in {image_dir}")
        print("Please add images to the data/raw/ directory first.")
        return
    
    print(f"✓ Found {len(image_paths)} images")
    
    # Build the index
    print(f"\n[3/4] Processing images and building Milvus collection...")
    print("This will:")
    print("  - Extract CLIP embeddings (768-dim)")
    print("  - Collect metadata (filename, size, timestamps)")
    print("  - Insert into Milvus collection")
    print("\nThis may take a few minutes depending on the number of images...")
    
    try:
        engine.build_index(
            image_dir=image_dir,
            batch_size=16,  # Process 16 images at a time
            save_path="data/index",
            drop_existing=True  # Start fresh
        )
        print("✓ Index built successfully!")
        
    except Exception as e:
        print(f"\n✗ Error building index: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Show statistics
    print(f"\n[4/4] Collection Statistics:")
    stats = engine.get_stats()
    print(f"  Collection name: {stats['name']}")
    print(f"  Total images indexed: {stats['num_entities']}")
    print(f"  Model: {stats['model_name']}")
    print(f"  Embedding dimension: {stats['feature_dim']}")
    print(f"  Database location: ./data/index/milvus_lite.db")
    
    # Test search with first image
    print(f"\n[Test] Performing a test search with the first image...")
    try:
        query_image = image_paths[0]
        results = engine.search(query_image, top_k=5)
        
        print(f"\nQuery: {query_image.name}")
        print(f"Top 5 similar images:")
        for result in results:
            print(f"  {result['rank']}. {result['filename']} "
                  f"(score: {result['score']:.4f})")
    except Exception as e:
        print(f"Test search failed: {e}")
    
    print("\n" + "="*60)
    print("Index building complete! You can now run:")
    print("  python examples/basic_usage.py")
    print("  python examples/search_with_visualization.py")
    print("="*60)


if __name__ == "__main__":
    main()
