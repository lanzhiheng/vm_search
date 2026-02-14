"""Basic usage example for the image search engine with CLIP and Milvus."""

from pathlib import Path
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ImageSearchEngine


def main():
    """Run a basic example of the image search engine."""
    
    # Initialize the search engine with CLIP and Milvus
    print("Initializing search engine with CLIP and Milvus...")
    engine = ImageSearchEngine(
        model_name="openai/clip-vit-large-patch14",
        collection_name="image_search",
        db_path="./data/index/milvus_lite.db"
    )
    
    # Path to your image directory
    image_dir = Path("data/raw")
    
    # Check if directory exists and has images
    if not image_dir.exists():
        print(f"Error: Directory {image_dir} does not exist.")
        print("Please add images to the data/raw/ directory first.")
        return
    
    # Build the index
    print("\nBuilding index from images with CLIP embeddings...")
    try:
        engine.build_index(
            image_dir=image_dir,
            batch_size=16,  # Smaller batch for CLIP
            save_path="data/index",
            drop_existing=True  # Set to True to rebuild from scratch
        )
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Get the first image as a query example
    from src.utils import load_image_paths
    image_paths = load_image_paths(image_dir)
    
    if len(image_paths) == 0:
        print("No images found for testing.")
        return
    
    query_image = image_paths[0]
    print(f"\nSearching for images similar to: {query_image.name}")
    
    # Search for similar images
    results = engine.search(query_image, top_k=5)
    
    # Display results with metadata
    print("\nTop 5 similar images:")
    for result in results:
        # Convert timestamps to readable format
        mod_time = datetime.fromtimestamp(result['modified_time']).strftime('%Y-%m-%d %H:%M')
        file_size_mb = result['file_size'] / (1024 * 1024)
        
        print(f"  {result['rank']}. {result['filename']}")
        print(f"      Score: {result['score']:.4f}  Similarity: {result['similarity']:.1f}%")
        print(f"      Size: {file_size_mb:.2f} MB")
        print(f"      Modified: {mod_time}")
        print()


if __name__ == "__main__":
    main()
