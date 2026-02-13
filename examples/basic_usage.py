"""Basic usage example for the image search engine."""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ImageSearchEngine


def main():
    """Run a basic example of the image search engine."""
    
    # Initialize the search engine
    print("Initializing search engine...")
    engine = ImageSearchEngine(
        model_name="resnet50",
        index_type="cosine"  # Use cosine similarity
    )
    
    # Path to your image directory
    image_dir = Path("data/raw")
    
    # Check if directory exists and has images
    if not image_dir.exists():
        print(f"Error: Directory {image_dir} does not exist.")
        print("Please add images to the data/raw/ directory first.")
        return
    
    # Build the index
    print("\nBuilding index from images...")
    try:
        engine.build_index(
            image_dir=image_dir,
            batch_size=32,
            save_path="data/index"
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
    
    # Display results
    print("\nTop 5 similar images:")
    for result in results:
        print(f"  {result['rank']}. {Path(result['path']).name} "
              f"(score: {result['score']:.4f})")


if __name__ == "__main__":
    main()
