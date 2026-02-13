"""Example with visualization of search results."""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ImageSearchEngine
from src.utils import load_image_paths


def visualize_results(query_path: str, results: list, save_path: str = None):
    """
    Visualize query image and search results.
    
    Args:
        query_path: Path to query image
        results: Search results from engine
        save_path: Optional path to save visualization
    """
    n_results = len(results)
    fig, axes = plt.subplots(1, n_results + 1, figsize=(3 * (n_results + 1), 3))
    
    # Show query image
    query_img = Image.open(query_path)
    axes[0].imshow(query_img)
    axes[0].set_title("Query", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Show results with metadata
    for idx, result in enumerate(results, 1):
        img = Image.open(result['path'])
        axes[idx].imshow(img)
        # Show rank, score, and file size
        size_mb = result.get('file_size', 0) / (1024 * 1024)
        axes[idx].set_title(
            f"#{result['rank']}\nScore: {result['score']:.3f}\n{size_mb:.1f}MB",
            fontsize=9
        )
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def main():
    """Run example with visualization using CLIP and Milvus."""
    
    # Initialize search engine with CLIP and Milvus
    print("Initializing search engine with CLIP and Milvus...")
    engine = ImageSearchEngine(
        model_name="openai/clip-vit-large-patch14",
        collection_name="image_search"
    )
    
    # Build or load index
    index_path = Path("data/index")
    
    if index_path.exists():
        print("Loading existing index...")
        engine.load(index_path)
    else:
        print("Building new index with CLIP embeddings...")
        image_dir = Path("data/raw")
        
        if not image_dir.exists() or len(list(image_dir.iterdir())) == 0:
            print(f"Error: No images found in {image_dir}")
            print("Please add images to data/raw/ directory first.")
            return
        
        engine.build_index(image_dir, batch_size=16, save_path=index_path)
    
    # Select a query image
    image_paths = load_image_paths(Path("data/raw"))
    if len(image_paths) == 0:
        print("No images available for querying.")
        return
    
    query_image = image_paths[0]
    print(f"\nSearching for images similar to: {query_image.name}")
    
    # Search
    results = engine.search(query_image, top_k=5)
    
    # Print results with metadata
    print("\nSearch Results:")
    for result in results:
        print(f"  {result['rank']}. {result['filename']} "
              f"(score: {result['score']:.4f}, "
              f"size: {result['file_size']/(1024*1024):.2f}MB)")
    
    # Visualize
    print("\nGenerating visualization...")
    visualize_results(query_image, results, save_path="search_results.png")


if __name__ == "__main__":
    main()
