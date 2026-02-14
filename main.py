"""CLI to search for images by query image or text using the existing Milvus index."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from src import ImageSearchEngine


def main():
    parser = argparse.ArgumentParser(
        description="Search for images by query image or text."
    )
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the query image file (use --text for text search instead)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        metavar="QUERY",
        help="Text query to search for similar images (e.g. 'a red bus')",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="K",
        help="Number of similar images to return (default: 5)",
    )
    parser.add_argument(
        "--index",
        dest="load_path",
        type=Path,
        default=Path("data/index"),
        metavar="DIR",
        help="Directory containing engine_metadata.pkl from build_index (default: data/index)",
    )
    args = parser.parse_args()

    if args.text is not None and args.image is not None:
        print("Error: Provide either an image path or --text, not both.", file=sys.stderr)
        sys.exit(1)
    if args.text is None and args.image is None:
        print("Error: Provide an image path or --text for the search query.", file=sys.stderr)
        sys.exit(1)

    if args.image is not None:
        query_image = args.image
        if not query_image.exists():
            print(f"Error: Image file not found: {query_image}", file=sys.stderr)
            sys.exit(1)
        if not query_image.is_file():
            print(f"Error: Not a file: {query_image}", file=sys.stderr)
            sys.exit(1)

    load_path = args.load_path
    if not load_path.exists() or not load_path.is_dir():
        print(f"Error: Index directory not found: {load_path}", file=sys.stderr)
        print("Run build_index.py first to create the index.", file=sys.stderr)
        sys.exit(1)
    metadata_file = load_path / "engine_metadata.pkl"
    if not metadata_file.exists():
        print(f"Error: No engine_metadata.pkl in {load_path}", file=sys.stderr)
        print("Run build_index.py first to create the index.", file=sys.stderr)
        sys.exit(1)

    engine = ImageSearchEngine(
        model_name="openai/clip-vit-large-patch14",
        collection_name="image_search",
        db_path="./data/index/milvus_lite.db",
    )
    engine.load(str(load_path))

    if args.text is not None:
        results = engine.search_by_text(args.text, top_k=args.top_k)
        query_label = f'"{args.text}"'
    else:
        results = engine.search(args.image, top_k=args.top_k)
        query_label = str(args.image)

    print(f"\nTop {len(results)} similar images for: {query_label}")
    for result in results:
        mod_time = datetime.fromtimestamp(result["modified_time"]).strftime("%Y-%m-%d %H:%M")
        file_size_mb = result["file_size"] / (1024 * 1024)
        print(f"  {result['rank']}. {result['filename']}")
        print(f"      Score: {result['score']:.4f}")
        print(f"      Size: {file_size_mb:.2f} MB")
        print(f"      Modified: {mod_time}")
        print(f"      Path: {result['path']}")
        print()


if __name__ == "__main__":
    main()
