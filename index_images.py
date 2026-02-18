#!/usr/bin/env python3
"""脚本：指定图片目录，将图片向量化并存储到 Milvus Lite。

用法:
    python index_images.py /path/to/images
    python index_images.py /path/to/images --db-path ./my_index/milvus.db
    python index_images.py /path/to/images --append  # 追加到现有索引
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src import ImageSearchEngine
from src.utils import load_image_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="将目录中的图片向量化并存储到 Milvus Lite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法：索引 data/raw 目录
  python index_images.py data/raw

  # 指定数据库路径
  python index_images.py ~/Pictures/photos --db-path ./photos_index/milvus.db

  # 追加到已有索引（不删除原有数据）
  python index_images.py ./new_images --append
        """,
    )
    parser.add_argument(
        "image_dir",
        type=str,
        help="包含图片的目录路径（支持递归子目录）",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./data/index/milvus_lite.db",
        help="Milvus Lite 数据库文件路径 (默认: ./data/index/milvus_lite.db)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="image_search",
        help="Milvus collection 名称 (默认: image_search)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="元数据保存目录（可选，不指定则不保存）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="每批处理的图片数量 (默认: 16)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="追加到已有索引，而非删除重建",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP 模型名称 (默认: openai/clip-vit-large-patch14)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    image_dir = Path(args.image_dir).resolve()
    if not image_dir.exists():
        print(f"错误: 目录不存在: {image_dir}")
        sys.exit(1)
    if not image_dir.is_dir():
        print(f"错误: 不是目录: {image_dir}")
        sys.exit(1)

    # 确保数据库所在目录存在
    db_path = Path(args.db_path).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VM Search - 图片向量化并写入 Milvus Lite")
    print("=" * 60)

    # 扫描图片
    print(f"\n[1/4] 扫描目录: {image_dir}")
    image_paths = load_image_paths(image_dir)
    if not image_paths:
        print(f"错误: 在 {image_dir} 中未找到图片")
        print("支持的格式: jpg, jpeg, png, bmp, gif, tiff, webp")
        sys.exit(1)
    print(f"找到 {len(image_paths)} 张图片")

    # 初始化引擎
    print("\n[2/4] 初始化 CLIP 模型和 Milvus 连接...")
    engine = ImageSearchEngine(
        model_name=args.model,
        collection_name=args.collection,
        db_path=str(db_path),
    )
    print("✓ 引擎已初始化")

    # 构建索引
    drop_existing = not args.append
    print(f"\n[3/4] 处理图片并构建 Milvus 索引...")
    if drop_existing:
        print("  - 将删除已有 collection 并重建")
    else:
        print("  - 追加到现有 collection")
    print(f"  - 批次大小: {args.batch_size}")
    print("  - 提取 CLIP 嵌入向量 (768 维)")
    print("  - 收集元数据并写入 Milvus\n")

    try:
        engine.build_index(
            image_dir=image_dir,
            batch_size=args.batch_size,
            save_path=args.save_path,
            drop_existing=drop_existing,
        )
        print("✓ 索引构建成功")
    except Exception as e:
        print(f"\n✗ 构建索引失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 统计信息
    print(f"\n[4/4] Collection 统计:")
    stats = engine.get_stats()
    print(f"  Collection: {stats['name']}")
    print(f"  索引图片数: {stats['num_entities']}")
    print(f"  模型: {stats['model_name']}")
    print(f"  向量维度: {stats['feature_dim']}")
    print(f"  数据库: {db_path}")

    # 简单测试搜索
    print(f"\n[测试] 使用第一张图片进行相似度搜索...")
    try:
        results = engine.search(image_paths[0], top_k=min(5, len(image_paths)))
        print("Top 5 相似图片:")
        for r in results:
            print(f"  {r['rank']}. {r['filename']} (score: {r['score']:.4f})")
    except Exception as e:
        print(f"测试搜索失败: {e}")

    print("\n" + "=" * 60)
    print("完成！可通过 main.py 或 examples 进行搜索。")
    print("=" * 60)


if __name__ == "__main__":
    main()
