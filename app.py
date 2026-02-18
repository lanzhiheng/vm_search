"""Flask 服务：通过文字搜索图片。"""

import os
import sys
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file

# 添加项目根目录到路径
sys_path = Path(__file__).parent
sys.path.insert(0, str(sys_path))

from src import ImageSearchEngine

app = Flask(__name__)

# 配置（可通过环境变量覆盖）
INDEX_DIR = (sys_path / os.environ.get("VM_SEARCH_INDEX", "data/index")).resolve()
DB_PATH = str((sys_path / os.environ.get("VM_SEARCH_DB", "data/index/milvus_lite.db")).resolve())
COLLECTION_NAME = os.environ.get("VM_SEARCH_COLLECTION", "image_search")
MODEL_NAME = os.environ.get("VM_SEARCH_MODEL", "openai/clip-vit-large-patch14")

# 图片服务安全配置
# VM_SEARCH_IMAGE_BASE: 允许的图片根路径，可设置多个（逗号分隔），如 /Users/xxx/Pictures
# VM_SEARCH_IMAGE_PERMISSIVE=1: 允许任意存在的图片文件（本地开发时使用）
_image_base = os.environ.get("VM_SEARCH_IMAGE_BASE", "")
IMAGE_BASE_PATHS = [p.strip() for p in _image_base.split(",") if p.strip()]
IMAGE_PERMISSIVE = os.environ.get("VM_SEARCH_IMAGE_PERMISSIVE", "").lower() in ("1", "true", "yes")

engine = None


def get_engine():
    """延迟加载搜索引擎。"""
    global engine
    if engine is None:
        engine = ImageSearchEngine(
            model_name=MODEL_NAME,
            collection_name=COLLECTION_NAME,
            db_path=DB_PATH,
        )
        metadata_file = INDEX_DIR / "engine_metadata.pkl"
        if metadata_file.exists():
            engine.load(str(INDEX_DIR))
        else:
            engine.milvus.create_collection(drop_existing=False)
        app.logger.info("Search engine initialized")
    return engine


def is_safe_image_path(path: str) -> bool:
    """检查路径是否在允许的服务范围内。"""
    try:
        resolved = Path(path).resolve()
        if not resolved.exists() or not resolved.is_file():
            return False
        if IMAGE_PERMISSIVE:
            return True
        if IMAGE_BASE_PATHS:
            resolved_str = str(resolved)
            for base in IMAGE_BASE_PATHS:
                if resolved_str.startswith(str(Path(base).resolve())):
                    return True
            return False
        return str(resolved).startswith(str(sys_path.resolve()))
    except (ValueError, OSError):
        return False


@app.route("/")
def index():
    """搜索页面。"""
    return render_template("search.html")


@app.route("/api/search", methods=["GET", "POST"])
def search():
    """文字搜索 API。"""
    if request.method == "POST":
        data = request.get_json() or {}
        q = data.get("q", "").strip()
    else:
        q = request.args.get("q", "").strip()

    if not q:
        return jsonify({"error": "请提供搜索关键词 (q)"}), 400

    try:
        eng = get_engine()
        top_k = int(request.args.get("top_k", request.form.get("top_k", 12)))
        top_k = min(max(top_k, 1), 50)
        results = eng.search_by_text(q, top_k=top_k)
    except Exception as e:
        app.logger.exception("Search failed")
        return jsonify({"error": str(e)}), 500

    # 转换为可序列化格式，根据路径生成图片 URL（供前端展示）
    items = []
    for r in results:
        img_path = r.get("path") or r.get("image_path")
        items.append({
            "rank": r["rank"],
            "filename": r["filename"],
            "score": float(r["score"]),
            "path": img_path,
            "image_url": f"/api/image?path={_encode_path(img_path)}" if img_path else None,
        })
    return jsonify({"query": q, "results": items})


def _encode_path(path: str) -> str:
    """对路径进行 URL 安全编码。"""
    import base64
    return base64.urlsafe_b64encode(path.encode("utf-8")).decode("ascii")


def _decode_path(encoded: str) -> str:
    """解码路径。"""
    import base64
    return base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8")


@app.route("/api/image")
def serve_image():
    """根据路径安全地返回图片文件。"""
    encoded = request.args.get("path")
    if not encoded:
        return jsonify({"error": "Missing path"}), 400
    try:
        path = _decode_path(encoded)
    except Exception:
        return jsonify({"error": "Invalid path"}), 400
    if not is_safe_image_path(path):
        return jsonify({"error": "Path not allowed"}), 403
    return send_file(path, mimetype=None)  # 自动根据扩展名推断


def main():
    """启动 Flask 服务。"""
    port = int(os.environ.get("PORT", 5000))
    if not IMAGE_BASE_PATHS and not IMAGE_PERMISSIVE:
        print("提示: 图片在项目外时，需设置 VM_SEARCH_IMAGE_BASE 或 VM_SEARCH_IMAGE_PERMISSIVE=1")
    print(f"启动服务: http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
    main()
