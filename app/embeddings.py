"""本地中文 embedding：模型下载、缓存与 bge-small-zh-v1.5 推理。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from functools import lru_cache
from pathlib import Path

import requests
from fastembed import TextEmbedding

from app.config import settings
from app.db_types import VECTOR_DIMENSIONS

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
MODEL_REVISION = "677b527935023b31f309ecb027745526aa5f680f"
MODEL_REPO_PATH = "Qdrant/bge-small-zh-v1.5"
MODEL_BASE_URL = (
    f"https://modelscope.cn/models/{MODEL_REPO_PATH}/resolve/{MODEL_REVISION}"
)
MODEL_MANIFEST_FILE = ".healthtrace-model.json"

# fastembed 指定本地目录时会直接使用这些文件；SHA-256 防止残缺或错误模型进入缓存。
MODEL_FILES: dict[str, tuple[int, str]] = {
    "config.json": (739, "9088751d39abbf86ec3d19ffca92ad62ad19075f7e59712e6c71217fa125d1d3"),
    "model_optimized.onnx": (
        94781076,
        "1294ea4b6331115a353d81f96b85e8c8d7fdcc284453d5b2fab5b016230aad38",
    ),
    "ort_config.json": (1234, "97e78d1d21c2eb719e865b018f17915df6a12ed987446eb7f3f3a783a5afb1e1"),
    "special_tokens_map.json": (
        125,
        "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    ),
    "tokenizer.json": (
        439125,
        "48cea5d44424912a6fd1ea647bf4fe50b55ab8b1e5879c3275f80e339e8fae26",
    ),
    "tokenizer_config.json": (
        367,
        "e6f3b96db926a37d4039995fbf5ad17de158dfb8f6343d607e4dbaad18d75f5a",
    ),
    "vocab.txt": (109540, "45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c"),
}

_EMBED_LOCK = threading.Lock()


def _sha256(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_is_current(model_dir: Path) -> bool:
    """快速检查缓存清单与文件大小，避免每次启动都重新哈希 91MB 模型。"""
    manifest_path = model_dir / MODEL_MANIFEST_FILE
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    for name, (size, checksum) in MODEL_FILES.items():
        path = model_dir / name
        if not path.is_file() or path.stat().st_size != size:
            return False
        if manifest.get(name) != checksum:
            return False
    return True


def _download_file(url: str, destination: Path, expected_size: int, expected_hash: str) -> None:
    """流式下载并原子替换单个模型文件，校验大小与 SHA-256。"""
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    with requests.Session() as session:
        # ModelScope 是公开只读模型源；显式绕过失效的本机代理，保证首次初始化可复现。
        session.trust_env = False
        for attempt in range(3):
            try:
                with session.get(url, stream=True, timeout=(10, 120)) as response:
                    response.raise_for_status()
                    with partial.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            handle.write(chunk)
                            digest.update(chunk)
                break
            except requests.RequestException:
                partial.unlink(missing_ok=True)
                digest = hashlib.sha256()
                if attempt == 2:
                    raise

    size = partial.stat().st_size
    checksum = digest.hexdigest()
    if size != expected_size or checksum != expected_hash:
        partial.unlink(missing_ok=True)
        raise ValueError(
            f"模型文件校验失败：{destination.name} size={size} sha256={checksum}"
        )
    partial.replace(destination)


def ensure_embedding_model(cache_dir: str | Path | None = None) -> Path:
    """确保本地模型完整可用，返回 bge-small-zh-v1.5 模型目录。"""
    root = Path(cache_dir or settings.embedding_model_cache_dir).expanduser()
    model_dir = root / "bge-small-zh-v1.5"
    model_dir.mkdir(parents=True, exist_ok=True)
    if _manifest_is_current(model_dir):
        return model_dir

    for name, (size, checksum) in MODEL_FILES.items():
        path = model_dir / name
        if path.is_file() and path.stat().st_size == size and _sha256(path) == checksum:
            continue
        _download_file(f"{MODEL_BASE_URL}/{name}", path, size, checksum)

    manifest = {name: checksum for name, (_, checksum) in MODEL_FILES.items()}
    (model_dir / MODEL_MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return model_dir


@lru_cache(maxsize=1)
def _load_model(cache_dir: str) -> TextEmbedding:
    """加载本地 ONNX 模型；同一进程内复用推理会话。"""
    model_dir = ensure_embedding_model(cache_dir)
    model = TextEmbedding(
        model_name=MODEL_NAME,
        cache_dir=str(Path(cache_dir).expanduser()),
        specific_model_path=str(model_dir),
        threads=settings.embedding_threads,
    )
    if model.embedding_size != VECTOR_DIMENSIONS:
        raise ValueError(
            f"模型维度不匹配：expected={VECTOR_DIMENSIONS}, actual={model.embedding_size}"
        )
    return model


def embed_texts_sync(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """同步批量生成归一化向量；空输入不加载模型。"""
    if not texts:
        return []
    model = _load_model(settings.embedding_model_cache_dir)
    with _EMBED_LOCK:
        vectors = [embedding.tolist() for embedding in model.embed(texts, batch_size=batch_size)]
    if any(len(vector) != VECTOR_DIMENSIONS for vector in vectors):
        raise ValueError("embedding 维度与 pgvector 列定义不一致")
    return vectors


async def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """在线程池中批量生成向量，避免阻塞 FastAPI 事件循环。"""
    return await asyncio.to_thread(embed_texts_sync, texts, batch_size)


async def embed_query(text: str) -> list[float]:
    """生成单条查询向量；空查询由调用层拦截。"""
    vectors = await embed_texts([text])
    return vectors[0]
