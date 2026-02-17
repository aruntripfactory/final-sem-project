# ingestion/embed_store.py

import os
import logging
import threading
from typing import List, Dict, Any, Union

from sentence_transformers import SentenceTransformer
import chromadb
import chromadb.config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# CHROMA PATH
# ─────────────────────────────────────────────────────────────
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
CHROMA_DIR = os.path.abspath(CHROMA_DIR)

# ─────────────────────────────────────────────────────────────
# SHARED SINGLETON: CHROMA CLIENT
# ─────────────────────────────────────────────────────────────
_CLIENT = None
_CLIENT_LOCK = threading.Lock()


def get_shared_client() -> chromadb.PersistentClient:
    """
    Single Chroma client used across ENTIRE application.
    Prevents:
    - readonly DB
    - instance conflicts
    - multiple sqlite writers
    """
    global _CLIENT

    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                os.makedirs(CHROMA_DIR, exist_ok=True)

                settings = chromadb.config.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )

                _CLIENT = chromadb.PersistentClient(
                    path=CHROMA_DIR,
                    settings=settings
                )

                logger.info("[EMBED_STORE] Shared Chroma client created.")

    return _CLIENT


# ─────────────────────────────────────────────────────────────
# SHARED SINGLETON: EMBEDDING MODEL
# ─────────────────────────────────────────────────────────────
_MODEL: SentenceTransformer | None = None
_MODEL_LOCK = threading.Lock()


def get_embedding_model() -> SentenceTransformer:
    """Loads embedding model only once for whole app."""
    global _MODEL

    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                logger.info("[EMBED_STORE] Loading embedding model: all-MiniLM-L6-v2")
                _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("[EMBED_STORE] Embedding model loaded.")

    return _MODEL


# ─────────────────────────────────────────────────────────────
# COLLECTION MANAGEMENT
# ─────────────────────────────────────────────────────────────
def get_or_create_collection(collection_name: str):
    client = get_shared_client()

    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(collection_name: str):
    """Delete + recreate collection safely."""
    client = get_shared_client()

    try:
        client.delete_collection(collection_name)
        logger.info(f"[EMBED_STORE] Collection '{collection_name}' deleted.")
    except Exception:
        pass

    client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    logger.info(f"[EMBED_STORE] Collection '{collection_name}' recreated.")


def invalidate_client():
    """Force reset of singleton client."""
    global _CLIENT
    _CLIENT = None
    logger.warning("[EMBED_STORE] Chroma client invalidated.")


# ─────────────────────────────────────────────────────────────
# METADATA SANITIZER
# ─────────────────────────────────────────────────────────────
def _sanitize_metadata(metadata: dict) -> dict:
    if not metadata or not isinstance(metadata, dict):
        return {}

    sanitized: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            sanitized[key] = ""
        elif isinstance(value, (bool, int, float)):
            sanitized[key] = value
        elif isinstance(value, str):
            sanitized[key] = value if value else "unknown"
        elif isinstance(value, list):
            sanitized[key] = ", ".join(str(v) for v in value)
        else:
            sanitized[key] = str(value)

    return sanitized


# ─────────────────────────────────────────────────────────────
# MAIN STORAGE FUNCTION
# ─────────────────────────────────────────────────────────────
def store_documents_enhanced(
    chunks: List[Union[Dict[str, Any], Any]],
    ids: List[str],
    collection_name: str = "research_papers",
    embedding_model: SentenceTransformer | None = None,
) -> chromadb.Collection:

    if not chunks:
        logger.warning("[EMBED_STORE] No documents to store.")
        return

    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    valid_ids: List[str] = []

    for i, chunk in enumerate(chunks):
        if hasattr(chunk, "text") and hasattr(chunk, "metadata"):
            text = chunk.text
            metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
        elif isinstance(chunk, dict) and "text" in chunk:
            text = chunk["text"]
            metadata = chunk.get("metadata", {})
        else:
            text = str(chunk)
            metadata = {}

        if not text or not text.strip():
            continue

        texts.append(text.strip())
        metadatas.append(_sanitize_metadata(metadata))
        valid_ids.append(ids[i] if i < len(ids) else f"chunk_{i}")

    if not texts:
        logger.warning("[EMBED_STORE] No valid text after filtering.")
        return

    model = embedding_model if embedding_model else get_embedding_model()

    logger.info(f"[EMBED_STORE] Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).tolist()

    collection = get_or_create_collection(collection_name)

    BATCH_SIZE = 500
    for start in range(0, len(texts), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(texts))
        collection.upsert(
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
            ids=valid_ids[start:end],
        )

        logger.info(f"[EMBED_STORE] Stored batch {start // BATCH_SIZE + 1}: {end-start} chunks")

    logger.info(
        f"[EMBED_STORE] Stored {len(texts)} docs in '{collection_name}'. "
        f"Total now: {collection.count()}"
    )

    return collection


# ─────────────────────────────────────────────────────────────
# BACKWARD COMPAT
# ─────────────────────────────────────────────────────────────
def store_documents(
    docs: List[Union[str, Dict[str, Any]]],
    ids: List[str],
    collection_name: str = "research_papers",
) -> chromadb.Collection:
    chunks = [
        doc if isinstance(doc, dict) else {"text": str(doc), "metadata": {}}
        for doc in docs
    ]
    return store_documents_enhanced(chunks, ids, collection_name)

# Backward compatibility (old code may still call this)
def create_client():
    return get_shared_client()
