# ingestion/embed_store.py

import os
import logging
import threading
from typing import List, Dict, Any, Union

from sentence_transformers import SentenceTransformer
import chromadb

logger = logging.getLogger(__name__)

# Location for chroma persistence (absolute)
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
CHROMA_DIR = os.path.abspath(CHROMA_DIR)

# ── Thread-safe singleton for embedding model ──────────────────────────────
_MODEL: SentenceTransformer | None = None
_MODEL_LOCK = threading.Lock()


def get_embedding_model() -> SentenceTransformer:
   
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:  # double-checked locking
                logger.info("[EMBED_STORE] Loading embedding model: all-MiniLM-L6-v2")
                _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("[EMBED_STORE] Embedding model loaded.")
    return _MODEL


def create_client() -> chromadb.PersistentClient:
    
    try:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        return chromadb.PersistentClient(path=CHROMA_DIR)
    except Exception as e:
        raise RuntimeError(
            f"[EMBED_STORE] ChromaDB init failed at '{CHROMA_DIR}': {e}"
        ) from e


def _sanitize_metadata(metadata: dict) -> dict:
    
    if not metadata or not isinstance(metadata, dict):
        return {}

    sanitized: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            sanitized[key] = ""          
        elif isinstance(value, bool):
            sanitized[key] = value
        elif isinstance(value, (int, float)):
            sanitized[key] = value
        elif isinstance(value, str):
            sanitized[key] = value if value else "unknown"
        elif isinstance(value, list):
            sanitized[key] = ", ".join(str(v) for v in value) if value else ""
        else:
            sanitized[key] = str(value)

    return sanitized


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
            # TextChunk dataclass
            text = chunk.text
            metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
        elif isinstance(chunk, dict) and "text" in chunk:
            text = chunk["text"]
            metadata = chunk.get("metadata", {})
        else:
            text = str(chunk)
            logger.warning(f"[EMBED_STORE] Unexpected chunk type at index {i}: {type(chunk)}")

        if not text or not text.strip():
            logger.warning(f"[EMBED_STORE] Skipping empty chunk at index {i}")
            continue

        texts.append(text.strip())
        metadatas.append(_sanitize_metadata(metadata))
        valid_ids.append(ids[i] if i < len(ids) else f"chunk_{i}")

    if not texts:
        logger.warning("[EMBED_STORE] No valid text content to store after filtering.")
        return

    model = embedding_model if embedding_model else get_embedding_model()

    logger.info(f"[EMBED_STORE] Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).tolist()

    client = create_client()

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    BATCH_SIZE = 500
    for start in range(0, len(texts), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(texts))
        collection.upsert(
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
            ids=valid_ids[start:end],
        )
        logger.info(
            f"[EMBED_STORE] Stored batch {start // BATCH_SIZE + 1}: "
            f"{end - start} chunks"
        )

    chunk_type_counts: Dict[str, int] = {}
    for meta in metadatas:
        ct = meta.get("chunk_type", "unknown")
        chunk_type_counts[ct] = chunk_type_counts.get(ct, 0) + 1

    logger.info(
        f"[EMBED_STORE] Stored {len(texts)} docs in '{collection_name}'. "
        f"Total in collection: {collection.count()}"
    )
    for ct, count in chunk_type_counts.items():
        logger.info(f"[EMBED_STORE]   {ct}: {count} chunks")

    return collection


def store_documents(
    docs: List[Union[str, Dict[str, Any]]],
    ids: List[str],
    collection_name: str = "research_papers",
) -> chromadb.Collection:
    """Backward-compatible wrapper."""
    chunks = [
        doc if isinstance(doc, dict) else {"text": str(doc), "metadata": {}}
        for doc in docs
    ]
    return store_documents_enhanced(chunks, ids, collection_name)