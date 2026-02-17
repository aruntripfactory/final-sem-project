"""
Public API:
  ingest_uploaded_files()   ← called by app.py (Streamlit uploads)
  ingest_pdfs_enhanced()    ← called by CLI / scheduled jobs

Everything below these two functions is internal.
app.py imports NOTHING from any other ingestion module.
"""

import os
import sys
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dataclasses import dataclass

from .pdf_loader import extract_text_with_pages, list_pdfs, batch_extract
from .preprocess import ResearchPaperChunker
from .embed_store import (
    store_documents_enhanced,
    CHROMA_DIR,
    get_shared_client,
)
from .image_extractor import process_pdf_images

__all__ = ["ingest_uploaded_files", "ingest_pdfs_enhanced", "CHROMA_DIR", "IngestResult"]

logger = logging.getLogger(__name__)

PDF_DIR = Path(
    os.getenv("PDF_DIR", str(Path(__file__).parent.parent / "data" / "pdfs"))
)


# ─────────────────────────────────────────────────────────────
# RESULT MODEL
# ─────────────────────────────────────────────────────────────
@dataclass
class IngestResult:
    processed: int
    skipped: List[str]
    failed: List[str]
    total_chunks: int
    image_chunks: int
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


# ─────────────────────────────────────────────────────────────
# METADATA TEXT OPTIMIZATION
# ─────────────────────────────────────────────────────────────
def _extract_core_sections(full_text: str) -> str:
    """Extract abstract/introduction/conclusion to reduce LLM tokens."""
    full_text_lower = full_text.lower()
    sections = []

    if "abstract" in full_text_lower:
        start = full_text_lower.find("abstract")
        sections.append(full_text[start:start+1500])

    if "introduction" in full_text_lower:
        start = full_text_lower.find("introduction")
        sections.append(full_text[start:start+2000])

    for keyword in ["conclusion", "conclusions", "future work"]:
        if keyword in full_text_lower:
            start = full_text_lower.find(keyword)
            sections.append(full_text[start:start+2000])
            break

    if not sections:
        return full_text[:4000]

    return "\n\n".join(sections)


# ─────────────────────────────────────────────────────────────
# MAIN INGESTION PIPELINE (STREAMLIT)
# ─────────────────────────────────────────────────────────────
def ingest_uploaded_files(
    files: List[Dict[str, Any]],
    *,
    doc_manager=None,
    embedding_model=None,
    collection_name: str = "research_papers",
    chunk_size: int = 1000,
    overlap: int = 150,
    max_images: int = 10,
    max_workers: int = 4,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> IngestResult:

    if not files:
        return IngestResult(0, [], [], 0, 0)

    def _notify(frac: float, msg: str):
        logger.info(f"[INGEST] {msg}")
        if progress_cb:
            progress_cb(frac, msg)

    total = len(files)
    max_workers = min(max_workers, os.cpu_count() or 4, total)

    chunker = ResearchPaperChunker(chunk_size=chunk_size, overlap=overlap)

    all_chunks = []
    all_chunk_ids = []
    phase2_meta = []

    processed_count = 0
    skipped_files = []
    failed_files = []

    # ───────── PHASE 1: TEXT EXTRACTION ─────────
    _notify(0.0, f"Extracting text from {total} file(s)…")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_extract_and_chunk, fd, chunker, doc_manager): fd["name"]
            for fd in files
        }

        done = 0
        for future in as_completed(futures):
            name = futures[future]
            done += 1

            try:
                result = future.result()

                if result is None:
                    skipped_files.append(name)
                    continue

                for chunk in result["chunks"]:
                    all_chunks.append(chunk)
                    all_chunk_ids.append(chunk.chunk_id)

                phase2_meta.append(result)
                processed_count += 1

            except Exception as e:
                failed_files.append(name)
                logger.error(f"[INGEST] Failed {name}: {e}")

            _notify(done / total * 0.5, f"Processed {name}")

    # ───────── PHASE 2: IMAGE ANALYSIS ─────────
    image_chunk_count = 0

    if max_images:
        for idx, meta in enumerate(phase2_meta):
            try:
                image_chunks = process_pdf_images(
                    pdf_source=meta["content"],
                    paper_id=meta["file_hash"],
                    max_images=max_images,
                )
                for i, ic in enumerate(image_chunks):
                    all_chunks.append(ic)
                    all_chunk_ids.append(f"{meta['file_hash']}_img_{i}")
                    image_chunk_count += 1
            except Exception as e:
                logger.warning(f"[INGEST] Image extraction skipped: {e}")

    # ───────── PHASE 3: DEDUP + EMBEDDING ─────────
    if not all_chunks:
        return IngestResult(processed_count, skipped_files, failed_files, 0, 0)

    _notify(0.75, "Checking for duplicate embeddings…")

    client = get_shared_client()
    collection = client.get_or_create_collection(collection_name)

    existing_ids = set(collection.get(ids=all_chunk_ids)["ids"] or [])

    new_chunks = []
    new_ids = []

    for chunk, cid in zip(all_chunks, all_chunk_ids):
        if cid not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(cid)

    if new_chunks:
        _notify(0.85, f"Embedding {len(new_chunks)} chunks…")
        store_documents_enhanced(
            chunks=new_chunks,
            ids=new_ids,
            collection_name=collection_name,
            embedding_model=embedding_model,
        )
    else:
        logger.info("[INGEST] All chunks already exist in Chroma.")

    # ───────── PHASE 4: METADATA EXTRACTION ─────────
    _notify(0.93, "Extracting research metadata…")

    try:
        from ingestion.metadata_extractor import extract_paper_metadata
        from db.paper_repository import save_paper_metadata
        from db.database import SessionLocal
        from db.models import ResearchPaperMetadata
    except Exception as e:
        logger.error(f"[INGEST] Metadata modules missing: {e}")
    else:
        db = SessionLocal()

        for meta in phase2_meta:
            try:
                file_hash = meta["file_hash"]

                existing = db.query(ResearchPaperMetadata)\
                    .filter(ResearchPaperMetadata.paper_id == file_hash)\
                    .first()

                if existing:
                    continue

                full_text = "\n\n".join(
                    f"[PAGE {p['page_num']}]\n{p['text']}"
                    for p in meta.get("pages", [])
                )

                core_text = _extract_core_sections(full_text)

                metadata_json = extract_paper_metadata(
                    paper_text=core_text,
                    paper_id=file_hash,
                    file_name=meta["file_name"],
                )

                save_paper_metadata(metadata_json)

            except Exception as e:
                logger.warning(f"[INGEST] Metadata extraction failed: {e}")

        db.close()

    stats = IngestResult(
        processed=processed_count,
        skipped=skipped_files,
        failed=failed_files,
        total_chunks=len(all_chunks),
        image_chunks=image_chunk_count
    )

    logger.info(f"[INGEST] Final stats: {stats}")
    return stats


# ─────────────────────────────────────────────────────────────
# INTERNAL WORKER
# ─────────────────────────────────────────────────────────────
def _extract_and_chunk(fd, chunker, doc_manager=None):

    file_name = fd["name"]
    file_bytes = fd["content"]
    file_size = len(file_bytes)
    file_hash = hashlib.md5(file_bytes).hexdigest()[:12]

    if doc_manager and doc_manager.document_exists(file_name, file_size):
        return None

    pages = extract_text_with_pages(file_bytes)
    if not pages:
        return None

    full_text = "\n\n".join(
        f"[PAGE {p['page_num']}]\n{p['text']}" for p in pages
    )

    chunks = chunker.chunk_document({
        "content": full_text,
        "paper_id": file_hash,
        "title": file_name,
    })

    return {
        "file_name": file_name,
        "file_hash": file_hash,
        "content": file_bytes,
        "chunks": chunks,
        "pages": pages,
    }
