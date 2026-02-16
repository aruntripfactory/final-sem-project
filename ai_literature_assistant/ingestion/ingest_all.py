# ingestion/ingest_all.py
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
from typing import Any, Callable, Dict, List, Optional, Tuple

from dataclasses import dataclass, field

from .pdf_loader import extract_text_with_pages, list_pdfs, batch_extract
from .preprocess import ResearchPaperChunker
from .embed_store import store_documents_enhanced, CHROMA_DIR   # re-exported for app.py
from .image_extractor import process_pdf_images

__all__ = ["ingest_uploaded_files", "ingest_pdfs_enhanced", "CHROMA_DIR", "IngestResult"]

logger = logging.getLogger(__name__)

PDF_DIR = Path(
    os.getenv("PDF_DIR", str(Path(__file__).parent.parent / "data" / "pdfs"))
)


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


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC — app.py calls only this function
# ════════════════════════════════════════════════════════════════════════════

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
    """
    Full ingestion pipeline for files uploaded through the Streamlit UI.

    app.py passes in [{name, content}] dicts and a progress callback.
    Everything else — parsing, chunking, image analysis, embedding, storage —
    is handled here.  app.py never touches pdf_loader, preprocess,
    embed_store, or image_extractor directly.
    """
    if not files:
        return IngestResult(0, [], [], 0, 0)

    def _notify(fraction: float, msg: str) -> None:
        logger.info(f"[INGEST] {msg}")
        if progress_cb:
            progress_cb(fraction, msg)

    total = len(files)
    chunker = ResearchPaperChunker(chunk_size=chunk_size, overlap=overlap)

    all_chunks:    List[Any] = []
    all_chunk_ids: List[str] = []
    phase2_meta:   List[Dict] = []   # files that need image extraction
    
    processed_count = 0
    skipped_files: List[str] = []
    failed_files:  List[str] = []

    # ── Phase 1: Text extraction + chunking (parallel threads) ──────────────
    _notify(0.0, f"Extracting text from {total} file(s)…")

    with ThreadPoolExecutor(max_workers=min(max_workers, total)) as pool:
        futures = {
            pool.submit(_extract_and_chunk, fd, chunker, doc_manager): fd["name"]
            for fd in files
        }
        done = 0
        for future in as_completed(futures):
            file_name = futures[future]
            done += 1
            try:
                result = future.result()
                if result is None:
                    skipped_files.append(file_name)
                    _notify(done / total * 0.5,
                            f"Skipped {file_name} (already processed)")
                else:
                    for chunk in result["chunks"]:
                        all_chunks.append(chunk)
                        all_chunk_ids.append(chunk.chunk_id)
                    phase2_meta.append(result)
                    processed_count += 1
                    _notify(done / total * 0.5,
                            f"Processed {file_name} — {len(result['chunks'])} chunks")
            except Exception as e:
                failed_files.append(file_name)
                logger.error(f"[INGEST] Failed '{file_name}': {e}")
                _notify(done / total * 0.5, f"Failed {file_name}: {e}")

    # ── Phase 2: Image extraction + Vision analysis (sequential per file) ────
    image_chunk_count = 0

    if max_images and max_images > 0:
        for idx, meta in enumerate(phase2_meta):
            file_bytes = meta.get("content")
            if not file_bytes:
                continue

            file_hash = meta["file_hash"]
            file_name = meta["file_name"]
            _notify(0.5 + idx / max(len(phase2_meta), 1) * 0.3,
                    f"Analyzing images in {file_name}…")

            try:
                image_chunks = process_pdf_images(
                    pdf_source=file_bytes,
                    paper_id=file_hash,
                    max_images=max_images,
                )
                for i, ic in enumerate(image_chunks):
                    # IDs are scoped to file_hash — stable and collision-free
                    all_chunks.append(ic)
                    all_chunk_ids.append(f"{file_hash}_img_{i:04d}")
                    image_chunk_count += 1
            except Exception as e:
                logger.warning(f"[INGEST] Image extraction skipped for '{file_name}': {e}")

    # ── Phase 3: Embed + store ───────────────────────────────────────────────
    if not all_chunks:
        logger.warning("[INGEST] No chunks to store.")
        if failed_files and not processed_count and not skipped_files:
             return IngestResult(0, skipped_files, failed_files, 0, 0, "Processing failed for all files")
        return IngestResult(processed_count, skipped_files, failed_files, 0, 0)

    _notify(0.85, f"Embedding {len(all_chunks)} chunks…")

    store_documents_enhanced(
        chunks=all_chunks,
        ids=all_chunk_ids,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

    _notify(1.0, "Ingestion complete.")

    stats = IngestResult(
        processed=processed_count,
        skipped=skipped_files,
        failed=failed_files,
        total_chunks=len(all_chunks),
        image_chunks=image_chunk_count
    )
    logger.info(f"[INGEST] Final stats: {stats}")
    return stats


# ════════════════════════════════════════════════════════════════════════════
# INTERNAL — not imported outside this module
# ════════════════════════════════════════════════════════════════════════════

def _extract_and_chunk(
    fd: Dict[str, Any],
    chunker: ResearchPaperChunker,
    doc_manager=None,
) -> Optional[Dict[str, Any]]:
    """
    Single-file worker: one PDF parse → chunk.
    Returns None if doc_manager reports file already processed.
    Thread-safe; never calls Streamlit.
    """
    file_name:  str   = fd["name"]
    file_bytes: bytes = fd["content"]
    file_size         = len(file_bytes)
    file_hash         = hashlib.md5(file_bytes).hexdigest()[:12]

    if doc_manager and doc_manager.document_exists(file_name, file_size):
        return None

    if doc_manager:
        doc_manager.add_document(file_name, file_size, file_hash)

    # Single parse: extract_text_with_pages gives text AND page count together
    pages = extract_text_with_pages(file_bytes)
    if not pages:
        logger.warning(f"[INGEST] No pages extracted from '{file_name}'")
        return None

    num_pages = pages[-1]["total_pages"]
    full_text = "\n\n".join(
        f"[PAGE {p['page_num']}]\n{p['text']}" for p in pages
    )

    if len(full_text.strip()) < 100:
        logger.warning(f"[INGEST] Too little text in '{file_name}'")
        return None

    chunks = chunker.chunk_document({
        "content":  full_text,
        "paper_id": file_hash,
        "title":    file_name,
    })

    if doc_manager:
        doc_manager.update_document(
            file_hash,
            status="completed",
            num_chunks=len(chunks),
            num_pages=num_pages,
        )

    logger.info(
        f"[INGEST] '{file_name}': "
        f"{num_pages} pages, {len(chunks)} chunks, {len(full_text)} chars"
    )
    return {
        "file_name": file_name,
        "file_hash": file_hash,
        "content":   file_bytes,   # passed to Phase 2 for image extraction
        "chunks":    chunks,
        "num_pages": num_pages,
    }


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC — disk-based batch ingestion (CLI / cron jobs, not app.py)
# ════════════════════════════════════════════════════════════════════════════

def validate_environment() -> List[Path]:
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"[INGEST] PDF directory not found: {PDF_DIR}")
    pdf_files = list_pdfs(str(PDF_DIR))
    if not pdf_files:
        raise ValueError(f"[INGEST] No PDFs in {PDF_DIR}")
    return [Path(p) for p in pdf_files]


def ingest_pdfs_enhanced(
    collection_name: str = "research_papers",
    chunk_size: int = 1000,
    overlap: int = 150,
    max_workers: int = 4,
) -> Dict[str, Any]:
    """Batch ingestion from PDF_DIR — for CLI / scheduled jobs."""
    try:
        pdf_files = validate_environment()
    except Exception as e:
        logger.error(f"[INGEST] {e}")
        return {"error": str(e)}

    logger.info(f"[INGEST] Ingesting {len(pdf_files)} PDF(s) from {PDF_DIR}")
    chunker = ResearchPaperChunker(chunk_size=chunk_size, overlap=overlap)
    all_chunks: List[Any] = []
    all_ids:    List[str] = []
    processed = failed = 0

    text_by_path = {
        r["path"]: r["text"]
        for r in batch_extract(str(PDF_DIR), max_workers=max_workers)
    }

    for pdf_path in pdf_files:
        raw_text = text_by_path.get(str(pdf_path), "")
        if len(raw_text.strip()) < 100:
            logger.warning(f"[INGEST] Skipping {pdf_path.name}: insufficient text")
            failed += 1
            continue
        try:
            chunks = chunker.chunk_text(raw_text, pdf_path.stem)
            if not chunks:
                failed += 1
                continue
            all_chunks.extend(chunks)
            all_ids.extend(c.chunk_id for c in chunks)
            processed += 1
            logger.info(f"[INGEST] ✔ {pdf_path.name} → {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"[INGEST] Chunking failed for {pdf_path.name}: {e}")
            failed += 1

    if not all_chunks:
        return {"error": "No chunks generated"}

    store_documents_enhanced(chunks=all_chunks, ids=all_ids,
                             collection_name=collection_name)

    return {
        "total_pdfs":         len(pdf_files),
        "processed_pdfs":     processed,
        "failed_pdfs":        failed,
        "total_chunks":       len(all_chunks),
        "avg_chunks_per_pdf": round(len(all_chunks) / processed, 1) if processed else 0,
    }


def ingest_pdfs() -> None:
    """Legacy simple ingestion (backward compat)."""
    from .preprocess import chunk_text
    from .embed_store import store_documents
    all_chunks: List[str] = []
    ids: List[str] = []
    for i, pdf in enumerate(list_pdfs(str(PDF_DIR))):
        pages  = extract_text_with_pages(pdf)
        joined = "\n\n".join(f"[PAGE {p['page_num']}]\n{p['text']}" for p in pages)
        for j, chunk in enumerate(chunk_text(joined, 800, 120)):
            all_chunks.append(chunk)
            ids.append(f"legacy_{i:04d}_{j:04d}")
    store_documents(all_chunks, ids)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        ingest_pdfs()
    else:
        result = ingest_pdfs_enhanced()
        sys.exit(0 if "error" not in result else 1)