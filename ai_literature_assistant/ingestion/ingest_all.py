# ingestion/ingest_all.py

import sys
from typing import List, Dict, Any, Tuple
from pathlib import Path

from .pdf_loader import extract_text_from_pdf, list_pdfs
from .preprocess import ResearchPaperChunker
from .embed_store import store_documents_enhanced

PDF_DIR = Path(__file__).parent.parent / "data" / "pdfs"


def validate_environment() -> List[Path]:
    """
    Validate PDF directory and return list of PDFs.
    """
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"PDF directory not found: {PDF_DIR}")

    pdf_files = list_pdfs(str(PDF_DIR))

    if not pdf_files:
        raise ValueError(f"No PDF files found in {PDF_DIR}")

    return [Path(p) for p in pdf_files]


def process_single_pdf(
    pdf_path: Path,
    chunker: ResearchPaperChunker
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Extract text → chunk → generate IDs
    """
    try:
        text = extract_text_from_pdf(str(pdf_path))

        if not text or len(text.strip()) < 100:
            print(f"[WARN] Skipping {pdf_path.name}: insufficient text")
            return [], []

        paper_id = pdf_path.stem

        chunks = chunker.chunk_text(text, paper_id)
        chunk_ids = [
            f"{paper_id}_chunk_{i:04d}"
            for i in range(len(chunks))
        ]

        return chunks, chunk_ids

    except Exception as e:
        print(f"[ERROR] Failed to process {pdf_path.name}: {e}")
        return [], []


def ingest_pdfs_enhanced(
    collection_name: str = "research_papers",
    chunk_size: int = 1000,
    overlap: int = 150
) -> Dict[str, Any]:
    """
    Main ingestion pipeline.
    """
    try:
        pdf_files = validate_environment()
    except Exception as e:
        print(f"[FATAL] {e}")
        return {"error": str(e)}

    print("=" * 60)
    print(f"Starting ingestion of {len(pdf_files)} PDF(s)")
    print("=" * 60)

    chunker = ResearchPaperChunker(
        chunk_size=chunk_size,
        overlap=overlap
    )

    all_chunks = []
    all_ids = []

    processed = 0
    failed = 0

    for idx, pdf_path in enumerate(pdf_files, start=1):
        print(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_path.name}")

        chunks, ids = process_single_pdf(pdf_path, chunker)

        if chunks:
            all_chunks.extend(chunks)
            all_ids.extend(ids)
            processed += 1
            print(f"  ✔ Created {len(chunks)} chunks")
        else:
            failed += 1
            print(f"  ✖ No chunks created")

    if not all_chunks:
        print("[ERROR] No chunks generated from any PDF")
        return {"error": "No chunks generated"}

    print("\n" + "-" * 60)
    print(f"Storing {len(all_chunks)} chunks in ChromaDB...")
    print("-" * 60)

    store_documents_enhanced(
        chunks=all_chunks,
        ids=all_ids,
        collection_name=collection_name
    )


    stats = {
        "total_pdfs": len(pdf_files),
        "processed_pdfs": processed,
        "failed_pdfs": failed,
        "total_chunks": len(all_chunks),
        "avg_chunks_per_pdf": len(all_chunks) / processed if processed else 0,
    }

    print("\nIngestion Summary")
    print("-" * 60)
    for k, v in stats.items():
        print(f"{k.replace('_', ' ').title()}: {v}")

    return stats


# Backward compatibility
def ingest_pdfs():
    from .preprocess import chunk_text
    from .embed_store import store_documents

    all_chunks = []
    ids = []
    idx = 0

    pdf_files = list_pdfs(str(PDF_DIR))

    for pdf in pdf_files:
        print(f"Processing: {Path(pdf).name}")
        text = extract_text_from_pdf(pdf)
        chunks = chunk_text(text, chunk_size=800, overlap=120)

        for chunk in chunks:
            all_chunks.append(chunk)
            ids.append(f"legacy_{idx:04d}")
            idx += 1

    store_documents(all_chunks, ids)


if __name__ == "__main__":
    print("=" * 60)
    print("AI Research Literature Assistant - Ingestion")
    print("=" * 60)

    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        print("Running legacy ingestion...")
        ingest_pdfs()
    else:
        ingest_pdfs_enhanced()
