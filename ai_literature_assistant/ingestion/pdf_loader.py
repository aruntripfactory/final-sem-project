# ingestion/pdf_loader.py

import io
import os
from typing import Union, List
from pypdf import PdfReader


def list_pdfs(pdf_dir: str) -> List[str]:
    "Lists all PDF files in a directory."
    pdfs = [
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    ]

    print(f"[PDF_LOADER] Found {len(pdfs)} PDF file(s).")
    return pdfs


def extract_text_from_pdf(pdf_source: Union[str, bytes]) -> str:
    """Accepts a file path or raw bytes. 
    Extracts and returns raw text only."""

    try:
        if isinstance(pdf_source, str):
            reader = PdfReader(pdf_source)
            pdf_name = os.path.basename(pdf_source)
        else:
            reader = PdfReader(io.BytesIO(pdf_source))
            pdf_name = "bytes_input"

        print(f"[PDF_LOADER] Processing: {pdf_name}")
        print(f"[PDF_LOADER] Pages: {len(reader.pages)}")

        pages_text = []
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                # Try standard extraction first
                text = page.extract_text(extraction_mode="layout") or ""
                if not text.strip():
                    # Fallback to plain extraction if layout mode fails
                    text = page.extract_text() or ""
                pages_text.append(text)
            except Exception as e:
                print(f"[PDF_LOADER] Warning: Failed to extract page {page_num}: {e}")
                # Continue with other pages even if one fails
                pages_text.append("")
                continue

        full_text = "\n".join(pages_text)
        
        if not full_text.strip():
            raise ValueError("No text could be extracted from PDF. The PDF might be image-based or corrupted.")
        
        return full_text
        
    except Exception as e:
        print(f"[PDF_LOADER] Error processing PDF: {e}")
        raise ValueError(f"Failed to process PDF: {str(e)}")
