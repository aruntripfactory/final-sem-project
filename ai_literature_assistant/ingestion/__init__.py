# ingestion package exports (clean, minimal, stable)

from .ingest_all import ingest_uploaded_files, CHROMA_DIR, IngestResult
from .embed_store import get_shared_client
