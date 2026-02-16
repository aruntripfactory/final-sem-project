# ai_literature_assistant/config.py
import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
TEST_RESULTS_DIR = os.path.join(PROJECT_ROOT, "test_results")
QUERY_RESULTS_DIR = os.path.join(PROJECT_ROOT, "query_results")

# ChromaDB settings
COLLECTION_NAME = "research_papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# OpenAI settings (model only - no API key here!)
OPENAI_MODEL = "gpt-4o-mini"  
OPENAI_TEMPERATURE = 0.3
OPENAI_MAX_TOKENS = 1500

# RAG settings
MAX_RETRIEVAL_RESULTS = 25
MAX_CONTEXT_LENGTH = 12000
MIN_SIMILARITY_THRESHOLD = 0.3

# Flask settings (non-sensitive)
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

# Create directories if they don't exist
for directory in [DATA_DIR, CHROMA_DIR, PDF_DIR, TEST_RESULTS_DIR, QUERY_RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)
    # print(f"Directory ensured: {directory}")