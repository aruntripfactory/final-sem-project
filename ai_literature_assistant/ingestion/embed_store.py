# ingestion/embed_store.py
import os
from typing import List, Dict, Any, Union
from sentence_transformers import SentenceTransformer
import chromadb

# Import the TextChunk dataclass if your chunks might be this type
# from .preprocess import TextChunk  # Uncomment if you have this import

# Location for chroma persistence (absolute)
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
CHROMA_DIR = os.path.abspath(CHROMA_DIR)

_MODEL = None

def get_embedding_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-mpnet-base-v2")
    return _MODEL


def create_client():
    """
    Create a Chroma client using the new PersistentClient API.
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client

def store_documents_enhanced(chunks: List[Union[Dict[str, Any], Any]], ids: List[str], 
                           collection_name: str = "research_papers",
                           embedding_model = None):
    """
    Store documents with metadata in ChromaDB.
    Handles both dictionaries AND TextChunk dataclass objects.
    
    Args:
        chunks: List of dictionaries OR TextChunk objects
        ids: List of document IDs
        collection_name: Name of the collection
        embedding_model: Optional pre-loaded SentenceTransformer model
    """
    if not chunks:
        print("No documents to store.")
        return
    
    # Extract texts and metadata
    texts = []
    metadatas = []
    
    for chunk in chunks:
        # Check if it's a TextChunk dataclass (has .text and .metadata attributes)
        if hasattr(chunk, 'text') and hasattr(chunk, 'metadata'):
            # Handle TextChunk object
            texts.append(chunk.text)
            metadatas.append(chunk.metadata if hasattr(chunk.metadata, 'get') else {})
        # Check if it's a dictionary
        elif isinstance(chunk, dict) and 'text' in chunk:
            texts.append(chunk['text'])
            metadatas.append(chunk.get('metadata', {}))
        else:
            # Fallback - convert to string
            texts.append(str(chunk))
            metadatas.append({})
            print(f"Warning: Unexpected chunk type: {type(chunk)}")
    
    # Load embedding model if not provided
    model = embedding_model if embedding_model else get_embedding_model()

    
    # Create embeddings
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    ).tolist()

    
    # Create client and collection
    client = create_client()
    
    # Create collection with metadata schema
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Add documents with metadata
    collection.upsert(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    # Print statistics
    metadata_types = {}
    for metadata in metadatas:
        if metadata and isinstance(metadata, dict) and 'chunk_type' in metadata:
            chunk_type = metadata['chunk_type']
            metadata_types[chunk_type] = metadata_types.get(chunk_type, 0) + 1
    
    print(f"Stored {len(texts)} documents in collection '{collection_name}'")
    if metadata_types:
        print("Metadata distribution:")
        for chunk_type, count in metadata_types.items():
            print(f"  - {chunk_type}: {count}")
    
    return collection

def store_documents(docs: List[Union[str, Dict[str, Any]]], ids: List[str], 
                   collection_name: str = "research_papers"):
    """
    Backward compatible function for storing documents.
    """
    # Convert to enhanced format
    chunks = []
    for doc in docs:
        if isinstance(doc, dict):
            chunks.append(doc)
        else:
            chunks.append({"text": str(doc), "metadata": {}})
    
    return store_documents_enhanced(chunks, ids, collection_name)