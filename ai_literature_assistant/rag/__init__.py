# ai_literature_assistant/rag/__init__.py
from .retriever import RAGRetriever
from .response_generator import ResponseGenerator
from .pipeline import RAGPipeline

# Optionally, you can alias RAGRetriever as Retriever for backward compatibility
Retriever = RAGRetriever

__all__ = ['RAGRetriever', 'Retriever', 'ResponseGenerator', 'RAGPipeline']