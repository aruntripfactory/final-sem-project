# ai_literature_assistant/rag/pipeline.py
import os
import sys
from datetime import datetime
from typing import Dict, Optional, List

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    MAX_RETRIEVAL_RESULTS,
)
from rag.retriever import RAGRetriever
from rag.response_generator import ResponseGenerator


class RAGPipeline:
    """Complete RAG pipeline from query to response"""

    # Initializes retriever and response generator using config settings
    def __init__(
        self,
        openai_model: str = OPENAI_MODEL,
        temperature: float = OPENAI_TEMPERATURE,
    ):
        print("Initializing RAG Pipeline...")

        self.retriever = RAGRetriever()

        try:
            self.chatgpt = ResponseGenerator(
                model=openai_model,
                temperature=temperature,
            )
            self.use_chatgpt = True
        except Exception as e:
            print(f"ChatGPT initialization failed: {e}")
            print("Falling back to retrieval-only mode")
            self.use_chatgpt = False

        print("RAG Pipeline ready")
        print(f"ChatGPT enabled: {self.use_chatgpt}")
        print(f"Model: {openai_model}")
        print(f"Documents in database: {self.retriever.collection.count()}")

    # Executes full RAG flow: retrieval, generation, and response formatting
    def query(
        self,
        question: str,
        n_retrieve: int = MAX_RETRIEVAL_RESULTS,
        filter_by: Optional[Dict] = None,
        include_sources: bool = True,
        generate_response: bool = True,
    ) -> Dict:

        print(f"\nProcessing query: {question}")

        retrieval_results = self.retriever.retrieve(
            query=question,
            n_results=n_retrieve,
            filter_by=filter_by,
            include_metadata=True,
        )

        if retrieval_results.get("error"):
            return {
                "query": question,
                "error": retrieval_results["error"],
                "success": False,
                "timestamp": datetime.now().isoformat(),
            }

        print(f"Documents retrieved: {retrieval_results['total']}")

        if (
            generate_response
            and self.use_chatgpt
            and retrieval_results["total"] > 0
        ):
            chatgpt_response = self.chatgpt.generate_from_retrieved_docs(
                query=question,
                retrieved_docs=retrieval_results["results"],
            )

            answer = chatgpt_response.get("response", "")
            response_success = chatgpt_response.get("success", False)
        else:
            answer = self._create_simple_summary(
                question, retrieval_results["results"]
            )
            response_success = True
            chatgpt_response = {"model": "retrieval-only"}

        response_data = {
            "query": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "success": response_success and retrieval_results["total"] > 0,
            "model": chatgpt_response.get("model", "retrieval-only"),
            "retrieval": {
                "documents_retrieved": retrieval_results["total"],
                "average_similarity": retrieval_results.get(
                    "average_similarity", 0
                ),
            },
        }

        if include_sources and retrieval_results["total"] > 0:
            sources = []
            for doc in retrieval_results["results"]:
                src = {
                    "id": doc["id"],
                    "similarity": f"{doc['similarity_score']:.1%}",
                    "content_preview": doc["content"][:200] + "...",
                }

                if doc.get("metadata"):
                    m = doc["metadata"]
                    src["metadata"] = {
                        "chunk_type": m.get("chunk_type", "unknown"),
                        "paper_id": m.get("paper_id", "unknown"),
                        "section": m.get("section", "unknown"),
                        "source": m.get("source", "unknown"),
                    }

                sources.append(src)

            response_data["sources"] = sources

        print("Response generated successfully")
        return response_data

    # Creates a basic text summary when LLM generation is disabled or unavailable
    def _create_simple_summary(
        self, query: str, documents: List[Dict]
    ) -> str:
        summary = [
            f"Based on {len(documents)} documents related to '{query}':\n"
        ]

        for i, doc in enumerate(documents, 1):
            meta = doc.get("metadata", {})
            paper_id = meta.get("paper_id", f"Paper {i}")
            chunk_type = meta.get("chunk_type", "content")
            section = meta.get("section", "")

            summary.append(f"\n[{paper_id}] ({chunk_type})")
            if section and section != "unknown":
                summary.append(f"Section: {section}")

            content = doc["content"]
            preview = (
                ". ".join(content.split(".")[:2]) + "."
                if "." in content
                else content[:150] + "..."
            )
            summary.append(preview)

        return "\n".join(summary)

    # Runs a query with metadata-based filters applied
    def query_with_filters(
        self,
        question: str,
        chunk_type: str = None,
        paper_id: str = None,
        section: str = None,
    ) -> Dict:
        filter_by = {}
        if chunk_type:
            filter_by["chunk_type"] = chunk_type
        if paper_id:
            filter_by["paper_id"] = paper_id
        if section:
            filter_by["section"] = section

        return self.query(
            question,
            filter_by=filter_by if filter_by else None,
        )
