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
    MAX_CONTEXT_LENGTH,
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
        search_mode: str = "Semantic",
        alpha: float = 1.0,
    ) -> Dict:

        print(f"\nProcessing query: {question}")
        
        # --- AUTO-INTELLIGENCE ROUTING ---
        mode = "MULTI_PAPER"
        target_paper_id = None
        
        # 1. Check for Catalog Query (Explicit list requests)
        low_q = question.lower().strip().rstrip('?')
        catalog_keywords = [
            "list papers", "all papers", "list all papers", "what papers", "show papers", 
            "titles", "list of papers", "available papers", "title of the papers", 
            "titles of the papers", "paper titles", "list titles"
        ]
        
        is_catalog_query = any(keyword in low_q for keyword in catalog_keywords) or low_q == "papers"
        
        if is_catalog_query:
            mode = "CATALOG"
        else:
            title_search = self.retriever.retrieve(
                query=question,
                n_results=1,
                filter_by={"chunk_type": "title_reference"}
            )
            
            if title_search["success"] and title_search["results"]:
                top_match = title_search["results"][0]
                similarity = top_match["similarity_score"]
                
                print(f"Auto-Intel: Top title match: '{top_match['content']}' (Score: {similarity:.2f})")
                
                # If high confidence match, assume Single Paper Mode
                if similarity > 0.75:
                    mode = "SINGLE_PAPER"
                    target_paper_id = top_match["metadata"]["paper_id"]
                    print(f"Auto-Intel: Detected SINGLE_PAPER intent for '{top_match['content']}'")

        # --- EXECUTION BASED ON MODE ---
        
        if mode == "CATALOG":
            print(f"Routing to CATALOG retrieval.")
            retrieval_results = self.retriever.retrieve_by_metadata(
                where={"chunk_type": "title_reference"},
                n_results=1000 
            )
            
        elif mode == "SINGLE_PAPER" and target_paper_id:
            print(f"Routing to SINGLE_PAPER retrieval (Paper ID: {target_paper_id}).")
            # Deep dive: Retrieve more chunks, but scoped to this paper
            # Merge with existing filter_by if present
            scoped_filter = filter_by.copy() if filter_by else {}
            scoped_filter["paper_id"] = target_paper_id
            
            retrieval_results = self.retriever.retrieve(
                query=question,
                n_results=15, # Fetch more context for single paper depth
                filter_by=scoped_filter,
                include_metadata=True,
            )
            
        else:
            print(f"Routing to MULTI_PAPER retrieval (Standard).")
            
            # Dynamic Retrieval Adjustment based on Intent
            query_lower = question.lower()
            is_compare = any(k in query_lower for k in ["compare", "difference", " vs ", "versus", "comparison", "table"])
            is_all_papers = any(k in query_lower for k in ["all papers", "every paper", "each paper", "uploaded papers", "summarize"])
            is_diagram = any(k in query_lower for k in ["diagram", "figure", "image", "chart", "visual"])

            # Increase recall for multi-paper tasks
            adjusted_n = n_retrieve
            if is_compare or is_all_papers:
                adjusted_n = 30  # Boost to ensure we get chunks from multiple papers
                print(f"Booster: Increasing retrieval to {adjusted_n} for multi-paper context.")
            
            if is_diagram:
                adjusted_n = 25
                print(f"Booster: Increasing retrieval to {adjusted_n} for diagram search.")
                # We don't strictly filter by image_description because we also want text context about the images
                # But the prompt content handling in response_generator.py will highlight images

            # Standard Semantic Search
            retrieval_results = self.retriever.retrieve(
                query=question,
                n_results=adjusted_n,
                filter_by=filter_by,
                include_metadata=True,
            )
        # ---------------------

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
                max_context_length=MAX_CONTEXT_LENGTH,
            )

            answer = chatgpt_response.get("response", "")
            response_success = chatgpt_response.get("success", False)
            
            if not response_success:
                print(f"ChatGPT generation failed: {chatgpt_response.get('error')}")
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
                    "content_preview": doc["content"][:300] + "...",
                }

                if doc.get("metadata"):
                    m = doc["metadata"]
                    src["metadata"] = {
                        "chunk_type": m.get("chunk_type", "unknown"),
                        "paper_id": m.get("paper_id", "unknown"),
                        "section": m.get("section", "unknown"),
                        "source": m.get("source", "unknown"),
                        "page_start": m.get("page_start", 0),
                        "page_end": m.get("page_end", 0),
                        "image_path": m.get("image_path", ""),
                        "image_hash": m.get("image_hash", ""),
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
