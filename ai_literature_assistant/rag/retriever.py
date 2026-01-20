import os
import sys
import chromadb
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

# Define project root relative to this file
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# Ensure project root is in Python path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class RAGRetriever:
    """Core RAG retriever for querying research papers from ChromaDB"""

    # Initializes ChromaDB client, collection, and embedding model
    def __init__(
        self,
        chroma_db_path: str = None,
        collection_name: str = "research_papers",
        embedding_model: str = "all-mpnet-base-v2",
    ):
        if chroma_db_path is None:
            chroma_db_path = os.path.join(PROJECT_ROOT, "data", "chroma_db")

        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        self.client = None
        self.collection = None
        self.embedding_model = None

        print(f"Looking for ChromaDB at: {self.chroma_db_path}")
        self._initialize()

    # Connects to ChromaDB and loads the embedding model
    def _initialize(self):
        print("Initializing RAG Retriever...")

        if not os.path.exists(self.chroma_db_path):
            raise FileNotFoundError(
                f"ChromaDB directory not found: {self.chroma_db_path}"
            )

        try:
            self.client = chromadb.PersistentClient(
                path=self.chroma_db_path
            )

            collections = self.client.list_collections()
            collection_names = [c.name for c in collections]
            print(f"Available collections: {collection_names}")

            if self.collection_name not in collection_names:
                raise ValueError(
                    f"Collection '{self.collection_name}' not found. "
                    f"Available: {collection_names}"
                )

            self.collection = self.client.get_collection(
                name=self.collection_name
            )

            print(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name
            )

            embedding_dim = (
                self.embedding_model.get_sentence_embedding_dimension()
            )

            print("RAG Retriever initialized successfully")
            print(f"Database: {self.chroma_db_path}")
            print(f"Collection: {self.collection_name}")
            print(f"Documents: {self.collection.count()}")
            print(f"Embedding dimension: {embedding_dim}")

        except Exception as e:
            print(f"Initialization failed: {e}")
            raise

    # Retrieves top-k relevant documents for a given query
    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        filter_by: Optional[Dict] = None,
        include_metadata: bool = True,
    ) -> Dict:
        try:
            query_embedding = self.embedding_model.encode(
                [query]
            ).tolist()

            query_kwargs = {
                "query_embeddings": query_embedding,
                "n_results": min(n_results, self.collection.count()),
            }

            if filter_by:
                query_kwargs["where"] = filter_by

            include = ["documents", "distances"]
            if include_metadata:
                include.append("metadatas")

            query_kwargs["include"] = include

            results = self.collection.query(**query_kwargs)
            return self._format_results(query, results)

        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "results": [],
                "success": False,
            }

    # Formats raw ChromaDB results into a structured response
    def _format_results(self, query: str, raw_results: Dict) -> Dict:
        if not raw_results or not raw_results.get("ids"):
            return {
                "query": query,
                "results": [],
                "total": 0,
                "success": True,
            }

        formatted = []

        for i in range(len(raw_results["ids"][0])):
            item = {
                "id": raw_results["ids"][0][i],
                "content": raw_results["documents"][0][i],
                "distance": raw_results["distances"][0][i],
                "similarity_score": 1
                - raw_results["distances"][0][i],
            }

            if raw_results.get("metadatas"):
                item["metadata"] = raw_results["metadatas"][0][i]

            formatted.append(item)

        scores = [r["similarity_score"] for r in formatted]

        return {
            "query": query,
            "results": formatted,
            "total": len(formatted),
            "average_similarity": float(np.mean(scores))
            if scores
            else 0,
            "min_similarity": float(min(scores)) if scores else 0,
            "max_similarity": float(max(scores)) if scores else 0,
            "success": True,
        }

    # Builds a concatenated context string from retrieved documents
    def get_context_for_query(
        self,
        query: str,
        n_results: int = 5,
        max_context_length: int = 4000,
    ) -> str:
        results = self.retrieve(query, n_results)

        if not results["results"]:
            return "No relevant documents found."

        context = []
        total_len = 0

        for i, r in enumerate(results["results"], 1):
            meta = r.get("metadata", {})
            source = " | ".join(
                f"{k}: {v}"
                for k, v in meta.items()
                if v not in ("unknown", None)
            )

            block = f"[Document {i} | {source}]\n{r['content']}"

            if total_len + len(block) > max_context_length:
                break

            context.append(block)
            total_len += len(block)

        return "\n\n---\n\n".join(context)

    # Returns statistics about the ChromaDB collection
    def get_collection_stats(self) -> Dict:
        total_docs = self.collection.count()
        sample_size = min(100, total_docs)
        sample = self.collection.peek(limit=sample_size)

        chunk_types = {}
        if sample.get("metadatas"):
            for m in sample["metadatas"]:
                if m and "chunk_type" in m:
                    chunk_types[m["chunk_type"]] = (
                        chunk_types.get(m["chunk_type"], 0) + 1
                    )

        return {
            "total_documents": total_docs,
            "chunk_types": chunk_types,
            "sample_size": sample_size,
        }
