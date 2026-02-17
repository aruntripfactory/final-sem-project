import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

# 🔥 IMPORTANT: use shared Chroma client
from ingestion.embed_store import get_shared_client, get_embedding_model


class RAGRetriever:
    """Core RAG retriever for querying research papers from ChromaDB"""

    def __init__(
        self,
        collection_name: str = "research_papers",
    ):
        self.collection_name = collection_name

        print("Initializing RAG Retriever...")

        # ✅ Use shared Chroma client (singleton)
        self.client = get_shared_client()

        # ✅ Always get or create collection safely
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # ✅ Use shared embedding model (no reload)
        self.embedding_model = get_embedding_model()

        print("RAG Retriever initialized successfully")
        print(f"Collection: {self.collection_name}")
        print(f"Documents: {self.collection.count()}")

    # ─────────────────────────────────────────────────────────────
    # Retrieval
    # ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        filter_by: Optional[Dict] = None,
        include_metadata: bool = True,
    ) -> Dict:
        try:
            query_embedding = self.embedding_model.encode([query]).tolist()

            query_kwargs = {
                "query_embeddings": query_embedding,
                "n_results": min(n_results, max(self.collection.count(), 1)),
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

    # ─────────────────────────────────────────────────────────────

    def retrieve_by_metadata(
        self,
        where: Dict,
        n_results: int = 100,
    ) -> Dict:
        try:
            results = self.collection.get(
                where=where,
                limit=n_results,
                include=["documents", "metadatas"],
            )

            formatted = []
            if results["ids"]:
                for i in range(len(results["ids"])):
                    formatted.append({
                        "id": results["ids"][i],
                        "content": results["documents"][i],
                        "metadata": results["metadatas"][i],
                        "similarity_score": 1.0,
                        "distance": 0.0,
                    })

            return {
                "query": f"metadata_filter={where}",
                "results": formatted,
                "total": len(formatted),
                "success": True,
            }

        except Exception as e:
            return {
                "error": str(e),
                "results": [],
                "total": 0,
                "success": False,
            }

    # ─────────────────────────────────────────────────────────────

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
                "similarity_score": 1 - raw_results["distances"][0][i],
            }

            if raw_results.get("metadatas"):
                item["metadata"] = raw_results["metadatas"][0][i]

            formatted.append(item)

        scores = [r["similarity_score"] for r in formatted]

        return {
            "query": query,
            "results": formatted,
            "total": len(formatted),
            "average_similarity": float(np.mean(scores)) if scores else 0,
            "min_similarity": float(min(scores)) if scores else 0,
            "max_similarity": float(max(scores)) if scores else 0,
            "success": True,
        }

    # ─────────────────────────────────────────────────────────────

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
