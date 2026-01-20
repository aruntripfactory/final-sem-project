import os
import sys
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ResponseGenerator:
    """Generate responses using an LLM with retrieved RAG context"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")

        self.client = OpenAI(api_key=api_key)

    def generate_response(
        self,
        query: str,
        context: str,
        system_prompt: str = None,
    ) -> Dict:

        if system_prompt is None:
            system_prompt = (
                "You are an expert AI research assistant. "
                "Answer strictly using the provided research paper context only. "
                "If the answer is not present, say so clearly. "
                "Cite sources using document numbers and paper IDs."
            )

        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"RESEARCH CONTEXT:\n{context}\n\nQUESTION: {query}",
                    },
                ],
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )

            usage = response.usage

            return {
                "response": response.output_text,
                "model": self.model,
                "tokens_used": usage.total_tokens if usage else 0,
                "success": True,
            }

        except Exception as e:
            return {
                "response": "",
                "error": str(e),
                "success": False,
            }

    def generate_from_retrieved_docs(
        self,
        query: str,
        retrieved_docs: List[Dict],
        max_context_length: int = 4000,
    ) -> Dict:

        context_parts = []
        total_length = 0

        for i, doc in enumerate(retrieved_docs, 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            source_info = []
            if metadata.get("paper_id"):
                source_info.append(f"Paper: {metadata['paper_id']}")
            if metadata.get("chunk_type"):
                source_info.append(metadata["chunk_type"])
            if metadata.get("section") and metadata["section"] != "unknown":
                source_info.append(f"Section: {metadata['section']}")

            doc_block = (
                f"[Document {i}: {' | '.join(source_info)}]\n{content}"
            )

            if total_length + len(doc_block) > max_context_length:
                break

            context_parts.append(doc_block)
            total_length += len(doc_block)

        context = "\n\n---\n\n".join(context_parts)

        return self.generate_response(query, context)
