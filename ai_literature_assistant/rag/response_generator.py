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
                "Do not include in-text citations, document numbers, or paper IDs in your response. "
                "When listing items (like paper titles), provide a clean list with bullet points only. "
                "Just provide the answer directly."
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"RESEARCH CONTEXT:\n{context}\n\nQUESTION: {query}",
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            usage = response.usage

            return {
                "response": response.choices[0].message.content,
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
        # Detect Intent
        query_lower = query.lower()
        
        is_compare = any(k in query_lower for k in ["compare", "difference", " vs ", "versus", "comparison", "table"])
        is_diagram = any(k in query_lower for k in ["diagram", "figure", "image", "chart", "visual", "illustration", "picture"])
        is_summary = any(k in query_lower for k in ["summarize", "summary", "overview", "explain"])
        is_all_papers = any(k in query_lower for k in ["all papers", "every paper", "each paper", "uploaded papers"])

        # Construct System Prompt based on Intent
        system_prompt = (
            "You are an expert AI research assistant. "
            "Answer strictly using the provided research paper context only. "
            "If the answer is not present, say so clearly. "
            "Do not include in-text citations like (Author, Year) or [1]. "
        )

        if is_compare:
            system_prompt += (
                "\n\nFORMATTING REQUIREMENT: The user has asked for a comparison. "
                "You MUST present the answer as a Markdown table. "
                "The table columns should represent the different papers or approaches being compared. "
                "The rows should represent the criteria or features being compared. "
                "Ensure the table is well-structured and easy to read. "
                "After the table, you may provide a brief textual synthesis."
            )
        elif is_diagram:
             system_prompt += (
                "\n\nFORMATTING REQUIREMENT: The user is asking about diagrams, figures, or images. "
                "Check the context for descriptions of figures (chunk_type='image_description'). "
                "For each relevant figure found in the context, describe it clearly and mention which paper it belongs to. "
                "If the context contains image paths or figure numbers, reference them. "
                "If no figures are found in the context, explicitly state that no visual information was retrieved."
            )
        elif is_summary or is_all_papers:
            system_prompt += (
                "\n\nFORMATTING REQUIREMENT: The user wants a summary or explanation covering the uploaded papers. "
                "Address EACH paper mentioned in the context separately if possible, followed by a synthesis. "
                "Use distinct headers (### Paper Title) or bullet points for each paper. "
                "Ensure you cover *all* papers provided in the context, not just one. "
                "If the context only contains information about one paper, state that you only have information on that one."
            )
        else:
            system_prompt += (
                "\n\nProvide a direct, comprehensive answer. "
                "If the context includes multiple papers, synthesize the information across them. "
                "Use bullet points for lists."
            )

        context_parts = []
        total_length = 0

        # Enhance context formatting to include more metadata for the LLM
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # Extract key metadata
            paper_id = metadata.get("paper_id", "Unknown Paper")
            chunk_type = metadata.get("chunk_type", "text")
            section = metadata.get("section", "unknown")
            image_path = metadata.get("image_path", "")
            
            source_info = [f"Paper ID: {paper_id}", f"Type: {chunk_type}"]
            
            if section != "unknown":
                source_info.append(f"Section: {section}")
            
            # Special handling for image descriptions to make them obvious to the LLM
            if chunk_type == "image_description":
                doc_block = f"[IMAGE DESCRIPTION from {paper_id}, Image Path: {image_path}]\n{content}"
            else:
                doc_block = f"[Document {i} | {' | '.join(source_info)}]\n{content}"

            if total_length + len(doc_block) > max_context_length:
                break

            context_parts.append(doc_block)
            total_length += len(doc_block)

        context = "\n\n---\n\n".join(context_parts)

        return self.generate_response(query, context, system_prompt)
