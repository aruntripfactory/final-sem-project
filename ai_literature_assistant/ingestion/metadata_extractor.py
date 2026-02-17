# ingestion/metadata_extractor.py

import json
import logging
from typing import Dict, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

client = OpenAI()


SYSTEM_PROMPT = """
You are a research paper analysis engine.

Extract structured metadata from the given research paper.

Return ONLY valid JSON.
Do NOT include explanation.
Do NOT include markdown.
Do NOT include text outside JSON.

JSON schema:

{
  "title": "",
  "authors": "",
  "year": "",
  "domain": "",
  "research_problem": "",
  "methodology": "",
  "dataset": "",
  "evaluation_metrics": "",
  "baseline_models": "",
  "key_results": "",
  "contributions": "",
  "limitations": "",
  "future_work": "",
  "keywords": ""
}
"""


def _safe_parse_json(text: str) -> Optional[Dict]:
    """Safely parse model output into JSON."""
    if not text or not text.strip():
        return None

    try:
        return json.loads(text)
    except Exception:
        # Try to recover JSON from markdown-style responses
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end != -1:
                return json.loads(text[start:end])
        except Exception:
            pass

    return None


def extract_paper_metadata(paper_text: str, paper_id: str, file_name: str) -> Optional[Dict]:
    """
    Extract structured metadata from paper text using OpenAI.

    Never crashes ingestion:
    - returns None if extraction fails
    - logs error instead
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},  #forces JSON output
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": paper_text[:12000]}
            ],
        )

        raw = response.choices[0].message.content

        metadata = _safe_parse_json(raw)

        if not metadata:
            logger.warning("[METADATA] Model returned non-JSON output. Skipping.")
            return None

        # attach system fields
        metadata["paper_id"] = paper_id
        metadata["file_name"] = file_name

        return metadata

    except Exception as e:
        logger.warning(f"[METADATA] Extraction failed: {e}")
        return None
