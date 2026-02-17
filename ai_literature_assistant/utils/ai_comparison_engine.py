import json
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_comparison_insight(papers_metadata):
    """
    papers_metadata: list of dicts from PostgreSQL
    """

    try:
        system_prompt = """
You are an AI research analyst.

Compare multiple research papers based on:
- research problem
- methodology
- dataset
- evaluation metrics
- key results
- contributions
- limitations
- future work

Return analysis in MARKDOWN format with clear headings.

Structure:

## Executive Summary
## Key Differences
## Methodology Comparison
## Dataset Comparison
## Performance Insights
## Strengths vs Weaknesses
## Research Gap Identified
## Future Research Directions
"""

        user_prompt = f"""
Compare the following research papers:

{json.dumps(papers_metadata[:5], indent=2)}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI comparison failed: {str(e)}"
