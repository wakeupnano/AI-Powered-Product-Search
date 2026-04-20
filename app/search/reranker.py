"""LLM-based re-ranking of search candidates against the original query."""

import json
import logging
import re
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Re-rank candidates using Claude to score relevance against the query.

    Falls back to pass-through ordering if the API is unavailable.

    Args:
        query: Original user query.
        candidates: Product dicts from hybrid search.
        top_k: Number of results to return.

    Returns:
        Re-ordered list of product dicts with 'relevance_score' added.
    """
    result = _rerank_with_llm(query, candidates, top_k)
    if result is not None:
        return result

    return _fallback_rank(candidates, top_k)


def _rerank_with_llm(
    query: str,
    candidates: list[dict],
    top_k: int,
) -> Optional[list[dict]]:
    """Use Claude to score and re-order candidates."""
    settings = get_settings()

    if not settings.anthropic_api_key:
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

        # Build concise product summaries
        product_summaries = []
        for i, c in enumerate(candidates[: top_k * 2]):
            summary = (
                f"[{i}] {c['product_name']} | SKU: {c['sku']} | "
                f"${c['price']:,.2f} | Type: {c.get('product_type', 'unknown')}"
            )
            product_summaries.append(summary)

        products_text = "\n".join(product_summaries)

        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": f"""A customer searched for: "{query}"

Candidate products:
{products_text}

Select the {top_k} most relevant products. Respond with ONLY a JSON array:
[{{"index": 0, "relevance_score": 0.95}}, ...]

Order by relevance. Score 0.0-1.0 based on how well each matches the query.""",
                }
            ],
        )

        text = response.content[0].text.strip()
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        rankings = json.loads(text)

        results = []
        for entry in rankings[:top_k]:
            idx = entry["index"]
            if idx < len(candidates):
                product = candidates[idx].copy()
                product["relevance_score"] = min(max(entry["relevance_score"], 0.0), 1.0)
                results.append(product)

        logger.info("LLM re-ranked %d results", len(results))
        return results

    except Exception as e:
        logger.warning("LLM re-ranking failed: %s", e)
        return None


def _fallback_rank(candidates: list[dict], top_k: int) -> list[dict]:
    """Pass-through ranking using RRF order with synthetic scores."""
    results = []
    for i, c in enumerate(candidates[:top_k]):
        product = c.copy()
        product["relevance_score"] = round(0.9 - (i * 0.08), 2)
        results.append(product)
    return results
