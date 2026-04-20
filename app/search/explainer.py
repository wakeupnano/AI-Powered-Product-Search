"""Generate natural language explanations for why each result matches the query."""

import json
import logging
import re
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


def explain_results(
    query: str,
    results: list[dict],
) -> list[dict]:
    """Add an 'explanation' field to each result.

    Uses Claude to generate grounded explanations. Falls back to
    template-based explanations if the API is unavailable.

    Args:
        query: Original user query.
        results: Product dicts with relevance_score already set.

    Returns:
        Same list with 'explanation' field added to each result.
    """
    explained = _explain_with_llm(query, results)
    if explained is not None:
        return explained

    return _fallback_explain(query, results)


def _explain_with_llm(
    query: str,
    results: list[dict],
) -> Optional[list[dict]]:
    """Use Claude to generate per-result explanations."""
    settings = get_settings()

    if not settings.anthropic_api_key:
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

        product_lines = []
        for i, r in enumerate(results):
            product_lines.append(
                f"[{i}] {r['product_name']} | ${r['price']:,.2f} | "
                f"Type: {r.get('product_type', 'unknown')}"
            )

        products_text = "\n".join(product_lines)

        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=600,
            messages=[
                {
                    "role": "user",
                    "content": f"""A customer searched for: "{query}"

These products were selected as matches:
{products_text}

For each product, write a 1-2 sentence explanation of why it matches the query.
Reference specific product features relevant to the query.

Respond with ONLY a JSON array:
[{{"index": 0, "explanation": "This product matches because..."}}, ...]""",
                }
            ],
        )

        text = response.content[0].text.strip()
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        explanations = json.loads(text)

        output = []
        for r in results:
            product = r.copy()
            product["explanation"] = ""
            output.append(product)

        for entry in explanations:
            idx = entry.get("index", -1)
            if 0 <= idx < len(output):
                output[idx]["explanation"] = entry.get("explanation", "")

        logger.info("LLM generated %d explanations", len(explanations))
        return output

    except Exception as e:
        logger.warning("LLM explanation failed: %s", e)
        return None


def _fallback_explain(query: str, results: list[dict]) -> list[dict]:
    """Template-based explanations when LLM is unavailable."""
    output = []

    for r in results:
        product = r.copy()

        price = r["price"]
        if price < 500:
            tier = "Budget-friendly"
        elif price < 1000:
            tier = "Mid-range"
        elif price < 2000:
            tier = "Premium"
        else:
            tier = "High-end"

        product_type = r.get("product_type", "product")
        product["explanation"] = (
            f"{tier} {product_type} at ${price:,.0f}. "
            f"Matched based on relevance to your search for \"{query}\"."
        )
        output.append(product)

    return output
