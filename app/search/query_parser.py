"""Query understanding: extract structured filters and semantic intent from natural language."""

import json
import logging
import re
from typing import Optional

from app.config import get_settings
from app.models.schemas import ParsedQuery

logger = logging.getLogger(__name__)

# Category keywords mapped to product_type metadata values
_CATEGORY_KEYWORDS = {
    "tv": "tv",
    "television": "tv",
    "oled": "tv",
    "qned": "tv",
    "screen": "tv",
    "washer": "laundry",
    "washing machine": "laundry",
    "dryer": "laundry",
    "laundry": "laundry",
    "washtower": "laundry",
    "styler": "styler",
    "steam closet": "styler",
}


def _parse_with_llm(query: str) -> Optional[ParsedQuery]:
    """Use Claude to extract structured filters from a natural language query."""
    settings = get_settings()

    if not settings.anthropic_api_key:
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": f"""Extract structured search filters from this product search query.

Query: "{query}"

Respond with ONLY a JSON object (no markdown, no explanation):
{{
  "semantic_query": "<the core search intent, without price/category constraints>",
  "category": "<one of: tv, washer, dryer, laundry, washtower, styler, or null if unclear>",
  "min_price": <number or null>,
  "max_price": <number or null>
}}

Rules:
- "semantic_query" should capture the user's intent in searchable terms
- Only set category if the query clearly specifies a product type
- Extract price bounds from phrases like "under $2000", "between $500 and $1000", "around $1500"
- For "around" prices, set min to 20% below and max to 20% above
- If no price is mentioned, set both to null""",
                }
            ],
        )

        text = response.content[0].text.strip()
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        data = json.loads(text)
        return ParsedQuery(
            semantic_query=data.get("semantic_query", query),
            category=data.get("category"),
            min_price=data.get("min_price"),
            max_price=data.get("max_price"),
        )

    except Exception as e:
        logger.warning("LLM query parsing failed: %s", e)
        return None


def _parse_with_regex(query: str) -> ParsedQuery:
    """Fallback parser using regex for price extraction and keyword matching."""
    q_lower = query.lower()

    min_price = None
    max_price = None

    # "under $X" / "below $X" / "less than $X"
    match = re.search(r"(?:under|below|less than|cheaper than|max)\s*\$?([\d,]+)", q_lower)
    if match:
        max_price = float(match.group(1).replace(",", ""))

    # "over $X" / "above $X" / "more than $X"
    match = re.search(r"(?:over|above|more than|at least|min)\s*\$?([\d,]+)", q_lower)
    if match:
        min_price = float(match.group(1).replace(",", ""))

    # "between $X and $Y"
    match = re.search(r"between\s*\$?([\d,]+)\s*(?:and|-)\s*\$?([\d,]+)", q_lower)
    if match:
        min_price = float(match.group(1).replace(",", ""))
        max_price = float(match.group(2).replace(",", ""))

    # "around $X"
    match = re.search(r"around\s*\$?([\d,]+)", q_lower)
    if match:
        center = float(match.group(1).replace(",", ""))
        min_price = center * 0.8
        max_price = center * 1.2

    # Detect category
    category = None
    for keyword, cat in _CATEGORY_KEYWORDS.items():
        if keyword in q_lower:
            category = cat
            break

    # Build semantic query by stripping price phrases
    semantic = re.sub(
        r"(?:under|below|less than|over|above|more than|at least|around|between)"
        r"\s*\$?[\d,]+(?:\s*(?:and|-)\s*\$?[\d,]+)?",
        "",
        query,
        flags=re.IGNORECASE,
    )
    semantic = re.sub(r"\$[\d,]+", "", semantic)
    semantic = re.sub(r"\s+", " ", semantic).strip()

    if not semantic:
        semantic = query

    return ParsedQuery(
        semantic_query=semantic,
        category=category,
        min_price=min_price,
        max_price=max_price,
    )


def parse_query(query: str) -> ParsedQuery:
    """Parse a natural language query into structured filters.

    Tries LLM first, falls back to regex if unavailable.
    """
    result = _parse_with_llm(query)
    if result:
        logger.info("LLM parsed query: %s", result.model_dump())
        return result

    result = _parse_with_regex(query)
    logger.info("Regex parsed query: %s", result.model_dump())
    return result


def build_chroma_filter(parsed: ParsedQuery) -> Optional[dict]:
    """Convert parsed query filters into a ChromaDB where clause."""
    conditions = []

    if parsed.category:
        conditions.append({"product_type": {"$eq": parsed.category}})

    if parsed.min_price is not None:
        conditions.append({"price": {"$gte": parsed.min_price}})

    if parsed.max_price is not None:
        conditions.append({"price": {"$lte": parsed.max_price}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
