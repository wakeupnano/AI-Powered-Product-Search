"""Product catalog parsing: CSV ingestion, spec flattening, and document enrichment.

This module handles the Extract and Transform (ET) phases of the data pipeline.
It reads dirty CSV data, safely parses stringified JSON, removes noisy attributes,
and constructs semantically rich prose documents optimized for vector embeddings.
"""

import csv
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Domain Specific Filters
# Generic tags that don't help with search filtering
_GENERIC_CATEGORIES = {
    "b2c", "appliances", "collections", "home_appliances", "gift_guide",
    "best_sellers_gift_guide", "gifts_under_1000", "gifts_under_500",
    "cleaning_gift_guide", "reliable_appliances", "lifestyle_products",
}

# Spec groups most useful for search, by product domain
_TV_SPEC_GROUPS = {
    "Picture (Panel)", "Picture (Processing)", "Picture Quality",
    "Gaming", "Audio", "Smart TV", "Connectivity", "Power", "General",
}

_LAUNDRY_SPEC_GROUPS = {
    "Laundry Appliance Capacity", "Capacity", "Energy",
    "Washer Programs", "Dryer Programs", "Programs", "Motor",
    "Laundry Appliance Convenience Features", "Convenience Features",
    "Fabric Care Features", "Fabric Care",
    "Laundry Appliance ThinQ® Smart Features", "Summary",
}

# Spec attributes to skip (noise that doesn't help search)
_SKIP_SPEC_NAMES = {
    "UPC", "Ratings", "E-Manual", "Quick Start Guide",
    "Power Cable", "Remote Control Battery",
}


def parse_csv(filepath: str) -> list[dict]:
    """Parse the headerless product CSV into structured dicts.

    Uses defensive programming to gracefully skip malformed rows rather
    than crashing the entire ingestion process.
    """
    products = []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 7:
                logger.warning("Skipping malformed row: %s fields", len(row))
                continue

            try:
                categories = json.loads(row[3])
                specs = json.loads(row[5])
                features = json.loads(row[6])
            except json.JSONDecodeError as e:
                logger.warning("Skipping row %s: JSON parse error: %s", row[0], e)
                continue

            products.append({
                "model_id": row[0],
                "sku": row[1],
                "product_name": row[2],
                "categories": categories,
                "price": float(row[4]),
                "specs": specs,
                "features": features,
            })

    logger.info("Parsed %d products from %s", len(products), filepath)
    return products


def classify_product(categories: list[str]) -> str:
    """Determine the primary product type from category tags."""
    cat_set = set(categories)

    if cat_set & {"tvs", "tv_and_home_theater", "oled_tvs", "qned_tvs", "4k_uhd_tvs", "hd_tvs"}:
        return "tv"
    if cat_set & {"styler", "styler_steam_closets"}:
        return "styler"
    if cat_set & {"laundry_pedestals", "laundry_accessories", "appliances_accessories"}:
        return "accessory"
    return "laundry"


def get_meaningful_categories(categories: list[str]) -> list[str]:
    """Filter out generic tags, return human-readable category names."""
    meaningful = [c for c in categories if c not in _GENERIC_CATEGORIES]
    return [c.replace("_", " ") for c in meaningful]


def flatten_specs(specs: dict[str, Any], product_type: str) -> dict[str, str]:
    """Extract key spec name-value pairs from the nested spec JSON."""
    relevant_groups = _TV_SPEC_GROUPS if product_type == "tv" else _LAUNDRY_SPEC_GROUPS
    flat = {}

    for _key, group in specs.items():
        group_name = group.get("spec_group_name", "")

        if group_name not in relevant_groups:
            # Still grab universally useful fields from any group
            for attr in group.get("spec_attributes", []):
                name = attr.get("spec_name", "")
                if name in {"Capacity", "ENERGY STAR Certified", "Wi-Fi"}:
                    flat[name] = attr.get("spec_value", "")
            continue

        for attr in group.get("spec_attributes", []):
            name = attr.get("spec_name", "")
            value = attr.get("spec_value", "")

            if name in _SKIP_SPEC_NAMES:
                continue
            if not value or value.lower() == "no":
                continue

            flat[name] = value

    return flat


def get_sorted_features(features: list[dict]) -> list[str]:
    """Extract bullet features sorted by priority."""
    sorted_feats = sorted(features, key=lambda f: int(f.get("priority", 99)))
    return [f["bullet_feature"] for f in sorted_feats if f.get("bullet_feature")]


def build_enriched_document(product: dict) -> str:
    """Build a natural language document for embedding and keyword search.

    This is the core of search quality. The enriched text reads like a product
    description, blending name, category, price, specs, and features into a
    coherent paragraph that embeds well.
    """
    name = product["product_name"]
    sku = product["sku"]
    price = product["price"]
    categories = get_meaningful_categories(product["categories"])
    product_type = classify_product(product["categories"])
    specs = flatten_specs(product["specs"], product_type)
    features = get_sorted_features(product["features"])

    parts = []

    # Opening: name + SKU + category context
    cat_str = ", ".join(categories) if categories else "LG product"
    parts.append(f"{name} (SKU: {sku}). Category: {cat_str}. Price: ${price:,.2f}.")

    # Price tier (helps with queries like "budget TV" or "premium washer")
    if price < 500:
        parts.append("This is a budget-friendly option.")
    elif price < 1000:
        parts.append("This is a mid-range option.")
    elif price < 2000:
        parts.append("This is a premium option.")
    else:
        parts.append("This is a high-end, top-of-the-line option.")

    # Key specs as readable text
    if specs:
        spec_lines = [f"{k}: {v}" for k, v in specs.items()]
        parts.append("Specifications: " + "; ".join(spec_lines) + ".")

    # Bullet features, filtering out promotional text
    if features:
        search_relevant = [
            f for f in features
            if not any(
                kw in f.lower()
                for kw in ["warranty", "thinq offer", "terms apply", "through 12/31"]
            )
        ]
        if search_relevant:
            parts.append("Key features: " + ". ".join(search_relevant) + ".")

    return " ".join(parts)


def build_metadata(product: dict) -> dict:
    """Build ChromaDB metadata for filtering. Values must be simple types."""
    product_type = classify_product(product["categories"])
    categories = product["categories"]

    # Determine the most specific category for filtering
    primary_category = product_type
    if product_type == "tv":
        if "oled_tvs" in categories:
            primary_category = "oled_tvs"
        elif "qned_tvs" in categories:
            primary_category = "qned_tvs"
        elif "4k_uhd_tvs" in categories:
            primary_category = "4k_uhd_tvs"
        else:
            primary_category = "tvs"
    elif product_type == "laundry":
        if "washers" in categories and "dryers" not in categories:
            primary_category = "washers"
        elif "dryers" in categories and "washers" not in categories:
            primary_category = "dryers"
        elif "washer_dryer_combos" in categories:
            primary_category = "washer_dryer_combos"
        elif "washtower" in categories:
            primary_category = "washtower"
        elif "styler" in categories:
            primary_category = "styler"
        else:
            primary_category = "washers_and_dryers"

    return {
        "price": product["price"],
        "product_type": product_type,
        "primary_category": primary_category,
        "sku": product["sku"],
        "product_name": product["product_name"],
        "categories_json": json.dumps(product["categories"]),
    }
