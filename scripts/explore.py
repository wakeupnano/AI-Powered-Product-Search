"""Explore and validate the product catalog CSV.

Run this before ingestion to understand the data shape, spot issues,
and verify the parser's category/spec grouping logic.

Usage:
    python -m scripts.explore
    python -m scripts.explore --catalog path/to/catalog.csv
"""

import argparse
import csv
import json
import logging
import sys
from collections import Counter


def load_rows(filepath: str) -> list[list[str]]:
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.reader(f))


def explore(filepath: str) -> None:
    rows = load_rows(filepath)

    # ---- Basic Structure ----
    print("=" * 60)
    print("CATALOG OVERVIEW")
    print("=" * 60)
    print(f"File: {filepath}")
    print(f"Total rows: {len(rows)}")
    print(f"Columns per row: {len(rows[0]) if rows else 0}")
    print()

    # Check for header
    first_field = rows[0][0] if rows else ""
    has_header = not first_field.startswith("MD")
    if has_header:
        print(f"Header row detected: {rows[0]}")
        data_rows = rows[1:]
    else:
        print("No header row (first field is a model ID)")
        print("Assumed columns: model_id, sku, product_name, categories, price, specs, features")
        data_rows = rows

    # ---- Column Validation ----
    print()
    print("=" * 60)
    print("COLUMN VALIDATION")
    print("=" * 60)
    malformed = [i for i, r in enumerate(data_rows) if len(r) != 7]
    if malformed:
        print(f"WARNING: {len(malformed)} rows have != 7 columns: rows {malformed[:10]}")
    else:
        print("All rows have 7 columns ✓")

    # Check for parse errors
    json_errors = []
    for i, row in enumerate(data_rows):
        if len(row) != 7:
            continue
        for col, idx in [("categories", 3), ("specs", 5), ("features", 6)]:
            try:
                json.loads(row[idx])
            except json.JSONDecodeError:
                json_errors.append((i, col, row[0]))

    if json_errors:
        print(f"WARNING: {len(json_errors)} JSON parse errors:")
        for row_i, col, model_id in json_errors[:5]:
            print(f"  Row {row_i}, column '{col}', model_id: {model_id}")
    else:
        print("All JSON columns parse successfully ✓")

    # ---- Price Distribution ----
    print()
    print("=" * 60)
    print("PRICE DISTRIBUTION")
    print("=" * 60)
    prices = []
    for row in data_rows:
        try:
            prices.append(float(row[4]))
        except (ValueError, IndexError):
            pass

    if prices:
        prices.sort()
        print(f"Min:    ${min(prices):,.2f}")
        print(f"Max:    ${max(prices):,.2f}")
        print(f"Mean:   ${sum(prices)/len(prices):,.2f}")
        print(f"Median: ${prices[len(prices)//2]:,.2f}")
        print()

        # Price buckets
        buckets = {"Under $500": 0, "$500-$999": 0, "$1000-$1999": 0, "$2000-$4999": 0, "$5000+": 0}
        for p in prices:
            if p < 500:
                buckets["Under $500"] += 1
            elif p < 1000:
                buckets["$500-$999"] += 1
            elif p < 2000:
                buckets["$1000-$1999"] += 1
            elif p < 5000:
                buckets["$2000-$4999"] += 1
            else:
                buckets["$5000+"] += 1

        for bucket, count in buckets.items():
            bar = "█" * (count // 2)
            print(f"  {bucket:15s} {count:3d} {bar}")

    # ---- Categories ----
    print()
    print("=" * 60)
    print("ALL CATEGORY TAGS")
    print("=" * 60)
    all_cats = Counter()
    for row in data_rows:
        try:
            cats = json.loads(row[3])
            all_cats.update(cats)
        except (json.JSONDecodeError, IndexError):
            pass

    for cat, count in all_cats.most_common():
        print(f"  {cat:40s} {count:3d}")

    print(f"\n  Total unique tags: {len(all_cats)}")

    # ---- Product Type Classification ----
    print()
    print("=" * 60)
    print("PRODUCT TYPE CLASSIFICATION (what parser.py assigns)")
    print("=" * 60)
    from app.ingestion.parser import classify_product

    type_counts = Counter()
    type_examples = {}
    for row in data_rows:
        try:
            cats = json.loads(row[3])
            ptype = classify_product(cats)
            type_counts[ptype] += 1
            if ptype not in type_examples:
                type_examples[ptype] = []
            if len(type_examples[ptype]) < 3:
                type_examples[ptype].append(row[2][:60])
        except (json.JSONDecodeError, IndexError):
            pass

    for ptype, count in type_counts.most_common():
        print(f"\n  {ptype}: {count} products")
        for example in type_examples.get(ptype, []):
            print(f"    → {example}")

    # ---- Spec Groups ----
    print()
    print("=" * 60)
    print("SPEC GROUPS (all unique group names across products)")
    print("=" * 60)
    spec_groups = Counter()
    for row in data_rows:
        try:
            specs = json.loads(row[5])
            for _key, group in specs.items():
                spec_groups[group.get("spec_group_name", "UNKNOWN")] += 1
        except (json.JSONDecodeError, IndexError):
            pass

    for group, count in spec_groups.most_common():
        print(f"  {group:50s} {count:3d}")

    # ---- Features ----
    print()
    print("=" * 60)
    print("FEATURE BULLETS")
    print("=" * 60)
    feature_counts = []
    empty_features = 0
    promo_features = 0
    for row in data_rows:
        try:
            features = json.loads(row[6])
            feature_counts.append(len(features))
            if not features:
                empty_features += 1
            for f in features:
                text = f.get("bullet_feature", "").lower()
                if any(kw in text for kw in ["warranty", "thinq offer", "terms apply", "through 12/31"]):
                    promo_features += 1
        except (json.JSONDecodeError, IndexError):
            pass

    if feature_counts:
        print(f"  Products with features: {len(feature_counts)}")
        print(f"  Products with 0 features: {empty_features}")
        print(f"  Features per product: {min(feature_counts)}-{max(feature_counts)} (avg {sum(feature_counts)/len(feature_counts):.1f})")
        print(f"  Promotional/warranty bullets (filtered during enrichment): {promo_features}")

    # ---- Missing Data ----
    print()
    print("=" * 60)
    print("MISSING / EMPTY DATA")
    print("=" * 60)
    missing = {"empty_name": 0, "zero_price": 0, "empty_specs": 0, "empty_features": 0, "empty_sku": 0}
    for row in data_rows:
        if len(row) != 7:
            continue
        if not row[2].strip():
            missing["empty_name"] += 1
        if not row[1].strip():
            missing["empty_sku"] += 1
        try:
            if float(row[4]) == 0:
                missing["zero_price"] += 1
        except ValueError:
            missing["zero_price"] += 1
        try:
            if not json.loads(row[5]):
                missing["empty_specs"] += 1
        except json.JSONDecodeError:
            missing["empty_specs"] += 1
        try:
            if not json.loads(row[6]):
                missing["empty_features"] += 1
        except json.JSONDecodeError:
            missing["empty_features"] += 1

    all_clean = True
    for field, count in missing.items():
        if count > 0:
            print(f"  WARNING: {count} products with {field.replace('_', ' ')}")
            all_clean = False
    if all_clean:
        print("  No missing data detected ✓")

    # ---- Sample Enriched Documents ----
    print()
    print("=" * 60)
    print("SAMPLE ENRICHED DOCUMENTS (what gets embedded)")
    print("=" * 60)
    from app.ingestion.parser import build_enriched_document

    samples = [data_rows[0], data_rows[len(data_rows) // 2], data_rows[-1]]
    for row in samples:
        try:
            product = {
                "model_id": row[0],
                "sku": row[1],
                "product_name": row[2],
                "categories": json.loads(row[3]),
                "price": float(row[4]),
                "specs": json.loads(row[5]),
                "features": json.loads(row[6]),
            }
            doc = build_enriched_document(product)
            print(f"\n  [{product['sku']}] {product['product_name'][:50]}")
            print(f"  {doc[:200]}...")
        except Exception as e:
            print(f"  Error building document for row: {e}")

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Explore the product catalog CSV")
    parser.add_argument(
        "--catalog",
        type=str,
        default="./data/lg_products.csv",
        help="Path to product catalog CSV",
    )
    args = parser.parse_args()
    explore(args.catalog)


if __name__ == "__main__":
    main()
