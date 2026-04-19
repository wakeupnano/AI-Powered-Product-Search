"""CLI entry point for product catalog ingestion.

Usage:
    python -m scripts.ingest
    python -m scripts.ingest --catalog path/to/catalog.csv
"""

import argparse
import logging
import sys
import time

from app.config import get_settings
from app.ingestion.indexer import get_product_index


def main():
    parser = argparse.ArgumentParser(description="Ingest LG product catalog")
    parser.add_argument(
        "--catalog",
        type=str,
        default=None,
        help="Path to product catalog CSV (default: from .env)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    catalog_path = args.catalog or settings.product_catalog_path

    start = time.time()
    index = get_product_index()
    count = index.build_index(catalog_path)
    elapsed = time.time() - start

    if count == 0:
        print("ERROR: No products ingested. Check the catalog path and format.")
        sys.exit(1)

    print(f"\nDone: {count} products ingested in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
