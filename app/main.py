"""LG Product Semantic Search API — FastAPI app and route definitions."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config import get_settings
from app.ingestion.indexer import get_product_index
from app.models.schemas import (
    HealthResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.search.explainer import explain_results
from app.search.query_parser import parse_query
from app.search.reranker import rerank
from app.search.retriever import hybrid_search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load search indexes on startup."""
    index = get_product_index()
    loaded = index.load_from_disk()

    if loaded:
        logger.info("Search indexes loaded: %d products", index.product_count)
    else:
        logger.warning(
            "No indexes found. Run ingestion first: python -m scripts.ingest"
        )
    yield


app = FastAPI(
    title="LG Product Semantic Search",
    description="AI-powered natural language search for LG products",
    version="0.1.0",
    lifespan=lifespan,
)


# ---- Routes ----


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    index = get_product_index()
    return HealthResponse(
        status="healthy",
        product_count=index.product_count,
        index_loaded=index.is_loaded,
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Semantic product search.

    Accepts a natural language query and returns the top matching products
    with relevance scores and AI-generated explanations.
    """
    start = time.time()
    settings = get_settings()
    index = get_product_index()

    if not index.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Search index not loaded. Run: python -m scripts.ingest",
        )

    query = request.query.strip()
    logger.info("Search query: %s", query)

    # 1. Parse query into structured filters + semantic intent
    parsed = parse_query(query)

    # 2. Hybrid retrieval (vector + BM25 + RRF fusion)
    candidates = hybrid_search(parsed, n_results=settings.search_top_k)

    if not candidates:
        return SearchResponse(
            query=query, parsed_query=parsed, results=[], total_results=0
        )

    # 3. Hydrate candidates with product data
    hydrated = []
    for doc_id, rrf_score in candidates:
        product = index.get_product(doc_id)
        if product:
            hydrated.append({
                "model_id": doc_id,
                "product_name": product["product_name"],
                "sku": product["sku"],
                "price": product["price"],
                "product_type": product.get("product_type", "unknown"),
                "rrf_score": rrf_score,
            })

    # 4. Re-rank
    ranked = rerank(query=query, candidates=hydrated, top_k=settings.final_results)

    # 5. Generate explanations
    explained = explain_results(query=query, results=ranked)

    # 6. Build response
    results = [
        SearchResult(
            product_name=r["product_name"],
            sku=r["sku"],
            price=r["price"],
            relevance_score=r["relevance_score"],
            explanation=r["explanation"],
        )
        for r in explained
    ]

    elapsed = time.time() - start
    logger.info("Search completed in %.2fs, returning %d results", elapsed, len(results))

    return SearchResponse(
        query=query,
        parsed_query=parsed,
        results=results,
        total_results=len(results),
    )
