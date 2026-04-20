"""Hybrid retrieval: vector search + BM25 keyword search fused via RRF."""

import logging

from app.config import get_settings
from app.ingestion.indexer import get_product_index
from app.models.schemas import ParsedQuery
from app.search.query_parser import build_chroma_filter

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for document d = sum(1 / (k + rank_i)) across all lists
    where rank_i is the 1-indexed rank of d in list i.

    Args:
        ranked_lists: Each list is [(doc_id, score)] already sorted by relevance.
        k: Smoothing constant (default 60, standard in IR literature).

    Returns:
        Fused list of (doc_id, rrf_score) sorted descending.
    """
    rrf_scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked_list, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(
    parsed_query: ParsedQuery,
    n_results: int = 20,
) -> list[tuple[str, float]]:
    """Run hybrid search: vector + BM25, fused with RRF.

    Args:
        parsed_query: Structured query with filters and semantic intent.
        n_results: Number of candidates to retrieve from each source.

    Returns:
        Fused list of (doc_id, rrf_score) sorted by relevance.
    """
    settings = get_settings()
    index = get_product_index()
    provider = index.get_embedding_provider()

    chroma_filter = build_chroma_filter(parsed_query)
    semantic_query = parsed_query.semantic_query

    # 1. Vector search (with metadata filters)
    query_embedding = provider.embed_query(semantic_query)
    vector_results = index.vector_search(
        query_embedding=query_embedding,
        n_results=n_results,
        where=chroma_filter,
    )
    logger.info(
        "Vector search: %d results (filter: %s)", len(vector_results), chroma_filter
    )

    # 2. BM25 keyword search (unfiltered — RRF handles rank blending)
    keyword_results = index.keyword_search(
        query=semantic_query,
        n_results=n_results,
    )
    logger.info("BM25 search: %d results", len(keyword_results))

    # 3. If filters returned too few vector results, add unfiltered fallback
    ranked_lists = [vector_results, keyword_results]
    if chroma_filter and len(vector_results) < 5:
        logger.info("Few filtered results, adding unfiltered vector fallback")
        unfiltered = index.vector_search(
            query_embedding=query_embedding,
            n_results=n_results,
        )
        ranked_lists.append(unfiltered)

    # 4. Fuse with RRF
    fused = reciprocal_rank_fusion(ranked_lists, k=settings.rrf_k)
    logger.info("RRF fusion: %d candidates", len(fused))

    return fused
