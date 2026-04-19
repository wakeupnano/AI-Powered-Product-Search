"""Embedding generation, ChromaDB vector store, and BM25 keyword index."""

import json
import logging
import re
from pathlib import Path
from typing import Optional, Protocol

from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.ingestion.parser import (
    build_enriched_document,
    build_metadata,
    classify_product,
    parse_csv,
)

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25: lowercase, strip punctuation/symbols, split."""
    text = text.lower()
    text = re.sub(r"[™®©°ᶲ""''\"',.:;!?(){}\[\]/\\|—–-]", " ", text)
    return [t for t in text.split() if len(t) > 1]


# ---- Embedding Providers ----


class EmbeddingProvider(Protocol):
    def embed_documents(self, documents: list[str]) -> list[list[float]]: ...
    def embed_query(self, query: str) -> list[float]: ...


class SentenceTransformerProvider:
    """Production embeddings using sentence-transformers (runs locally)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        logger.info("Loading sentence-transformers model: %s", model_name)
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return self._model.encode(documents, show_progress_bar=True).tolist()

    def embed_query(self, query: str) -> list[float]:
        return self._model.encode(query).tolist()


class TfidfEmbeddingProvider:
    """Fallback embeddings using TF-IDF + SVD. No downloads needed."""

    def __init__(self, n_components: int = 256):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(max_features=8000, stop_words="english")
        self._svd_class = TruncatedSVD
        self._svd = None
        self._fitted = False
        self._n_components = n_components

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        tfidf = self._vectorizer.fit_transform(documents)
        n_comp = min(self._n_components, tfidf.shape[1] - 1)
        self._svd = self._svd_class(n_components=n_comp)
        embeddings = self._svd.fit_transform(tfidf)
        self._fitted = True
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        if not self._fitted:
            raise RuntimeError("Must call embed_documents before embed_query")
        tfidf = self._vectorizer.transform([query])
        return self._svd.transform(tfidf)[0].tolist()

    def save(self, path: Path) -> None:
        import pickle

        with open(path / "tfidf_provider.pkl", "wb") as f:
            pickle.dump(
                {"vectorizer": self._vectorizer, "svd": self._svd, "fitted": self._fitted},
                f,
            )

    def load(self, path: Path) -> bool:
        import pickle

        pkl_path = path / "tfidf_provider.pkl"
        if not pkl_path.exists():
            return False
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self._vectorizer = data["vectorizer"]
        self._svd = data["svd"]
        self._fitted = data["fitted"]
        return True


def create_embedding_provider() -> EmbeddingProvider:
    """Create the best available embedding provider."""
    settings = get_settings()
    try:
        return SentenceTransformerProvider(settings.embedding_model)
    except Exception as e:
        logger.warning(
            "sentence-transformers unavailable (%s), falling back to TF-IDF", e
        )
        return TfidfEmbeddingProvider()


# ---- Product Index ----


class ProductIndex:
    """Manages vector and keyword indexes for product search."""

    def __init__(self):
        self._settings = get_settings()
        self._embedding_provider: Optional[EmbeddingProvider] = None
        self._chroma_client = None
        self._collection = None
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_doc_ids: list[str] = []
        self._products: dict[str, dict] = {}

    @property
    def is_loaded(self) -> bool:
        return self._collection is not None and self._bm25_index is not None

    @property
    def product_count(self) -> int:
        if self._collection is None:
            return 0
        return self._collection.count()

    def get_product(self, model_id: str) -> Optional[dict]:
        return self._products.get(model_id)

    def get_embedding_provider(self) -> EmbeddingProvider:
        if self._embedding_provider is None:
            self._embedding_provider = create_embedding_provider()
        return self._embedding_provider

    def _get_collection(self):
        if self._collection is None:
            import chromadb

            persist_dir = self._settings.chroma_persist_dir
            logger.info("Connecting to ChromaDB at: %s", persist_dir)
            self._chroma_client = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._chroma_client.get_or_create_collection(
                name=self._settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ---- Ingestion ----

    def build_index(self, catalog_path: str) -> int:
        """Full ingestion pipeline: parse → enrich → embed → index.

        Returns the number of products indexed.
        """
        # 1. Parse
        logger.info("Parsing product catalog: %s", catalog_path)
        products = parse_csv(catalog_path)
        if not products:
            logger.error("No products found in catalog")
            return 0

        # 2. Enrich
        logger.info("Building enriched documents for %d products", len(products))
        documents = [build_enriched_document(p) for p in products]
        logger.info("Sample document:\n%s", documents[0][:300])

        # 3. Embed
        provider = self.get_embedding_provider()
        logger.info("Generating embeddings...")
        embeddings = provider.embed_documents(documents)

        # 4. Build metadata + IDs
        metadatas = [build_metadata(p) for p in products]
        ids = [p["model_id"] for p in products]
        product_records = [
            {
                "model_id": p["model_id"],
                "sku": p["sku"],
                "product_name": p["product_name"],
                "price": p["price"],
                "categories": p["categories"],
                "product_type": classify_product(p["categories"]),
            }
            for p in products
        ]

        # 5. Load into ChromaDB
        collection = self._get_collection()
        existing = collection.count()
        if existing > 0:
            logger.info("Clearing %d existing documents", existing)
            collection.delete(ids=collection.get()["ids"])

        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            collection.upsert(
                ids=ids[i:end],
                documents=documents[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
            )
        logger.info("Loaded %d products into ChromaDB", len(ids))

        # 6. Build BM25 index
        tokenized_docs = [tokenize(doc) for doc in documents]
        self._bm25_index = BM25Okapi(tokenized_docs)
        self._bm25_doc_ids = ids
        self._products = {p["model_id"]: p for p in product_records}
        logger.info("Built BM25 index over %d documents", len(ids))

        # 7. Persist auxiliary data
        self._save_auxiliary_data(tokenized_docs)
        if hasattr(provider, "save"):
            provider.save(Path(self._settings.chroma_persist_dir))

        logger.info("Ingestion complete: %d products indexed", len(products))
        return len(products)

    # ---- Search ----

    def vector_search(
        self,
        query_embedding: list[float],
        n_results: int = 20,
        where: Optional[dict] = None,
    ) -> list[tuple[str, float]]:
        """Returns (id, distance) pairs. Lower distance = more similar."""
        collection = self._get_collection()

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, max(collection.count(), 1)),
        }
        if where:
            kwargs["where"] = where

        try:
            results = collection.query(**kwargs)
        except Exception as e:
            logger.warning("Vector search failed with filter %s: %s", where, e)
            return []

        return list(zip(results["ids"][0], results["distances"][0]))

    def keyword_search(
        self,
        query: str,
        n_results: int = 20,
    ) -> list[tuple[str, float]]:
        """Returns (id, score) pairs. Higher score = more relevant."""
        if self._bm25_index is None:
            return []

        tokenized_query = tokenize(query)
        scores = self._bm25_index.get_scores(tokenized_query)

        scored = sorted(
            zip(self._bm25_doc_ids, scores), key=lambda x: x[1], reverse=True
        )
        return [(doc_id, score) for doc_id, score in scored[:n_results] if score > 0]

    # ---- Persistence ----

    def _save_auxiliary_data(self, tokenized_docs: list[list[str]]) -> None:
        persist_dir = Path(self._settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "doc_ids": self._bm25_doc_ids,
            "tokenized_docs": tokenized_docs,
            "products": self._products,
        }
        with open(persist_dir / "auxiliary_data.json", "w") as f:
            json.dump(data, f)
        logger.info("Saved auxiliary data to %s", persist_dir)

    def load_from_disk(self) -> bool:
        """Load existing indexes from disk. Returns True if successful."""
        persist_dir = Path(self._settings.chroma_persist_dir)
        aux_path = persist_dir / "auxiliary_data.json"

        if not persist_dir.exists() or not aux_path.exists():
            return False

        try:
            collection = self._get_collection()
            if collection.count() == 0:
                return False

            with open(aux_path, "r") as f:
                data = json.load(f)

            self._bm25_doc_ids = data["doc_ids"]
            self._bm25_index = BM25Okapi(data["tokenized_docs"])
            self._products = data["products"]

            # Restore embedding provider if saved
            provider = self.get_embedding_provider()
            if hasattr(provider, "load"):
                provider.load(persist_dir)

            logger.info("Loaded indexes from disk: %d products", collection.count())
            return True

        except Exception as e:
            logger.warning("Failed to load from disk: %s", e)
            return False


# Singleton
_index: Optional[ProductIndex] = None


def get_product_index() -> ProductIndex:
    global _index
    if _index is None:
        _index = ProductIndex()
    return _index
