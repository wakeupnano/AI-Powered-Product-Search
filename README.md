# LG Product Semantic Search

An AI-powered search system that lets users find LG products using natural language queries with a sample product catalog CSV. The system understands intent, extracts filters, and explains why each result matches. 

## Features

- **Semantic Retrieval:** Uses high-dimensional text embeddings to understand the contextual meaning of queries.
- **Intelligent Query Parsing:** Leverages an LLM as a "Query Router" to dynamically extract hard filters (like price maximums and product categories) from natural language.
- **Hybrid Search Capabilities:** Combines metadata filtering with vector similarity to guarantee accurate results (e.g., strictly enforcing a budget while finding the most semantically relevant items).
- **Explainable AI:** Generates short, dynamic explanations for *why* a specific product matches the user's query.
- **Production-Ready:** Containerized with Docker, complete with input validation (Pydantic), error handling, and a health-check endpoint.

## Architecture

1. **Data Ingestion & Preprocessing:** 
2. **Embedding & Storage:** 
3. **Query Understanding:** 
    - Claude Sonnet
    - Structured Filters (price, category)
    - Refined Search
4. **Hybrid Retrieval:** 
    - ChromaDB (Vector)
    - BM25 (Keyword)
    - RRF Fusion
5. Reranking & Explanation Generation
    - Claude Sonnet 
6. **Result Generation:** 
    - name, sku, price, relevance_score, explanation  

### Pipeline Steps

1. **Query Understanding** — Claude parses the natural language query into structured filters (price bounds, category) and a refined semantic query. *"OLED TV under $3000"* becomes `{category: "tv", max_price: 3000, semantic_query: "OLED TV"}`. Falls back to regex extraction when no API key is configured.

2. **Hybrid Retrieval** — Two retrieval paths run together:
   - **Vector search** (ChromaDB + `all-MiniLM-L6-v2`) finds semantically similar products with metadata filters applied at the database level.
   - **Keyword search** (BM25) catches exact matches that semantic search misses — model numbers, feature names like "TurboSteam," or technical specs.
   - **Reciprocal Rank Fusion (RRF)** combines both ranked lists without needing score normalization.

3. **Re-ranking** — Claude re-scores the fused candidates against the original query, reordering by true relevance rather than vector distance alone.

4. **Explanation** — Claude generates a short, grounded explanation for each result referencing the product's actual specs and features.

## Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Embedding model | `all-MiniLM-L6-v2` | Runs locally, no API key needed. Quality is strong at this catalog size; document enrichment matters more than model size. |
| Vector store | ChromaDB | Zero-infrastructure, native metadata filtering for price/category, persists to disk. |
| Hybrid search | Vector + BM25 + RRF | Vector fails on model numbers; keywords fail on intent. RRF fusion is simple and well-established. |
| Query understanding | Claude Sonnet | Structured extraction from natural language is fragile with regex. Falls back to regex when unavailable. |
| Document enrichment | Template-based | Deterministic, fast, reproducible. LLM-generated enrichment would be richer but adds cost and non-determinism. |


All LLM steps gracefully degrade to rule-based fallbacks when no API key is set, so the system is fully functional without one.

## Dataset

219 LG products across three product types:

| Type | Count | Examples |
|------|-------|----------|
| Laundry | 151 | Front-load washers, top-load washers, dryers (gas & electric), WashTowers, combos, pedestals |
| TVs | 59 | OLED, OLED evo, QNED, QNED evo, 4K UHD |
| Stylers | 6 | Steam closets |
| Accessories | 3 | Pedestal storage drawers |

Price range: $40–$24,900 (median $949). No header row in CSV. Specs are deeply nested JSON with 47 unique spec group names.


## Setup

### Prerequisites
- Python 3.10+
- Anthropic API key (optional — system works without it using fallbacks)

### Install

```bash
git clone https://github.com/your-username/lg-product-search.git
cd lg-product-search

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY (optional)
```

### Ingest & Run

```bash
# Build the search indexes from the product catalog
python -m scripts.ingest

# Start the API server
uvicorn app.main:app --port 8000

# Or with auto-reload during development (exclude .venv to avoid reload loops)
uvicorn app.main:app --reload --reload-dir app --reload-dir scripts --port 8000
```

You can also browse the interactive API docs at **http://localhost:8000/docs** once the server is running.

### Usage

```bash
# Health check
curl http://localhost:8000/health

# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "best TV for gaming in a dark room"}'

# More example queries
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "quiet washer under $1000"}'

curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "energy efficient dryer between $800 and $1500"}'

curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "OLED77M3YUA"}'
```

### Response Format

```json
{
  "query": "best TV for gaming in a dark room",
  "parsed_query": {
    "semantic_query": "TV for gaming dark room",
    "category": "tv",
    "min_price": null,
    "max_price": null
  },
  "results": [
    {
      "product_name": "LG 65-Inch Class OLED evo C4 Series TV",
      "sku": "OLED65C4PUA",
      "price": 1799.99,
      "relevance_score": 0.92,
      "explanation": "This OLED evo TV excels in dark rooms thanks to self-lit pixels that produce perfect blacks. Its 144Hz refresh rate and low input lag make it highly responsive for gaming."
    }
  ],
  "total_results": 5
}
```
## Project Structure

```
lg-product-search/
├── app/
│   ├── main.py              # FastAPI app, route definitions, startup
│   ├── config.py            # Settings, env vars, model paths
│   ├── ingestion/
│   │   ├── parser.py        # CSV parsing, spec JSON flattening, enrichment
│   │   └── indexer.py       # Embedding generation, ChromaDB + BM25 storage
│   ├── search/
│   │   ├── query_parser.py  # LLM-powered query understanding (+ regex fallback)
│   │   ├── retriever.py     # Hybrid vector + BM25 retrieval with RRF fusion
│   │   ├── reranker.py      # LLM-based re-ranking
│   │   └── explainer.py     # Result explanation generation
│   └── models/
│       └── schemas.py       # Pydantic request/response models
├── data/
│   └── lg_products.csv      # Product catalog (219 products, no header)
├── scripts/
│   └── ingest.py            # Build search indexes from catalog
├── .env.example             # Environment variable template
├── .gitignore
├── requirements.txt         # Pinned Python dependencies
└── README.md
```

## Tech Stack

- **Python 3.11+**
- **FastAPI** — async web framework with auto-generated OpenAPI docs
- **ChromaDB** — vector database with metadata filtering