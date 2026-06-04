# Solution Overview

Multimodal product similarity search on the Amazon Fashion dataset (30k products).
Given a product `uniq_id`, returns the most similar products ranked by a weighted combination of text, image, and structured features.

| Part | Deliverable | Path |
|------|-------------|------|
| **Part 1** | `find_similar_products` function | `src/sap_cxii_tech_ex_01/search.py` |
| **Part 2** | FastAPI microservice (`GET /find_similar_products`) | `src/sap_cxii_tech_ex_01/api/` |
| **Part 2** | Dockerfile (K8s-ready) | `Dockerfile` |
| **Part 2** | Kubernetes manifests | `k8s/` |
| **Part 3** | Vector search optimizations (FAISS HNSW, ScaNN, TurboQuant, Two-Stage) | `src/sap_cxii_tech_ex_01/vector_index.py`, `src/sap_cxii_tech_ex_01/backends/`, `src/sap_cxii_tech_ex_01/retrieval.py` |

---

## 1. Setup & Running

### Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — fast Python package manager

### Install Dependencies

```bash
cd sap-cxii-tech-ex-01
uv sync                     # creates .venv/ and installs all deps
cp .env.example .env         # copy default configuration
```

### Run Locally

```bash
# Option A — shell script (recommended)
./start.sh                   # boots API at http://localhost:8000
./start.sh --reload          # with auto-reload for development

# Option B — uv directly
uv run uvicorn sap_cxii_tech_ex_01.api.app:app --host 0.0.0.0 --port 8000

# Option C — CLI entry point
uv run sap-cxii-tech-ex-01
```

### Run with Docker

```bash
./start_container.sh                    # builds image + runs container
./start_container.sh --build-only       # build only, don't run
SAP_API_PORT=9000 ./start_container.sh  # custom host port
```

The Dockerfile uses a multi-stage build (`python:3.12-slim`), runs as non-root `appuser`, and exposes port 8000.

### API Usage

Once running, visit **http://localhost:8000/docs** for the interactive Swagger UI.

```bash
# Find 5 similar products
curl "http://localhost:8000/find_similar_products?product_id=<UNIQ_ID>&num_similar=5"

# Health / readiness probes
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

### Run Tests

```bash
# Full suite (153 tests — unit, integration, regression)
uv run pytest

# Verbose with short tracebacks
uv run pytest -v --tb=short

# Run a specific test layer
uv run pytest tests/unit/
uv run pytest tests/integration/
uv run pytest tests/regression/

# Run a single file
uv run pytest tests/unit/test_search.py -v

# With coverage (if pytest-cov installed)
uv run pytest --cov=sap_cxii_tech_ex_01
```

All tests run without the real dataset or network access — fixtures inject controlled data via patching or direct `SearchState` construction.

---

## 2. Data & Design Decisions

### Data Exploration

The 30k-row Amazon Fashion dataset was profiled in [`notebooks/data_exploration.ipynb`](notebooks/data_exploration.ipynb). Key findings are documented in [`docs/data_quality_assumptions.md`](docs/data_quality_assumptions.md):

- **5 columns dropped** (`weight`, `discount_percentage`, `left_in_stock`, `no__of_reviews`, `colour`) — all >60% missing
- **284 rows dropped** (>50% remaining columns null after column pruning)
- `sales_price` and `rating` are the usable numeric features; `brand`, `delivery_type`, `amazon_prime`, `best_seller_tag` are categorical
- `weight` column is 100% sentinel value `999999999` — entirely unusable

### Architectural Decisions

Each non-trivial choice is recorded in a Decision Record (`decisions/` directory). Summary:

| Decision | Why |
|----------|-----|
| Column/row drop thresholds (60%/50%) | Low signal-to-noise in sparse columns; 284 rows lost is <1% |
| sklearn-style `fit`/`transform` Preprocessor | Stateful transforms (fitted scalers/encoders) need persist across fit and transform |
| Centralized `pydantic-settings` config | Single `.env` file; no hardcoded values; testable via `Settings(...)` override |
| Three independent extractors (Text, Image, Structured) | Each modality is independently testable; zero-vector fallback for missing inputs |
| `all-MiniLM-L6-v2` for text embeddings | 384-dim semantic embeddings; captures "Blue Denim Jacket" ≈ "Navy Jean Coat" unlike TF-IDF |
| EfficientNet-B0 for image features | 1280-dim visual embeddings; classifier head removed for transfer learning |
| Weighted concatenation (0.4/0.3/0.3) | Simplest multimodal fusion; configurable weights; fallback weights when images unavailable |
| Module-boundary patching for API tests | Lifespan patches inject test fixtures without loading real models |
| `VectorIndex` Protocol (build/query/save/load) | Any ANN backend is swappable without touching calling code |
| HNSW over IVF | Near-perfect recall without tuning; no training phase; O(log n) query |
| PCA over t-SNE/UMAP | Preserves global Euclidean structure needed for ANN; t-SNE is non-invertible |
| In-process LRU cache over Redis | Zero dependencies; sub-microsecond cache hits; sufficient for single-process deployment |
| ScaNN → FAISS HNSW graceful fallback | ScaNN is Linux-only; fallback + warning keeps portability |
| ScaNN as optional high-throughput backend | 5–9× QPS over HNSW at 100k+ scale via anisotropic vector quantization |
| TurboQuant for extreme compression | 4-bit: 12.5% memory; 2-bit: 6.2% memory; no codebook training required |
| Two-stage retrieval (ANN over-fetch + exact rerank) | Combines fast ANN throughput with near brute-force recall |

For algorithm details and paper references, see [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md).

---

## 3. Solution Strengths

### 1. Protocol-Driven Backend Swappability

The `VectorIndex` Protocol defines a 4-method contract (`build`, `query`, `save`, `load`). Four backends (FAISS HNSW, ScaNN, TurboQuant, Two-Stage) implement it interchangeably — switching from HNSW to ScaNN at scale is a one-line config change, not a rewrite.

### 2. Test-Driven Development with Layered Coverage

153 tests across three layers (unit → integration → regression) written **before** implementation (TDD red-green-refactor). Tests never depend on the real dataset, network access, or ML model loading — all use fixtures or direct `SearchState` construction. Regression tests lock recall tolerances per backend.

### 3. Centralized, Type-Safe Configuration

Every tunable parameter (model names, weights, thresholds, ports) lives in a single `Settings` class backed by `.env` + `pydantic-settings`. No hardcoded values in source code. Tests override settings via constructor injection — no monkeypatching environment variables.

### 4. Precomputed Feature Matrix at Startup

Feature extraction (sentence-transformers + structured encoding for 30k products) runs **once** during application lifespan startup. Query-time similarity is a single vector × matrix dot product — sub-millisecond response regardless of dataset size. The previous per-request approach was re-extracting all features on every call (~90s).

### 5. Production-Ready API with Observability

FastAPI with structured error handling (400/404/422/500), health/readiness probes for Kubernetes, multi-stage Docker build with non-root user, and Kubernetes manifests (`k8s/`) with ConfigMap for tunable parameters. Startup logs show each extraction phase with shapes and timings.

---

## 4. Performance & Image Feature Notes

### System Performance

| Metric | Value |
|--------|-------|
| **Dataset** | 30,000 Amazon Fashion products |
| **Features used** | Text (384-dim) + Structured (6-dim) = 390-dim combined |
| **Query latency** | <1 ms (precomputed normalized matrix, single BLAS dot product) |
| **Startup time** | ~30–40s (model load + 30k text encoding on GPU) |
| **Test suite** | 153 tests, all passing in ~38s |
| **FAISS HNSW recall@10** | 1.000 (perfect at 30k scale) |
| **ScaNN recall@10** | ≥0.85 (tunable via `num_leaves_to_search_ratio`) |
| **TurboQuant 4-bit recall@10** | ≥0.93 (with 5× over-fetch + exact rerank) |
| **Two-Stage recall@10** | ≥0.98 (approaches brute-force) |
| **TurboQuant memory savings** | 4-bit: 87.5% reduction; 2-bit: 93.8% reduction vs f32 |

Full benchmark data at all scales (30k–1M) is in `benchmarks/backends_results.json`, with analysis in [`docs/ALGORITHMS.md` §6](docs/ALGORITHMS.md).

### Why Image Features Are Built But Skipped

The image pipeline is **fully implemented and tested** — `ImageExtractor` downloads product images, runs EfficientNet-B0, and produces 1280-dim embeddings with in-memory caching and zero-vector fallback for unreachable URLs.

It is **skipped by default** (`SAP_SKIP_IMAGES=true`) for two reasons:

1. **Dead URLs**: The dataset contains Amazon CDN image URLs from 2020 that are largely expired. Probing succeeds intermittently, but downloading all 30k images serially (10s timeout each) would take hours and block startup entirely.
2. **Startup latency**: Even with valid URLs, 30k image downloads + EfficientNet forward passes add ~60–90 minutes to startup — impractical for local development and CI.

When images are skipped, the system automatically uses **fallback weights** (text: 0.6, structured: 0.4) instead of the full multimodal split (text: 0.4, image: 0.3, structured: 0.3). This is handled transparently by `SimilarityEngine.combine()`.

To enable image features with a dataset that has reachable URLs, set `SAP_SKIP_IMAGES=false` in `.env`.
