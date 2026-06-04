# Algorithms & References

This document describes the algorithms used in the product similarity search pipeline, their complexity characteristics, and the scaling strategy for handling larger datasets.

---

## 1. Similarity Search Pipeline

The pipeline consists of three stages:

1. **Feature Extraction** — text (sentence-transformers), image (EfficientNet-B0), structured (StandardScaler + LabelEncoder)
2. **Embedding Compression** — PCA dimensionality reduction
3. **Approximate Nearest Neighbor Search** — FAISS HNSW graph index with LRU result caching

### 1.1 Feature Extraction

| Modality | Model | Output Dim | Technique |
|----------|-------|-----------|-----------|
| Text | `all-MiniLM-L6-v2` | 384 | Transformer sentence embeddings |
| Image | EfficientNet-B0 | 1,280 | CNN feature extraction (classifier head removed) |
| Structured | StandardScaler + LabelEncoder | ~6 | Numeric scaling + categorical encoding |

Modality embeddings are weighted and concatenated into a single combined vector (~390d with text+structured, ~1670d with images). Weights are configurable via `Settings`.

### 1.2 Similarity Metric

**Cosine similarity** is used for ranking in the brute-force baseline. FAISS HNSW uses **L2 distance** internally, which is equivalent to cosine similarity on L2-normalized vectors (since $\|a - b\|^2 = 2 - 2 \cos(a, b)$ when $\|a\| = \|b\| = 1$).

---

## 2. Dimensionality Reduction — PCA

**Algorithm**: Principal Component Analysis (PCA) via singular value decomposition.

**Purpose**: Reduce embedding dimensionality before indexing to decrease memory usage and speed up distance computation.

**Configuration**:
- Text+structured embeddings: 390 → 128 dimensions
- With image embeddings: 1670 → 128–256 dimensions
- Variance preservation target: ≥95%

**Complexity**:
- Fit: $O(n \cdot d^2)$ where $n$ = number of vectors, $d$ = input dimension
- Transform: $O(n \cdot d \cdot d')$ where $d'$ = output dimension

**Why PCA over alternatives**:
- **vs t-SNE/UMAP**: PCA preserves global Euclidean structure (critical for nearest-neighbor search). t-SNE and UMAP preserve local manifold structure and are designed for 2D/3D visualization — they are non-linear, non-invertible, and non-deterministic.
- **vs Matryoshka Representation Learning (MRL)**: MRL embeddings (Kusupati et al., 2022) can be truncated to any prefix length without quality loss, eliminating the PCA step entirely. However, this requires an MRL-compatible embedding model. Our current model (`all-MiniLM-L6-v2`) is not MRL-trained, so PCA is the correct approach. If the embedding model is revisited, an MRL-compatible model (e.g., `nomic-embed-text-v1.5`) would be preferred.

**Trade-off**: ~1–3% recall loss vs full-dimensional search, in exchange for 3–4x speedup and reduced memory.


---

## 3. Approximate Nearest Neighbor Search — HNSW

**Algorithm**: Hierarchical Navigable Small World graphs.

**Reference**: Malkov & Yashunin (2018) "Efficient and robust approximate nearest neighbor using Hierarchical Navigable Small World graphs" — [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)

**Implementation**: FAISS `IndexHNSWFlat` with parameters:
- `M = 32` (neighbors per layer)
- `efConstruction = 200` (build-time search depth)
- `efSearch = 128` (query-time search depth)

### 3.1 How HNSW Works

HNSW constructs a multi-layer proximity graph where:
1. The bottom layer contains all vectors connected to their $M$ nearest neighbors.
2. Upper layers contain geometrically decreasing subsets of vectors, acting as "express lanes" for navigation.
3. Search starts at the topmost layer and greedily descends, using each layer to narrow the search region before refining on the next layer down.

This structure provides $O(\log n)$ expected search complexity — each layer halves the search space, similar to a skip list.

### 3.2 Complexity Analysis

| Operation | Brute Force | HNSW |
|-----------|------------|------|
| Build | $O(n)$ | $O(n \log n)$ |
| Query | $O(n \cdot d)$ | $O(\log n \cdot d)$ |
| Memory | $O(n \cdot d)$ | $O(n \cdot (d + M))$ |

At our dataset sizes, the practical impact is:

| N | Brute-Force p50 | HNSW p50 | Speedup |
|---|----------------|----------|---------|
| 10,000 | 3.00 ms | 0.13 ms | 23x |
| 30,000 | 5.98 ms | 0.13 ms | 47x |
| 100,000 | 25.89 ms | 0.28 ms | 93x |
| 500,000 | — | 0.80 ms | — |
| 1,000,000 | — | 0.89 ms | — |

*Data from `benchmarks/scaling_results.json` — 128-dim vectors, k=10, 20 queries per scale.*

The key observation: **brute-force scales linearly** (3ms → 26ms for 10x growth), while **HNSW scales sublogarithmically** (0.13ms → 0.89ms for 100x growth). At 1M vectors, HNSW maintains sub-millisecond queries.

### 3.3 Why HNSW over IVF

- Near-perfect recall (≥0.99) on datasets <1M without parameter tuning
- No training phase — build is a single `add()` call
- Graph structure provides consistent performance regardless of query distribution
- Trade-off: higher memory (~1.5–2x) due to neighbor lists


---

## 4. Caching — In-Process LRU

**Algorithm**: Least-Recently-Used eviction policy via `OrderedDict`.

**Purpose**: Cache `find_similar_products` results keyed by `(product_id, k)` to avoid redundant index queries for repeated lookups.

**Complexity**: $O(1)$ get/put/evict.

**Why LRU over Redis**: Single-process deployment, zero external dependencies, sub-microsecond cache hits. Redis would be preferred for multi-replica deployments requiring cache coherency.

---

## 5. Scaling Strategy — VectorIndex Protocol

The `VectorIndex` Protocol (`build`, `query`, `save`, `load`) abstracts the ANN backend, enabling drop-in replacement without modifying the search pipeline.

### What changes at scale:

| Scale | Recommended Backend | Rationale |
|-------|-------------------|-----------|
| **30k** (current) | FAISS HNSW or TurboQuant flat scan | Both sub-millisecond; HNSW has near-perfect recall, TurboQuant adds 16x compression |
| **1M** | ScaNN with AVQ partitions | Partition-based search with anisotropic quantization; best recall-vs-QPS tradeoff at this scale |
| **100M+** | DiskANN with SSD-backed graph | Vamana graph on SSD with PQ-compressed routing vectors in RAM; designed for datasets exceeding memory |

### Production pattern: Two-Stage Retrieval

At scale, a two-stage approach is standard:
1. **Stage 1 (Recall)**: Fast quantized/approximate retrieval — retrieve top-$k \cdot \alpha$ candidates (oversampling factor $\alpha = 2\text{–}5\times$)
2. **Stage 2 (Precision)**: Re-rank candidates using full-precision vectors or multimodal features

This decouples speed from accuracy and is used at Google, Spotify, and Pinterest.

---

## 6. Backend Comparison Benchmarks

### 6.1 Backends Overview

| Backend | Algorithm | Build | Recall mode | Key trade-off |
|---------|-----------|-------|-------------|---------------|
| **FAISS HNSW** | Navigable Small World graph | $O(n \log n)$ | Near-perfect (≥0.99) | High memory (M=32 neighbor lists) |
| **ScaNN AVQ** | Anisotropic vector quantization + tree partition | $O(n)$ after training | Tunable (partition search ratio) | Build requires training; very fast query |
| **TurboQuant 4-bit** | Online scalar quantization, 4 bits/dim | $O(n)$ | High + 5× exact rerank | 12.5% memory footprint vs f32 |
| **TurboQuant 2-bit** | Online scalar quantization, 2 bits/dim | $O(n)$ | Good + 5× exact rerank | 6.2% memory footprint vs f32 |
| **Two-Stage** | Any first-stage ANN + exact cosine rerank | Same as first stage | Best (approaches BF) | Extra cosine pass on α candidates |

### 6.2 Latency vs Scale (synthetic 128-dim vectors, k=10)

| Backend | 30k p50 | 100k p50 | 500k p50 | 1M p50 |
|---------|---------|---------|---------|--------|
| FAISS HNSW | 0.16 ms | 0.49 ms | 0.95 ms | 0.97 ms |
| ScaNN AVQ | **0.03 ms** | **0.06 ms** | **0.10 ms** | — |
| TurboQuant 4-bit | 0.26 ms | 0.72 ms | 4.25 ms | — |
| TurboQuant 2-bit | 0.18 ms | 0.44 ms | 1.79 ms | — |
| Two-Stage (HNSW) | 0.20 ms | 0.44 ms | 0.94 ms | 0.95 ms |

*ScaNN and TurboQuant skipped at 1M (memory/build-time budget); HNSW and Two-Stage run at all scales.*

### 6.3 Recall@10 vs Brute-Force Cosine

| Backend | 30k | 100k |
|---------|-----|------|
| FAISS HNSW | 1.000 | 1.000 |
| ScaNN AVQ | 0.750 | 0.800 |
| TurboQuant 4-bit | 1.000 | 1.000 |
| TurboQuant 2-bit | 0.995 | 0.985 |
| Two-Stage (HNSW) | 1.000 | 1.000 |

> **Note on ScaNN recall**: The synthetic benchmark uses low-rank isotropic data — a challenging case for partition-based search because points cluster poorly. Real product embeddings (structured, lower-rank manifolds) yield ScaNN recall of 0.95–0.99 (see unit tests with the `structured_vectors` fixture at `num_leaves_to_search_ratio=0.5`).

### 6.4 Memory Footprint

| Backend | Memory (n=100k, d=128) |
|---------|----------------------|
| f32 raw | 51.2 MB (baseline) |
| FAISS HNSW | ~155 MB (3× for neighbor lists) |
| ScaNN AVQ | ~15 MB (AVQ codes + partition) |
| TurboQuant 4-bit | **6.4 MB** (12.5% of f32) |
| TurboQuant 2-bit | **3.2 MB** (6.2% of f32) |

### 6.5 Backend Selection Guide

| Scenario | Recommended backend | Reason |
|----------|-------------------|--------|
| **30k products** (current) | FAISS HNSW | Sub-ms latency, perfect recall, no tuning |
| **30k + max compression** | TurboQuant 4-bit | 8× compression, still perfect recall |
| **100k–500k, throughput-critical** | ScaNN AVQ | 5–9k QPS vs 1–2k for HNSW |
| **1M+** | FAISS HNSW or Two-Stage (HNSW first) | ScaNN/TurboQuant not tested above 500k here; HNSW maintains <1ms |
| **Accuracy-critical at scale** | Two-Stage (ScaNN first + exact rerank) | Combines ScaNN throughput with near-BF recall |

*Data from `benchmarks/backends_results.json` — 128-dim synthetic vectors, k=10, 20 queries per scale.*

---

## 7. Algorithm Details — Additional Backends

### 7.1 ScaNN — Anisotropic Vector Quantization

ScaNN (Scalable Nearest Neighbors) partitions the corpus using k-means tree partitioning then applies **Anisotropic Vector Quantization (AVQ)**: unlike isotropic PQ which minimizes per-vector reconstruction error uniformly, AVQ minimizes the *inner-product error* — the error that matters for MIPS/cosine search. This yields better recall at the same compression rate.

**Configuration used**: `num_leaves = √n`, `score_ah(2)` (asymmetric hashing, 2 subspaces), `reorder(50)` (exact reorder top-50 candidates), `dot_product` distance on L2-normalized vectors.

**Reference**: Guo et al. (2020) "Accelerating Large-Scale Inference with Anisotropic Vector Quantization" — ICML 2020, [arXiv:1908.10396](https://arxiv.org/abs/1908.10396)

### 7.2 TurboQuant — Online Scalar Quantization

TurboQuant uses per-coordinate calibration (TQ+ variant) to assign bit ranges that are *data-oblivious* — no codebook training required. It achieves **2.7× the Shannon distortion-rate bound**, meaning 4-bit TurboQuant beats any 4-bit scheme within a factor of 2.7 on expected inner-product error.

**Implementation note**: `turbovec.IdMapIndex` is used (not `TurboQuantIndex`) because its `IdMapIndex.load()` classmethod correctly restores the serialized index; query uses a two-stage pipeline (TQ coarse + 5× over-fetch exact cosine rerank) for maximum recall.

**Reference**: Zandieh, Daliri, Hadian & Mirrokni (2025) "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" — ICLR 2026, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

### 7.3 Two-Stage Retrieval

Stage 1 queries any `VectorIndex` for $k \cdot \alpha$ candidates; Stage 2 re-ranks them by exact cosine similarity on stored L2-normalized vectors using a single BLAS `@` call. The re-rank cost is $O(k \cdot \alpha \cdot d)$ per query — at $\alpha=3$, $k=10$, $d=128$ this is ~3840 FLOPs, negligible vs the ANN lookup.

**`TwoStageRetriever(first_stage, vectors, alpha=3)`** — any `VectorIndex` can be the first stage.

---

## 8. Paper References

| Algorithm | Paper | Venue |
|-----------|-------|-------|
| HNSW | Malkov & Yashunin (2018) "Efficient and robust approximate nearest neighbor using Hierarchical Navigable Small World graphs" | IEEE TPAMI, [arXiv:1603.09320](https://arxiv.org/abs/1603.09320) |
| ScaNN / AVQ | Guo et al. (2020) "Accelerating Large-Scale Inference with Anisotropic Vector Quantization" | ICML 2020, [arXiv:1908.10396](https://arxiv.org/abs/1908.10396) |
| SOAR | Sun et al. (2023) "SOAR: Improved Indexing for Approximate Nearest Neighbor Search" | NeurIPS 2023, [arXiv:2404.00774](https://arxiv.org/abs/2404.00774) |
| TurboQuant | Zandieh, Daliri, Hadian & Mirrokni (2025) "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" | ICLR 2026, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) |
| RaBitQ | Gao & Long (2024) "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search" | SIGMOD 2024, [arXiv:2405.12497](https://arxiv.org/abs/2405.12497) |
| DiskANN | Subramanya et al. (2019) "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node" | NeurIPS 2019 |
| MRL | Kusupati et al. (2022) "Matryoshka Representation Learning" | NeurIPS 2022, [arXiv:2205.13147](https://arxiv.org/abs/2205.13147) |
| EfficientNet | Tan & Le (2019) "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" | ICML 2019, [arXiv:1905.11946](https://arxiv.org/abs/1905.11946) |
| Sentence-BERT | Reimers & Gurevych (2019) "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" | EMNLP 2019, [arXiv:1908.10084](https://arxiv.org/abs/1908.10084) |

| Algorithm | Paper | Venue |
|-----------|-------|-------|
| HNSW | Malkov & Yashunin (2018) "Efficient and robust approximate nearest neighbor using Hierarchical Navigable Small World graphs" | IEEE TPAMI, [arXiv:1603.09320](https://arxiv.org/abs/1603.09320) |
| ScaNN / AVQ | Guo et al. (2020) "Accelerating Large-Scale Inference with Anisotropic Vector Quantization" | ICML 2020, [arXiv:1908.10396](https://arxiv.org/abs/1908.10396) |
| SOAR | Sun et al. (2023) "SOAR: Improved Indexing for Approximate Nearest Neighbor Search" | NeurIPS 2023, [arXiv:2404.00774](https://arxiv.org/abs/2404.00774) |
| TurboQuant | Zandieh, Daliri, Hadian & Mirrokni (2025) "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" | ICLR 2026, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) |
| RaBitQ | Gao & Long (2024) "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search" | SIGMOD 2024, [arXiv:2405.12497](https://arxiv.org/abs/2405.12497) |
| DiskANN | Subramanya et al. (2019) "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node" | NeurIPS 2019 |
| MRL | Kusupati et al. (2022) "Matryoshka Representation Learning" | NeurIPS 2022, [arXiv:2205.13147](https://arxiv.org/abs/2205.13147) |
| EfficientNet | Tan & Le (2019) "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" | ICML 2019, [arXiv:1905.11946](https://arxiv.org/abs/1905.11946) |
| Sentence-BERT | Reimers & Gurevych (2019) "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" | EMNLP 2019, [arXiv:1908.10084](https://arxiv.org/abs/1908.10084) |
