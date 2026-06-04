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

## 6. Paper References

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
