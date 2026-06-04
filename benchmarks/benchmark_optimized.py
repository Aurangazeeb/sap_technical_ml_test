"""Optimized benchmark using PCA + FAISS HNSW + caching.

Compares the optimized pipeline against the brute-force baseline from
``benchmark_search.py``.  Reports latency, throughput, memory, and recall@k.

Usage
-----
    cd sap-cxii-tech-ex-01
    uv run python -m benchmarks.benchmark_optimized --sample-size 500 --num-queries 5 --skip-images
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sap_cxii_tech_ex_01.cache import SearchCache  # noqa: E402
from sap_cxii_tech_ex_01.config import get_settings  # noqa: E402
from sap_cxii_tech_ex_01.dim_reduction import DimensionalityReducer  # noqa: E402
from sap_cxii_tech_ex_01.features.structured import StructuredExtractor  # noqa: E402
from sap_cxii_tech_ex_01.features.text import TextExtractor  # noqa: E402
from sap_cxii_tech_ex_01.search import load_dataset  # noqa: E402
from sap_cxii_tech_ex_01.similarity import SimilarityEngine  # noqa: E402
from sap_cxii_tech_ex_01.vector_index import FAISSHNSWIndex  # noqa: E402


def _select_query_ids(
    df: pd.DataFrame, num_queries: int, seed: int = 42
) -> list[str]:
    rng = np.random.default_rng(seed)
    ids = df["uniq_id"].tolist()
    indices = rng.choice(len(ids), size=min(num_queries, len(ids)), replace=False)
    return [ids[int(i)] for i in indices]


def _percentile(values: list[float], p: int) -> float:
    return float(np.percentile(values, p))


def _brute_force_top_k(
    combined: np.ndarray, query_idx: int, k: int
) -> list[int]:
    """Brute-force kNN for recall comparison."""
    query = combined[query_idx]
    norms = np.linalg.norm(combined, axis=1, keepdims=False)
    norms = np.where(norms == 0.0, 1.0, norms)
    normed = combined / norms[:, None]
    q_norm = query / (np.linalg.norm(query) or 1.0)
    scores = normed @ q_norm
    scores[query_idx] = -np.inf
    return np.argsort(scores)[::-1][:k].tolist()


def run_optimized_benchmark(
    num_queries: int = 50,
    k: int = 10,
    warmup: int = 2,
    sample_size: int | None = None,
    skip_images: bool = False,
    pca_target_dim: int | None = 128,
) -> dict:
    """Run the optimized pipeline benchmark."""

    settings = get_settings()
    df = load_dataset(settings)

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    if skip_images and "image_urls__small" in df.columns:
        df["image_urls__small"] = None

    dataset_size = len(df)
    ids: list[str] = df["uniq_id"].tolist()
    query_ids = _select_query_ids(df, num_queries)

    print(f"Dataset size       : {dataset_size:,}")
    print(f"Query IDs sampled  : {len(query_ids)}")
    print(f"k (num_similar)    : {k}")
    print(f"PCA target dim     : {pca_target_dim}")
    print()

    # ── Feature extraction (one-time cost) ────────────────────────────────
    print("Extracting features…")
    t_feat_start = time.perf_counter()

    text_col = "product_name" if "product_name" in df.columns else "uniq_id"
    texts: list[str | None] = df[text_col].tolist()
    text_feats = TextExtractor(settings).extract(texts)

    image_col = "image_urls__small"
    has_images = image_col in df.columns and df[image_col].notna().any()
    image_feats = None
    if has_images:
        from sap_cxii_tech_ex_01.features.image import ImageExtractor
        image_feats = ImageExtractor(settings).extract(df[image_col].tolist())

    struct_feats = StructuredExtractor(settings).fit_transform(df)

    engine = SimilarityEngine(settings)
    combined = engine.combine(text_feats, image_feats, struct_feats)

    t_feat_elapsed = time.perf_counter() - t_feat_start
    print(f"Feature extraction : {t_feat_elapsed:.2f}s")
    print(f"Combined shape     : {combined.shape}")

    # ── PCA reduction ─────────────────────────────────────────────────────
    if pca_target_dim is not None and pca_target_dim < combined.shape[1]:
        print(f"Applying PCA ({combined.shape[1]} → {pca_target_dim})…")
        reducer = DimensionalityReducer()
        combined = reducer.fit_transform(combined, target_dim=pca_target_dim)
        print(f"Reduced shape      : {combined.shape}")
    else:
        print("Skipping PCA (target_dim >= combined dim or disabled)")

    # ── Build FAISS HNSW index ────────────────────────────────────────────
    print("Building HNSW index…")
    t_build_start = time.perf_counter()
    index = FAISSHNSWIndex()
    index.build(combined)
    t_build_elapsed = time.perf_counter() - t_build_start
    print(f"Index build time   : {t_build_elapsed:.2f}s")

    # ── Warmup ────────────────────────────────────────────────────────────
    cache = SearchCache(max_size=1024)
    print(f"\nWarming up ({warmup} queries)…")
    for pid in query_ids[:warmup]:
        qi = ids.index(pid)
        ann_indices, _ = index.query(combined[qi], k=k + 1)
        result = [ids[int(j)] for j in ann_indices if j != qi][:k]
        cache.put(pid, k, result)

    # ── Timed queries ─────────────────────────────────────────────────────
    latencies: list[float] = []
    results_map: dict[str, list[str]] = {}

    tracemalloc.start()

    print(f"Running {len(query_ids)} queries…")
    wall_start = time.perf_counter()

    for pid in query_ids:
        t0 = time.perf_counter()
        cached = cache.get(pid, k)
        if cached is not None:
            result = cached
        else:
            qi = ids.index(pid)
            ann_indices, _ = index.query(combined[qi], k=k + 1)
            result = [ids[int(j)] for j in ann_indices if j != qi][:k]
            cache.put(pid, k, result)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)
        results_map[pid] = result

    wall_elapsed = time.perf_counter() - wall_start
    _, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ── Recall@k vs brute-force on combined vectors ───────────────────────
    print("Computing recall@k vs brute force…")
    hits = 0
    total = 0
    for pid in query_ids:
        qi = ids.index(pid)
        bf_top = set(_brute_force_top_k(combined, qi, k))
        ann_top = set()
        ann_indices, _ = index.query(combined[qi], k=k + 1)
        for j in ann_indices:
            if int(j) != qi:
                ann_top.add(int(j))
        ann_top = set(list(ann_top)[:k])
        hits += len(ann_top & bf_top)
        total += k
    recall_at_k = round(hits / total, 4) if total > 0 else 0.0

    throughput = len(query_ids) / wall_elapsed if wall_elapsed > 0 else 0.0

    report = {
        "backend": "pca_hnsw_cached",
        "dataset_size": dataset_size,
        "num_queries": len(query_ids),
        "k": k,
        "pca_target_dim": pca_target_dim,
        "feature_extraction_s": round(t_feat_elapsed, 2),
        "index_build_s": round(t_build_elapsed, 2),
        "latency_p50_ms": round(_percentile(latencies, 50) * 1000, 4),
        "latency_p95_ms": round(_percentile(latencies, 95) * 1000, 4),
        "latency_p99_ms": round(_percentile(latencies, 99) * 1000, 4),
        "latency_mean_ms": round(float(np.mean(latencies)) * 1000, 4),
        "throughput_qps": round(throughput, 2),
        "wall_time_s": round(wall_elapsed, 4),
        "peak_memory_mb": round(mem_peak / (1024 * 1024), 2),
        "recall_at_k": recall_at_k,
        "cache_size": len(cache),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized pipeline benchmark (PCA + HNSW + cache)")
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--pca-dim", type=int, default=128, help="PCA target dimension (0 to disable)")
    args = parser.parse_args()

    pca_dim = args.pca_dim if args.pca_dim > 0 else None

    report = run_optimized_benchmark(
        num_queries=args.num_queries,
        k=args.k,
        warmup=args.warmup,
        sample_size=args.sample_size,
        skip_images=args.skip_images,
        pca_target_dim=pca_dim,
    )

    print()
    print("=" * 55)
    print("  OPTIMIZED PIPELINE RESULTS (PCA + HNSW + Cache)")
    print("=" * 55)
    for key, val in report.items():
        print(f"  {key:<26s}: {val}")
    print("=" * 55)

    # ── Compare with baseline if available ────────────────────────────────
    baseline_path = Path(__file__).resolve().parent / "baseline_results.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        print()
        print("─" * 55)
        print("  COMPARISON vs BRUTE-FORCE BASELINE")
        print("─" * 55)
        for metric in ["latency_p50_ms", "latency_p95_ms", "throughput_qps", "peak_memory_mb"]:
            bv = baseline.get(metric, 0)
            ov = report.get(metric, 0)
            if bv and ov:
                speedup = bv / ov if ov > 0 else float("inf")
                print(f"  {metric:<26s}: {bv:>10.2f} → {ov:>10.4f}  ({speedup:>8.1f}x)")
        print(f"  {'recall_at_k':<26s}: {baseline.get('recall_at_k', 'N/A')} → {report['recall_at_k']}")
        print("─" * 55)

    out_path = Path(__file__).resolve().parent / "optimized_results.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
