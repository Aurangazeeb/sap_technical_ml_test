"""Baseline benchmark for brute-force similarity search.

Measures latency (p50/p95/p99), throughput (queries/sec), peak memory,
and recall@k for the current ``find_similar_products`` implementation.

Usage
-----
    cd sap-cxii-tech-ex-01
    uv run python -m benchmarks.benchmark_search [--num-queries 50] [--k 10]
    uv run python -m benchmarks.benchmark_search --sample-size 500 --num-queries 5

Results are printed to stdout and saved to ``benchmarks/baseline_results.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

# Ensure the package is importable when run as ``python -m benchmarks.benchmark_search``
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import sap_cxii_tech_ex_01.search as _search_mod  # noqa: E402
from sap_cxii_tech_ex_01.config import get_settings  # noqa: E402
from sap_cxii_tech_ex_01.search import find_similar_products, load_dataset  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _select_query_ids(
    df: pd.DataFrame, num_queries: int, seed: int = 42
) -> list[str]:
    """Sample *num_queries* product IDs deterministically."""
    rng = np.random.default_rng(seed)
    ids = df["uniq_id"].tolist()
    indices = rng.choice(len(ids), size=min(num_queries, len(ids)), replace=False)
    return [ids[int(i)] for i in indices]


def _percentile(values: list[float], p: int) -> float:
    return float(np.percentile(values, p))


# ── Main benchmark ───────────────────────────────────────────────────────────

def run_benchmark(
    num_queries: int = 50,
    k: int = 10,
    warmup: int = 2,
    sample_size: int | None = None,
    skip_images: bool = False,
) -> dict:
    """Run the brute-force baseline benchmark and return a results dict."""

    settings = get_settings()
    df = load_dataset(settings)

    # Optionally sample a subset for faster runs
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Null out image URLs to avoid network I/O during benchmarking
    if skip_images and "image_urls__small" in df.columns:
        df["image_urls__small"] = None

    dataset_size = len(df)
    query_ids = _select_query_ids(df, num_queries)

    print(f"Dataset size       : {dataset_size:,}")
    print(f"Query IDs sampled  : {len(query_ids)}")
    print(f"k (num_similar)    : {k}")
    print()

    # Patch load_dataset so find_similar_products uses our (possibly subsetted) df
    with patch.object(_search_mod, "load_dataset", return_value=df):
        # ── Warmup — prime model caches (sentence-transformers, etc.) ─────
        print(f"Warming up ({warmup} queries)…")
        for pid in query_ids[:warmup]:
            find_similar_products(pid, k)
        print()

        # ── Timed queries ─────────────────────────────────────────────────
        latencies: list[float] = []
        results_map: dict[str, list[str]] = {}

        tracemalloc.start()
        mem_before = tracemalloc.get_traced_memory()[1]

        print(f"Running {len(query_ids)} queries…")
        wall_start = time.perf_counter()

        for pid in query_ids:
            t0 = time.perf_counter()
            result = find_similar_products(pid, k)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
            results_map[pid] = result

        wall_elapsed = time.perf_counter() - wall_start
        _, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    # ── Recall@k — brute force is its own ground truth, so recall=1.0 by
    #    definition.  This placeholder will be compared against ANN indices
    #    in later steps.
    recall_at_k = 1.0

    # ── Aggregate ─────────────────────────────────────────────────────────
    throughput = len(query_ids) / wall_elapsed if wall_elapsed > 0 else 0.0

    report = {
        "backend": "brute_force",
        "dataset_size": dataset_size,
        "num_queries": len(query_ids),
        "k": k,
        "latency_p50_ms": round(_percentile(latencies, 50) * 1000, 2),
        "latency_p95_ms": round(_percentile(latencies, 95) * 1000, 2),
        "latency_p99_ms": round(_percentile(latencies, 99) * 1000, 2),
        "latency_mean_ms": round(float(np.mean(latencies)) * 1000, 2),
        "throughput_qps": round(throughput, 2),
        "wall_time_s": round(wall_elapsed, 2),
        "peak_memory_mb": round(mem_peak / (1024 * 1024), 2),
        "recall_at_k": recall_at_k,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Brute-force baseline benchmark")
    parser.add_argument("--num-queries", type=int, default=50, help="Number of query products")
    parser.add_argument("--k", type=int, default=10, help="num_similar per query")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup queries (not timed)")
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Subset dataset to N products (default: use full dataset)",
    )
    parser.add_argument(
        "--skip-images", action="store_true",
        help="Null out image URLs to avoid network I/O (use text+structured only)",
    )
    args = parser.parse_args()

    report = run_benchmark(
        num_queries=args.num_queries, k=args.k,
        warmup=args.warmup, sample_size=args.sample_size,
        skip_images=args.skip_images,
    )

    # ── Print ─────────────────────────────────────────────────────────────
    print()
    print("=" * 50)
    print("  BRUTE-FORCE BASELINE RESULTS")
    print("=" * 50)
    for key, val in report.items():
        print(f"  {key:<22s}: {val}")
    print("=" * 50)

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = Path(__file__).resolve().parent / "baseline_results.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
