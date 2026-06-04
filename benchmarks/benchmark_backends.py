"""Backend comparison benchmark: FAISS HNSW vs ScaNN vs TurboQuant vs Two-Stage.

Measures latency (p50/p95/p99), throughput (QPS), memory footprint, and
recall@k for each backend at synthetic scales (30k → 1M).

Usage
-----
    cd sap-cxii-tech-ex-01
    uv run python -m benchmarks.benchmark_backends [--dim 128] [--k 10] [--num-queries 20]

Results are saved to ``benchmarks/backends_results.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sap_cxii_tech_ex_01.vector_index import FAISSHNSWIndex  # noqa: E402
from sap_cxii_tech_ex_01.retrieval import TwoStageRetriever  # noqa: E402

try:
    from sap_cxii_tech_ex_01.backends.scann_index import ScaNNIndex
    _SCANN_AVAILABLE = True
except ImportError:
    _SCANN_AVAILABLE = False
    warnings.warn("ScaNN not available — skipping ScaNN benchmarks.")

try:
    from sap_cxii_tech_ex_01.backends.turboquant_index import TurboQuantIndex
    _TURBOQUANT_AVAILABLE = True
except ImportError:
    _TURBOQUANT_AVAILABLE = False
    warnings.warn("TurboQuant not available — skipping TurboQuant benchmarks.")


SCALES = [30_000, 100_000, 500_000, 1_000_000]
RESULTS_FILE = Path(__file__).resolve().parent / "backends_results.json"


def _percentile(values: list[float], p: int) -> float:
    return float(np.percentile(values, p))


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (vecs / norms).astype(np.float32)


def _brute_force_top_k(
    normed: np.ndarray, query_idx: int, k: int
) -> set[int]:
    scores = normed @ normed[query_idx]
    scores[query_idx] = -np.inf
    return set(np.argsort(scores)[::-1][:k].tolist())


def _recall_at_k(
    index,
    normed_vecs: np.ndarray,
    query_indices: list[int],
    k: int,
    query_vecs: np.ndarray | None = None,
) -> float:
    """Compute recall@k vs cosine brute-force for given query_indices."""
    if query_vecs is None:
        query_vecs = normed_vecs
    hits = 0
    for qi in query_indices:
        bf = _brute_force_top_k(normed_vecs, qi, k)
        ann_idx, _ = index.query(query_vecs[qi], k + 1)
        ann_set = set(int(j) for j in ann_idx if j != qi)
        hits += len(set(list(ann_set)[:k]) & bf)
    return hits / (len(query_indices) * k) if query_indices else 0.0


def _latency_stats(
    index, query_vecs: np.ndarray, k: int, n_warmup: int = 5, n_measure: int = 50
) -> dict[str, float]:
    """Measure query latency: returns p50/p95/p99 in ms and QPS."""
    n = len(query_vecs)
    for i in range(n_warmup):
        index.query(query_vecs[i % n], k)

    times: list[float] = []
    for i in range(n_measure):
        t0 = time.perf_counter()
        index.query(query_vecs[i % n], k)
        times.append(time.perf_counter() - t0)

    p50 = _percentile(times, 50)
    return {
        "p50_ms": round(p50 * 1000, 4),
        "p95_ms": round(_percentile(times, 95) * 1000, 4),
        "p99_ms": round(_percentile(times, 99) * 1000, 4),
        "qps": round(1.0 / p50 if p50 > 0 else 0.0, 1),
    }


def _make_synthetic(n: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rank = min(dim, 64)
    latent = rng.standard_normal((n, rank)).astype(np.float32)
    proj = rng.standard_normal((rank, dim)).astype(np.float32)
    vecs = latent @ proj
    vecs += 0.05 * rng.standard_normal(vecs.shape).astype(np.float32)
    return vecs.astype(np.float32)


def benchmark_at_scale(
    n: int,
    dim: int,
    k: int,
    num_queries: int,
    seed: int = 42,
) -> dict:
    """Benchmark all backends for *n* vectors of *dim* dimensions."""
    print(f"\n{'='*60}")
    print(f"Scale: {n:,} vectors × {dim} dims  (k={k}, {num_queries} queries)")
    print(f"{'='*60}")

    rng = np.random.default_rng(seed)
    vecs = _make_synthetic(n, dim, seed=seed)
    normed = _normalize(vecs)

    query_indices = rng.choice(n, size=min(num_queries, n), replace=False).tolist()
    query_vecs = normed  # queries are L2-normalised

    results: dict = {"n": n, "dim": dim, "k": k, "backends": {}}

    # ── FAISS HNSW ────────────────────────────────────────────────────────────
    print("  [FAISS HNSW] building...", end=" ", flush=True)
    t0 = time.perf_counter()
    hnsw = FAISSHNSWIndex()
    hnsw.build(normed)
    build_s = time.perf_counter() - t0
    print(f"done ({build_s:.1f}s)")
    recall = _recall_at_k(hnsw, normed, query_indices, k) if n <= 100_000 else None
    lat = _latency_stats(hnsw, query_vecs, k)
    results["backends"]["faiss_hnsw"] = {
        "build_s": round(build_s, 2),
        "recall_at_k": round(recall, 4) if recall is not None else "skipped",
        **lat,
    }
    print(f"         latency p50={lat['p50_ms']}ms  recall={recall:.3f}" if recall is not None else f"         latency p50={lat['p50_ms']}ms")

    # ── ScaNN ─────────────────────────────────────────────────────────────────
    if _SCANN_AVAILABLE and n <= 500_000:
        print("  [ScaNN]     building...", end=" ", flush=True)
        t0 = time.perf_counter()
        scann = ScaNNIndex()
        scann.build(normed)
        build_s = time.perf_counter() - t0
        print(f"done ({build_s:.1f}s)")
        recall = _recall_at_k(scann, normed, query_indices, k) if n <= 100_000 else None
        lat = _latency_stats(scann, query_vecs, k)
        results["backends"]["scann"] = {
            "build_s": round(build_s, 2),
            "recall_at_k": round(recall, 4) if recall is not None else "skipped",
            **lat,
        }
        print(f"         latency p50={lat['p50_ms']}ms  recall={recall:.3f}" if recall is not None else f"         latency p50={lat['p50_ms']}ms")

    # ── TurboQuant 4-bit ─────────────────────────────────────────────────────
    if _TURBOQUANT_AVAILABLE and n <= 500_000:
        print("  [TurboQ 4b] building...", end=" ", flush=True)
        t0 = time.perf_counter()
        tq4 = TurboQuantIndex(bit_width=4)
        tq4.build(normed)
        build_s = time.perf_counter() - t0
        print(f"done ({build_s:.1f}s)")
        recall = _recall_at_k(tq4, normed, query_indices, k) if n <= 100_000 else None
        lat = _latency_stats(tq4, query_vecs, k)
        compressed_ratio = tq4.compressed_size_bytes() / (n * dim * 4)
        results["backends"]["turboquant_4bit"] = {
            "build_s": round(build_s, 2),
            "recall_at_k": round(recall, 4) if recall is not None else "skipped",
            "compressed_ratio": round(compressed_ratio, 4),
            **lat,
        }
        print(f"         latency p50={lat['p50_ms']}ms  recall={recall:.3f}  compression={compressed_ratio:.1%}" if recall is not None else f"         latency p50={lat['p50_ms']}ms  compression={compressed_ratio:.1%}")

    # ── TurboQuant 2-bit ─────────────────────────────────────────────────────
    if _TURBOQUANT_AVAILABLE and n <= 500_000:
        print("  [TurboQ 2b] building...", end=" ", flush=True)
        t0 = time.perf_counter()
        tq2 = TurboQuantIndex(bit_width=2)
        tq2.build(normed)
        build_s = time.perf_counter() - t0
        print(f"done ({build_s:.1f}s)")
        recall = _recall_at_k(tq2, normed, query_indices, k) if n <= 100_000 else None
        lat = _latency_stats(tq2, query_vecs, k)
        compressed_ratio = tq2.compressed_size_bytes() / (n * dim * 4)
        results["backends"]["turboquant_2bit"] = {
            "build_s": round(build_s, 2),
            "recall_at_k": round(recall, 4) if recall is not None else "skipped",
            "compressed_ratio": round(compressed_ratio, 4),
            **lat,
        }
        print(f"         latency p50={lat['p50_ms']}ms  recall={recall:.3f}  compression={compressed_ratio:.1%}" if recall is not None else f"         latency p50={lat['p50_ms']}ms  compression={compressed_ratio:.1%}")

    # ── Two-Stage (FAISS first-pass + exact cosine re-rank) ───────────────────
    print("  [Two-Stage] building...", end=" ", flush=True)
    t0 = time.perf_counter()
    hnsw2 = FAISSHNSWIndex()
    hnsw2.build(normed)
    ts = TwoStageRetriever(hnsw2, normed, alpha=3)
    build_s = time.perf_counter() - t0
    print(f"done ({build_s:.1f}s)")
    recall = _recall_at_k(ts, normed, query_indices, k) if n <= 100_000 else None
    lat = _latency_stats(ts, query_vecs, k)
    results["backends"]["two_stage_faiss"] = {
        "build_s": round(build_s, 2),
        "recall_at_k": round(recall, 4) if recall is not None else "skipped",
        **lat,
    }
    print(f"         latency p50={lat['p50_ms']}ms  recall={recall:.3f}" if recall is not None else f"         latency p50={lat['p50_ms']}ms")

    return results


def print_comparison_table(all_results: list[dict]) -> None:
    """Print a compact comparison table to stdout."""
    backends = ["faiss_hnsw", "scann", "turboquant_4bit", "turboquant_2bit", "two_stage_faiss"]
    labels = {
        "faiss_hnsw": "FAISS HNSW",
        "scann": "ScaNN AVQ",
        "turboquant_4bit": "TurboQ 4-bit",
        "turboquant_2bit": "TurboQ 2-bit",
        "two_stage_faiss": "Two-Stage",
    }
    print("\n" + "=" * 90)
    print(f"{'Backend':<16}  {'Scale':>10}  {'p50 (ms)':>10}  {'p95 (ms)':>10}  {'QPS':>8}  {'Recall@k':>10}")
    print("-" * 90)
    for res in all_results:
        n = res["n"]
        for be in backends:
            if be not in res["backends"]:
                continue
            b = res["backends"][be]
            recall_str = f"{b['recall_at_k']:.3f}" if isinstance(b.get("recall_at_k"), float) else "  —"
            print(
                f"{labels[be]:<16}  {n:>10,}  {b['p50_ms']:>10.3f}  {b['p95_ms']:>10.3f}"
                f"  {b['qps']:>8,.0f}  {recall_str:>10}"
            )
    print("=" * 90)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backend comparison benchmark")
    parser.add_argument("--dim", type=int, default=128, help="Vector dimension")
    parser.add_argument("--k", type=int, default=10, help="Neighbors to retrieve")
    parser.add_argument("--num-queries", type=int, default=20, help="Queries per scale")
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=SCALES,
        help="Dataset sizes to benchmark",
    )
    args = parser.parse_args()

    all_results: list[dict] = []
    for n in args.scales:
        r = benchmark_at_scale(n=n, dim=args.dim, k=args.k, num_queries=args.num_queries)
        all_results.append(r)

    print_comparison_table(all_results)

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE.relative_to(Path.cwd()) if RESULTS_FILE.is_relative_to(Path.cwd()) else RESULTS_FILE}")


if __name__ == "__main__":
    main()
