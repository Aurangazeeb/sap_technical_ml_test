"""Synthetic scaling benchmark: brute-force vs HNSW at 30k–1M scale.

Generates synthetic feature matrices at increasing sizes and measures
query latency for both brute-force cosine similarity and FAISS HNSW,
demonstrating sublinear ANN scaling.

Usage
-----
    cd sap-cxii-tech-ex-01
    uv run python -m benchmarks.benchmark_scaling [--dim 128] [--k 10] [--num-queries 20]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sap_cxii_tech_ex_01.vector_index import FAISSHNSWIndex  # noqa: E402


SCALES = [10_000, 30_000, 100_000, 500_000, 1_000_000]


def _percentile(values: list[float], p: int) -> float:
    return float(np.percentile(values, p))


def _brute_force_query(
    matrix: np.ndarray, query: np.ndarray, k: int, query_idx: int
) -> list[int]:
    """Cosine similarity brute-force."""
    norms = np.linalg.norm(matrix, axis=1)
    norms = np.where(norms == 0.0, 1.0, norms)
    normed = matrix / norms[:, None]
    q_norm = query / (np.linalg.norm(query) or 1.0)
    scores = normed @ q_norm
    scores[query_idx] = -np.inf
    return np.argsort(scores)[::-1][:k].tolist()


def benchmark_at_scale(
    n: int,
    dim: int,
    k: int,
    num_queries: int,
    seed: int = 42,
) -> dict:
    """Run brute-force and HNSW benchmarks for *n* vectors of *dim* dimensions."""
    rng = np.random.default_rng(seed)

    # Generate structured synthetic data (low-rank + noise, like real embeddings)
    rank = min(dim, 64)
    latent = rng.standard_normal((n, rank)).astype(np.float32)
    proj = rng.standard_normal((rank, dim)).astype(np.float32)
    vectors = (latent @ proj).astype(np.float32)
    vectors += 0.05 * rng.standard_normal(vectors.shape).astype(np.float32)

    query_indices = rng.choice(n, size=min(num_queries, n), replace=False)

    # ── Build HNSW index ──────────────────────────────────────────────────
    t_build_start = time.perf_counter()
    index = FAISSHNSWIndex()
    index.build(vectors)
    hnsw_build_s = time.perf_counter() - t_build_start

    # ── HNSW queries ──────────────────────────────────────────────────────
    hnsw_latencies: list[float] = []
    for qi in query_indices:
        t0 = time.perf_counter()
        index.query(vectors[qi], k=k + 1)
        hnsw_latencies.append(time.perf_counter() - t0)

    # ── Brute-force queries (skip for n > 100k — too slow) ────────────────
    bf_latencies: list[float] = []
    if n <= 100_000:
        for qi in query_indices:
            t0 = time.perf_counter()
            _brute_force_query(vectors, vectors[qi], k, int(qi))
            bf_latencies.append(time.perf_counter() - t0)

    result = {
        "n": n,
        "dim": dim,
        "k": k,
        "num_queries": len(query_indices),
        "hnsw_build_s": round(hnsw_build_s, 3),
        "hnsw_p50_ms": round(_percentile(hnsw_latencies, 50) * 1000, 4),
        "hnsw_p95_ms": round(_percentile(hnsw_latencies, 95) * 1000, 4),
        "hnsw_mean_ms": round(float(np.mean(hnsw_latencies)) * 1000, 4),
    }

    if bf_latencies:
        result["bf_p50_ms"] = round(_percentile(bf_latencies, 50) * 1000, 2)
        result["bf_p95_ms"] = round(_percentile(bf_latencies, 95) * 1000, 2)
        result["bf_mean_ms"] = round(float(np.mean(bf_latencies)) * 1000, 2)
        result["speedup_p50"] = round(result["bf_p50_ms"] / result["hnsw_p50_ms"], 1) if result["hnsw_p50_ms"] > 0 else None
    else:
        result["bf_p50_ms"] = None
        result["bf_p95_ms"] = None
        result["bf_mean_ms"] = None
        result["speedup_p50"] = None

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaling benchmark: brute-force vs HNSW")
    parser.add_argument("--dim", type=int, default=128, help="Vector dimensionality (post-PCA)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num-queries", type=int, default=20)
    parser.add_argument(
        "--scales", type=str, default=None,
        help="Comma-separated dataset sizes (default: 10000,30000,100000,500000,1000000)",
    )
    args = parser.parse_args()

    scales = [int(s) for s in args.scales.split(",")] if args.scales else SCALES

    all_results: list[dict] = []

    print("=" * 70)
    print("  SCALING BENCHMARK: Brute-Force vs HNSW")
    print(f"  dim={args.dim}  k={args.k}  queries={args.num_queries}")
    print("=" * 70)
    print()

    for n in scales:
        print(f"▸ N = {n:>10,} …", end="", flush=True)
        result = benchmark_at_scale(n, args.dim, args.k, args.num_queries)
        all_results.append(result)

        bf_str = f"bf_p50={result['bf_p50_ms']:.2f}ms" if result["bf_p50_ms"] else "bf=skipped"
        speedup_str = f"  speedup={result['speedup_p50']}x" if result.get("speedup_p50") else ""
        print(
            f"  hnsw_p50={result['hnsw_p50_ms']:.4f}ms  "
            f"{bf_str}{speedup_str}  build={result['hnsw_build_s']:.1f}s"
        )

    # ── Summary table ─────────────────────────────────────────────────────
    print()
    print("─" * 70)
    print(f"  {'N':>10s}  {'HNSW p50':>12s}  {'BF p50':>12s}  {'Speedup':>10s}  {'Build':>8s}")
    print("─" * 70)
    for r in all_results:
        bf = f"{r['bf_p50_ms']:.2f}ms" if r["bf_p50_ms"] else "—"
        sp = f"{r['speedup_p50']}x" if r.get("speedup_p50") else "—"
        print(
            f"  {r['n']:>10,}  "
            f"{r['hnsw_p50_ms']:>10.4f}ms  "
            f"{bf:>12s}  "
            f"{sp:>10s}  "
            f"{r['hnsw_build_s']:>6.1f}s"
        )
    print("─" * 70)

    out_path = Path(__file__).resolve().parent / "scaling_results.json"
    out_path.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
