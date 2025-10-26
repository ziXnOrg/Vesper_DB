#!/usr/bin/env python3
"""
FAISS reference comparison for IVF-PQ on SIFT-128-euclidean.

- Loads TEXMEX-format files from data/fvecs:
  * <dataset>_base.fvecs
  * <dataset>_query.fvecs
  * <dataset>_groundtruth.ivecs
- Trains FAISS IndexIVFPQ with given parameters (default: nlist=4096, m=16, nbits=8)
- Searches with nprobe=256, k=10 and computes Recall@10 vs provided groundtruth (k_gt=100)
- Reports Recall@10 and query latency (P50/P99) in microseconds

Usage:
  python scripts/faiss_reference_compare.py --data-dir data/fvecs --dataset sift-128-euclidean \
      --nlist 4096 --m 16 --nbits 8 --nprobe 256 --k 10 --max-base 1000000 --max-train 200000 --max-queries 10000

Notes:
- Determinism: sets FAISS OMP threads to 1 and numpy RNG seed; FAISS internal kmeans seeding may vary by version.
- Dependencies: numpy, faiss-cpu (or faiss-gpu). Will exit with a helpful message if FAISS is not installed.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    print("ERROR: faiss is not installed (pip install faiss-cpu). Aborting.")
    print(f"Import error: {e}")
    sys.exit(2)


def fvecs_read(fname: str) -> np.ndarray:
    a = np.fromfile(fname, dtype=np.int32)
    if a.size == 0:
        raise FileNotFoundError(f"Empty or missing file: {fname}")
    d = a[0]
    if d <= 0:
        raise ValueError(f"Invalid dimension header in {fname}: {d}")
    a = a.reshape(-1, d + 1)
    # reinterpret trailing int32 as float32 without copy
    return a[:, 1:].view(np.float32)


def ivecs_read(fname: str) -> np.ndarray:
    a = np.fromfile(fname, dtype=np.int32)
    if a.size == 0:
        raise FileNotFoundError(f"Empty or missing file: {fname}")
    d = a[0]
    if d <= 0:
        raise ValueError(f"Invalid dimension header in {fname}: {d}")
    return a.reshape(-1, d + 1)[:, 1:]


def recall_at_k(results: np.ndarray, gt: np.ndarray, k: int, k_gt: int) -> float:
    # results: [nq, k], gt: [nq, k_gt]
    assert results.shape[0] == gt.shape[0]
    nq = results.shape[0]
    k = min(k, results.shape[1])
    k_gt = min(k_gt, gt.shape[1])
    correct = 0.0
    for i in range(nq):
        gt_set = set(gt[i, :k_gt].tolist())
        hits = sum(1 for id_ in results[i, :k] if id_ in gt_set)
        correct += hits / float(k)
    return correct / float(nq)


def p50_p99(values: np.ndarray) -> Tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    v = np.sort(values)
    p50 = v[int(0.50 * (v.size - 1))]
    p99 = v[int(0.99 * (v.size - 1))]
    return float(p50), float(p99)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/fvecs")
    parser.add_argument("--dataset", default="sift-128-euclidean")
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--nprobe", type=int, default=256)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-base", type=int, default=1_000_000)
    parser.add_argument("--max-train", type=int, default=200_000)
    parser.add_argument("--max-queries", type=int, default=10_000)
    parser.add_argument("--batch", type=int, default=1000, help="search batch size for timing")
    args = parser.parse_args()

    np.random.seed(42)
    try:
        faiss.omp_set_num_threads(1)
    except Exception:
        pass

    base_path = os.path.join(args.data_dir, f"{args.dataset}_base.fvecs")
    query_path = os.path.join(args.data_dir, f"{args.dataset}_query.fvecs")
    gt_path = os.path.join(args.data_dir, f"{args.dataset}_groundtruth.ivecs")

    print(f"FAISS version: {getattr(faiss, '__version__', 'unknown')}")
    print(f"Loading dataset from: {args.data_dir}")
    xb = fvecs_read(base_path)
    xq = fvecs_read(query_path)
    gt = ivecs_read(gt_path)

    d = xb.shape[1]
    xb = xb[: min(len(xb), args.max_base)]
    xq = xq[: min(len(xq), args.max_queries)]

    # Build FAISS IVF-PQ (L2)
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, args.nlist, args.m, args.nbits)

    # Train
    train_n = min(len(xb), args.max_train)
    print(f"Training IVFPQ: nlist={args.nlist} m={args.m} nbits={args.nbits} train_n={train_n}")
    t0 = time.perf_counter()
    index.train(xb[:train_n])
    t_train = (time.perf_counter() - t0) * 1000.0
    print(f"Train time: {t_train:.1f} ms")

    # Add
    t0 = time.perf_counter()
    index.add(xb)
    t_add = (time.perf_counter() - t0) * 1000.0
    print(f"Add time: {t_add:.1f} ms for {len(xb)} vectors")

    # Search
    index.nprobe = args.nprobe
    k = args.k

    nq = len(xq)
    print(f"Searching {nq} queries, k={k}, nprobe={args.nprobe} ...")
    lat_us = []
    all_I = np.empty((nq, k), dtype=np.int64)
    bs = args.batch
    for i in range(0, nq, bs):
        q = xq[i : i + bs]
        t0 = time.perf_counter()
        D, I = index.search(q, k)
        dt = (time.perf_counter() - t0) * 1e6
        per_query = dt / float(len(q))
        lat_us.extend([per_query] * len(q))
        all_I[i : i + len(q)] = I

    lat_us = np.array(lat_us, dtype=np.float64)
    p50, p99 = p50_p99(lat_us)

    # Recall@10 vs first 100 GT for each query
    k_gt = gt.shape[1]
    r_at_10 = recall_at_k(all_I.astype(np.int32), gt[:nq], k=k, k_gt=k_gt)

    print("\n==== FAISS IVF-PQ Reference ====")
    print(f"Recall@10: {r_at_10:.3f}  (nq={nq}, k_gt={k_gt})")
    print(f"Latency per query (us): P50={p50:.0f}, P99={p99:.0f}")
    print(f"Config: nlist={args.nlist}, m={args.m}, nbits={args.nbits}, nprobe={args.nprobe}")

    # Exit code indicates whether target >= 0.70 (useful for CI gating in the future)
    if r_at_10 >= 0.70:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

