
# Vesper Benchmark Plan (v0)

## Goals
Quantify latency/throughput/recall tradeoffs and the effects of filters, PQ/OPQ, and index families on commodity CPUs.

## Datasets
- **SIFT1M / SIFT1B** (128‑D) for legacy comparability.
- **DEEP1B subset** (96‑D), **MSMARCO‑Passage** embeddings (768‑D), **MTEB text** (384/768‑D), **GIST1M** (960‑D).
- Synthetic filtered datasets: add categorical/range metadata with controlled selectivity (1%, 5%, 20%, 50%, 80%).

## Metrics
- Latency: P50, P90, P99.
- Throughput: QPS at fixed recall targets (0.9/0.95/0.98).
- Quality: recall@k (k∈{1,10,100}).
- Efficiency: CPU cycles/query, SSD reads/query, cache hit ratios.
- Build/ingest rate and compaction debt.
- Recovery time (WAL replay) and snapshot publish latency.
- Filter cost: delta in distance evaluations and adjacency fetches.

## Protocol
- Warm‑up, then steady‑state runs (3× 3 min) with randomized seeds and shuffled queries.
- NUMA pinned workers; isolated cores; governor=performance.
- Fixed compiler flags and instruction sets per run (e.g., `-mavx2` vs `-mavx512f`).

## Comparisons
- IVF‑PQ/OPQ vs HNSW vs Disk‑graph at matched recall/footprint.
- ADC `fp16` LUTs vs `fp32` LUTs.
- Filter‑aware traversal vs naive post‑filtering.
- Top‑k heap vs Floyd–Rivest partial selection.

## Outputs
- CSV for all metrics; plots (CDFs, Pareto recall‑latency curves).
- Flamegraphs and perf maps of hot kernels.
