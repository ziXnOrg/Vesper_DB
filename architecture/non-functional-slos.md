# Non‑Functional SLOs (Phase 1)

Status: Proposed
Date: 2025-08-28
Source of Truth: blueprint.md • Bench methodology: benchmark-plan.md

## Objectives
Define measurable service level objectives (SLOs) for latency, quality (recall), throughput, memory/footprint, and recovery/durability. Provide acceptance tests and measurement protocols to gate releases and performance regressions.

## Environment & Determinism (baseline)
- Hardware: x86_64 CPU with AVX2 (AVX-512 optional); 8–16 logical cores; NVMe SSD
- OS: Ubuntu LTS runner (exact image pinned in CI later)
- Toolchain: GCC≥12 or Clang≥15; CMake≥3.24
- Determinism: fixed seeds; decoding policy (temperature=0.0, top_p=1.0, n=1)
- CPU policy: feature-gated; record SIMD level (scalar/AVX2/AVX-512) in outputs

## Datasets (per benchmark-plan.md)
- SIFT1M (128-D), MSMARCO/MTEB (384–768-D), filtered synthetic datasets with selectivity 1%/5%/20%/50%/80%

## SLOs (initial targets)
- Latency (search): p50 ≤ 3 ms; p99 ≤ 20 ms on 128–1536D at recall@10 ≥ 0.95 (IVF‑PQ or HNSW as appropriate)
- Quality: recall@10 ≥ 0.95 (tunable per workload); report recall@1 and recall@100 additionally
- Throughput: ≥ 10k QPS aggregate on 8–16 logical cores for 128D SIFT1M at recall gate above (exact target refined in Phase 8)
- Ingest/build: ≥ 50k vectors/s (dataset-dependent; report with confidence intervals)
- Recovery: WAL replay ≤ 60 s for 100M ops; snapshot restore ≤ 5 min for typical snapshot size (report data size)
- Memory footprint: configurable; document peak RSS during build/search; target ≤ budget B configured per run

## Acceptance tests (per release gate)
- Perf run: execute standard suite with warmup (2 min) + 3×3 min steady state per query set; record percentiles and confidence
- Quality run: compute recall@k using held-out ground truth; must meet or exceed gates
- Filter cost analysis: measure distance evals and adjacency fetch deltas across selectivity tiers
- Recovery test: power-failure simulation (torn/truncated WAL frame); replay is idempotent; time-to-ready ≤ SLO
- Footprint test: report peak RSS and SSD read sizes; must be ≤ configured budgets

## Measurement protocol
- Pin worker affinities; isolate CPUs; set governor=performance
- Record: CPU model/features, kernel version, compiler flags, SIMD path, dataset commit/ID, seeds
- Export: CSV of metrics, JSON summary, and flamegraphs for hot kernels

## Release gates (initial)
- G1: latency_p50_ms ≤ 3.0 AND latency_p99_ms ≤ 20.0
- G2: recall_at_10 ≥ 0.95
- G3: recovery_wal_seconds ≤ 60 AND snapshot_restore_minutes ≤ 5
- G4: tests pass; no regressions > 5% vs baseline on any primary metric

## Reporting schema linkage
- Perf/quality metrics to be summarized into `experiments/examples/eval.report.json` shape for CI trends

## Risks & mitigations
- Hardware variability → pin runners; capture full env fingerprints
- Dataset drift → pin dataset versions/URLs; checksum
- SIMD cliffs → always report both scalar and SIMD reference runs periodically

## References
- blueprint.md (§3 Performance Targets, §6 Storage, §16 Testing)
- benchmark-plan.md

