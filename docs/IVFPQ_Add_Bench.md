# IVF-PQ add() micro-benchmark (Task 18 acceptance)

The ivfpq_add_bench tool measures add() throughput for IVF-PQ under two modes and emits JSONL or human-readable text:
- mode=bruteforce: ANN disabled (use_centroid_ann=false)
- mode=ann: ANN-enabled centroid assignment (HNSW), sampled correctness validation, and Stats capture

It is intended to serve as the acceptance harness for Task 18.

## Build

The target is enabled by default in the build and lives under tools/:
- Target: ivfpq_add_bench (CMake)
- Output: build-<cfg>/Release/ivfpq_add_bench(.exe)

## Usage

```
ivfpq_add_bench [--json|--text|--both] [--acceptance]
  --dim=128 --nvec=200000 --nlist=2048 --m=16 --nbits=8 --chunk=20000 --seed=123456789
  --ef_search=64 --ef_construction=100 --validate_rate=0.01
  --speedup_threshold=3.0 --max_mismatch_rate=0.01
```

- --json (default): emits one JSON object per mode plus a final summary object
- --text: prints brief human-readable metrics
- --acceptance: exits with non-zero code if thresholds are not met (see below)

## Output fields (JSONL)

Per-mode entries include:
- mode: "bruteforce" | "ann"
- params: dim, nvec, nlist, m, nbits, chunk, seed, ef_search, ef_construction, validate_rate
- metrics: train_ms, add_ms, add_Mvec_per_s
- stats: ann_enabled, ann_assignments, ann_validated, ann_mismatches

Summary entry includes:
- base_add_ms, ann_add_ms
- speedup: ann_add_throughput / base_add_throughput
- ann_assignments, ann_validated, ann_mismatches, mismatch_rate

## Recommended acceptance parameters

For robust, representative validation (nlist ≥ 1024 as per Phase 1 exit criteria):
- dim: 128 (or your target)
- nvec: ≥ 200,000
- nlist: 2048 (min 1024)
- m: 16, nbits: 8
- chunk: 20,000 (batches of add())
- seed: fixed (e.g., 123456789)
- ef_search: 200
- ef_construction: 200
- validate_rate: 0.01 (1% sampled brute-force validation)
- speedup_threshold: 3.0 (ANN must be ≥3x faster than brute-force add())
- max_mismatch_rate: 0.01 (≤1% among validated samples)

Example (JSONL + acceptance gating):
```
ivfpq_add_bench --json --acceptance \
  --dim=128 --nvec=200000 --nlist=2048 --m=16 --nbits=8 --chunk=20000 --seed=123456789 \
  --ef_search=200 --ef_construction=200 --validate_rate=0.01 \
  --speedup_threshold=3.0 --max_mismatch_rate=0.01
```

Exit codes:
- 0: success (criteria met or acceptance not requested)
- 2: acceptance failed (speedup and/or mismatch-rate criteria not met)
- 1: invalid parameters

## Notes

- The tool uses synthetic uniform vectors with deterministic seeding for reproducibility.
- ANN performance is sensitive to ef_search and dataset; raise ef_search for quality and consider higher nvec for stable timing signals.
- Stats are provided by IvfPqIndex::get_stats(): ann_enabled, ann_assignments, ann_validated, ann_mismatches.
- This tool does not persist the index; it focuses on add() path throughput and ANN coarse-assignment correctness telemetry.

