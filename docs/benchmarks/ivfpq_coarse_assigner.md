# IVF-PQ Coarse Assigner Grid (HNSW vs KD-tree vs Projection)

This document summarizes add() throughput and accuracy when using different ANN coarse assignment strategies for IVF-PQ centroid assignment.

Common setup: dim=128, nvec=200k, nlist=2048, m=16, nbits=8, chunk=20k, seed=123456789, validate_rate=1%.

- HNSW search parameters: ef_search ∈ {96, 128, 192}
- Refinement: refine_k ∈ {64, 96, 128, 192}
- Additional HNSW construction variants explored: ef_construction ∈ {100, 200, 300}, hnsw_M ∈ {8, 16, 32}
- Projection assigner used projection_dim=16

Raw CSV: tools/results/ivfpq_coarse_assigner_grid.csv

> Coarse assigner defaults
>
> - Default coarse quantizer: KDTree (exact)
> - `use_centroid_ann = true` by default to enable non-brute assigners. For KDTree this means using the exact KD assignment path; HNSW/Projection toggles only apply when those assigners are selected.
> - HNSW defaults (if selected): `ef_search=96`, `ef_construction=200`, `M=16`, `refine_k=96`.

## Update (2025-09-12)

- New CSV with KD parameters: tools/results/ivfpq_coarse_assigner_grid_v2.csv (adds kd_leaf_size and kd_split columns).
- KD-tree micro-opts re-bench at ef_search=128, refine_k=128 with kd_leaf_size:
  - 64 → speedup 0.556×, mismatch 0.00%
  - 128 → 0.685×, 0.00%
  - 256 → 0.718×, 0.00%
  - 512 → 0.803×, 0.00% (best of the four on this setup)
- Projection assigner crash at ef=128, k=128 on Windows is fixed (tail remainder handling in AVX2 path); still very inaccurate and slow.
- HNSW 12-cell grid re-run confirms prior conclusion: still slower than brute with 8.8–22.2% mismatch across the tested cells.

## Headline findings

- KD-tree: 0.0% mismatch across all measured cells, but consistently slower than brute (speedup ~0.68–0.85×).
- HNSW: Faster than KD-tree in assignment stage but still slower overall than brute (speedup ~0.26–0.38× in our subset) with significant mismatch (10–17%). Construction tweaks (ef_construction, hnsw_M) did not change the order-of-magnitude.
- Projection assigner: Substantially slower than brute with very high mismatch (83–95%) and one crash at ef_search=128, refine_k=128 on Windows (seh_unhandled:0xC0000005). Not recommended.

Acceptance targets (≥3× add() speedup vs brute and ≤1% mismatch) were not met by any assigner/setting in this matrix.

## Representative cells (speedup vs brute, mismatch)

- KD-tree
  - ef=96, k=192 → 0.686×, 0.00%
  - ef=128, k=128 → 0.810×, 0.00%
  - ef=192, k=128 → 0.799×, 0.00%
- HNSW (variants)
  - ef=128, k=96, (efc=200, M=32) → 0.367×, 16.68%
  - ef=192, k=128, (efc=300, M=8) → 0.318×, 13.76%
  - ef=192, k=128, (efc=100, M=32) → 0.305×, 10.63%
- Projection (dim=16)
  - ef=96, k=128 → 0.073×, 89.37%
  - ef=128, k=192 → 0.074×, 83.17%
  - ef=192, k=96 → 0.092×, 92.73%

## Notes

- All runs include the new HNSW index-level shared_mutex with shared/exclusive lock wiring, and centroid-index read-only enforcement during assignment.
- The previously failing cell (ef=192, k=128) is now stable but remains far from acceptance.
- Projection assigner produced one crash on this environment; entry recorded in CSV.

## Next steps (suggested)

- Prefer KD-tree as default coarse quantizer (params.coarse_assigner=KDTree) for correctness; consider batching/implementation optimizations to improve throughput.
- Keep HNSW as optional/fallback; if pursuing it further, explore:
  - Higher ef_search, multi-EP search, or stronger candidate verification (but expect further slowdown)
  - Heavier construction (efc, M) showed limited gains here
- Remove projection assigner from recommended paths (or gate behind an experimental flag) until accuracy/perf issues are resolved.
