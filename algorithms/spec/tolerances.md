# Numerical Tolerances and Feature Flags

## SIMD vs Scalar tolerances (ULP)
- L2: ≤ 1 ULP per 256 elements (accumulated); final sum within 2 ULPs
- IP: ≤ 1 ULP per 256 elements; final dot within 2 ULPs
- ADC (PQ): table lookups exact; accumulated sum within 1 ULP (integer LUT indices)

## Seeds and determinism
- Default seeds: [17, 23, 43]
- All stochastic components (kmeans init, HNSW level assignment) use fixed seeds

## Feature flags
- Scalar reference path required
- AVX2 and AVX‑512 gated by build flags and runtime checks

