# Distance Kernels — Pseudocode Spec

## Pre‑conditions
- Data aligned/padded; SIMD feature flag detected

## L2 (scalar reference)
```
Input: a[0..d), b[0..d)
Output: sum (float32)

s ← 0
for i in 0..d-1:
  dx ← a[i] - b[i]
  s ← s + dx*dx
return s
```

## IP (scalar reference)
```
Input: a[0..d), b[0..d)
Output: dot (float32)

d ← 0
for i in 0..d-1:
  d ← d + a[i]*b[i]
return d
```

## ADC lookup (PQ)
```
Input: q', code u[0..m), LUTs t[m][256]
Output: dist

s ← 0
for i in 0..m-1:
  s ← s + t[i][ u[i] ]
return s
```

## Failure modes
- ULP drift vs SIMD; enforce tolerances in tests
- Misaligned buffers → performance cliffs

