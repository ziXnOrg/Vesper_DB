# Filters — Pseudocode Spec

## Pre‑conditions
- Per-field Roaring bitmaps; optional dictionaries for string/enum

## Compile and intersect
```
Input: FilterExpr AST, segment indexes
Output: bitmap mask F

1: F ← compile(FilterExpr)
2: for each term T in FilterExpr:
3:   B_T ← bitmap_for_term(T)
4:   F ← F AND B_T
5: return F
```

## Failure modes
- High-cardinality terms inflate memory
- Degenerate filters (tiny masks) → switch to brute-force path

