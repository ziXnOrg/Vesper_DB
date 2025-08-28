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


## Preconditions / Postconditions
- Preconditions: bitmap indexes built per field; dictionary encoding for high-cardinality strings
- Postconditions: `F` size matches segment cardinality; AND sequence is associative/commutative

## Edge cases & fallbacks
- Empty result mask → planner short-circuits search
- Very large enumerations → switch to range encoding or tiered bitmaps

## References
- docs/blueprint.md §7 Filtering

