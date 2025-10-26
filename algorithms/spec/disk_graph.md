# Disk‑Oriented Graph — Pseudocode Spec

## Pre‑conditions
- SSD‑resident adjacency; RAM cache for medoids, entry points, codebooks

## Build (RobustPrune)
```
Input: vectors V, params {degree, alpha}
Output: graph G (on disk)

1: G ← empty flat graph
2: for each v in V:
3:   N ← candidate neighbors by heuristic
4:   E ← robust_prune(N, degree, alpha)
5:   persist adjacency list for v (sorted by id)
```

## Search (SSD‑aware)
```
Input: query q, beam, cache, filter mask F
Output: top-k ids

1: frontier ← beam entry points
2: while frontier not empty:
3:   fetch adjacency lists in order (minimize random reads)
4:   for n in neighbors:
5:     if F[n] == 0: continue
6:     evaluate distance (ADC-assisted if PQ)
7:     update frontier
8: return topk_select(C, k)
```

## Failure modes
- Poor locality → excessive random reads
- Cache too small → latency cliffs


## Preconditions / Postconditions
- Preconditions: SSD with sufficient IOPS; cache size configured
- Postconditions: adjacency lists persisted sorted by id; cache warms deterministically

## Edge cases & fallbacks
- Excessive random reads → reorder fetch plan; increase beam or cache
- Filter skipping causes beam starvation → inject unfiltered pivots periodically

## References
- docs/blueprint.md §5.3 Disk‑graph

