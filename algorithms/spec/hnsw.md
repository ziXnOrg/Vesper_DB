# HNSW — Pseudocode Spec

## Pre‑conditions
- Capacity and memory budget determine M, efConstruction

## Build
```
Input: vectors V, params {M, efConstruction}
Output: graph G

1: G ← empty multi-layer graph
2: for v in V (random level assignment):
3:   enter ← select_entry_point(G)
4:   greedy_search from enter with efConstruction
5:   connect v to best M neighbors per layer
```

## Search
```
Input: query q, params {efSearch}, filter mask F
Output: top-k ids

1: enter ← select_entry_point(G)
2: C ← best-first search with efSearch
3: for each neighbor n popped:
4:   if F[n] == 0: continue
5:   relax edges and update candidate set
6: return topk_select(C, k)
```

## Failure modes
- Low efSearch reduces recall
- High M increases memory footprint and build time

