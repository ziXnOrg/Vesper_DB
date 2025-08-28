# Top‑k and Query Planner — Pseudocode Spec

## Planner outline
```
Input: params {metric, k, target_recall, time_budget}, filter expr, segment set S
Output: top-k results

1: F ← compile_and_intersect(filter expr)
2: for seg in S:
3:   if small(seg,F): path ← brute_force
4:   else if seg.mutable: path ← HNSW
5:   else if seg.size < THRESH: path ← IVF-PQ
6:   else: path ← Disk-graph
7:   R_seg ← execute(path, F)
8: return merge_topk(R_seg, k)
```

## Top‑k selection (partial selection)
```
Input: candidates C
Output: top-k

return quickselect_k(C, k)
```

## Failure modes
- Misselection of path → perf/recall regression; log planner decisions
- Merge bias across segments → ensure deterministic tie‑breakers

