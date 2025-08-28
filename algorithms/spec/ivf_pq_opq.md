# IVF‑PQ/OPQ — Pseudocode Spec

## Pre‑conditions
- Trained coarse quantizer `C` with `nlist` centroids
- PQ codebooks `Q = {q_1..q_m}` with `nbits` per subquantizer
- Optional OPQ rotation matrix `R`

## Build (train + encode)
```
Input: vectors V (N×d), params {nlist, m, nbits}
Output: centroids C, codebooks Q, codes U, assignments A

1: C ← kmeans(V, nlist)
2: if OPQ: V' ← R · V else V' ← V
3: Partition dims into m blocks; learn sub-codebooks Q_i on V'
4: For each v in V':
5:   a ← argmin_c ||v - C[c]||^2 ; A.append(a)
6:   u ← encode_pq(v, {Q_i}); U.append(u)
```

## Search (ADC)
```
Input: query q, params {nprobe, R?}
Output: top-k ids

1: if OPQ: q' ← R · q else q' ← q
2: t[i][j] ← LUTs for each subspace i and codebook entry j
3: P ← select nprobe closest centroids to q'
4: Candidates ← []
5: for each c in P:
6:   for each (id,u) in inverted_list[c]:
7:     d ← adc_distance(q', u, t)
8:     Candidates.push((id,d))
9: return topk_select(Candidates, k)
```

## Complexity
- Build: O(N·d·iters) for k‑means; PQ learning O(N·d)
- Search: O(nprobe·L + k log k), where L is postings per list

## Failure modes
- Poor training (unbalanced lists) → degraded recall/latency
- Mismatch between OPQ rotation and codebooks → quality loss

