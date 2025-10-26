# ADR-0001: Hybrid Sparse+Dense Search (BM25 + Vector) with Fusion

Date: 2025-10-20
Status: Accepted
Authors: Vesper Team

## Context
- We need a first-class hybrid searcher that combines sparse lexical (BM25) and dense vector results.
- Goals:
  - Support text-only, embedding-only, and combined queries
  - Provide deterministic, reproducible behavior with fixed seeds
  - Meet performance budgets (P50 1–3ms, P99 10–20ms at target recall)
  - Preserve existing IVF-PQ APIs and IndexManager orchestration
- Constraints:
  - CPU-only, C++20; no exceptions on hot paths; explicit error codes
  - Backward-compatible serialization across versions (BM25 pending)

## Decision
Implement `HybridSearcher` that executes sparse and dense searches (parallel or staged) and fuses ranked lists using pluggable fusion algorithms:
- Reciprocal Rank Fusion (RRF)
- Weighted Sum normalization-based fusion
- Max Score (simple fallback)
- Late Interaction stub (fallback to Weighted for now)

Expose configuration via `HybridSearchConfig`:
- strategy: AUTO | PARALLEL | DENSE_FIRST | SPARSE_FIRST
- k, rerank_factor
- fusion_algorithm and params (rrf_k, dense_weight, sparse_weight)

## Architecture Mapping
- Dense path: IndexManager → IVF-PQ (and others) using QueryConfig
- Sparse path: BM25Index → tokenize → term stats → scoring
- Fusion: `vesper::search::fusion::*` implementations
- Filters: optional Roaring bitmaps propagated to both paths

## Alternatives Considered
1) Dense-only with lexical re-ranking
   - Pros: Simpler end-to-end; fewer moving parts
   - Cons: Loses sparse-only queries; weak lexical coverage; lower user-perceived quality on some corpora
2) Sparse-only with dense re-ranking
   - Pros: Strong lexical control; simple integration
   - Cons: Fails pure embedding use-cases; recall suffers for semantic queries
3) Single late-interaction model (ColBERT-style) only
   - Pros: Unified scoring
   - Cons: Heavier compute; token-level machinery not ready; higher complexity

Decision: Hybrid with pluggable fusion provides best coverage, incremental complexity, and clear evolution path.

## First Principles & Invariants
- Numerical stability: no NaN/Inf in fusion or scoring; clamp/normalize as needed
- Determinism: stable ordering ties; fixed seeds where randomness exists
- Correctness: recall targets per CLAUDE.md; fusion preserves rank coherently
- Performance: parallel execution when both modalities provided; bounded allocations, SoA-friendly inner loops

## Compatibility & ABI
- Public C++ APIs preserved; C API additions are optional wrappers (future)
- No breaking changes to IVF-PQ serialization; BM25 serialization is new (versioned)

## Testing & Validation
- Unit: fusion correctness (RRF/Weighted), tie-stability, normalization bounds
- Integration: HybridSearcher end-to-end on synthetic fixtures; filters pass-through
- Property: randomized ranked-list fusion invariants (id stability, monotonicity under weights)
- Bench: latency vs strategy (AUTO/PARALLEL), effect of rerank_factor

## Security & Reliability
- Strict input validation for configs and query content
- Defensive bounds on k and rerank_factor; fail-closed on invalid inputs

## Consequences
- Additional code paths (sparse+dense) to maintain; requires BM25 persistence to be production-ready
- Parallel execution can increase CPU usage; mitigated by configurable strategy

## Rollout Plan
- Land HybridSearcher + fusion algorithms (done)
- Add ADR (this doc) and API reference sections (follow-up PR)
- Implement BM25 serialization + roundtrip tests; wire into docs and CI (follow-up PR)

## Open Issues / Follow-ups
- BM25 serialization (versioned, checksummed) + roundtrip tests [P1]
- Late Interaction: token-level scoring integration (future)
- C API coverage for hybrid queries (future)

## References
- CLAUDE.md (budgets, testing)
- docs/Open_issues.md (roadmap)
- src/search/hybrid_searcher.cpp; src/search/fusion_algorithms.cpp; src/index/bm25.cpp

