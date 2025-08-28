# ADR-0004: Concurrency and Memory Reclamation

Status: Proposed
Date: 2025-08-28

## Context
High read concurrency with safe, predictable writes; immutable segments must be wait-free for readers.

## Decision
Use RCU-style epoch pinning for readers of immutable structures; single-writer per collection with coarse locks for mutable state; memory reclamation via epochs/hazard pointers for transient graph data.

## Consequences
- Readers avoid blocking writers
- Safe reclamation avoids ABA/UAF issues
- Predictable contention profile


## Traceability to blueprint.md
- §12 Concurrency & Correctness → readers/writers, reclamation
- §9 SIMD, Caches & NUMA → pmr arenas and alignment implications

## References
- `docs/blueprint.md#12-concurrency-correctness`

