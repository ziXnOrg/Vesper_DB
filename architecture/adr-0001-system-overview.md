# ADR-0001: System Overview and Goals

Status: Proposed
Date: 2025-08-28

## Context
Vesper is an embeddable, CPU-only vector search engine emphasizing crash-safety, filtered ANN, and predictable performance. It targets on-device and air-gapped deployments across Linux/macOS/Windows.

## Decision
Adopt the single-process embeddable library architecture with three pluggable index families (IVF-PQ/OPQ, HNSW, Disk-oriented graph) over a WAL+snapshot storage engine, as specified in `docs/blueprint.md`. No network IO by default.

## Consequences
- Clear separation of hot (HNSW) vs compact (IVF-PQ) vs scale (Disk-graph) paths
- Deterministic crash recovery via WAL checksums + atomic snapshot publish
- Portable C++20 codebase with SIMD feature gating and scalar fallbacks


## Traceability to blueprint.md
- §4 High‑Level Architecture → overall component layout
- §5 Index Families → rationale for three families
- §6 Storage Engine → WAL/snapshots decisions

## References
- `docs/blueprint.md`
- `docs/prompt-dev-roadmap.md`

