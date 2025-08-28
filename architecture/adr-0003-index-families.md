# ADR-0003: Index Families and Segment Lifecycle

Status: Proposed
Date: 2025-08-28

## Context
We need flexible indexing to balance latency, memory, and scale while supporting metadata filters.

## Decision
Provide three index families selectable per collection and even per segment: IVF-PQ/OPQ (compact), HNSW (hot/in-memory), Disk-oriented graph (scale). Segment lifecycle: mutable -> sealed -> compacted.

## Consequences
- Predictable latency for compact segments (IVF-PQ)
- Low-latency hot shards (HNSW)
- Billion-scale on SSD (Disk-graph)

## References
- `docs/blueprint.md#5-index-families-search-algorithms`
- `docs/blueprint.md#7-filtering-payload-indexing`

