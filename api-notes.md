
# Vesper API Notes (no code)

## Core Types
- `Collection` — owns configuration, segments, and storage directory.
- `Segment` — `mutable` or `sealed`, index‑family specific.
- `SearchParams` — metric, `k`, `target_recall`, beam/ef, `nprobe`, rerank size.
- `FilterExpr` — boolean AST (`And`, `Or`, `Not`, `Term`, `Range`), compiled to roaring bitmaps.

## Critical Operations
- `insert(id, vector, metadata)` — appends WAL; updates mutable index.
- `remove(id)` — tombstone; WAL append.
- `search(query, params, filter)` — unified planner + executor, returns `(id, score)`.
- `seal_segment()` — flush mutable; finalize on disk without blocking readers.
- `compact()` — merge sealed segments; publish atomically.
- `snapshot()` — point‑in‑time copy (manifest + hardlinks/copies).
- `recover()` — WAL replay + snapshot load.

## Stability
- No exceptions on hot paths: return `expected<T, error_code>`.
- Thread safety: multiple concurrent readers; single writer per collection (configurable).

## FFI
- Stable C ABI surface with POD structs; versioned; zero‑copy buffers where safe.
