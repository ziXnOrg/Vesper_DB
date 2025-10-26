# Component Map (Phase 1)

## Diagram
```mermaid
flowchart LR
  subgraph Host
    APP[Host App]
  end
  subgraph VESPER[VESPER LIB]
    QP[Query Planner]
    IDX[Index APIs\nIVF-PQ/OPQ | HNSW | Disk-Graph]
    FIL[Filters\nRoaring Bitmaps]
    STM[Storage\nWAL | Segments | Snapshots]
    SIMD[SIMD Kernels\nDistance | PQ LUTs | Top-k]
    TEL[Telemetry\nMetrics | Histograms | Logging]
    CRY[Crypto (optional)\nXChaCha20-Poly1305 | AES-GCM]
  end

  APP --> QP
  QP --> IDX
  QP --> FIL
  IDX <---> STM
  IDX --> SIMD
  QP --> TEL
  STM --> TEL
```

## Subsystems
- Query planner
- Index APIs: IVF-PQ/OPQ, HNSW, Disk-graph
- Storage: WAL, immutable segments, snapshots/compaction
- Filters: Roaring bitmap compiler and intersection
- SIMD kernels: distance, PQ LUTs, top-k selection
- Telemetry: metrics, histograms, logging
- Crypto (optional): XChaCha20-Poly1305; AES-GCM (FIPS mode)

## Threading & memory ownership (LLD excerpt)
- Thread pools with work-stealing; per-thread pmr arenas
- Immutable segment maps shared read-only; mutable guarded by locks
- Ownership documented per component; error boundaries via `std::expected`

## Error boundaries
- Planner: returns `expected<Result, error_code>`; never throws in hot paths
- Storage: fs ops return `expected<size_t, io_error>`; classify retryable vs fatal
- Index: search/build return `expected<Stats, error_code>` with structured stats
- Crypto: optional; errors isolated and never fail-open on integrity checks

## Interfaces & ownership
- Query planner owns transient scratch arenas (pmr) per thread; callers own query buffers
- Indexes own their segment memory/maps; storage owns file descriptors and mmaps
- Filters own bitmap indexes per field/segment; shared by planner via read-only views
- Telemetry sinks are owned by the app; library exposes counters/histograms via handles

