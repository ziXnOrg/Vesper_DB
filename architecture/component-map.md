# Component Map (Phase 1)

- Query planner
- Index APIs: IVF-PQ/OPQ, HNSW, Disk-graph
- Storage: WAL, immutable segments, snapshots/compaction
- Filters: Roaring bitmap compiler and intersection
- SIMD kernels: distance, PQ LUTs, top-k selection
- Telemetry: metrics, histograms, logging
- Crypto (optional): XChaCha20-Poly1305; AES-GCM (FIPS mode)

Threading & Memory ownership (LLD excerpt)
- Thread pools with work-stealing; per-thread pmr arenas
- Immutable segment maps shared read-only; mutable guarded by locks
- Ownership documented per component; error boundaries via expected<>

