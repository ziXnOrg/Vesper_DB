# Vesper Project Roadmap (Single Source of Truth)

This document tracks status, priorities, acceptance criteria, success metrics, dependencies, and release readiness for the IVF-PQ subsystem and related components. It is intended to guide development from now until "ship-ready" status while leaving space for opportunistic optimizations.

---

## Current Status (as of today)

Completed milestones:
- [x] Task 15: Persistence performance benchmarks (tools/ivfpq_persist_bench)
- [x] Task 16: Robustness and fuzz tests for v1.1 loader (deterministic + randomized multi-section; permutations/duplications/overlaps)
- [x] Task 17: Documentation for v1.1 serialization format, versioning policy, CLI/API usage (docs/IVFPQ_Serialization_v11.md)

- [x] Projection assigner AVX2 tail remainder fix; unit tail test and stability smoke test added; integrated in aggregate and discoverable via CTest (MSVC multi-config)
- [x] Catch2/CTest on MSVC multi-config fixed (CONFIGURATIONS applied); CMake aggregate corrected to exclude GoogleTest sources
- [x] KD-tree coarse assigner params exposed (kd_leaf_size, kd_batch_assign, kd_split heuristic); OpenMP parallel leaf scans implemented; env overrides preserved

Open PRs:
- None (previous PR closed). Commits will be created after Task 18 and roadmap finalize.

Format/features highlights:

Component status snapshot:
- HNSW index: Shared_mutex synchronization integrated; aggregate tests green; performance vs baselines pending revalidation; optional coarse assigner track only (non-default)
- IVF-PQ: v1.1 serialization stable with fuzz/robust tests; OPQ supported; coarse assigner selector plumbed; add/search pipelines stable
- KD-tree coarse assigner: Default candidate; 0% mismatch; throughput below target; knobs exposed (kd_leaf_size, kd_batch_assign, kd_split); OpenMP leaf scans enabled
- Projection coarse assigner: Stability fixed (tail AVX2 remainder) with tests; accuracy not acceptable; experimental only
- Hybrid search (BM25 + fusion): implemented and integrated; BM25 serialization (save/load) + roundtrip tests pending

- Test/Build: Catch2 discovery fixed for MSVC multi-config; CTest discovery/filters working; projection tests integrated into aggregate

- v1.1 sectioned binary with optional Zstd, FNV-1a checksums, mmap-friendly
- Optional metadata JSON (size-limited, roundtrip tests)
- OPQ support with PCA init and alternating optimization (guardrails + early-stop)

---
## Project Scope and Components

Vesper is a complete vector search system composed of interoperable subsystems. This roadmap spans the whole project:

- Indexing backends
  - IVF-PQ (current focus)
  - HNSW in-memory graph (include/vesper/index/hnsw.hpp; src/index/hnsw.cpp)
  - Disk Graph (on-disk/searchable graph; include/vesper/index/disk_graph.hpp)
  - Matryoshka and related quantization variants (include/vesper/index/matryoshka.hpp)
  - Product Quantizer and variants (product_quantizer.hpp, pq_fastscan.hpp, pq_simple.hpp)
  - KMeans (standard and Elkan) for training (kmeans*.{hpp,cpp})

- Core infrastructure
  - Kernels: distance, batch top-k, dispatcher (include/vesper/kernels/*)
  - Memory: NUMA-aware allocation (memory/numa_allocator)
  - I/O: async I/O, prefetching, io_uring (io/*)
  - Serialization: sectioned formats (IVF-PQ v1.1), checksums, optional compression
  - Platform abstractions: SIMD, threading, CPU features, filesystem
  - Metadata store (metadata/metadata_store)

- Query planning and execution
  - Query planner (src/index/query_planner.cpp)
  - Filters (bitmap/roaring) and predicate evaluation (filter/*, filter_eval)
  - Multi-index orchestration (index_manager)

- Durability and lifecycle
  - WAL (frame, manifest, replay, retention, snapshot, checkpoint)
  - Tombstone management using roaring bitmaps (tombstone_manager)

- SDKs, C API, and tooling
  - Public C API header (include/vesper/vesper_c.h), examples (include/vesper/examples.md)
  - Benchmarks and tools (tools/*)

---

## Cross-Component Integration Map

- Ingestion path
  - index_manager orchestrates adds/updates; tombstone_manager tracks deletions
  - WAL ensures durability; checkpoints/snapshots coordinate with index persistence
  - Metadata_store records collection/index schema and runtime knobs

- Query path
  - query_planner selects index backend(s), probes/ef settings, and merges results
  - Filters (roaring) applied pre/post ranking; uses filter_eval and bitmap filters
  - Kernels/dispatcher provide optimized distance/top-k; NUMA/memory placement honored

- Persistence and recovery
  - Per-index serialization format (IVF-PQ v1.1 today); WAL replays outstanding ops
  - Retention and snapshot policies bound replay time; recovery validated by tests

- Platform and performance
  - SIMD/AVX2 kernels, OpenMP guarded parallelism; optional io_uring on Linux
  - NUMA allocator used in memory-heavy components; prefetch manager for cold I/O

Dependencies to track (non-exhaustive):
- index_manager ↔ query_planner ↔ filter_eval/roaring
- index_manager ↔ tombstone_manager ↔ WAL (manifest, retention, snapshot)
- index backends ↔ kernels/dispatcher ↔ memory/numa ↔ platform features
- persistence formats ↔ IO layer (async, mmap, io_uring) ↔ checksums/compression

---

## Architecture & Design Decisions (Open Items)

- API surface and stability
  - Consolidate C++ API for index lifecycle and queries; stabilize C API (vesper_c.h)
  - Versioning policy for on-disk formats across all backends (IVF-PQ done)

- Index backend abstraction
  - Common interface for train/add/search/save/load; plugin-style registration in index_manager
  - Capability descriptors for planner (supports filters, supports delete/tombstones, etc.)

- Concurrency model
  - Thread-safety guarantees for add()/search()/persist(); per-backend locking strategies
  - Background jobs (compaction, ANN construction), cooperative yielding, cancellation

- Memory & NUMA
  - Standardize ownership boundaries; zero-copy where possible; NUMA pinning policies
  - Bounded-memory modes (spill, streaming build) and backpressure

- Durability & recovery
  - WAL sequencing across multi-index writes; idempotency; snapshot atomicity contract
  - Crash consistency matrix and recovery guarantees documented and tested

- Observability
  - Unified telemetry: counters/timers for ingestion, search, persistence, recovery
  - Diagnostics dumps for indexes and planner decisions (rate-limited)

- Multi-tenancy & namespaces
  - Define collection/tenant namespace model and directory layout; propagate namespace IDs through index_manager, metadata_store, tombstone_manager, and WAL manifest
  - Resource isolation plan: per-tenant memory/CPU/query-rate limits; admission control hooks; background job scheduling fairness
  - Naming/versioning conventions for multi-tenant persistence (snapshots, manifests, retention policies)

- Replication & high availability (design-first)
  - Choose initial model aligned with existing WAL+snapshot: snapshot shipping + WAL tailing (asynchronous)
  - Consistency options: eventual (default) with bounded lag; optional read-your-writes within shard (future)
  - Replica roles awareness in APIs (leader/follower), backpressure on lag; failover orchestration is longer-term


---

## System-wide Performance & Scalability Goals

- Indexing throughput
  - IVF-PQ add(): ≥1.5x vs brute-force for nlist≥1024 with ANN assignment
  - HNSW: target parity with common baselines on SIFT-1M; tunable efConstruction
  - Disk Graph: sustained build throughput with bounded memory and prefetching

- Query latency and throughput
  - P50/P95/P99 targets at defined recall@k; planner selects backends and params
  - Filtered search overhead ≤15% vs unfiltered for typical predicates

- Availability & consistency
  - Recovery time objectives (RTO): ≤ N minutes per 100M vectors via snapshot + WAL
  - Recovery point objectives (RPO): 0 for single-node crash; ≤ S seconds lag target for async replicas (future)
  - Replica lag metrics and alerting (future)

- Multi-tenant isolation
  - p95 per-tenant latency degradation cap under noisy-neighbor scenarios (e.g., ≤ 15%)
  - Enforce memory/CPU/query-rate quotas; measurable adherence in stress tests


- Memory budget
  - NUMA-aware allocations; peak RSS targets by dataset scale (documented per backend)
  - Streaming/external-memory modes enable datasets ≥10× RAM

- Persistence and recovery
  - Save/load I/O throughput reported by bench; cold-load times documented
  - Recovery time ≤ N minutes per 100M vectors with snapshot+WAL; tunable retention

- Scale-out
  - Multi-index sharding demo shows near-linear speedup across sockets/nodes

---

## Prioritization Framework

- Impact × Effort scoring with categories P0 (now), P1 (next), P2 (later)
- Tie-breakers: production risk, user-facing value, and de-risking for future phases
- All phases below are project-wide; each has per-subsystem tracks.


## Phase 1: Quick Wins (Project-wide)

1) v1.1 serialization polish
- Acceptance: No-crash/no-leak under corpus from fuzz tests, CI passes (Windows/Linux). Trailer/section checksums validated on random samples.
- Metrics: 0 crashes; <1% false-accept under randomized corruption corpus of 10k cases; load time parity with v1.0 in no-compression path.

2) Task 18: ANN-based centroid assignment (KD-tree default, HNSW optional; brute-force fallback retained)
- Acceptance: Verified via add() micro-bench (nlist ≥ 1024) capturing `Stats` counters (assignments, validated, mismatches); toggles via `IvfPqTrainParams`: `use_centroid_ann`, `centroid_ann_ef_search`, `centroid_ann_ef_construction`; sampled correctness validation; hard fallback to brute-force assignment retained.
- Metrics: ≥1.5x add() throughput improvement vs brute-force baseline for nlist ≥ 1024; ≤1% sampled assignment mismatch.


#### Task 18: Current findings (coarse assigners) and next actions
- KD-tree assigner: 0% mismatch across grid; initial benchmarks showed 0.56–0.85× performance. After optimizations (kd_leaf_size=64, batch mode default): achieved 1.026× speedup. Status: kd_leaf_size changed to 64 (from 256), batch mode enabled by default, kd_batch_assign, kd_split heuristic params added to IvfPqTrainParams; OpenMP parallel leaf scans implemented; env overrides preserved. Note: Performance target relaxed from 3× to 1.5× based on practical limitations. Auto-disable ANN for nlist < 1024 (bruteforce is faster).
- HNSW assigner: slower than brute with ≈8.8–22.2% assignment mismatch in tested ranges. Decision: keep optional (non-default) for experimentation; not recommended for nlist ≤4096 in current state.
- Projection assigner: severe accuracy issues; prior Windows crash fixed via AVX2 tail remainder handling. Unit test (tail) and stability smoke test added and passing in aggregate (CTest discovery fixed). Keep as experimental only.

Planned work (now):
- Re-benchmark KD/HNSW/Projection grids and persist CSV/Markdown under tools/results and docs/benchmarks (v2+ artifacts).
- Pursue KD-tree throughput optimizations (prefetch hints, reduced branching, tighter SoA buffers) while preserving 0% mismatch and stability.
- Wire coarse_assigner selector: params.coarse_assigner ∈ {Brute, KDTree (default), HNSW} with documented defaults and flags.

3) Bench harness improvements (non-invasive)
- Acceptance: Persist-bench outputs JSON lines with timings, optional memory peak; seed/reproducibility plumbing; add() throughput micro-bench wired to `Stats` capture and baseline comparison.
- Metrics: Ability to produce repeatable runs across machines (±5%).

4) Metadata & tooling (moved up from Phase 2)
- Scope: Add helper to set structured metadata; optional schema check hook; v1.1 docs updated.
- Acceptance: Examples in docs; size limit remains enforced; round-trip tests (save→load→equality) including metadata-bearing sections.
- Metrics: N/A (DX improvement).

Reserved quick wins slots:
- [ ] Small LUT prefetch tweak in ADC loop (compile-time flag)
- [ ] More robust mmap fallback heuristics on non-Windows

### Phase 1 Exit Criteria
- Performance:
  - Add() micro-bench shows ≥1.5x vs brute-force baseline for nlist ≥ 1024 on reference dataset/hardware; `Stats` counters surfaced in reports.
- Stability:
  - 0 crashes across unit/integration/fuzz for loader and writer; false-accept <1% on 10k randomized corruptions, including metadata-bearing cases.
- Format & DX:
  - v1.1 doc updated with metadata rules; API reference documents ANN toggles and metadata helper; defaults stated and tested.
- CI:
  - Windows + Linux CI passing with warnings-as-errors for relevant targets.

---

## Phase 2: Medium Priority (Project-wide)

1) ANN coarse quantizer alternatives and validation
- Scope: Expose KD-tree micro-index as a compile-time alternative (if light). Keep HNSW as default. Improve sampled correctness logging and counters.
- Acceptance: Configurable strategy; correctness sample logs gated by `verbose`; counters surfaced in `Stats`.
- Metrics: Maintain ≥1.5x add() throughput on large nlist, ≤1% mismatch rate on sample validation.

2) OPQ tuning and ergonomics
- Scope: Expose `opq_iter`, `opq_sample_n` (done), document best practices, cap time budget; add optional early exit when unrotated error < threshold.
- Acceptance: Clear guidance in docs; predictable run-times on large training sets.
- Metrics: ≤2 min OPQ wall-time default; ≥5% recall@k improvement on SIFT-128 at same latency.

3) Zstd revisit (off by default)
- Scope: Re-evaluate per-section compressibility on real corpora; keep off unless clear gains in load speed (I/O-bound) with acceptable CPU.
- Acceptance: Data-backed recommendation; gating via env.
- Metrics: If enabled: ≥1.2x faster cold-load on HDD/SATA with <10% CPU overhead.

Reserved medium-priority slots:
- [ ] Background build of centroid ANN during train vs post-train
- [ ] Pluggable distance metrics for coarse assignment (L2 only today)

---

## Phase 3: Longer Term (Project-wide)

1) Advanced recall/latency improvements
- Scope: Multi-level quantization or IVF-Flat hybrid for tails; cached PQ LUTs across probes.
- Acceptance: Feature flagged; benchmarked on public datasets.
- Metrics: +1–2% recall at same latency or ~10% latency reduction at same recall.

2) Streaming build / external-memory paths
- Scope: Allow large-scale add() with bounded memory; spill lists.
- Acceptance: Functional prototype behind flag; robust error handling.
- Metrics: Able to index >10x RAM size with predictable throughput.

3) Sharding & multi-index manager integrations
- Scope: Partitioning and routing; consistency with tombstones and WAL.
- Acceptance: Simple shard manager demo; coordinated search.
- Metrics: Linear scale-out on multi-socket / multi-node demo.

Reserved longer-term slots:
- [ ] Quantization-aware re-ranking
- [ ] Learned coarse quantizer variants

---

## Integration Points & Dependencies

- Build flags: VESPER_WITH_ZSTD, VESPER_ENABLE_OPENMP, VESPER_ENABLE_KERNEL_DISPATCH, VESPER_SERIALIZE_BASE_LAYER; CPU SIMD (AVX2); macOS Accelerate (optional)
- Env controls: VESPER_IVFPQ_SAVE_V11, VESPER_IVFPQ_ZSTD_LEVEL, VESPER_IVFPQ_LOAD_MMAP; WAL retention/snapshot via config (to be documented)
- Tests: Catch2-based unit/integration/fuzz suites for IVF-PQ, WAL (frames/manifest/replay), kernels, planner, tombstones
- Bench: tools/ivfpq_persist_bench; add() throughput micro (planned); query latency/recal bench (planned)
- API/ABI: C API (vesper_c.h) stability policy; header checks build target; examples compile
- Versioning: On-disk format versioning per backend; cross-version compat tests
- Namespacing: Tenant/collection namespace plumbed through index_manager, metadata_store, tombstone_manager, WAL manifest
- Replication: Replica-coordination layer interfaces with WAL checkpoints and snapshot shipping (future)

---

## Acceptance Criteria & Success Metrics (Summary)

- Stability: 0 crashes across fuzz suites; strict checksum validation
- Performance: ≥1.5x add() throughput (nlist≥1024) with ANN vs brute-force
- Quality: Recall within agreed targets; correctness validation sampling shows ≤1% mismatch
- DX: Clear docs; easy toggles; reproducible benches

---

## Ship-Ready Definition

- Stability: All unit/integration/fuzz suites green on CI (Windows + Linux)
- Performance SLOs: Query P50/P95/P99 within targets at agreed recall@k; indexing throughput targets met (per-backend); memory usage within budgets
- Durability: WAL replay correctness; snapshot/retention policies verified; tombstone consistency under concurrent ops
- Persistence: Save/load formats documented and tested; cross-version compat validated for supported versions
- API: C++ and C (vesper_c.h) APIs documented and stable; examples compile and run
- Observability: Basic telemetry counters/timers exposed for ingestion/search/persist/recovery
- Documentation: Architecture guide, user guides, format specs, and tuning docs available
- CI/CD: Matrix builds, tests, and release pipeline prepared

### Final Release Checklist
- [ ] CI green on main for 3 consecutive runs across Windows/Linux toolchains
- [ ] Bench report attached (save/load, add(), search) with hardware details and dataset
- [ ] Docs: Architecture overview, IVF-PQ spec, planner/filters guide, API examples validated
- [ ] Default flags reviewed (Zstd off; ANN on; safe OPQ defaults); CI matrix recorded
- [ ] Version bump and tag; changelog and upgrade notes
- [ ] Release notes with compatibility matrix and tuning guidance

---

## Release Readiness Assessment (current gaps)

- Performance
  - KD-tree coarse assigner: 0% mismatch; with optimizations achieved 1.16× throughput; approaching ≥1.5× add() target (P0 gap)
  - HNSW coarse assigner: slower than brute with 8.8–22.2% mismatch; kept optional (P1 gap)
  - Projection coarse assigner: stability fixed, accuracy remains poor; experimental only (P2 gap)
- API & Usability
  - Coarse assigner selection plumbed (params.coarse_assigner) but needs final defaulting and documentation pass (P0)
  - Public C API coverage/examples for IVF-PQ lifecycle and coarse-assigner toggles (P1)
- Testing/Validation
  - Add() throughput micro-bench in place; need formal acceptance runs tied to ≥1.5× target and report artifacts (P0)
  - Multi-index integration test currently not built in this environment; must be fixed and validated (P1)
- CI & Tooling
  - CTest discovery fixed on MSVC; need CI matrix (Windows + Linux), sanitizers, warnings-as-errors for core (P0/P1)
- Docs
  - Architecture overview, planner/filter guides, and performance tuning cookbook pending (P1)
- Durability & Recovery
  - WAL fuzz beyond frames/manifest and recovery SLO harness pending (P1)


## Work Queue (Living List)

### Index Backends

IVF-PQ
- Quick Wins
  - [x] Add ANN telemetry counters in Stats (assignments, validated, mismatches)
  - [x] Document ANN toggles and defaults in API reference (docs/API_REFERENCE.md)
  - [x] Add add() throughput micro-bench (nlist≥1024) (tools/ivfpq_add_bench)
    - Outputs JSONL with timings and peak memory; includes brute-force vs ANN speedup comparison
    - Captures `Stats` counters (assignments, validated, mismatches) and writes to report
    - Deterministic seed and dataset spec recorded for reproducibility
    - Acceptance: Used by Task 18 to validate ≥1.5x add() throughput at nlist ≥ 1024
  - [x] Expand fuzz corpus for randomized multi-section corruptions (metadata-bearing v1.1 + multi-section)
- Medium Priority
  - [ ] Optional KD-tree over centroids (compile-time or small dependency)
  - [ ] Zstd revisit with data-backed recommendation and env gating
  - [ ] Streaming build prototype with bounded memory
- Longer Term
  - [ ] IVF-Flat hybrid tails; cached PQ LUTs across probes
  - [ ] Learned/advanced coarse quantizer variants

HNSW
- Quick Wins
  - [ ] Document defaults and recommended ranges (efConstruction/efSearch)
  - [ ] Validate connectivity and recall targets in tests
- Medium Priority
  - [ ] Thread-pool ergonomics and partitioned-base locks benchmarking
  - [ ] Batch insert path and memory optimizations
- Longer Term
  - [ ] Lock-free variant exploration; disk-bake/mmap search path

Disk Graph
- Quick Wins
  - [ ] Document API and expected performance envelope
  - [ ] Prefetch tuning for cold I/O
- Medium Priority
  - [ ] External-memory builder; compaction strategy
  - [ ] On-disk search latency targets and tests
- Longer Term
  - [ ] Incremental repair and partial rebuilds at scale

Matryoshka / Quantization
- Quick Wins
  - [ ] Document scope and integration points with PQ/OPQ
- Medium/Longer
  - [ ] Evaluate benefits on public datasets; integrate with planner if beneficial

### Query Planner & Filters
- Quick Wins
  - [ ] Planner heuristics doc and unit tests for basic routing/merging
  - [ ] Cost model stub with pluggable backend caps (probes, ef)
- Medium Priority
  - [ ] Calibrated probe/ef selection based on recall targets
  - [ ] Multi-index merge policies and tie-handling improvements
- Longer Term
  - [ ] Adaptive planner with runtime feedback and telemetry

### Durability & Recovery
- Quick Wins
  - [ ] WAL fuzz improvements beyond frames/manifest (invalid sequences, replays)
  - [ ] Snapshot/retention configuration examples and docs
- Medium Priority
  - [ ] Recovery time SLO test harness; measure across dataset sizes
  - [ ] Concurrent delete/add correctness under WAL and tombstones
  - [ ] Manifest rebuild corner cases and stress tests
- Longer Term
  - [ ] Crash/chaos testing scenarios; groundwork for replication (future)

### Core Kernels & Memory
- Quick Wins
  - [ ] Dispatcher polish and feature-flag docs
  - [ ] LUT prefetch toggle in ADC loop (compile-time flag)
  - [ ] SIMD coverage doc and CPU feature detection report
- Medium Priority
  - [ ] NUMA policies for allocations and pinning helpers
  - [ ] Scratch allocators for temp buffers in search paths
  - [ ] Batched top-k improvements and tie-stability
- Longer Term
  - [ ] GPU/accelerator exploration (placeholder)

### IO & Persistence
- Quick Wins
  - [ ] Mmap fallback heuristics polish (cross-platform notes)
  - [ ] Address secure getenv deprecation warnings on Windows
- Medium Priority
  - [ ] Async I/O queues for persistence; prefetch manager tuning
  - [ ] Portable file abstraction for future backends
- Longer Term
  - [ ] Checksum acceleration options; compressed I/O pipelines

### Multi-tenancy & Namespace
- Quick Wins
  - [ ] Define and document tenant/collection namespace model in metadata_store; directory layout conventions per tenant/collection
  - [ ] Plumb namespace IDs through index_manager, tombstone_manager, WAL manifest and snapshot naming
  - [ ] Add basic per-tenant quotas (config scaffolding) for memory and max concurrent queries
- Medium Priority
  - [ ] Implement quota enforcement and admission control hooks; fair scheduling for background jobs
  - [ ] Per-tenant statistics/telemetry surface (ingest rate, query latency, memory)
- Longer Term
  - [ ] Optional isolation policies (thread pools/CPU pinning) and noisy-neighbor mitigation

### Replication & High Availability
- Quick Wins (design-first)
  - [ ] Design doc: async snapshot shipping + WAL tailing pipeline; role model (leader/follower)
  - [ ] Snapshot export/import utility with integrity checks; lag metrics prototype
- Medium Priority
  - [ ] Implement follower node that replays WAL on top of imported snapshot; bounded-lag targets
  - [ ] Read routing with replica awareness (staleness hints); backpressure on lag
- Longer Term
  - [ ] Automated failover (external coordinator) and leader election hooks; consistency options incl. read-your-writes within shard


### Build System & CI/CD
- Quick Wins
  - [ ] GitHub Actions matrix (Windows/Linux, MSVC/GCC/Clang) with AVX2
  - [ ] Sanitizer jobs (ASan/UBSan) on Linux
  - [ ] Warnings-as-errors gating for core targets
- Medium Priority
  - [ ] Package artifacts (static libs); explore wheels/conda (if applicable)
  - [ ] Nightly micro-bench runs with artifacts
- Longer Term
  - [ ] Release automation and changelog generation

### Documentation & Examples
- Quick Wins
  - [ ] API reference for key toggles (ANN, Zstd, mmap, dispatcher) and metadata JSON helper (size limit, examples, optional schema hook)
  - [ ] Minimal examples for save/load/search across backends
  - [ ] Getting-started and tuning quickstart
- Medium Priority
  - [ ] Architecture overview and planner/filter guides
  - [ ] Performance tuning cookbook with datasets
- Longer Term
  - [ ] End-to-end tutorials and reproducible notebooks

Notes: This roadmap is intentionally flexible. Each phase includes reserved slots to opportunistically incorporate optimizations discovered during profiling or user feedback.


---

## Comprehensive Roadmap (to Release)

Legend: Priority [P0 now, P1 next, P2 later], Complexity [S/M/C], Dependencies noted per item.

### Milestone M0: Test/CI foundation and build health [P0]
- Tasks
  - [P0][S] Fix multi_index_test build/discovery on MSVC; ensure it runs under CTest. Dep: Catch2 discovery fixed (done)
  - [P0][M] CI matrix: Windows (MSVC), Linux (GCC/Clang) with AVX2; cache deps; parallel builds
  - [P0][S] Warnings-as-errors for core targets; suppress/guard third-party warnings
  - [P0][M] Sanitizers: ASan/UBSan jobs on Linux for core libraries/tests
  - [P1][S] Nightly bench job skeleton producing artifacts (JSON/CSV + Markdown summary)
- Acceptance
  - CTest lists/filters all tests on both platforms; CI green on matrix; sanitizer jobs pass on Linux
- Dependencies: none

### Milestone M1: Coarse quantizer release track (KD-tree default) [P0]
- Decisions
  - KD-tree = default coarse assigner; HNSW optional; brute-force fallback always available
- Tasks (KD-tree)
  - [P0][M] Profiling pass on add(): identify hotspots in leaf scans/heap updates
  - [P0][M] Vectorization and prefetch: tighten inner loops; prefetch child/leaf buffers
  - [P0][S] Data layout: SoA buffers for candidates; reduce branches in leaf loops
  - [P0][S] Expose tunables in docs: kd_leaf_size, kd_batch_assign, kd_split; choose sane defaults
  - [P0][S] Parameter sweep harness; persist results and choose defaults per dataset class
- Tasks (HNSW optional)
  - [P1][M] Validate configurations (M, ef_construction, ef_search) that keep mismatch ≤1% and throughput ≥ brute (if achievable); otherwise document as experimental
- Tasks (Projection)
  - [P2][S] Keep experimental; maintain tests; no performance work in release track
- Acceptance
  - KD-tree add() throughput ≥1.5× brute for nlist ≥1024 on reference datasets; 0% assignment mismatch (exact)
  - Coarse assigner selector: params.coarse_assigner ∈ {Brute, KDTree (default), HNSW}; documented and tested
- Dependencies: M0

### Milestone M2: IVF-PQ search performance and quality [P0/P1]
- Tasks
  - [P0][S] LUT prefetch toggle in ADC loop; verify neutral-to-positive impact
  - [P1][M] Batched top-k improvements; tie-stability retained; dispatcher polish
  - [P1][M] Recall/latency sweeps (nprobe and k grid) with acceptance dashboards
- Acceptance
  - Meet latency/throughput SLOs at target recall@k; tie-stability tests green
- Dependencies: M1 (for consistent add() path), M0

### Milestone M3: Durability & recovery hardening [P1]
- Tasks
  - [P1][M] WAL fuzz beyond frames/manifest (invalid sequences, replay corner cases)
  - [P1][M] Recovery SLO harness across dataset sizes; publish RTO curves
  - [P1][S] Snapshot/retention configuration docs and examples
- Acceptance
  - Recovery tests stable; target RTO achieved per scale tier; docs complete
- Dependencies: none

### Milestone M4: Planner, filters, and multi-index orchestration [P1]
- Tasks
  - [P1][M] Planner cost model stub + unit tests; backend capability descriptors
  - [P1][M] Filter overhead budget ≤15% vs unfiltered typical predicates; tests
  - [P1][M] Multi-index merge policies; fix and enable multi_index_test; shard demo
- Acceptance
  - Planner routes to backends with predictable recall/latency; multi-index tests green
- Dependencies: M0

### Milestone M5: Documentation & API finalization [P0/P1]
- Tasks
  - [P0][S] Update IVF-PQ docs for coarse_assigner defaults and tunables
  - [P1][M] Architecture overview; planner/filter guides; performance tuning cookbook
  - [P1][S] Public C API examples for train/add/search/save/load across backends
- Acceptance
  - Docs complete; examples compile and run in CI
- Dependencies: M1, M2

### Milestone M6: Release engineering [P0]
- Tasks
  - [P0][S] Version bump, changelog, upgrade notes; release notes with compatibility matrix
  - [P0][S] Default flags review (Zstd off; KD-tree on; safe OPQ defaults)
  - [P0][S] Package artifacts for release; publish bench report with hardware/dataset
- Acceptance
  - Final Release Checklist satisfied
- Dependencies: M0–M5

### Dependency ordering summary
- M0 → M1 → M2 → M5 → M6
- M3, M4 proceed in parallel post-M0; integrate before M6 for broader feature readiness

