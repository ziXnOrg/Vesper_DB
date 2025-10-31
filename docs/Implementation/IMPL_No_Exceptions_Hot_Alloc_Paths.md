# IMPL: Eliminate exceptions from core allocation hot paths (MemoryArena/ArenaResource/TempBuffer)

Owner: Vesper Engineering (Task 25)
Date: 2025-10-31
Scope: include/vesper/core/memory_pool.hpp (MemoryArena, ArenaResource, ThreadLocalPool, TempBuffer)

## 1) Context & Problem Statement

Audit flags high-priority violations of the "no exceptions on hot paths" policy:
- MemoryArena ctor throws std::bad_alloc on aligned_allocate failure (header: ~45)
- ArenaResource::do_allocate throws std::bad_alloc when arena_->allocate() returns nullptr (~163)
- TempBuffer<T> ctor throws std::bad_alloc on pool allocation failure (~341)

PMR containers expect memory_resource::allocate to either return memory or throw; returning nullptr is not permitted by the standard. Therefore, eliminating throws must be achieved by (a) moving failure detection to cold paths and (b) ensuring hot loops do not attempt allocations (pre-sizing / zero-allocation invariants), while preserving PMR contract.

## 2) Constraints & Invariants (Vesper)
- No exceptions on hot paths; zero allocations inside steady-state hot loops (k-means inner loops, kernels, top-k, staging).
- Determinism; zero Release overhead for guards (debug-only instrumentation OK).
- ABI/API stability for public surfaces; memory_pool.hpp is internal but widely used.
- PMR contract: do_allocate must either succeed or throw; cannot return nullptr.

## 3) Acceptance Criteria
- AC1: In representative hot loops (k-means Elkan M-step; PQ LUT compute; HNSW search), debug instrumentation records zero pmr allocations.
- AC2: Unit/property tests prove no exceptions are thrown during hot path execution at typical sizes (deterministic seeds).
- AC3: MemoryArena constructor failures are surfaced via a cold-path try-create API (std::expected) with no throws.
- AC4: TempBuffer provides a non-throwing construction path; code using TempBuffer in hot loops is adapted to a preflight that guarantees success (or fails early in cold path).
- AC5: Release builds have zero additional overhead; no performance regressions beyond noise; no new warnings.

## 4) Design Overview (Phase 1 – minimal, zero-Release-overhead)

A) ThreadLocalPool preflight/"prewarm"
- Add `static expected<void, error> prewarm(std::size_t arena_size = DEFAULT)` that constructs the thread-local pool (if not already) and returns error on OOM without throwing (cold path). Implementation: a guarded call to `instance()` on a background code path with try/catch that maps std::bad_alloc to error::oom; only used by callers explicitly (tests and initialization sequences). No changes to hot path behavior.

B) Debug-only allocation counters
- Add debug-only per-thread counters in ThreadLocalPool: `alloc_calls_`, `bytes_allocated_` incremented by ArenaResource::do_allocate. In NDEBUG these are compiled out.
- Provide debug helpers: `begin_hot_region()`/`end_hot_region()` wrappers for tests to assert zero allocations in regions.

C) TempBuffer non-throwing factory (internal)
- Add `static expected<TempBuffer, error> try_create(std::size_t count) noexcept;` which constructs via pool_.allocate and returns `unexpected(error::oom)` on failure instead of throwing.
- Keep existing throwing constructor for backward compatibility; mark as discouraged in docs for hot paths.

D) Guidance & pre-sizing helpers
- Document a required pattern for hot loops: create pooled containers and TempBuffer via preflight before entering the loop, with explicit `reserve`/`resize` to the maximum needed. In debug, assert that `alloc_calls_` remains unchanged while inside hot regions.

Notes: We intentionally do not change pmr::memory_resource throwing contract. Instead, we ensure hot code paths do not go through it after pre-sizing. 

## 5) Alternatives Considered
- A1: Non-throwing ArenaResource (return nullptr). Rejected: violates PMR contract and risks UB in STL containers.
- A2: Per-scope custom resource that never fails (falls back to a static emergency slab). Rejected for Phase 1: complex, risks silent data corruption; may be explored in an ADR if needed.
- A3: Replace PMR containers with bespoke SoA buffers. Deferred: larger refactor; not necessary to meet ACs.

## 6) Detailed Plan & Changes

1) ThreadLocalPool (memory_pool.hpp)
- Add (debug-only) counters; expose getters under NDEBUG guard.
- Add `static expected<void, error> prewarm(std::size_t)`; cold-path helper only (no dependency changes).

2) ArenaResource::do_allocate (memory_pool.hpp)
- Increment debug counters on success; keep throwing semantics on failure (PMR compliance). Add doxygen warning that this must not be hit in hot loops; use pre-sizing.

3) TempBuffer<T> (memory_pool.hpp)
- Add non-throwing `try_create(count) -> expected<TempBuffer, error>` that skips throwing on OOM. Existing ctor remains for legacy usage; docs steer hot paths to the factory.
- Add debug assert to detect construction inside a region where allocs are forbidden (optional).

4) Tests
- Add a `tests/unit/core_memory_pool_noexcept_hot_test.cpp`:
  - Instrumentation to begin/end a hot region; execute representative hot loops (k-means Elkan inner loops via small harness) and assert zero allocations and no exceptions.
  - Property-style tests varying sizes to exercise pre-sizing logic.
- Update docs with examples (kmeans_elkan.cpp pattern).

## 7) Risks & Mitigations
- Risk: Hidden allocations via STL/PMR despite pre-sizing. Mitigate with allocation counters and targeted tests.
- Risk: API surface change for TempBuffer. Mitigate with additive factory, keep ctor for compatibility; update internal call sites progressively.
- Risk: Overhead in Release. Avoid by NDEBUG guards for counters; factories are cold-path only.

## 8) Validation Plan
- Build Debug + Release (Ninja/MSVC); zero new warnings.
- Run new tests + existing [memory_pool] and [kmeans] subsets.
- Benchmark smoke to verify no perf regressions (micro: batch distances; macro optional).

## 9) Checkpoint Gate
This IMPL documents Phase 3–6. Request approval to implement Phase 7–9 changes:
- Add prewarm(), debug counters, TempBuffer::try_create(), docs
- Add tests asserting zero allocations/exceptions in hot regions
- Keep PMR throw semantics but ensure they’re not reached in hot loops

