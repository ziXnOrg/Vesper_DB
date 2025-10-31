# IMPL: Pooled Container Scope Safety (Task 24)

## Research summary
- C++ PMR containers (std::pmr::vector, etc.) obtain storage from an upstream std::pmr::memory_resource. Their lifetime is logically bound to the resource: destroying or invalidating the resource while the container remains in use causes undefined behavior (UB).
- Arena/monotonic-style resources (like our ArenaResource over MemoryArena) do bulk allocation and reset in one shot. They are intended for scope-bounded allocations: resetting the arena invalidates all previously returned storage.
- Our PoolScope calls ThreadLocalPool::reset() in its destructor, which resets the underlying MemoryArena. Any pmr container (or PooledVector) created within such a scope becomes invalid at scope exit. Therefore, pooled containers must not escape their creating PoolScope.

References:
- C++ PMR: cppreference “std::pmr::memory_resource” and “std::pmr::polymorphic_allocator” semantics
- Monotonic resource lifetime patterns; arena-style allocators (Herb Sutter, WG21 papers N3916/N4617 background)

## Architecture mapping
- MemoryArena: fixed-size bump allocator used by ThreadLocalPool
- ArenaResource: std::pmr::memory_resource adapter over MemoryArena
- ThreadLocalPool: owns MemoryArena and ArenaResource; thread-local instance
- PoolScope: RAII guard that calls ThreadLocalPool::reset() on destruction (invalidates arena storage)
- PooledVector<T>: alias to std::pmr::vector<T>
- make_pooled_vector<T>: constructs a PooledVector<T> using ThreadLocalPool::allocator()

## First principles & invariants
- Non-escape invariant: Any allocation/container using the pool must not outlive the PoolScope during which it was created.
- UAF prevention: After PoolScope::~PoolScope calls reset(), all pool-backed storage is invalid. Continued use beyond the scope boundary is UB.
- Thread affinity: ThreadLocalPool and its allocator are thread-local; do not use across threads.

## Alternatives considered
1) Enforce per-scope unique ArenaResource instance bound to PoolScope; type PooledVector would carry a reference/handle to forbid moves/returns.
   - Pros: Stronger prevention of scope escape at compile time if wrapped.
   - Cons: Breaks existing API (PooledVector alias), ABI surface changes, higher migration cost.
2) Runtime tracking of container lifetimes with registration and checks.
   - Pros: Can detect escapes dynamically.
   - Cons: Overhead, complexity, and intrusive hooks into container lifecycle; undesirable for hot paths.
3) Documentation + debug-only guard on construction.
   - Pros: Zero cost in Release, minimal changes, preserves ABI and current usage; catches common mistakes early in Debug.
   - Cons: Cannot detect use-after-scope at the point of use; relies on correct patterns and reviews.

Decision: Approach (3) for Task 24; consider (1) or scoped-resource wrapper as a future ADR if stronger guarantees are desired.

## Decision & specification
- Document explicit non-escape contract on PoolScope, PooledVector, make_pooled_vector, and ThreadLocalPool::allocator.
- Add debug-only scope-depth tracking to ThreadLocalPool:
  - Methods: debug_scope_enter(), debug_scope_exit(), debug_scope_depth()
  - Member: int scope_depth_{0}
- PoolScope ctor/dtor update: increment/decrement debug scope depth under !NDEBUG; destructor still calls reset().
- make_pooled_vector<T>: assert in Debug that a PoolScope is active on the current thread.
- No behavior change in Release builds (NDEBUG-gated).

## Testing plan
- Unit test (positive): "PooledVector works within PoolScope" demonstrating correct usage (reserve, push_back, size).
- Do not add a negative test that triggers debug assert (would abort test process); misuse scenarios documented in comments.
- Re-run existing [memory_pool] tests to ensure no regressions; build both Debug and Release.

## Acceptance criteria
- Documentation clearly states non-escape rule and thread-affinity constraints.
- Debug builds assert when constructing a pooled vector without an active PoolScope.
- All [memory_pool] tests pass deterministically in Debug and Release; zero warnings; no hot-path exceptions added.
- No Release overhead from the new checks.

