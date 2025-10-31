# IMPL — Memory Pool Overflow-Safe Allocation Guard (Task 23)

Scope
- Component: `include/vesper/core/memory_pool.hpp`
- Function: `vesper::core::MemoryArena::allocate(std::size_t bytes, std::size_t alignment)`
- Problem: Overflow-unsafe exhaustion check `if (offset + bytes > size_)` and unsafe `align_up()` usage can wrap on `size_t` and incorrectly succeed, enabling OOB writes or returning a non-null pointer for a huge request (bytes rounds to 0 on wrap).
- Non-goals: Changing public API, ABI, or PMR `ArenaResource` throwing semantics; altering constructor allocation or global pool policies.

First Principles & Invariants (Phase 3)
- Invariants
  - `used_ <= size_` always
  - `alignment` must be non-zero and power-of-two
  - `offset = align_up(used_, alignment)` must not overflow
  - `bytes = align_up(bytes, alignment)` must not overflow
  - Capacity check must be overflow-safe: prefer subtraction form
- Error model
  - Hot path: never throw; return `nullptr` on invalid alignment, overflow, or exhaustion
  - PMR wrapper (`ArenaResource`) may throw `std::bad_alloc()` — not modified in this task
- Performance
  - No observable regression: same number of branches; constant-time checks; cache behavior unchanged

Alternatives (Phase 4)
- A) Subtraction-based capacity check
  - Validate `alignment` power-of-two; compute `mask = alignment - 1`
  - Overflow-safe alignments: if `x > max - mask` → fail; else `x = (x + mask) & ~mask`
  - Compute `offset` from `used_` similarly; then `remaining = size_ - offset`; if `bytes > remaining` → fail
  - Pros: Clear, branch-predictable, avoids UB; no compiler builtins
  - Cons: A few extra comparisons
- B) Checked addition guard
  - Guard with `if (offset > max - bytes)` before `offset + bytes`
  - Pros: Minimal code churn
  - Cons: Still need separate checked-alignment for `used_` and `bytes`; readability slightly worse
- Decision: A — subtraction-based check plus overflow-safe alignment for `bytes` and `used_`; explicit alignment validation.

Design & Implementation (Phases 5–6)
- ADR: Not required (localized safety fix); rationale documented here
- Edits
  - Add `#include <limits>`
  - In `allocate()`:
    - Validate alignment: `alignment != 0 && (alignment & (alignment - 1)) == 0` else return `nullptr`
    - Compute `mask = alignment - 1`
    - Overflow-safe align for `bytes` and for `used_` → `offset`
    - Exhaustion check: return `nullptr` if `offset > size_` or `bytes > size_ - offset`
    - Update `used_ = offset + bytes`; return `buffer_ + offset`
  - Add contract-based Doxygen doc for `allocate()` (brief, params, pre/post, thread_safety, complexity, warnings)

Testing Strategy (Phase 7 — TDD)
- Unit tests: `tests/unit/core_memory_pool_test.cpp`
  1) Overflow boundary: with small arena, request `std::numeric_limits<size_t>::max()` → returns `nullptr`; `used()` unchanged
  2) Alignment validation: `alignment == 0` and non-power-of-two (e.g., 3) both return `nullptr`
  3) Exhaustion: multiple allocations to fill arena; next allocation fails (no wrap); `used()` reflects aligned consumption
  4) Normal path: returned pointer alignment is respected; `used()` increments by aligned size
- Build & run with Ninja; Catch2 test filter `[memory_pool]`

Acceptance Criteria
- All new tests pass locally (Debug, Ninja); no warnings; zero regressions to existing suites
- `MemoryArena::allocate()` returns `nullptr` on invalid alignment/overflow/exhaustion; never throws
- Behavior unchanged for valid requests; deterministic and thread-local semantics preserved

Risks & Mitigations
- Risk: Overly strict alignment validation could break callers relying on undefined inputs → Mitigation: validation limited to `allocate()`; PMR paths continue to throw as before, minimizing behavior change surface
- Risk: Edge-case alignment of `size_` and `used_` → Mitigation: subtraction-based checks and guards on both alignment steps

References
- Overflow-safe arithmetic patterns for allocators (subtraction checks)
- Nick Fitzgerald, “Always Bump Downwards”
- Common allocator guidance: power-of-two alignment masks; avoid wrap-around on unsigned arithmetic

