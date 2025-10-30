# IMPL: CAPQ Const-Correct View (Task 22)

Thought framework: First Principles with targeted ReAct lookups.

## Scope
- Fix const-correctness breach in CAPQ SoA views: `make_view() const` returns a mutable view via const_cast.
- Introduce a read-only view type and adjust APIs to enforce immutability from const storage.

## Modules/Files
- include/vesper/index/capq.hpp (new type + method signature updates)
- include/vesper/index/cgf_capq_bridge.hpp (bridge to accept/store const view)
- tests/unit/capq_const_view_test.cpp (new)

## Invariants & Error Model (Phase 3)
- Invariant: Const views must never confer mutability. No writes through a const storage reference.
- Invariant: Read-only views are safe for concurrent readers (documented), writers must use mutable view.
- Error model: Enforced at compile-time via types (`std::span<const T>`). No runtime error path added.
- Performance: Zero overhead. Views remain non-owning; no copies; same memory layout.

## Alternatives (Phase 4)
1) CapqSoAViewConst with const spans (selected). Minimal refactor, type-safe, preserves writer paths.
2) Templated view `CapqSoAView<TQ4,TQ8,TRes>` (flexible, larger refactor surface).
3) Single view always-const spans (breaks writers).
4) Docs-only (insufficient; violates invariant-first policy).

## ADR (Phase 5)
- Not required. Localized API enhancement with clear invariants; documented here.

## Design (Phase 6)
- Add `struct CapqSoAViewConst` mirroring `CapqSoAView` but with `std::span<const T>` fields.
- Update `CapqSoAStorage`:
  - `view() noexcept -> CapqSoAView` (unchanged)
  - `view() const noexcept -> CapqSoAViewConst` (changed)
  - `make_view() const noexcept -> CapqSoAViewConst` (changed)
- Validation helpers:
  - Overload `validate_capq_view(const CapqSoAViewConst&)` (same shape checks).
- Bridge:
  - Store `CapqSoAViewConst` internally.
  - Provide `initialize(const CapqSoAViewConst&, ...)` and a convenience overload `initialize(const CapqSoAView&, ...)` that converts to const view.
- Documentation: Contract-based Doxygen on new/modified declarations including `\brief`, `\pre/\post` (N/A), `\thread_safety`, `\complexity` (O(1)), `\warning`, `\see`.

## Testing Plan (TDD)
RED (new tests):
- Static assertions: `const CapqSoAStorage`\`view()` returns `CapqSoAViewConst` and spans point to const element types.
- Runtime: `validate_capq_view(CapqSoAViewConst)` passes on freshly allocated storage.
- Non-const: `view()` returns `CapqSoAView`; pointers returned by `*_ptr()` are mutable.

GREEN:
- Implement new type, signature updates, validate overload, bridge update.

REFACTOR:
- Polish Doxygen docs; ensure zero warnings.

## Acceptance Criteria
- Const view type prevents mutation at compile time.
- Existing writer call sites compile unchanged.
- Tests under `[capq]` pass; no new warnings.

## Risk & Mitigation
- Signature change for `view() const` may affect any code binding its exact type. Most call sites use `auto`. Bridge updated accordingly and provides compatibility overload.

