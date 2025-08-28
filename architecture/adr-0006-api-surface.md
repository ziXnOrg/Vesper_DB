# ADR-0006: Public API & ABI Contracts (Phase 3)

Status: Proposed
Date: 2025-08-28

## Context
We need a stable C++20 API and C ABI that reflect the blueprint while preserving portability and predictable performance.

## Decision
- C++20 headers under `include/vesper/` with Doxygen docs and `std::expected` for errors.
- Stable C ABI in `include/vesper/vesper_c.h` using only POD types; explicit ownership rules.
- Error taxonomy aligned with ADR‑0005; one place to map C codes.

## Consequences
- Headers must compile standalone on GCC/Clang/MSVC with `-pedantic`.
- ABI guarantees enable bindings and long‑term stability.

## References
- docs/blueprint.md §2, §11
- ADR‑0005 Error taxonomy

