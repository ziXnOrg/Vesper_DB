# Vesper Coding Standards (Staff+)

This guide sets enforceable, modern C++20 standards for Vesper. Use MUST/SHOULD/MAY language. See `.clang-format`, `.clang-tidy`, and `.editorconfig` for automated enforcement.

## Philosophy
- MUST prioritize correctness, crash‑safety, and predictable performance.
- SHOULD be portable across Linux/macOS/Windows and multiple compilers.
- MAY use advanced C++20 features when they do not harm clarity or performance.

## Design tenets (C++)
- MUST use RAII and value semantics by default; avoid shared mutable state.
- MUST avoid exceptions on hot paths; prefer `std::expected<T, error_code>`.
- MUST use `std::pmr` arenas for per‑thread scratch in hot paths; no global `new/delete`.
- SHOULD use `span`, `string_view`, and SoA layouts for hot data.
- SHOULD document pre/post‑conditions and complexity for all public APIs.

## ABI rules
- MUST expose C ABI without STL types; use POD structs and explicit ownership rules.
- MUST version the C ABI and document stability guarantees.

## Memory, layout, and performance
- MUST align hot structures to 64‑byte cache lines; avoid false sharing.
- SHOULD pad dimensions for SIMD (e.g., 32/64) when beneficial.
- SHOULD use SoA for hot scalar metadata to reduce cache pollution.

## SIMD dispatch policy
- MUST provide scalar reference kernels.
- SHOULD provide AVX2 and AVX‑512 implementations behind compile‑time flags and runtime detection.
- SHOULD guard feature use and provide deterministic fallbacks.

## Concurrency model
- Readers use RCU/epoch pinning for immutable segments.
- Writers: single‑writer per collection; coarse locks for mutable structures.
- Memory reclamation via epochs/hazard pointers to avoid ABA/UAF.

## Logging and telemetry
- MUST use structured logs and counters; no network IO by default.
- SHOULD provide histograms (latency, cache hits) and tracing hooks in hot paths.

## Header hygiene and includes
- MUST keep public headers under `include/vesper/` with minimal includes.
- SHOULD prefer forward declarations in headers; IWYU‑friendly includes.
- MUST avoid macros for constants; use `constexpr` or scoped enums.

## Namespaces and naming
- Namespaces: `vesper::<subsystem>::…`
- Types/Concepts: `PascalCase` (e.g., `VamanaGraph`, `SearchParams`)
- Functions/Methods: `lower_snake_case` (e.g., `seal_segment`, `compute_filter_mask`)
- Variables: `lower_snake_case`; private data members with trailing underscore
- Constants/Enums: `UPPER_SNAKE_CASE` or `enum class` in `PascalCase`
- Files: `lower_snake_case.{hpp,cpp}`; public headers under `include/vesper/`
- Acronyms: `PqCode` not `PQCode` (capitalize first letter only)

## Documentation style (Doxygen)
- Every public API must include: brief, detailed description, pre/post‑conditions, complexity, thread‑safety, errors/ownership.
- Module docs must include assumptions, failure modes, and performance notes; cross‑link to `blueprint.md` sections.
- Provide short, compilable examples where appropriate.

## Examples (abbreviated)
- Error handling with `expected`
- pmr arena scratch usage
- SIMD dispatch pattern with scalar fallback

## Enforcement
- `.clang-format` and `.clang-tidy` are authoritative. Violations block PRs.
- Local run: `clang-format -i` and `clang-tidy` per `docs/SETUP.md`.

## Rationale and references
- C++ Core Guidelines; Boost doc quality bar; internal SLOs and safety constraints (see `blueprint.md` as project source of truth, plus `prompt-dev-roadmap.md` for sequencing and `prompt-blueprint.md` for methodology).

