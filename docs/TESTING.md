# Testing Strategy and QA Process

Vesper follows a tests‑first workflow with deterministic seeds and reproducible runs. This guide defines test types, layout, policies, and how gates are enforced.

## Principles
- Tests precede code; CI gates PRs (lint → build → tests → evals).
- Deterministic: fixed seeds, stable outputs; numerical kernels use scalar oracles and ULP tolerances.
- Safety: includes power‑failure/WAL replay tests and fuzzing for parsers.

## Test types
- Unit tests (GoogleTest or Catch2) — fast, isolated, API‑level behavior.
- Property tests — invariants (e.g., distance metrics), seeds, shrinking guidance.
- Fuzz targets (libFuzzer) — WAL framing/parsers, manifest readers, bitmap decoders.
- Perf micro‑benchmarks (Google Benchmark) — kernel latency, cache behavior; pinned affinity, warm‑ups.

## Determinism and tolerances
- Use fixed seeds in fixtures and property tests.
- Compare SIMD vs scalar with ULP bounds; document tolerances per kernel.

## Coverage and quality bars
- Aim for ≥80% for core modules; justify exclusions (SIMD intrinsics, platform branches).
- Require tests for bug fixes (regressions) and new APIs.

## Power‑failure and recovery tests
- Fault injection around WAL writes: torn/truncated frames must be ignored.
- Idempotent WAL replay: repeated replays yield the same state.

## Layout and naming
```
/tests/
  unit/
  property/
  fuzz/
/bench/
  micro/
```
- Test file names mirror module names: `module_name_test.cpp`.

## Running locally
- Build with `-DVESPER_ENABLE_TESTS=ON -DVESPER_ENABLE_BENCH=ON` (when available).
- Run `ctest --output-on-failure`.
- Run micro‑benches and capture percentiles using Google Benchmark flags.

## CI enforcement (even before CI lands)
- PRs must include test changes alongside code changes.
- Where prompts generate code or docs, attach eval diffs and decoding params per `prompt-blueprint.md`.
- The CI pipeline (added in Phase 0) will block merges on failing tests/evals.

## References
- `blueprint.md` (project source of truth)
- `prompt-dev-roadmap.md` (execution plan)
- `prompt-blueprint.md` (methodology reference)
- `benchmark-plan.md`, `threat-model.md`.

