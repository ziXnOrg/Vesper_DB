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
- `experiments/VALIDATION.md` (schema validation)
- `benchmark-plan.md`, `threat-model.md`.



## Windows: Building and running libFuzzer targets with Clang + Ninja

This section shows how to build and execute the libFuzzer-based fuzz tests on Windows using the LLVM/Clang toolchain with Ninja. It focuses on the IVF-PQ v1.1 loader fuzzer (ivfpq_v11_fuzz) and applies equally to other fuzz targets.

1) Prerequisites
- Install LLVM/Clang 21+ for Windows (includes clang++, lld-link, libFuzzer, ASan/UBSan)
  - Typical path: C:\\Program Files\\LLVM
- Install Ninja build system (bundled with many LLVM installs or install separately)
- PATH: Ensure C:\\Program Files\\LLVM\\bin is on PATH so clang/clang++/lld-link are discoverable

2) Configure build (Clang + Ninja, fuzzing enabled)
Run from the repository root:

- Release with debug info recommended for stable symbols and speed
- VESPER_ENABLE_FUZZ=ON turns on fuzz targets
- Explicitly select clang/clang++ and Ninja generator

```
cmake -S . -B build-clang-ninja -G Ninja \
  -D CMAKE_C_COMPILER="C:/Program Files/LLVM/bin/clang.exe" \
  -D CMAKE_CXX_COMPILER="C:/Program Files/LLVM/bin/clang++.exe" \
  -D CMAKE_BUILD_TYPE=RelWithDebInfo \
  -D VESPER_ENABLE_FUZZ=ON

cmake --build build-clang-ninja --target ivfpq_v11_fuzz -j 8
```

Notes
- Fuzz builds link with -fsanitize=fuzzer,address,undefined on Windows via clang++ + lld-link
- The build system avoids heavy optional modules when fuzzing and wires in sanitizer flags automatically

3) AddressSanitizer runtime (DLL) on Windows
- The fuzzer requires the ASan runtime DLL at execution time
- Copy it next to the produced fuzzer .exe (needed once per build directory):

```
copy "C:\\Program Files\\LLVM\\lib\\clang\\21\\lib\\windows\\clang_rt.asan_dynamic-x86_64.dll" build-clang-ninja\
```

4) Running the fuzzer
- Create an optional seed corpus directory (can be empty):
```
mkdir fuzz-corpus
```
- Example runs (short and medium):
```
# Short smoke
build-clang-ninja/ivfpq_v11_fuzz.exe -runs=1000 -max_len=4096 -timeout=5 -print_final_stats=1 fuzz-corpus

# Longer run
build-clang-ninja/ivfpq_v11_fuzz.exe -runs=10000 -max_len=4096 -timeout=5 -print_final_stats=1 -artifact_prefix=./crash- fuzz-corpus
```

5) Exercising both load paths (mmap vs streaming)
- The fuzzer randomly flips the path across inputs, but you can force it via env var:
```
# Force mmap path
cmd /c "set VESPER_IVFPQ_LOAD_MMAP=1 && build-clang-ninja\ivfpq_v11_fuzz.exe -runs=200 -max_len=2048 -timeout=5 -print_final_stats=1 fuzz-corpus"

# Force streaming path
cmd /c "set VESPER_IVFPQ_LOAD_MMAP=0 && build-clang-ninja\ivfpq_v11_fuzz.exe -runs=200 -max_len=2048 -timeout=5 -print_final_stats=1 fuzz-corpus"
```

6) Interpreting output
- You should see libFuzzer banners like "INFO: Running with entropic power schedule" and lines showing cov/ft/corp evolution
- Coverage growth and new corpus entries indicate fuzzer is reaching new code paths (e.g., v1.1 section checks, metadata parsing)
- Sanitizer reports (ASan/UBSan) will appear as detailed stack traces; the run should be clean (no reports) on valid inputs and typical corruptions
- The fuzzer exits 0 on normal completion of a -runs session; crashes/timeouts will emit artifacts with the given -artifact_prefix

7) Troubleshooting
- Missing ASan DLL (process fails to start): copy clang_rt.asan_dynamic-x86_64.dll as shown above
- Link/runtime mismatch errors on Windows:
  - Examples: /failifmismatch for RuntimeLibrary (MD vs MT) or _ITERATOR_DEBUG_LEVEL
  - Use RelWithDebInfo instead of Debug when fuzzing; avoid mixing Debug CRT with sanitizer libs
  - Ensure all linked objects are built by the same toolchain/runtime; prefer a single Clang+Ninja build directory for fuzz targets
  - If using clang-cl (MSVC driver), align the CRT via CMAKE_MSVC_RUNTIME_LIBRARY (e.g., MultiThreaded for static) and rebuild all targets
- libFuzzer flags: use -help=1 to list options; -runs caps iteration count; -timeout controls per‑unit seconds; -max_len caps input size

Tips
- Keep -max_len small (<=4KB) for fast iterations; slowly raise if new sections are gated by size
- Use a seed corpus with a few minimal valid v1.1 samples (with/without metadata) to bootstrap coverage
