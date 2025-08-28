# Setup, Build, and Local Development

This document describes how to set up a portable C++20 environment to build, test, and benchmark Vesper on Linux, macOS, and Windows. It follows the deterministic, prompt‑first workflow described in `prompt-blueprint.md` and the execution plan in `prompt-dev-roadmap.md`.

## Prerequisites
- CMake ≥ 3.24 (recommended)
- A recent C++20 toolchain:
  - Linux: GCC ≥ 12 or Clang ≥ 15
  - macOS: AppleClang ≥ 15 (Xcode 15 toolchain)
  - Windows: MSVC ≥ 19.36 (VS 2022)
- Python 3.9+ (for helper scripts when added)
- Git (and Git LFS if you plan to fetch large benchmark datasets later)

Optional (not required by default)
- Package managers: vcpkg or Conan can be used, but are not assumed. Any usage must pin versions and preserve reproducibility.

## Platform matrix
| OS | Compiler | Notes |
|---|---|---|
| Ubuntu LTS | GCC 12+/Clang 15+ | AVX2 baseline with scalar fallback |
| macOS 13+ | AppleClang 15+ | codesign not required for CLI tools |
| Windows 10/11 | MSVC 19.36+ | long path support recommended |

## Build (out‑of‑source)
```
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
```
Common options (to be introduced as code lands):
- `-DVESPER_ENABLE_TESTS=ON` — build tests
- `-DVESPER_ENABLE_BENCH=ON` — build micro‑benchmarks
- `-DVESPER_FORCE_SCALAR=ON` — disable SIMD for debugging
- `-DVESPER_SIMD_MAX=AVX2|AVX512` — cap SIMD level

## Running tests and benchmarks
Once tests are added (Phase 4), you will be able to run:
```
ctest --output-on-failure
```
Benchmarks (Google Benchmark):
```
./bench/micro/<target> --benchmark_report_aggregates_only=true
```

## Environment variables
- `VESPER_LOG=info|debug|trace` — logging verbosity
- `VESPER_SIMD_MASK=` bitmask to restrict features at runtime
- `VESPER_SEED=` integer seed to control deterministic behavior of tests/tools

## Reproducible builds
- Use a fixed compiler and toolchain versions from the matrix above
- Pin seeds and set deterministic flags in tests/benches
- Avoid environment‑dependent paths in build artifacts

## Troubleshooting
- Missing AVX: ensure your CPU supports AVX2 or set `-DVESPER_FORCE_SCALAR=ON`
- Windows long paths: enable long path support in Group Policy/Registry
- Linux perf permissions for benchmarks: set `kernel.perf_event_paranoid=1` (or run with appropriate caps)
- macOS: if using Homebrew LLVM/Clang, ensure `CC`/`CXX` exported before CMake configure

## References
- Architecture and performance targets: `../blueprint.md`
- Prompt workflow and evals: `../prompt-blueprint.md`
- Roadmap and phases: `../prompt-dev-roadmap.md`

