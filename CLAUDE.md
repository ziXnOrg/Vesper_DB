# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vesper is a crash-safe, embeddable vector search engine written in C++20 for CPU-only environments. It targets on-device and air-gapped scenarios with ultra-low latency approximate nearest neighbor (ANN) search, metadata filters, and deterministic persistence (WAL + snapshots).

## Build Commands

```bash
# Out-of-source build (required)
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j

# Common CMake options
-DVESPER_ENABLE_TESTS=ON       # Build tests (default ON)
-DVESPER_ENABLE_BENCH=ON       # Build benchmarks (default ON)
-DVESPER_FORCE_SCALAR=ON       # Disable SIMD for debugging
-DVESPER_SIMD_MAX=AVX2|AVX512  # Cap SIMD level
-DVESPER_ENABLE_KERNEL_DISPATCH=ON  # Enable kernel dispatcher
```

## Test Commands

```bash
# Run all tests
cd build
ctest --output-on-failure

# Run specific test suite
./vesper_tests

# Run benchmarks
./bench/micro/vesper_bench_dummy --benchmark_report_aggregates_only=true

# Run fuzz tests (Clang only)
./wal_manifest_fuzz corpus/
./wal_frame_fuzz corpus/
```

## Code Architecture

### Core Components

- **Index Families**: Three pluggable index types per collection:
  - IVF-PQ/OPQ: Compact, disk-resident retrieval with asymmetric distance computation
  - HNSW: In-memory hot segments for high-recall, low-latency search
  - Disk-graph: DiskANN-style for billion-scale with SSD-resident neighbor lists

- **Storage Layer**: Log-structured with WAL for crash-safety
  - Write-ahead logging with checksummed frames
  - Copy-on-write snapshots for recovery
  - Atomic publish via rename()

- **Memory Management**: Uses std::pmr arenas for thread-local scratch
  - NUMA-aware memory pools
  - 64-byte cache line alignment to avoid false sharing

- **SIMD Acceleration**: AVX2/AVX-512 with scalar fallback
  - Runtime feature detection and dispatch
  - Distance kernels with FMA operations

### Key Directories

- `/include/vesper/`: Public C++20 headers and C ABI
- `/src/`: Implementation files
  - `/src/wal/`: WAL implementation (frame, io, manifest, replay, snapshot)
- `/tests/`: Test suites
  - `/tests/unit/`: Unit tests (Catch2)
  - `/tests/fuzz/`: Fuzz targets (libFuzzer)
- `/bench/micro/`: Micro-benchmarks (Google Benchmark)
- `/algorithms/spec/`: Algorithm specifications and pseudocode
- `/architecture/`: ADRs and design documents

## Development Guidelines

### Coding Standards

- Follow C++20 standards with RAII and value semantics
- Use `std::expected<T, error_code>` for error handling (no exceptions on hot paths)
- Naming conventions:
  - Types/Concepts: `PascalCase`
  - Functions/Methods: `lower_snake_case`
  - Private members: `trailing_underscore_`
  - Files: `lower_snake_case.{hpp,cpp}`

### Testing Approach

- Tests-first workflow with deterministic seeds
- Unit tests with Catch2 framework
- Property-based testing for invariants
- Fuzz testing for parsers and WAL components
- Performance micro-benchmarks with Google Benchmark

### Safety & Performance

- Crash-safety via WAL with checksummed frames and atomic operations
- No network IO by default (embedded library)
- Optional encryption at rest (XChaCha20-Poly1305)
- Performance targets:
  - P50 ≤ 1-3ms, P99 ≤ 10-20ms for 128-1536D embeddings
  - Recall@10 ≈ 0.95 (tunable)
  - 50-200k vectors/s ingestion rate

## Important References

- **Technical Blueprint**: `blueprint.md` - Comprehensive architecture and design
- **Development Roadmap**: `prompt-dev-roadmap.md` - Phased implementation plan
- **Coding Standards**: `CODING_STANDARDS.md` - Detailed C++ guidelines
- **Testing Strategy**: `TESTING.md` - QA process and test requirements
- **Setup Guide**: `docs/SETUP.md` - Build prerequisites and platform support