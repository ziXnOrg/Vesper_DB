# Changelog

All notable changes to Vesper will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Production-ready platform-specific optimizations
  - AVX-512 SIMD support for 16-wide vector operations
  - io_uring async I/O for Linux 5.1+ with zero-copy operations
  - NUMA-aware memory allocation with per-node pools
  - Prefetch instructions and loop unrolling in scalar kernels
- Comprehensive API reference documentation
- Detailed operator guide with performance tuning instructions
- LRU cache for hot data in disk-based indices
- Thread-safe sharded cache with TTL support
- Distance caching in incremental repair algorithms

### Changed
- Improved diversity calculation in incremental repair using actual distances
- Enhanced K-means clustering with proper statistical metrics
- Updated CMakeLists.txt to include new platform-specific sources
- Default OPQ rotation enabled for better recall
- Windows file mapping support for DiskGraph persistence
- Updated documentation and examples

### Fixed
- Placeholder implementation in incremental_repair.cpp
- K-means evaluation metrics calculation
- Proper distance computation in diversity pruning
- Memory-mapped file support for DiskGraph persistence on Windows

## [0.2.0] - 2024-01-07

### Added
- Multi-index architecture with IndexManager and QueryPlanner
- Product Quantizer (PQ) for vector compression
- Optimized Product Quantization (OPQ) support
- DiskANN (Vamana) graph index implementation
- Incremental repair algorithms for all index types
- Tombstone management for soft deletes
- Metadata store with Roaring bitmap support

### Changed
- Refactored index interface for pluggable backends
- Improved HNSW thread pool with work-stealing
- Enhanced WAL with retention policies

### Fixed
- HNSW connectivity issues in concurrent insertions
- Memory leaks in PQ codebook allocation
- Race conditions in batch operations

## [0.1.0] - 2024-01-01

### Added
- Initial release with core functionality
- HNSW index with configurable M and efConstruction
- IVF-PQ index with asymmetric distance computation
- WAL (Write-Ahead Log) for crash safety
- Snapshot mechanism for point-in-time recovery
- Basic SIMD support (AVX2)
- C++20 API with std::expected error handling
- CMake build system with FetchContent
- Unit tests with Catch2
- Micro-benchmarks with Google Benchmark
- Basic documentation and examples

### Security
- XChaCha20-Poly1305 encryption at rest (optional)
- Checksummed WAL frames for integrity

## [0.0.1] - 2023-12-15

### Added
- Project bootstrap and initial structure
- Architecture Decision Records (ADRs)
- Component map and design documents
- Development roadmap
- Coding standards documentation

[Unreleased]: https://github.com/vesper-arch/vesper/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/vesper-arch/vesper/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/vesper-arch/vesper/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/vesper-arch/vesper/releases/tag/v0.0.1
