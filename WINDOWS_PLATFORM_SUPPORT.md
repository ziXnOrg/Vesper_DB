# Windows Platform Support for Vesper

## Overview
This document describes the platform abstraction layer implemented to provide full Windows support for Vesper, a crash-safe embeddable vector search engine.

## Platform Abstraction Layer

### Architecture
The platform abstraction layer is organized into modular headers under `include/vesper/platform/`:

1. **compiler.hpp** - Compiler-specific attributes and hints
2. **memory.hpp** - Cross-platform aligned memory allocation
3. **intrinsics.hpp** - CPU intrinsics (prefetch, barriers, bit operations)
4. **filesystem.hpp** - File I/O and memory-mapped files
5. **parallel.hpp** - OpenMP compatibility and parallelization
6. **platform.hpp** - Central include and platform detection

### Key Features

#### Memory Management (`memory.hpp`)
- **Aligned allocation**: Abstracts `_aligned_malloc`/`std::aligned_alloc`
- **RAII wrappers**: `aligned_unique_ptr` for automatic cleanup
- **Type-safe allocation**: Template functions with alignment validation
- **Cache-line alignment**: Built-in support for 64-byte boundaries

```cpp
// Platform-agnostic aligned allocation
auto* buffer = aligned_allocate(size, 64);
// Automatically uses _aligned_malloc on Windows, std::aligned_alloc on POSIX

// RAII wrapper
aligned_unique_ptr<float, 32> data(1024);
```

#### CPU Intrinsics (`intrinsics.hpp`)
- **Prefetch operations**: Read/write prefetch with cache level hints
- **Memory barriers**: Compiler and CPU memory fences
- **Bit operations**: Leading/trailing zeros, population count
- **Timestamp counter**: Performance measurement support

```cpp
// Prefetch with cache hints
prefetch_read(data, prefetch_hint::all_levels);
prefetch_write(buffer, prefetch_hint::l2_l3);

// Memory barriers
compiler_barrier();  // Prevent compiler reordering
memory_fence();      // Full CPU memory barrier
```

#### Filesystem Operations (`filesystem.hpp`)
- **File handles**: RAII wrapper for Windows HANDLE/POSIX fd
- **Sync operations**: `FlushFileBuffers`/`fsync` abstraction
- **Memory-mapped files**: Unified interface for mmap/MapViewOfFile
- **File locking**: Cross-platform exclusive access

```cpp
// Open file with platform-specific flags
auto file = open_file(path, write_mode, create, direct_io);

// Memory-mapped file
auto mapped = map_file(path, size, write_access);
mapped->sync();  // FlushViewOfFile on Windows, msync on POSIX
```

#### Parallelization (`parallel.hpp`)
- **OpenMP compatibility**: Falls back to serial execution if unavailable
- **Signed index loops**: Handles MSVC's OpenMP requirements
- **Thread management**: Portable thread count and ID access
- **Parallel patterns**: for loops, reductions, critical sections

```cpp
// Parallel for with automatic index conversion
parallel_for(0, n, [](std::size_t i) {
    // Internally converts to signed for OpenMP
    process(i);
});

// Parallel reduction
auto sum = parallel_reduce(0, n, 0.0f, 
    [](std::size_t i) { return data[i]; },
    std::plus<>());
```

#### Compiler Attributes (`compiler.hpp`)
- **Platform detection**: Windows/Linux/macOS, x86/ARM
- **Function attributes**: inline hints, hot/cold paths
- **Branch prediction**: likely/unlikely hints
- **Warning suppression**: Cross-compiler warning control

```cpp
VESPER_ALWAYS_INLINE auto fast_path() -> void {
    if (VESPER_LIKELY(common_case)) {
        // Optimized for this branch
    }
}

VESPER_COLD auto error_handler() -> void {
    // Rarely executed, optimize for size
}
```

## Windows-Specific Considerations

### Build Configuration
The CMake configuration automatically detects Windows and applies appropriate settings:

```cmake
# Platform detection for PQ implementation
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|x64")
    # Use FastScan PQ on x86-64 (including Windows AMD64)
    list(APPEND VESPER_CORE_SOURCES src/index/pq_fastscan.cpp)
else()
    # Use Simple PQ on ARM
    list(APPEND VESPER_CORE_SOURCES src/index/pq_simple.cpp)
endif()
```

### Visual Studio Support
- **Tested with**: Visual Studio 2022 (MSVC 19.44)
- **C++ Standard**: C++20 required
- **OpenMP**: Supported via MSVC's OpenMP 2.0 implementation
- **SIMD**: AVX2 support on AMD Ryzen and Intel processors

### Performance Optimizations
1. **Memory alignment**: Uses `_aligned_malloc` for cache-line aligned allocations
2. **Prefetching**: Maps to `_mm_prefetch` intrinsic with appropriate hints
3. **Vectorization**: Leverages MSVC's auto-vectorization with `#pragma loop(ivdep)`
4. **OpenMP**: Parallel loops with proper signed index conversion

## Testing on Windows

### Build Instructions
```bash
# Configure with Visual Studio generator
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022"

# Build in Release mode
cmake --build . --config Release -j 8

# Run tests
ctest --output-on-failure -C Release
```

### Test Results
- Core library builds successfully
- SIMD benchmarks show ~1.3 GB/s throughput on AMD Ryzen 7 3700X
- 51/60 tests passing (WAL tests require filesystem adjustments)

## Future Improvements

### Pending Tasks
1. **Replace quick fixes**: Migrate remaining `#ifdef` blocks to use platform abstractions
2. **WAL filesystem**: Adjust WAL for Windows filesystem semantics
3. **Performance tuning**: Profile and optimize Windows-specific paths
4. **Documentation**: Expand Windows-specific usage examples

### Known Limitations
- WAL tests may fail due to different fsync behavior on Windows
- Some GNU-specific attributes generate warnings (harmless)
- OpenMP 2.0 on MSVC lacks some OpenMP 3.0+ features

## Contributing
When adding new platform-specific code:
1. Use the platform abstraction layer instead of direct `#ifdef`
2. Test on both Windows (MSVC) and Linux (GCC/Clang)
3. Document any platform-specific behavior
4. Add unit tests for platform abstractions

## References
- [MSVC Intrinsics](https://docs.microsoft.com/en-us/cpp/intrinsics/)
- [Windows Memory Management](https://docs.microsoft.com/en-us/windows/win32/memory/)
- [MSVC OpenMP](https://docs.microsoft.com/en-us/cpp/parallel/openmp/)