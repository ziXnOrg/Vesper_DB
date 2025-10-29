cmake_minimum_required(VERSION 3.22)
project(vesper LANGUAGES C CXX)

# Enforce C++20 globally
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add compatibility flags for GCC 11
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12.0)
    message(STATUS "GCC 11 detected, adding compatibility workarounds")
    # GCC 11 has broken std::span, disable it
    add_compile_options(-std=gnu++20)
endif()

# Optional features (platform-aware defaults)
if (APPLE)
  option(VESPER_ENABLE_ACCELERATE "Enable Apple Accelerate for vectorized math (macOS only)" ON)
  option(VESPER_ENABLE_OPENMP "Enable OpenMP parallelization (optional; macOS requires Homebrew LLVM)" OFF)
else()
  option(VESPER_ENABLE_ACCELERATE "Enable Apple Accelerate for vectorized math (macOS only)" OFF)
  # On Linux/UNIX (non-Apple), default to enabling OpenMP
  option(VESPER_ENABLE_OPENMP "Enable OpenMP parallelization" ON)
endif()
option(VESPER_SERIALIZE_BASE_LAYER "Serialize base-layer connection in HNSW (safer connectivity, slower)" ON)

# Experimental/guarded concurrency options (default OFF)
option(VESPER_PARTITIONED_BASE_LOCKS "Partitioned base-layer locking for HNSW inserts (experimental)" OFF)
option(VESPER_EXPERIMENTAL_WORK_STEALING "Enable experimental per-worker deques with basic work-stealing in HNSW thread pool" OFF)

# Optional CBLAS/BLAS for projection screening (disabled by default)
option(VESPER_ENABLE_CBLAS "Enable CBLAS (OpenBLAS/MKL) for projection screening" OFF)

# Configure OpenMP if requested
if (VESPER_ENABLE_OPENMP)
  find_package(OpenMP)
  if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found; will link OpenMP::OpenMP_CXX")
    set(VESPER_HAVE_OPENMP TRUE)
  else()
    message(STATUS "OpenMP requested but not found. Building without OpenMP parallelization")
    set(VESPER_HAVE_OPENMP FALSE)
  endif()
else()
  message(STATUS "Building without OpenMP parallelization (VESPER_ENABLE_OPENMP=OFF)")
  set(VESPER_HAVE_OPENMP FALSE)
endif()

# Note for macOS users
if (APPLE)
  message(STATUS "Note: For OpenMP on macOS, use Homebrew LLVM (clang++) and install libomp")
endif()

# Find required packages
find_package(Threads REQUIRED)

# Header-only interface target to compile public headers with -pedantic
add_library(vesper_headers INTERFACE)
target_include_directories(vesper_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  target_compile_features(vesper_headers INTERFACE cxx_std_20)
  target_compile_options(vesper_headers INTERFACE -Wall -Wextra -Wpedantic)
  # Enable AVX2 on x86-64 platforms
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|x64")
    target_compile_options(vesper_headers INTERFACE -mavx2 -mfma)
  endif()
elseif (MSVC)
  target_compile_features(vesper_headers INTERFACE cxx_std_20)
  target_compile_options(vesper_headers INTERFACE /W4)
  # Enable AVX2 on x86-64 platforms for MSVC
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|x64")
    target_compile_options(vesper_headers INTERFACE /arch:AVX2)
  endif()
endif()


# Build option: guarded dispatcher integration (default OFF)
option(VESPER_ENABLE_KERNEL_DISPATCH "Enable kernel dispatcher in in-memory search" OFF)
if (VESPER_ENABLE_KERNEL_DISPATCH)
  add_compile_definitions(VESPER_ENABLE_KERNEL_DISPATCH=1)
else()
  add_compile_definitions(VESPER_ENABLE_KERNEL_DISPATCH=0)
# Use static runtime on Windows when fuzzing with Clang to match libFuzzer (MT)
if (WIN32 AND VESPER_ENABLE_FUZZ AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # CMake 3.15+: controls /MT vs /MD for MSVC-like toolchains
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

endif()

# Fetch CRoaring bitmap library early (needed by tombstone module)
include(FetchContent)
set(ENABLE_ROARING_TESTS OFF CACHE BOOL "" FORCE)  # Disable CRoaring tests
FetchContent_Declare(
  croaring
  GIT_REPOSITORY https://github.com/RoaringBitmap/CRoaring.git
  GIT_TAG        v2.0.4
  CMAKE_ARGS -DENABLE_ROARING_TESTS=OFF
)
# Skip building CRoaring entirely for Clang+fuzz (we remove tombstone sources)
if (NOT (VESPER_ENABLE_FUZZ AND CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT MSVC))
  FetchContent_MakeAvailable(croaring)
endif()

# Library (minimal in-memory impl) to prove linkage in CI
set(VESPER_CORE_SOURCES
    src/inmemory_collection.cpp
    src/filter_eval.cpp
    src/kernels/dispatch.cpp
    src/index/kmeans.cpp
    src/index/kmeans_elkan.cpp

    src/index/hnsw.cpp
    # src/index/hnsw_lockfree.cpp (archived to experimental/)
    src/index/ivf_pq.cpp
    src/index/disk_graph.cpp
    src/index/product_quantizer.cpp
    src/index/index_manager.cpp
    src/index/query_planner.cpp
    src/index/rabitq_quantizer.cpp
    src/index/fast_hadamard.cpp
    src/index/capq_encode.cpp
    src/index/capq_util.cpp
    src/index/capq_dist_avx2.cpp
    src/index/capq_calibration.cpp
    src/index/capq_q4.cpp
    src/index/capq_opq.cpp
    src/index/matryoshka.cpp
    src/index/capq.cpp
    src/index/bm25.cpp
    # CGF (Cascaded Geometric Filtering) components
    # Temporarily disabled while fixing compilation issues
    # src/index/coarse_filter.cpp
    # src/index/hybrid_storage.cpp
    # src/index/smart_ivf.cpp
    # src/index/mini_hnsw.cpp
    src/search/hybrid_searcher.cpp
    src/search/fusion_algorithms.cpp
    src/io/async_io.cpp
    src/io/prefetch_manager.cpp
    src/tombstone/tombstone_manager.cpp
    src/metadata/metadata_store.cpp
    src/memory/numa_allocator.cpp
    )
list(APPEND VESPER_CORE_SOURCES src/index/projection_assigner.cpp)

# When fuzzing with Clang, exclude optional prefetch/async I/O modules and DiskANN/tombstone to simplify sanitizer build
if (VESPER_ENABLE_FUZZ AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  list(REMOVE_ITEM VESPER_CORE_SOURCES src/io/prefetch_manager.cpp)
  list(REMOVE_ITEM VESPER_CORE_SOURCES src/io/async_io.cpp)
  list(REMOVE_ITEM VESPER_CORE_SOURCES src/index/disk_graph.cpp)
  list(REMOVE_ITEM VESPER_CORE_SOURCES src/tombstone/tombstone_manager.cpp)
  list(REMOVE_ITEM VESPER_CORE_SOURCES src/index/index_manager.cpp)
  list(REMOVE_ITEM VESPER_CORE_SOURCES src/index/query_planner.cpp)
  list(REMOVE_ITEM VESPER_CORE_SOURCES src/metadata/metadata_store.cpp)
endif()


# Add platform-specific sources
# Windows reports AMD64, Linux/macOS report x86_64
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|x64")
    message(STATUS "Using FastScan PQ implementation for x86-64 architecture")
    list(APPEND VESPER_CORE_SOURCES src/index/pq_fastscan.cpp)
else()
    # Use simple PQ implementation on non-x86 platforms (ARM, etc.)
    message(STATUS "Using Simple PQ implementation for ${CMAKE_SYSTEM_PROCESSOR}")
    list(APPEND VESPER_CORE_SOURCES src/index/pq_simple.cpp)
endif()

# Add platform-specific I/O implementations
if(WIN32)
    # list(APPEND VESPER_CORE_SOURCES src/io/async_io_windows.cpp)  # Temporarily disabled
endif()

# Optional Zstd compression support
option(VESPER_WITH_ZSTD "Enable zstd compression support" OFF)

add_library(vesper_core STATIC ${VESPER_CORE_SOURCES})
target_link_libraries(vesper_core PRIVATE vesper_headers)
if(TARGET roaring)
  target_link_libraries(vesper_core PUBLIC roaring)
endif()

if (MSVC)
  target_compile_options(vesper_core PRIVATE /EHa)
endif()

# Optional BLAS linkage for projection screening
if (VESPER_ENABLE_CBLAS)
  # Define compile-time guard for CBLAS usage in code
  target_compile_definitions(vesper_core PUBLIC VESPER_HAS_CBLAS=1)
  # Prefer OpenBLAS config package (provides cblas.h and imported target)
  find_package(OpenBLAS CONFIG QUIET)
  if (OpenBLAS_FOUND)
    message(STATUS "OpenBLAS found; linking OpenBLAS::OpenBLAS")
    target_link_libraries(vesper_core PUBLIC OpenBLAS::OpenBLAS)
  else()
    # Fallback to generic BLAS (MKL/Accelerate/etc.). Note: may not provide cblas.h include dirs.
    find_package(BLAS)
    if (BLAS_FOUND)
      message(STATUS "BLAS found; linking BLAS::BLAS")
      target_link_libraries(vesper_core PUBLIC BLAS::BLAS)
    else()
      message(WARNING "VESPER_ENABLE_CBLAS=ON but no BLAS found; building without BLAS linkage")
      target_compile_definitions(vesper_core PUBLIC VESPER_HAS_CBLAS=0)
    endif()
  endif()
endif()

# If fuzzing with Clang (GNU driver), compile core with ASan/UBSan so link is consistent
if (VESPER_ENABLE_FUZZ AND CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT MSVC)
  target_compile_options(vesper_core PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(vesper_core PRIVATE -fsanitize=address,undefined)
  target_compile_definitions(vesper_core PUBLIC VESPER_NO_ROARING=1)
endif()

if (VESPER_WITH_ZSTD)
  find_package(ZSTD QUIET)
  if (NOT ZSTD_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      zstd_proj
      GIT_REPOSITORY https://github.com/facebook/zstd.git
      GIT_TAG        v1.5.6
    )
    set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_SHARED   OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(zstd_proj)
    # zstd CMake exports may name the target differently; create a consistent alias if needed
    if (TARGET libzstd_static)
      add_library(ZSTD::ZSTD ALIAS libzstd_static)
    elseif (TARGET zstd::libzstd_static)
      add_library(ZSTD::ZSTD ALIAS zstd::libzstd_static)
    endif()
  endif()
  if (TARGET ZSTD::ZSTD)
    target_link_libraries(vesper_core PRIVATE ZSTD::ZSTD)
    target_compile_definitions(vesper_core PRIVATE VESPER_HAS_ZSTD=1)
    # Propagate define so benchmarks/tests can use zstd conditionals
    add_compile_definitions(VESPER_HAS_ZSTD=1)
  endif()
endif()

# Enable AVX2 for vesper_core on x86-64
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|x64")
  if(MSVC)
    target_compile_options(vesper_core PRIVATE /arch:AVX2)
  else()
    target_compile_options(vesper_core PRIVATE -mavx2 -mfma)
  endif()
endif()
# ----------------------
# C API (shared)
# ----------------------
add_library(vesper_c SHARED src/c/vesper_c_api.cpp src/c/vesper_manager_c_api.cpp)
# Public headers for consumers of the C API
target_include_directories(vesper_c PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# Link core implementation
target_link_libraries(vesper_c PRIVATE vesper_headers vesper_core)
# Export macro for Windows
target_compile_definitions(vesper_c PRIVATE VESPER_C_API_EXPORTS=1)
if (MSVC)
  target_compile_options(vesper_c PRIVATE /EHa)
endif()


# Feature flags as compile definitions for core library
if (VESPER_ENABLE_ACCELERATE)
  target_compile_definitions(vesper_core PUBLIC VESPER_ENABLE_ACCELERATE=1)
else()
  target_compile_definitions(vesper_core PUBLIC VESPER_ENABLE_ACCELERATE=0)
endif()

if (VESPER_SERIALIZE_BASE_LAYER)
# Guarded concurrency compile definitions

  target_compile_definitions(vesper_core PUBLIC VESPER_SERIALIZE_BASE_LAYER=1)
else()
  target_compile_definitions(vesper_core PUBLIC VESPER_SERIALIZE_BASE_LAYER=0)
endif()
# Timing instrumentation (compile-time guarded)
option(VESPER_ENABLE_TIMING "Enable timing instrumentation in core" OFF)
if (VESPER_ENABLE_TIMING)
  target_compile_definitions(vesper_core PUBLIC VESPER_ENABLE_TIMING=1)
endif()


# Link optional dependencies
if (VESPER_HAVE_OPENMP)
  target_link_libraries(vesper_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if (VESPER_ENABLE_ACCELERATE AND APPLE)
  target_link_libraries(vesper_core PUBLIC "-framework Accelerate")
  message(STATUS "Accelerate enabled: linking '-framework Accelerate'")
else()
  message(STATUS "Accelerate disabled or non-Apple platform")
# Guarded concurrency compile definitions

endif()


# WAL library
add_library(vesper_wal STATIC src/wal/frame.cpp src/wal/io.cpp src/wal/manifest.cpp src/wal/snapshot.cpp src/wal/retention.cpp src/wal/replay.cpp src/wal/checkpoint.cpp)
target_link_libraries(vesper_wal PRIVATE vesper_headers)

# Dummy translation unit to force header checks in CI
add_executable(vesper_headers_check scripts/headers_check.cpp)
target_link_libraries(vesper_headers_check PRIVATE vesper_headers vesper_core)
# ----------------------
# Tests (Phase 4 scaffolding)
# Guarded concurrency compile definitions (apply regardless of Accelerate)
if (VESPER_PARTITIONED_BASE_LOCKS)
  target_compile_definitions(vesper_core PUBLIC VESPER_PARTITIONED_BASE_LOCKS=1)
else()
  target_compile_definitions(vesper_core PUBLIC VESPER_PARTITIONED_BASE_LOCKS=0)
endif()

if (VESPER_EXPERIMENTAL_WORK_STEALING)
  target_compile_definitions(vesper_core PUBLIC VESPER_EXPERIMENTAL_WORK_STEALING=1)
else()
  target_compile_definitions(vesper_core PUBLIC VESPER_EXPERIMENTAL_WORK_STEALING=0)
endif()

# ----------------------
include(CTest)
option(VESPER_ENABLE_TESTS "Build tests" ON)

if (VESPER_ENABLE_TESTS)
  # Propagate compile-time macro so code can expose test-only hooks under #ifdef VESPER_ENABLE_TESTS
  add_compile_definitions(VESPER_ENABLE_TESTS=1)

  include(FetchContent)
  set(CATCH2_TAG v3.5.3)
  FetchContent_Declare(
    catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        ${CATCH2_TAG}
  )
  FetchContent_MakeAvailable(catch2)

  file(GLOB TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/support/*.cpp)
  # Exclude GoogleTest-based and standalone executables from Catch2 aggregate
  list(REMOVE_ITEM TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/hnsw_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/hnsw_lockfree_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/ivf_pq_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/kmeans_algorithms_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/kmeans_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/simd_kernels_comprehensive_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/platform_agnostic_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/hnsw_thread_pool_stress_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/hnsw_connectivity_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/hnsw_basic_invariant_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_hnsw_batch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/simple_integration_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/minimal_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/basic_components_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/actual_index_performance_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/performance_validation_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_hnsw_simple.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/index_manager_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/query_planner_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/projection_assigner_avx2_parity_test.cpp
  )


  add_executable(vesper_tests ${TEST_SOURCES})
  target_link_libraries(vesper_tests PRIVATE Catch2::Catch2WithMain vesper_headers vesper_core vesper_wal)
  target_include_directories(vesper_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  # Standalone projection parity test (avoid linking vesper_wal)
  add_executable(vesper_proj_parity tests/unit/projection_assigner_avx2_parity_test.cpp)
  target_link_libraries(vesper_proj_parity PRIVATE Catch2::Catch2WithMain vesper_headers vesper_core)
  target_include_directories(vesper_proj_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME vesper_proj_parity COMMAND $<TARGET_FILE:vesper_proj_parity>)
  # Exclude GoogleTest-based and standalone executables from Catch2 aggregate
  list(REMOVE_ITEM TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/hnsw_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/hnsw_lockfree_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/ivf_pq_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/kmeans_algorithms_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/kmeans_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/simd_kernels_comprehensive_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/platform_agnostic_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/hnsw_thread_pool_stress_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/hnsw_connectivity_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_hnsw_batch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/simple_integration_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/projection_assigner_avx2_parity_test.cpp
  )


  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_features(vesper_tests PRIVATE cxx_std_20)
    target_compile_options(vesper_tests PRIVATE -Wall -Wextra -Wpedantic)
  elseif (MSVC)
    target_compile_features(vesper_tests PRIVATE cxx_std_20)
    target_compile_options(vesper_tests PRIVATE /W4)
  endif()

  include(Catch)
  catch_discover_tests(vesper_tests CONFIGURATIONS Release Debug RelWithDebInfo MinSizeRel)

  # Minimal BM25-only tests target to allow isolated builds
  add_executable(bm25_tests tests/unit/bm25_test.cpp)
  target_link_libraries(bm25_tests PRIVATE Catch2::Catch2WithMain vesper_headers vesper_core vesper_wal)
  target_include_directories(bm25_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  catch_discover_tests(bm25_tests CONFIGURATIONS Release Debug RelWithDebInfo MinSizeRel)


  # Lint: forbid std::getenv in tree (use vesper::core::safe_getenv instead)
  add_test(
    NAME lint_forbid_std_getenv
    COMMAND ${CMAKE_COMMAND} -DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake/CheckNoGetenv.cmake
  )

endif()

# ----------------------
# Benchmarks (Google Benchmark)
# ----------------------
option(VESPER_ENABLE_BENCH "Build micro-benchmarks" ON)
if (VESPER_ENABLE_BENCH)
  # Enable HNSW invariants in core when building benchmarks for better diagnostics
  target_compile_definitions(vesper_core PRIVATE VESPER_HNSW_INVARIANTS=1)

  include(FetchContent)
  set(GBENCH_TAG v1.8.3)
  FetchContent_Declare(
    gbench
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        ${GBENCH_TAG}
  )
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
# ----------------------
# Fuzz targets (Phase 4C scaffolding)
# ----------------------
option(VESPER_ENABLE_FUZZ "Build fuzz targets" OFF)
if (VESPER_ENABLE_FUZZ AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if (MSVC)
    # clang-cl path: use /fsanitize flags and link fuzzer runtime manually
    # Ensure core also built with ASan/UBSan to avoid CRT/sanitizer mismatches
    target_compile_definitions(vesper_core PRIVATE _DISABLE_STRING_ANNOTATION=1)
    target_compile_options(vesper_core PRIVATE /fsanitize=address)
    target_link_options(vesper_core PRIVATE /fsanitize=address)

    add_executable(wal_manifest_fuzz tests/fuzz/wal_manifest_fuzz.cpp)
    target_link_libraries(wal_manifest_fuzz PRIVATE vesper_headers vesper_core vesper_wal)
    target_compile_definitions(wal_manifest_fuzz PRIVATE _DISABLE_STRING_ANNOTATION=1)
    target_compile_options(wal_manifest_fuzz PRIVATE /fsanitize=address)
    target_link_options(wal_manifest_fuzz PRIVATE /fsanitize=address)

    add_executable(wal_frame_fuzz tests/fuzz/wal_frame_fuzz.cpp)
    target_link_libraries(wal_frame_fuzz PRIVATE vesper_headers vesper_core vesper_wal)
    target_compile_definitions(wal_frame_fuzz PRIVATE _DISABLE_STRING_ANNOTATION=1)
    target_compile_options(wal_frame_fuzz PRIVATE /fsanitize=address)
    target_link_options(wal_frame_fuzz PRIVATE /fsanitize=address)

    # IVF-PQ v1.1 loader fuzz (metadata-bearing, multi-section corruptions)
    add_executable(ivfpq_v11_fuzz tests/fuzz/ivfpq_v11_fuzz.cpp)
    target_link_libraries(ivfpq_v11_fuzz PRIVATE vesper_headers vesper_core)
    target_compile_definitions(ivfpq_v11_fuzz PRIVATE _DISABLE_STRING_ANNOTATION=1)
    target_compile_options(ivfpq_v11_fuzz PRIVATE /fsanitize=address)
    target_link_options(ivfpq_v11_fuzz PRIVATE /fsanitize=address)

    # Locate libFuzzer runtime and link it explicitly (clang_rt.fuzzer-x86_64.lib)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} --print-resource-dir
                    OUTPUT_VARIABLE CLANG_RESOURCE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(FUZZER_LIB "${CLANG_RESOURCE_DIR}/lib/windows/clang_rt.fuzzer-x86_64.lib")
    if (EXISTS "${FUZZER_LIB}")
      target_link_libraries(ivfpq_v11_fuzz PRIVATE "${FUZZER_LIB}")
      target_link_libraries(wal_manifest_fuzz PRIVATE "${FUZZER_LIB}")
      target_link_libraries(wal_frame_fuzz PRIVATE "${FUZZER_LIB}")
    endif()
  else()
    # clang++ (GNU driver) path
    add_executable(wal_manifest_fuzz tests/fuzz/wal_manifest_fuzz.cpp)
    target_link_libraries(wal_manifest_fuzz PRIVATE vesper_headers vesper_core vesper_wal)
    target_compile_options(wal_manifest_fuzz PRIVATE -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer)
    target_link_options(wal_manifest_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)

    add_executable(wal_frame_fuzz tests/fuzz/wal_frame_fuzz.cpp)
    target_link_libraries(wal_frame_fuzz PRIVATE vesper_headers vesper_core vesper_wal)
    target_compile_options(wal_frame_fuzz PRIVATE -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer)
    target_link_options(wal_frame_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)

    # IVF-PQ v1.1 loader fuzz (metadata-bearing, multi-section corruptions)
    add_executable(ivfpq_v11_fuzz tests/fuzz/ivfpq_v11_fuzz.cpp)
    target_link_libraries(ivfpq_v11_fuzz PRIVATE vesper_headers vesper_core)
    target_compile_options(ivfpq_v11_fuzz PRIVATE -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer)
    target_link_options(ivfpq_v11_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
  endif()
endif()
  FetchContent_MakeAvailable(gbench)

  add_executable(vesper_bench_dummy bench/micro/dummy_bench.cpp)
  target_link_libraries(vesper_bench_dummy PRIVATE benchmark::benchmark vesper_headers)

  add_executable(vesper_bench_simd bench/micro/simd_kernels_bench.cpp)
  target_link_libraries(vesper_bench_simd PRIVATE benchmark::benchmark vesper_headers vesper_core)
  add_executable(vesper_bench_hybrid bench/micro/hybrid_search_bench.cpp)
  target_link_libraries(vesper_bench_hybrid PRIVATE benchmark::benchmark vesper_headers vesper_core)

  add_executable(vesper_bench_index bench/vesper_bench_index.cpp)
  target_link_libraries(vesper_bench_index PRIVATE benchmark::benchmark vesper_headers vesper_core Threads::Threads)
  target_compile_features(vesper_bench_index PRIVATE cxx_std_20)

  add_executable(vesper_bench_fused_vs_materialized bench/micro/fused_vs_materialized_bench.cpp)
  target_link_libraries(vesper_bench_fused_vs_materialized PRIVATE benchmark::benchmark vesper_headers vesper_core)

  add_executable(vesper_bench_ivfpq_adc bench/micro/ivfpq_adc_bench.cpp)
  target_link_libraries(vesper_bench_ivfpq_adc PRIVATE benchmark::benchmark vesper_headers vesper_core)

  add_executable(vesper_bench_ivfpq_load bench/micro/ivfpq_load_bench.cpp)
  target_link_libraries(vesper_bench_ivfpq_load PRIVATE benchmark::benchmark vesper_headers vesper_core)
  target_compile_features(vesper_bench_ivfpq_load PRIVATE cxx_std_20)

  add_executable(vesper_bench_capq_scalar bench/micro/capq_scalar_bench.cpp)
  target_link_libraries(vesper_bench_capq_scalar PRIVATE benchmark::benchmark vesper_headers vesper_core)
  target_compile_features(vesper_bench_capq_scalar PRIVATE cxx_std_20)
  target_include_directories(vesper_bench_capq_scalar PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration)
  target_sources(vesper_bench_capq_scalar PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/dataset_loader.cpp)

  # Comprehensive full benchmark
  add_executable(vesper_full_benchmark bench/vesper_full_benchmark.cpp)
  target_link_libraries(vesper_full_benchmark PRIVATE benchmark::benchmark vesper_headers vesper_core Threads::Threads)
  target_compile_features(vesper_full_benchmark PRIVATE cxx_std_20)

  add_executable(vesper_bench_kmeans_init bench/micro/kmeans_init_bench.cpp)
  target_link_libraries(vesper_bench_kmeans_init PRIVATE benchmark::benchmark vesper_headers vesper_core)


endif()

# ----------------------
# Comprehensive Tests with GoogleTest
# ----------------------
option(VESPER_BUILD_GTESTS "Build GoogleTest-based tests" ON)

if(VESPER_BUILD_GTESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
  )
# Benchmark matrix (optional)
find_package(Python3 COMPONENTS Interpreter)
if (Python3_Interpreter_FOUND)
  add_test(NAME hnsw_bench_matrix COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/bench_matrix.py)
  # Keep this fast under CTest to avoid long hangs on CI/Windows
  set_tests_properties(hnsw_bench_matrix PROPERTIES TIMEOUT 120 ENVIRONMENT "VESPER_BENCH_FAST=1")
endif()

  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()

  # Comprehensive test executables
  add_executable(simd_kernels_test tests/unit/simd_kernels_comprehensive_test.cpp)
  target_link_libraries(simd_kernels_test PRIVATE vesper_headers vesper_core gtest_main)
  # Enable AVX2 for SIMD kernel tests
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|x64")
    if(MSVC)
      target_compile_options(simd_kernels_test PRIVATE /arch:AVX2)
    else()
      target_compile_options(simd_kernels_test PRIVATE -mavx2 -mfma)
    endif()
  endif()

  add_executable(hnsw_thread_pool_stress_test tests/unit/hnsw_thread_pool_stress_test.cpp)
  target_link_libraries(hnsw_thread_pool_stress_test PRIVATE vesper_headers vesper_core gtest_main Threads::Threads)

  add_executable(kmeans_test tests/unit/kmeans_algorithms_test.cpp)
  target_link_libraries(kmeans_test PRIVATE vesper_headers vesper_core gtest_main)

  add_executable(hnsw_index_test tests/unit/hnsw_test.cpp)
  add_executable(hnsw_basic_invariant_test tests/unit/hnsw_basic_invariant_test.cpp)
  target_link_libraries(hnsw_basic_invariant_test PRIVATE vesper_headers vesper_core gtest_main)

  target_link_libraries(hnsw_index_test PRIVATE vesper_headers vesper_core gtest_main)


  # Lock-free HNSW moved to experimental/
  # add_executable(hnsw_lockfree_test tests/unit/hnsw_lockfree_test.cpp)
  # target_link_libraries(hnsw_lockfree_test PRIVATE vesper_headers vesper_core gtest_main Threads::Threads)

  # add_executable(stress_test tests/unit/stress/scalability_stress_test.cpp)
  # target_link_libraries(stress_test PRIVATE vesper_headers vesper_core gtest_main Threads::Threads)

  # Simple integration test (no GTest dependency)
  add_executable(integration_test tests/unit/simple_integration_test.cpp)
  # Standalone connectivity executables
  add_executable(test_hnsw_batch tests/unit/test_hnsw_batch.cpp)
  target_link_libraries(test_hnsw_batch PRIVATE vesper_headers vesper_core Threads::Threads)

  add_executable(hnsw_connectivity_test tests/unit/hnsw_connectivity_test.cpp)
  target_link_libraries(hnsw_connectivity_test PRIVATE vesper_headers vesper_core Threads::Threads)


  # Dataset-based integration tests
  add_executable(vesper_integration_test tests/integration/vesper_integration_test.cpp tests/integration/dataset_loader.cpp)
  target_link_libraries(vesper_integration_test PRIVATE Catch2::Catch2WithMain vesper_headers vesper_core Threads::Threads)
  target_include_directories(vesper_integration_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_sources(vesper_integration_test PRIVATE tests/integration/capq_end_to_end_test.cpp)

  # Lightweight parity-only runner for CAPQ SIMD parity
  add_executable(vesper_capq_parity tests/unit/capq_avx2_parity_test.cpp)
  target_link_libraries(vesper_capq_parity PRIVATE vesper_headers vesper_core Catch2::Catch2WithMain)
  target_include_directories(vesper_capq_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(accuracy_test tests/integration/accuracy_test.cpp tests/integration/dataset_loader.cpp)
  target_link_libraries(accuracy_test PRIVATE Catch2::Catch2WithMain vesper_headers vesper_core Threads::Threads)
  target_include_directories(accuracy_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(performance_test tests/integration/performance_test.cpp tests/integration/dataset_loader.cpp)
  target_link_libraries(performance_test PRIVATE Catch2::Catch2WithMain vesper_headers vesper_core Threads::Threads)
  target_include_directories(performance_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  target_link_libraries(integration_test PRIVATE vesper_headers vesper_core Threads::Threads)
  catch_discover_tests(vesper_integration_test CONFIGURATIONS Release Debug RelWithDebInfo MinSizeRel)
  catch_discover_tests(accuracy_test CONFIGURATIONS Release Debug RelWithDebInfo MinSizeRel)
  add_executable(ivfpq_explain_test tests/integration/ivfpq_explain_test.cpp tests/integration/dataset_loader.cpp)
  target_link_libraries(ivfpq_explain_test PRIVATE Catch2::Catch2WithMain vesper_headers vesper_core Threads::Threads)
  target_include_directories(ivfpq_explain_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  catch_discover_tests(ivfpq_explain_test CONFIGURATIONS Release Debug RelWithDebInfo MinSizeRel)

  catch_discover_tests(performance_test CONFIGURATIONS Release Debug RelWithDebInfo MinSizeRel)

  add_executable(bruteforce_verifier_test tests/integration/bruteforce_verifier_test.cpp tests/integration/dataset_loader.cpp)
  target_link_libraries(bruteforce_verifier_test PRIVATE Catch2::Catch2WithMain vesper_headers vesper_core Threads::Threads)
  target_include_directories(bruteforce_verifier_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(ivfpq_sweep_test tests/integration/ivfpq_sweep_test.cpp tests/integration/dataset_loader.cpp)
  target_link_libraries(ivfpq_sweep_test PRIVATE Catch2::Catch2WithMain vesper_headers vesper_core Threads::Threads)
  target_include_directories(ivfpq_sweep_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  catch_discover_tests(bruteforce_verifier_test CONFIGURATIONS Release Debug RelWithDebInfo MinSizeRel)
  catch_discover_tests(ivfpq_sweep_test CONFIGURATIONS Release Debug RelWithDebInfo MinSizeRel)
  add_executable(ivfpq_serialize_roundtrip_test tests/integration/ivfpq_serialize_roundtrip_test.cpp)
  target_link_libraries(ivfpq_serialize_roundtrip_test PRIVATE Catch2::Catch2WithMain vesper_headers vesper_core Threads::Threads)
  target_include_directories(ivfpq_serialize_roundtrip_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  catch_discover_tests(ivfpq_serialize_roundtrip_test CONFIGURATIONS Release Debug RelWithDebInfo MinSizeRel)
  add_executable(ivfpq_v11_serialize_test tests/integration/ivfpq_v11_serialize_test.cpp)
  target_link_libraries(ivfpq_v11_serialize_test PRIVATE Catch2::Catch2WithMain vesper_headers vesper_core Threads::Threads)
  target_include_directories(ivfpq_v11_serialize_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  catch_discover_tests(ivfpq_v11_serialize_test CONFIGURATIONS Release Debug RelWithDebInfo MinSizeRel)

  add_executable(ivfpq_v11_error_handling_test tests/integration/ivfpq_v11_error_handling_test.cpp)
  target_link_libraries(ivfpq_v11_error_handling_test PRIVATE Catch2::Catch2WithMain vesper_headers vesper_core Threads::Threads)
  target_include_directories(ivfpq_v11_error_handling_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  catch_discover_tests(ivfpq_v11_error_handling_test CONFIGURATIONS Release Debug RelWithDebInfo MinSizeRel)
  add_executable(ivfpq_v11_stream_position_test tests/integration/ivfpq_v11_stream_position_test.cpp)
  target_link_libraries(ivfpq_v11_stream_position_test PRIVATE Catch2::Catch2WithMain vesper_headers vesper_core Threads::Threads)
  target_include_directories(ivfpq_v11_stream_position_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  catch_discover_tests(ivfpq_v11_stream_position_test CONFIGURATIONS Release Debug RelWithDebInfo MinSizeRel)
  add_executable(ivfpq_compat_matrix_test tests/integration/ivfpq_compat_matrix_test.cpp)
  target_link_libraries(ivfpq_compat_matrix_test PRIVATE Catch2::Catch2WithMain vesper_headers vesper_core Threads::Threads)
  target_include_directories(ivfpq_compat_matrix_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  catch_discover_tests(ivfpq_compat_matrix_test CONFIGURATIONS Release Debug RelWithDebInfo MinSizeRel)
  add_executable(ivfpq_v11_metadata_test tests/integration/ivfpq_v11_metadata_test.cpp)
  target_link_libraries(ivfpq_v11_metadata_test PRIVATE Catch2::Catch2WithMain vesper_headers vesper_core Threads::Threads)
  target_include_directories(ivfpq_v11_metadata_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  catch_discover_tests(ivfpq_v11_metadata_test CONFIGURATIONS Release Debug RelWithDebInfo MinSizeRel)
  add_executable(ivfpq_v11_metadata_limits_test tests/integration/ivfpq_v11_metadata_limits_test.cpp)
  target_link_libraries(ivfpq_v11_metadata_limits_test PRIVATE Catch2::Catch2WithMain vesper_headers vesper_core Threads::Threads)
  target_include_directories(ivfpq_v11_metadata_limits_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  catch_discover_tests(ivfpq_v11_metadata_limits_test CONFIGURATIONS Release Debug RelWithDebInfo MinSizeRel)


  add_executable(ivfpq_fuzz_robust_test tests/integration/ivfpq_fuzz_robust_test.cpp)
  target_link_libraries(ivfpq_fuzz_robust_test PRIVATE Catch2::Catch2WithMain vesper_headers vesper_core Threads::Threads)
  target_include_directories(ivfpq_fuzz_robust_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  catch_discover_tests(ivfpq_fuzz_robust_test CONFIGURATIONS Release Debug RelWithDebInfo MinSizeRel)







  # Multi-index integration test
  add_executable(multi_index_test tests/integration/multi_index_test.cpp)
  target_link_libraries(multi_index_test PRIVATE Catch2::Catch2WithMain vesper_headers vesper_core Threads::Threads)

  # Simple IndexManager test
  add_executable(test_index_manager test_index_manager.cpp)
  target_link_libraries(test_index_manager PRIVATE vesper_headers vesper_core Threads::Threads)

  # Reranking test
  add_executable(test_reranking tests/integration/test_reranking.cpp)
  target_link_libraries(test_reranking PRIVATE vesper_headers vesper_core Threads::Threads)

  # IVF-PQ Diagnostic test
  add_executable(ivfpq_diagnostic_test tests/integration/ivfpq_diagnostic_test.cpp)
  target_link_libraries(ivfpq_diagnostic_test PRIVATE Catch2::Catch2WithMain vesper_headers vesper_core Threads::Threads)
  target_include_directories(ivfpq_diagnostic_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  catch_discover_tests(ivfpq_diagnostic_test CONFIGURATIONS Release Debug RelWithDebInfo MinSizeRel)

  # CGF Proof-of-Concept test
  add_executable(cgf_poc_test tests/integration/cgf_poc_test.cpp)
  target_link_libraries(cgf_poc_test PRIVATE Catch2::Catch2WithMain vesper_headers vesper_core Threads::Threads)
  target_include_directories(cgf_poc_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  catch_discover_tests(cgf_poc_test CONFIGURATIONS Release Debug RelWithDebInfo MinSizeRel)

  # Reranking debug test
  add_executable(test_reranking_debug tests/integration/test_reranking_debug.cpp)
  target_link_libraries(test_reranking_debug PRIVATE vesper_headers vesper_core Threads::Threads)

  # Add tests to CTest
  include(GoogleTest)
  gtest_discover_tests(simd_kernels_test)
  gtest_discover_tests(kmeans_test)
  gtest_discover_tests(hnsw_index_test)
  # gtest_discover_tests(hnsw_lockfree_test)
  # gtest_discover_tests(stress_test)
  gtest_discover_tests(hnsw_basic_invariant_test)

  add_test(NAME integration_test COMMAND integration_test)
  # Relax perf targets during CTest to avoid environment variability
  set_tests_properties(integration_test PROPERTIES ENVIRONMENT "VESPER_RELAX_PERF=1")

  # Catch2 test discovery for multi-index test
  catch_discover_tests(multi_index_test CONFIGURATIONS Release Debug RelWithDebInfo MinSizeRel)
  gtest_discover_tests(hnsw_thread_pool_stress_test)

endif()


# ----------------------
# Tools / Smoke tests
# ----------------------
add_executable(connectivity_smoke tools/connectivity_smoke.cpp)
# Link against core library and headers; threads not strictly needed here
target_link_libraries(connectivity_smoke PRIVATE vesper_headers vesper_core Threads::Threads)

# Removed standalone test executables - tests are now in tests/ directory
add_executable(ivfpq_persist_bench tools/ivfpq_persist_bench.cpp)
# Link against core and headers; Threads for potential future use
if (WIN32)
  target_link_libraries(ivfpq_persist_bench PRIVATE vesper_headers vesper_core Threads::Threads Psapi)
else()
  target_link_libraries(ivfpq_persist_bench PRIVATE vesper_headers vesper_core Threads::Threads)
endif()

# ----------------------
# C API Example
# ----------------------
add_executable(vesper_c_example examples/c/vesper_c_example.c)
target_include_directories(vesper_c_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vesper_c_example PRIVATE vesper_c)
# On Windows, copy the DLL next to the example so it can run under CTest
if (WIN32)
  add_custom_command(TARGET vesper_c_example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:vesper_c> $<TARGET_FILE_DIR:vesper_c_example>)
endif()

# CTest for the example
add_test(NAME test_c_api_example COMMAND $<TARGET_FILE:vesper_c_example>)

# ----------------------
# IndexManager C API Example
# ----------------------
add_executable(vesper_manager_example examples/c/vesper_manager_example.c)
target_include_directories(vesper_manager_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vesper_manager_example PRIVATE vesper_c)

# Real-world test example
# ----------------------
add_executable(real_world_test examples/real_world_test.cpp)
target_include_directories(real_world_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(real_world_test PRIVATE vesper_core vesper_headers)
if (WIN32)
  add_custom_command(TARGET vesper_manager_example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:vesper_c> $<TARGET_FILE_DIR:vesper_manager_example>)
endif()

add_test(NAME test_c_index_manager_example COMMAND $<TARGET_FILE:vesper_manager_example>)

# add() throughput micro-bench for Task 18 acceptance
add_executable(ivfpq_add_bench tools/ivfpq_add_bench.cpp)
target_link_libraries(ivfpq_add_bench PRIVATE vesper_headers vesper_core Threads::Threads)
if (MSVC)
  target_compile_options(ivfpq_add_bench PRIVATE /EHa)
endif()

# IVF-PQ Auto-tuning utility
add_executable(ivfpq_autotune tools/ivfpq_autotune.cpp)
target_link_libraries(ivfpq_autotune PRIVATE vesper_headers vesper_core Threads::Threads)

# Hierarchical KD-tree test
add_executable(test_hierarchical_kd tools/test_hierarchical_kd.cpp)
target_link_libraries(test_hierarchical_kd PRIVATE vesper_headers vesper_core Threads::Threads)

# E2E runner: load dataset, build, persist, reload, query
add_executable(vesper_e2e_runner tools/vesper_e2e_runner.cpp)
target_link_libraries(vesper_e2e_runner PRIVATE vesper_headers vesper_core Threads::Threads)
target_include_directories(vesper_e2e_runner PRIVATE ${CMAKE_SOURCE_DIR})
if (MSVC)
  target_compile_options(vesper_e2e_runner PRIVATE /EHa)
target_sources(vesper_e2e_runner PRIVATE tests/integration/dataset_loader.cpp)

endif()

# CAPQ sweep tool
add_executable(capq_sweep tools/capq_sweep.cpp)
target_link_libraries(capq_sweep PRIVATE vesper_headers vesper_core Threads::Threads)
target_include_directories(capq_sweep PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_sources(capq_sweep PRIVATE tests/integration/dataset_loader.cpp)

# CAPQ diagnostics tool (JSON metrics)
add_executable(capq_diagnostics tools/capq_diagnostics.cpp)
target_link_libraries(capq_diagnostics PRIVATE vesper_headers vesper_core Threads::Threads)
target_include_directories(capq_diagnostics PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_sources(capq_diagnostics PRIVATE tests/integration/dataset_loader.cpp)


