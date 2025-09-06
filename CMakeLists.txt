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
    src/index/ivf_pq.cpp)

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

add_library(vesper_core STATIC ${VESPER_CORE_SOURCES})
target_link_libraries(vesper_core PRIVATE vesper_headers)

# Enable AVX2 for vesper_core on x86-64
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|x64")
  if(MSVC)
    target_compile_options(vesper_core PRIVATE /arch:AVX2)
  else()
    target_compile_options(vesper_core PRIVATE -mavx2 -mfma)
  endif()
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
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_hnsw_batch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/simple_integration_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/minimal_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/basic_components_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/actual_index_performance_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/performance_validation_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_hnsw_simple.cpp
  )


  add_executable(vesper_tests ${TEST_SOURCES})
  target_link_libraries(vesper_tests PRIVATE Catch2::Catch2WithMain vesper_headers vesper_core vesper_wal)
  target_include_directories(vesper_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
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
  )


  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_features(vesper_tests PRIVATE cxx_std_20)
    target_compile_options(vesper_tests PRIVATE -Wall -Wextra -Wpedantic)
  elseif (MSVC)
    target_compile_features(vesper_tests PRIVATE cxx_std_20)
    target_compile_options(vesper_tests PRIVATE /W4)
  endif()

  include(Catch)
  catch_discover_tests(vesper_tests)
endif()

# ----------------------
# Benchmarks (Google Benchmark)
# ----------------------
option(VESPER_ENABLE_BENCH "Build micro-benchmarks" ON)
if (VESPER_ENABLE_BENCH)
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
  add_executable(wal_manifest_fuzz tests/fuzz/wal_manifest_fuzz.cpp)
  target_link_libraries(wal_manifest_fuzz PRIVATE vesper_headers vesper_core vesper_wal)
  target_compile_options(wal_manifest_fuzz PRIVATE -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer)
  target_link_options(wal_manifest_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)

  add_executable(wal_frame_fuzz tests/fuzz/wal_frame_fuzz.cpp)
  target_link_libraries(wal_frame_fuzz PRIVATE vesper_headers vesper_core vesper_wal)
  target_compile_options(wal_frame_fuzz PRIVATE -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer)
  target_link_options(wal_frame_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
endif()
  FetchContent_MakeAvailable(gbench)

  add_executable(vesper_bench_dummy bench/micro/dummy_bench.cpp)
  target_link_libraries(vesper_bench_dummy PRIVATE benchmark::benchmark vesper_headers)

  add_executable(vesper_bench_simd bench/micro/simd_kernels_bench.cpp)
  target_link_libraries(vesper_bench_simd PRIVATE benchmark::benchmark vesper_headers vesper_core)

  add_executable(vesper_bench_index bench/vesper_bench_index.cpp)
  target_link_libraries(vesper_bench_index PRIVATE benchmark::benchmark vesper_headers vesper_core Threads::Threads)
  target_compile_features(vesper_bench_index PRIVATE cxx_std_20)

  add_executable(vesper_bench_fused_vs_materialized bench/micro/fused_vs_materialized_bench.cpp)
  target_link_libraries(vesper_bench_fused_vs_materialized PRIVATE benchmark::benchmark vesper_headers vesper_core)

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
  set_tests_properties(hnsw_bench_matrix PROPERTIES TIMEOUT 3600)
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
  target_link_libraries(hnsw_index_test PRIVATE vesper_headers vesper_core gtest_main)


  # Lock-free HNSW moved to experimental/
  # add_executable(hnsw_lockfree_test tests/unit/hnsw_lockfree_test.cpp)
  # target_link_libraries(hnsw_lockfree_test PRIVATE vesper_headers vesper_core gtest_main Threads::Threads)

  add_executable(stress_test tests/stress/scalability_stress_test.cpp)
  target_link_libraries(stress_test PRIVATE vesper_headers vesper_core gtest_main Threads::Threads)

  # Simple integration test (no GTest dependency)
  add_executable(integration_test tests/unit/simple_integration_test.cpp)
  # Standalone connectivity executables
  add_executable(test_hnsw_batch tests/unit/test_hnsw_batch.cpp)
  target_link_libraries(test_hnsw_batch PRIVATE vesper_headers vesper_core Threads::Threads)

  add_executable(hnsw_connectivity_test tests/unit/hnsw_connectivity_test.cpp)
  target_link_libraries(hnsw_connectivity_test PRIVATE vesper_headers vesper_core Threads::Threads)

  target_link_libraries(integration_test PRIVATE vesper_headers vesper_core Threads::Threads)

  # Add tests to CTest
  include(GoogleTest)
  gtest_discover_tests(simd_kernels_test)
  gtest_discover_tests(kmeans_test)
  gtest_discover_tests(hnsw_index_test)
  # gtest_discover_tests(hnsw_lockfree_test)
  gtest_discover_tests(stress_test)
  add_test(NAME integration_test COMMAND integration_test)
  gtest_discover_tests(hnsw_thread_pool_stress_test)

endif()


# ----------------------
# Tools / Smoke tests
# ----------------------
add_executable(connectivity_smoke tools/connectivity_smoke.cpp)
# Link against core library and headers; threads not strictly needed here
target_link_libraries(connectivity_smoke PRIVATE vesper_headers vesper_core Threads::Threads)

# Performance benchmark
add_executable(simple_benchmark simple_benchmark.cpp)
target_link_libraries(simple_benchmark PRIVATE vesper_headers vesper_core Threads::Threads)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|x64")
  if(MSVC)
    target_compile_options(simple_benchmark PRIVATE /arch:AVX2)
  else()
    target_compile_options(simple_benchmark PRIVATE -mavx2 -mfma)
  endif()
endif()

# IVF-PQ test
add_executable(test_ivfpq test_ivfpq.cpp)
target_link_libraries(test_ivfpq PRIVATE vesper_headers vesper_core Threads::Threads)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|x64")
  if(MSVC)
    target_compile_options(test_ivfpq PRIVATE /arch:AVX2)
  else()
    target_compile_options(test_ivfpq PRIVATE -mavx2 -mfma)
  endif()
endif()

