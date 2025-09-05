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

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling parallelization")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(STATUS "OpenMP not found, building without parallelization")
endif()

# Find required packages
find_package(Threads REQUIRED)

# Header-only interface target to compile public headers with -pedantic
add_library(vesper_headers INTERFACE)
target_include_directories(vesper_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  target_compile_features(vesper_headers INTERFACE cxx_std_20)
  target_compile_options(vesper_headers INTERFACE -Wall -Wextra -Wpedantic)
elseif (MSVC)
  target_compile_features(vesper_headers INTERFACE cxx_std_20)
  target_compile_options(vesper_headers INTERFACE /W4)
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
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    list(APPEND VESPER_CORE_SOURCES src/index/pq_fastscan.cpp)
else()
    # Use simple PQ implementation on non-x86 platforms (ARM, etc.)
    list(APPEND VESPER_CORE_SOURCES src/index/pq_simple.cpp)
endif()

add_library(vesper_core STATIC ${VESPER_CORE_SOURCES})
target_link_libraries(vesper_core PRIVATE vesper_headers)

# WAL library
add_library(vesper_wal STATIC src/wal/frame.cpp src/wal/io.cpp src/wal/manifest.cpp src/wal/snapshot.cpp src/wal/retention.cpp src/wal/replay.cpp)
target_link_libraries(vesper_wal PRIVATE vesper_headers)

# Dummy translation unit to force header checks in CI
add_executable(vesper_headers_check scripts/headers_check.cpp)
target_link_libraries(vesper_headers_check PRIVATE vesper_headers vesper_core)
# ----------------------
# Tests (Phase 4 scaffolding)
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

  add_executable(vesper_tests ${TEST_SOURCES})
  target_link_libraries(vesper_tests PRIVATE Catch2::Catch2WithMain vesper_headers vesper_core vesper_wal)
  target_include_directories(vesper_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

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
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  
  enable_testing()
  
  # Comprehensive test executables
  add_executable(simd_kernels_test tests/unit/simd_kernels_comprehensive_test.cpp)
  target_link_libraries(simd_kernels_test PRIVATE vesper_headers vesper_core gtest_main)
  
  add_executable(kmeans_test tests/unit/kmeans_algorithms_test.cpp)
  target_link_libraries(kmeans_test PRIVATE vesper_headers vesper_core gtest_main)
  
  add_executable(hnsw_index_test tests/unit/hnsw_test.cpp)
  target_link_libraries(hnsw_index_test PRIVATE vesper_headers vesper_core gtest_main)
  
  add_executable(hnsw_lockfree_test tests/unit/hnsw_lockfree_test.cpp)
  target_link_libraries(hnsw_lockfree_test PRIVATE vesper_headers vesper_core gtest_main Threads::Threads)
  
  add_executable(stress_test tests/stress/scalability_stress_test.cpp)
  target_link_libraries(stress_test PRIVATE vesper_headers vesper_core gtest_main Threads::Threads)
  
  # Simple integration test (no GTest dependency)
  add_executable(integration_test tests/unit/simple_integration_test.cpp)
  target_link_libraries(integration_test PRIVATE vesper_headers vesper_core Threads::Threads)
  
  # Add tests to CTest
  include(GoogleTest)
  gtest_discover_tests(simd_kernels_test)
  gtest_discover_tests(kmeans_test)
  gtest_discover_tests(hnsw_index_test)
  gtest_discover_tests(hnsw_lockfree_test)
  gtest_discover_tests(stress_test)
  add_test(NAME integration_test COMMAND integration_test)
endif()

