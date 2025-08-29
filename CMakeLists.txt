cmake_minimum_required(VERSION 3.24)
project(vesper LANGUAGES C CXX)

# Header-only interface target to compile public headers with -pedantic
add_library(vesper_headers INTERFACE)
target_include_directories(vesper_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  target_compile_features(vesper_headers INTERFACE cxx_std_23)
  target_compile_options(vesper_headers INTERFACE -Wall -Wextra -Wpedantic)
elseif (MSVC)
  target_compile_features(vesper_headers INTERFACE cxx_std_23)
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
add_library(vesper_core STATIC src/inmemory_collection.cpp src/filter_eval.cpp)
target_link_libraries(vesper_core PRIVATE vesper_headers)

# WAL library
add_library(vesper_wal STATIC src/wal/frame.cpp src/wal/io.cpp src/wal/manifest.cpp src/wal/snapshot.cpp src/wal/retention.cpp src/wal/replay.cpp)
target_link_libraries(vesper_wal PRIVATE vesper_headers)

# Dummy translation unit to force header checks in CI
add_executable(vesper_headers_check scripts/headers_check.cpp)
target_link_libraries(vesper_headers_check PRIVATE vesper_headers)
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
    target_compile_features(vesper_tests PRIVATE cxx_std_23)
    target_compile_options(vesper_tests PRIVATE -Wall -Wextra -Wpedantic)
  elseif (MSVC)
    target_compile_features(vesper_tests PRIVATE cxx_std_23)
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
  FetchContent_MakeAvailable(gbench)

  add_executable(vesper_bench_dummy bench/micro/dummy_bench.cpp)
  target_link_libraries(vesper_bench_dummy PRIVATE benchmark::benchmark vesper_headers)
endif()

