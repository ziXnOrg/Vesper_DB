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

# Dummy translation unit to force header checks in CI (will be replaced later)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/*.cpp)

  add_executable(vesper_tests ${TEST_SOURCES})
  target_link_libraries(vesper_tests PRIVATE Catch2::Catch2WithMain vesper_headers)
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
# Bench placeholders
# ----------------------
add_executable(vesper_bench_dummy bench/micro/dummy_bench.cpp)
target_link_libraries(vesper_bench_dummy PRIVATE vesper_headers)

