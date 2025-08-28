cmake_minimum_required(VERSION 3.24)
project(vesper LANGUAGES C CXX)

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

# Dummy translation unit to force header checks in CI (will be replaced later)
add_executable(vesper_headers_check scripts/headers_check.cpp)
target_link_libraries(vesper_headers_check PRIVATE vesper_headers)

