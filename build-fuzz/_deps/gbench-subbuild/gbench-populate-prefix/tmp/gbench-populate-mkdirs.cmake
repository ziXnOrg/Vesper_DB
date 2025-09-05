# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/shaiiko/Vesper/build-fuzz/_deps/gbench-src")
  file(MAKE_DIRECTORY "/Users/shaiiko/Vesper/build-fuzz/_deps/gbench-src")
endif()
file(MAKE_DIRECTORY
  "/Users/shaiiko/Vesper/build-fuzz/_deps/gbench-build"
  "/Users/shaiiko/Vesper/build-fuzz/_deps/gbench-subbuild/gbench-populate-prefix"
  "/Users/shaiiko/Vesper/build-fuzz/_deps/gbench-subbuild/gbench-populate-prefix/tmp"
  "/Users/shaiiko/Vesper/build-fuzz/_deps/gbench-subbuild/gbench-populate-prefix/src/gbench-populate-stamp"
  "/Users/shaiiko/Vesper/build-fuzz/_deps/gbench-subbuild/gbench-populate-prefix/src"
  "/Users/shaiiko/Vesper/build-fuzz/_deps/gbench-subbuild/gbench-populate-prefix/src/gbench-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/shaiiko/Vesper/build-fuzz/_deps/gbench-subbuild/gbench-populate-prefix/src/gbench-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/shaiiko/Vesper/build-fuzz/_deps/gbench-subbuild/gbench-populate-prefix/src/gbench-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
