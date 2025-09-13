# Fail if any source within selected project directories uses std::getenv(
# Whitelist: platform_utils.hpp (the safe wrapper implementation)

if(NOT DEFINED SRC_DIR)
  message(FATAL_ERROR "CheckNoGetenv.cmake requires -DSRC_DIR=<source_root>")
endif()

set(WHITELIST_FILE "${SRC_DIR}/include/vesper/core/platform_utils.hpp")

# Directories to scan (exclude third-party by construction)
set(SCAN_DIRS
  "${SRC_DIR}/include/vesper"
  "${SRC_DIR}/src"
  "${SRC_DIR}/tests"
  "${SRC_DIR}/bench"
  "${SRC_DIR}/tools"
  "${SRC_DIR}/examples"
)

set(FORBIDDEN_MATCHES)

foreach(dir IN LISTS SCAN_DIRS)
  if(EXISTS "${dir}")
    file(GLOB_RECURSE FILES
      LIST_DIRECTORIES false
      "${dir}/*.h" "${dir}/*.hpp" "${dir}/*.hh" "${dir}/*.inl"
      "${dir}/*.c" "${dir}/*.cc" "${dir}/*.cpp" "${dir}/*.cxx"
    )
    foreach(f IN LISTS FILES)
      # Skip generated or build artifacts conservatively
      if(f MATCHES "/_deps/" OR f MATCHES "/build" OR f MATCHES "/CMakeFiles/" OR f MATCHES ".*~$")
        continue()
      endif()
      if(f STREQUAL WHITELIST_FILE)
        continue()
      endif()
      file(READ "${f}" CONTENT)
      if(CONTENT MATCHES "std::getenv[ \t]*\\(")
        list(APPEND FORBIDDEN_MATCHES "${f}")
      endif()
    endforeach()
  endif()
endforeach()

list(REMOVE_DUPLICATES FORBIDDEN_MATCHES)

if(FORBIDDEN_MATCHES)
  message(STATUS "Forbidden std::getenv usages found:")
  foreach(x IN LISTS FORBIDDEN_MATCHES)
    message(STATUS "  ${x}")
  endforeach()
  message(FATAL_ERROR "std::getenv is forbidden; use vesper::core::safe_getenv instead")
else()
  message(STATUS "No forbidden std::getenv usages found.")
endif()

