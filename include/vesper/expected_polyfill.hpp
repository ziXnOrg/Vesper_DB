#pragma once

// Unified include for std::expected (C++23) or the local polyfill for C++20.
// If the toolchain provides <expected>, use it. Otherwise, our repository
// ships a minimal polyfill available as <expected> (include/expected).

#if defined(__has_include)
#  if __has_include(<expected>)
#    include <expected>
#  else
#    include <expected>  // resolved to repo-local polyfill (include/expected)
#  endif
#else
#  include <expected>
#endif

