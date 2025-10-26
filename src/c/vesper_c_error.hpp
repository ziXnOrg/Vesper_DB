#pragma once
#include <string>
#include <string_view>

namespace vesper_c {
  // Shared thread-local error buffer for all C API translation units
  extern thread_local std::string g_last_error;

  inline void set_error(std::string_view s) noexcept {
    g_last_error.assign(s.data(), s.size());
  }
  inline void clear_error() noexcept {
    g_last_error.clear();
  }
} // namespace vesper_c

