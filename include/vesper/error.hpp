#pragma once

#include <cstdint>
#include <string>
#include <system_error>
#include <utility>

namespace vesper::core {

// Error categories (stable codes) — sketch; finalized in ADR-0005
enum class error_code : std::uint32_t {
  ok = 0,
  io_failed = 1001,
  io_eof = 1002,
  config_invalid = 2001,
  data_integrity = 3001,
  precondition_failed = 4001,
  resource_exhausted = 5001,
  not_found = 6001,
  unavailable = 7001,
  cancelled = 8001,
  internal = 9001,
};

struct error {
  error_code code{error_code::internal};
  std::string message;    // human-readable (short)
  std::string component;  // e.g., "storage.wal"
};

} // namespace vesper::core

