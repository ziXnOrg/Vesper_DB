#pragma once

/**
 * \file error.hpp
 * \brief Error taxonomy and structured error type used with std::expected.
 *
 * Design:
 * - Stable error codes (see ADR‑0005 and architecture/errors.md) for programmatic handling.
 * - Human-readable message and originating component for diagnostics.
 */

#include <cstdint>
#include <string>
#include <vesper/expected_polyfill.hpp>

namespace vesper::core {

/** \brief Stable error codes used across the library. */
enum class error_code : std::uint32_t {
  ok = 0,
  io_failed = 1001,
  io_eof = 1002,
  io_error = 1003,
  config_invalid = 2001,
  data_integrity = 3001,
  precondition_failed = 4001,
  resource_exhausted = 5001,
  out_of_memory = 5002,
  not_found = 6001,
  unavailable = 7001,
  cancelled = 8001,
  internal = 9001,
  invalid_argument = 9002,
  not_initialized = 9003,
  out_of_range = 9004,
  unsupported = 9005,
};

/** \brief Structured error payload accompanying an error_code. */
struct error {
  error_code code{error_code::internal};   /**< machine-parseable code */
  std::string message;                     /**< short human-readable message */
  std::string component;                   /**< subsystem, e.g., "storage.wal" */
};

} // namespace vesper::core

