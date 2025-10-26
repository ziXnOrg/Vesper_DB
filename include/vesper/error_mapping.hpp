#pragma once

#include "vesper/error.hpp"
#include "vesper/vesper_c.h"

namespace vesper::core {

constexpr vesper_status_t to_c_status(error_code ec) {
  switch (ec) {
    case error_code::ok: return VESPER_OK;
    case error_code::io_failed: return VESPER_E_IO_FAILED;
    case error_code::config_invalid: return VESPER_E_CONFIG_INVALID;
    case error_code::data_integrity: return VESPER_E_DATA_INTEGRITY;
    case error_code::precondition_failed: return VESPER_E_PRECONDITION_FAILED;
    case error_code::resource_exhausted: return VESPER_E_RESOURCE_EXHAUSTED;
    case error_code::not_found: return VESPER_E_NOT_FOUND;
    case error_code::unavailable: return VESPER_E_UNAVAILABLE;
    case error_code::cancelled: return VESPER_E_CANCELLED;
    case error_code::internal: return VESPER_E_INTERNAL;
    case error_code::io_eof: return VESPER_OK; // treat EOF as non-fatal boundary in C API
  }
  return VESPER_E_INTERNAL;
}

constexpr error_code from_c_status(vesper_status_t st) {
  switch (st) {
    case VESPER_OK: return error_code::ok;
    case VESPER_E_IO_FAILED: return error_code::io_failed;
    case VESPER_E_CONFIG_INVALID: return error_code::config_invalid;
    case VESPER_E_DATA_INTEGRITY: return error_code::data_integrity;
    case VESPER_E_PRECONDITION_FAILED: return error_code::precondition_failed;
    case VESPER_E_RESOURCE_EXHAUSTED: return error_code::resource_exhausted;
    case VESPER_E_NOT_FOUND: return error_code::not_found;
    case VESPER_E_UNAVAILABLE: return error_code::unavailable;
    case VESPER_E_CANCELLED: return error_code::cancelled;
    case VESPER_E_INTERNAL: return error_code::internal;
    default: return error_code::internal;
  }
}

} // namespace vesper::core

