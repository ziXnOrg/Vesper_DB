#pragma once

/** \file retention.hpp
 *  \brief WAL retention helpers (purge rotated files covered by snapshot/LSN).
 */

#include <expected>
#include <filesystem>

#include "vesper/error.hpp"

namespace vesper::wal {

auto purge_wal(const std::filesystem::path& dir, std::uint64_t up_to_lsn)
    -> std::expected<void, vesper::core::error>;

} // namespace vesper::wal

