#pragma once

#include <expected>
#include <filesystem>
#include <string>

#include "vesper/error.hpp"
#include "vesper/wal/replay.hpp"

namespace vesper::wal::checkpoint {

struct Checkpoint { std::string consumer; std::uint64_t last_lsn{}; };

auto load(const std::filesystem::path& dir, std::string_view consumer)
    -> std::expected<Checkpoint, vesper::core::error>;

auto save(const std::filesystem::path& dir, std::string_view consumer, std::uint64_t last_lsn)
    -> std::expected<void, vesper::core::error>;

// Convenience wrapper: replay starting after stored checkpoint; filters by type mask
auto replay_from_checkpoint(const std::filesystem::path& dir,
                            std::string_view consumer,
                            std::uint32_t type_mask,
                            wal::ReplayCallback cb)
    -> std::expected<wal::RecoveryStats, vesper::core::error>;

} // namespace vesper::wal::checkpoint

