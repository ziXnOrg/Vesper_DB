#pragma once

#include <cstdint>
#include <filesystem>
#include <tests/support/replayer_payload.hpp>

namespace test_support {

// Deterministically reconstruct a ToyIndex by:
// 1) applying baseline frames with lsn <= cutoff by scanning wal-*.log files in order
// 2) applying post-cutoff frames (lsn > cutoff) via wal::recover_replay(dir, ...)
// No manifest assumptions; relies on file enumeration order only.
ToyIndex build_toy_index_baseline_then_replay(const std::filesystem::path& dir, std::uint64_t cutoff);

} // namespace test_support

