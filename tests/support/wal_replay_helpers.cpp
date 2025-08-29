#include "tests/support/wal_replay_helpers.hpp"

#include <regex>
#include <vector>
#include <algorithm>

#include <vesper/wal/io.hpp>
#include <vesper/wal/replay.hpp>

namespace test_support {

ToyIndex build_toy_index_baseline_then_replay(const std::filesystem::path& dir, std::uint64_t cutoff) {
  ToyIndex idx;
  // Enumerate wal-*.log files in ascending sequence order
  std::vector<std::pair<std::uint64_t, std::filesystem::path>> files;
  for (auto& de : std::filesystem::directory_iterator(dir)){
    if (!de.is_regular_file()) continue; auto name = de.path().filename().string();
    if (name.rfind("wal-", 0)==0 && name.size() >= 4+8+4) {
      // wal-XXXXXXXX.log
      try {
        auto seq = static_cast<std::uint64_t>(std::stoull(name.substr(4,8)));
        files.emplace_back(seq, de.path());
      } catch (...) {}
    }
  }
  std::sort(files.begin(), files.end(), [](auto& a, auto& b){ return a.first < b.first; });

  // Apply baseline frames (<=cutoff) by scanning each file
  for (auto& kv : files) {
    auto st0 = vesper::wal::recover_scan(kv.second.string(), [&](const vesper::wal::WalFrame& f){ if (f.lsn <= cutoff) apply_frame_payload(f.payload, idx); });
    (void)st0; // test harness requires has_value() at call sites if needed
  }

  // Apply >cutoff via replay
  (void)vesper::wal::recover_replay(dir, [&](std::uint64_t /*lsn*/, std::uint16_t /*type*/, std::span<const std::uint8_t> pl){ apply_frame_payload(pl, idx); });
  return idx;
}

} // namespace test_support

