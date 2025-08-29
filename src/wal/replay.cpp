#include "vesper/wal/replay.hpp"
#include "vesper/wal/io.hpp"

namespace vesper::wal {

auto recover_replay(const std::filesystem::path& dir, ReplayCallback on_payload)
    -> std::expected<RecoveryStats, vesper::core::error>
{
  return recover_scan_dir(dir, [&](const WalFrame& f){ on_payload(f.lsn, f.type, f.payload); });
}

} // namespace vesper::wal

