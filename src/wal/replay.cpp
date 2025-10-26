#include "vesper/wal/replay.hpp"
#include "vesper/wal/io.hpp"

namespace vesper::wal {

auto recover_replay(const std::filesystem::path& dir, ReplayCallback on_payload)
    -> std::expected<RecoveryStats, vesper::core::error>
{
  return recover_scan_dir(dir, [&](const WalFrame& f){ on_payload(f.lsn, f.type, f.payload); });
}


auto recover_replay(const std::filesystem::path& dir, std::uint32_t type_mask, ReplayCallback on_payload)
    -> std::expected<RecoveryStats, vesper::core::error>
{
  if (type_mask == 0) return std::unexpected(std::invalid_argument("type_mask==0")); // guard against empty callback
  return recover_scan_dir(dir, [&](const WalFrame& f){ if ((type_mask & (1u << f.type)) != 0) on_payload(f.lsn, f.type, f.payload); });
}

} // namespace vesper::wal

