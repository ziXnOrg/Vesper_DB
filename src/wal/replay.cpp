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
  if (type_mask == 0) return std::vesper_unexpected(vesper::core::error{vesper::core::error_code::invalid_argument, "replay: type_mask==0", "storage.wal.replay"});
  return recover_scan_dir(
    dir,
    type_mask,
    std::function<std::expected<DeliverDecision, vesper::core::error>(const WalFrame&)>(
      [&](const WalFrame& f) -> std::expected<DeliverDecision, vesper::core::error> {
        on_payload(f.lsn, f.type, f.payload);
        return std::expected<DeliverDecision, vesper::core::error>{DeliverDecision::DeliverAndContinue};
      }));
}


auto recover_replay(const std::filesystem::path& dir, ReplayResultCallback on_payload)
    -> std::expected<RecoveryStats, vesper::core::error>
{
  return recover_scan_dir(
    dir,
    std::function<std::expected<DeliverDecision, vesper::core::error>(const WalFrame&)>(
      [&](const WalFrame& f) -> std::expected<DeliverDecision, vesper::core::error> {
        return on_payload(f.lsn, f.type, f.payload);
      }));
}

auto recover_replay(const std::filesystem::path& dir, std::uint32_t type_mask, ReplayResultCallback on_payload)
    -> std::expected<RecoveryStats, vesper::core::error>
{
  if (type_mask == 0) return std::vesper_unexpected(vesper::core::error{vesper::core::error_code::invalid_argument, "replay: type_mask==0", "storage.wal.replay"});
  return recover_scan_dir(
    dir,
    type_mask,
    std::function<std::expected<DeliverDecision, vesper::core::error>(const WalFrame&)>(
      [&](const WalFrame& f) -> std::expected<DeliverDecision, vesper::core::error> {
        return on_payload(f.lsn, f.type, f.payload);
      }));
}

} // namespace vesper::wal

