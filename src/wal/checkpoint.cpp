#include "vesper/wal/checkpoint.hpp"

#include <fstream>

namespace vesper::wal::checkpoint {

static inline std::filesystem::path path_for(const std::filesystem::path& dir, std::string_view consumer){
  return dir / "wal.checkpoints" / (std::string(consumer) + ".ckpt");
}

auto load(const std::filesystem::path& dir, std::string_view consumer)
    -> std::expected<Checkpoint, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  Checkpoint ck{std::string(consumer), 0};
  auto p = path_for(dir, consumer);
  std::error_code ec;
  if (!std::filesystem::exists(p, ec)) return ck; // missing is ok
  std::ifstream in(p);
  if (!in.good()) return std::unexpected(error{error_code::io_failed, "open checkpoint failed", "wal.checkpoint"});
  std::string line; std::getline(in, line);
  auto pos = line.find("last_lsn=");
  if (pos != 0) return std::unexpected(error{error_code::data_integrity, "malformed checkpoint", "wal.checkpoint"});
  auto v = line.substr(std::string("last_lsn=").size());
  try { ck.last_lsn = std::stoull(v); } catch(...) { return std::unexpected(error{error_code::data_integrity, "malformed checkpoint value", "wal.checkpoint"}); }
  return ck;
}

auto save(const std::filesystem::path& dir, std::string_view consumer, std::uint64_t last_lsn)
    -> std::expected<void, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  auto folder = dir / "wal.checkpoints";
  std::error_code ec; std::filesystem::create_directories(folder, ec);
  if (ec) return std::unexpected(error{error_code::io_failed, "mkdir ckpt failed", "wal.checkpoint"});
  auto p = path_for(dir, consumer);
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  if (!out.good()) return std::unexpected(error{error_code::io_failed, "write checkpoint failed", "wal.checkpoint"});
  out << "last_lsn=" << last_lsn << "\n";
  return {};
}

auto replay_from_checkpoint(const std::filesystem::path& dir,
                            std::string_view consumer,
                            std::uint32_t type_mask,
                            wal::ReplayCallback cb)
    -> std::expected<wal::RecoveryStats, vesper::core::error> {
  auto ckx = load(dir, consumer); if (!ckx) return std::unexpected(ckx.error());
  auto ck = *ckx;
  // Replay with delivery cutoff override and mask
  DeliveryLimits lim{}; lim.cutoff_lsn = ck.last_lsn; lim.type_mask = type_mask;
  std::uint64_t last = ck.last_lsn;
  auto st = recover_scan_dir(dir, lim, [&](const wal::WalFrame& f){ last = f.lsn; cb(f.lsn, f.type, f.payload); });
  if (!st) return st;
  if (last != ck.last_lsn) {
    if (auto sx = save(dir, consumer, last); !sx) return std::unexpected(sx.error());
  }
  wal::RecoveryStats rs = *st;
  rs.last_lsn = last; // reflect last delivered
  return rs;
}

} // namespace vesper::wal::checkpoint

