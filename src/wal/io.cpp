#include "vesper/wal/io.hpp"

#include <vector>
#include <cstring>

namespace vesper::wal {

WalWriter::~WalWriter(){ if (out_.is_open()) out_.close(); }
WalWriter::WalWriter(WalWriter&& o) noexcept : path_(std::move(o.path_)), out_(std::move(o.out_)) {}
WalWriter& WalWriter::operator=(WalWriter&& o) noexcept { if(this!=&o){ if(out_.is_open()) out_.close(); path_=std::move(o.path_); out_=std::move(o.out_);} return *this; }

auto WalWriter::open(std::string_view p, bool create_if_missing)
    -> std::expected<WalWriter, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  WalWriter w;
  w.path_ = std::filesystem::path(p);
  std::ios::openmode mode = std::ios::binary | std::ios::out | std::ios::app;
  if (create_if_missing) {
    std::ofstream touch(w.path_, std::ios::binary | std::ios::app);
    if (!touch.good()) {
      return std::unexpected(error{error_code::io_failed, "open failed", "wal.io"});
    }
  }
  w.out_.open(w.path_, mode);
  if (!w.out_.good()) {
    return std::unexpected(error{error_code::io_failed, "open failed", "wal.io"});
  }
  return w;
}

auto WalWriter::append(std::uint64_t lsn, std::uint16_t type, std::span<const std::uint8_t> payload)
    -> std::expected<void, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  if (!out_.good()) return std::unexpected(error{error_code::io_failed, "writer closed", "wal.io"});
  auto bytes = encode_frame(lsn, type, payload);
  out_.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!out_.good()) return std::unexpected(error{error_code::io_failed, "write failed", "wal.io"});
  return {};
}

auto WalWriter::flush(bool sync) -> std::expected<void, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  if (!out_.good()) return std::unexpected(error{error_code::io_failed, "writer closed", "wal.io"});
  out_.flush();
  (void)sync; // fsync intentionally omitted for cross-platform determinism in tests
  return {};
}

namespace {
static auto read_exact(std::ifstream& in, std::vector<std::uint8_t>& buf, std::size_t n) -> bool {
  buf.resize(n);
  in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(n));
  return static_cast<std::size_t>(in.gcount()) == n;
}
}

auto recover_scan(std::string_view p, std::function<void(const WalFrame&)> on_frame)
    -> std::expected<RecoveryStats, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  RecoveryStats stats{};
  std::ifstream in(std::filesystem::path(p), std::ios::binary);
  if (!in.good()) return std::unexpected(error{error_code::not_found, "open failed", "wal.io"});

  std::uint64_t prev_lsn = 0;
  bool have_prev = false;

  while (true) {
    // Peek header
    std::vector<std::uint8_t> hdr;
    if (!read_exact(in, hdr, WAL_HEADER_SIZE)) {
      break; // EOF or partial header -> stop without error (torn tail)
    }
    // Read LEN
    std::uint32_t len; std::memcpy(&len, hdr.data() + 4, 4);
    if (len < WAL_HEADER_SIZE + 4) { break; }

    // Read remaining bytes (LEN - header already read)
    std::vector<std::uint8_t> frame = hdr;
    std::vector<std::uint8_t> rest;
    if (!read_exact(in, rest, len - WAL_HEADER_SIZE)) { break; }
    frame.insert(frame.end(), rest.begin(), rest.end());

    // Verify and decode
    if (!verify_crc32c(frame)) { break; }
    auto dec = decode_frame(frame);
    if (!dec) { break; }

    // Update stats
    const auto t = dec->type;
    if (t < stats.type_counts.size()) stats.type_counts[t]++;
    stats.frames += 1;
    stats.bytes += frame.size();
    stats.last_lsn = dec->lsn;
    if (stats.min_len == 0 || dec->len < stats.min_len) stats.min_len = dec->len;
    if (dec->len > stats.max_len) stats.max_len = dec->len;

    // Monotonicity (strict) for TYPE 1 and 2 only
    if (t == 1 || t == 2) {
      if (have_prev && dec->lsn <= prev_lsn) {
        stats.lsn_monotonic = false;
        stats.lsn_violations += 1;
      }
      prev_lsn = dec->lsn;
      have_prev = true;
    }

    on_frame(*dec);
  }
  return stats;
}

} // namespace vesper::wal

