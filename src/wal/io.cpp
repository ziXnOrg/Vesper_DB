#include "vesper/wal/io.hpp"
#include "vesper/wal/manifest.hpp"

#include <vector>
#include <cstring>
#include <regex>
#include <iomanip>
#include <sstream>

namespace vesper::wal {

WalWriter::~WalWriter(){ if (out_.is_open()) out_.close(); }
WalWriter::WalWriter(WalWriter&& o) noexcept : path_(std::move(o.path_)), dir_(std::move(o.dir_)), prefix_(std::move(o.prefix_)), max_file_bytes_(o.max_file_bytes_), seq_index_(o.seq_index_), cur_bytes_(o.cur_bytes_), cur_frames_(o.cur_frames_), cur_start_lsn_(o.cur_start_lsn_), cur_end_lsn_(o.cur_end_lsn_), out_(std::move(o.out_)) {}
WalWriter& WalWriter::operator=(WalWriter&& o) noexcept { if(this!=&o){ if(out_.is_open()) out_.close(); path_=std::move(o.path_); dir_=std::move(o.dir_); prefix_=std::move(o.prefix_); max_file_bytes_=o.max_file_bytes_; seq_index_=o.seq_index_; cur_bytes_=o.cur_bytes_; cur_frames_=o.cur_frames_; cur_start_lsn_=o.cur_start_lsn_; cur_end_lsn_=o.cur_end_lsn_; out_=std::move(o.out_);} return *this; }

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

auto WalWriter::open(const WalWriterOptions& opts)
    -> std::expected<WalWriter, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  WalWriter w;
  w.dir_ = opts.dir;
  w.prefix_ = opts.prefix;
  w.max_file_bytes_ = opts.max_file_bytes;
  if (!std::filesystem::exists(w.dir_)) {
    std::error_code ec; std::filesystem::create_directories(w.dir_, ec);
    if (ec) return std::unexpected(error{error_code::io_failed, "mkdir failed", "wal.io"});
  }
  // Determine next seq index by listing existing files
  std::uint64_t max_seq = 0;
  std::regex rx((w.prefix_ + std::string("([0-9]{8})\\.log")).c_str());
  for (auto& de : std::filesystem::directory_iterator(w.dir_)) {
    if (!de.is_regular_file()) continue;
    auto name = de.path().filename().string();
    std::smatch m; if (std::regex_match(name, m, rx) && m.size() == 2) {
      auto seq = static_cast<std::uint64_t>(std::stoull(m[1].str()));
      if (seq > max_seq) max_seq = seq;
    }
  }
  w.seq_index_ = max_seq; // start from existing max; open_seq(++seq) before first append
  w.cur_bytes_ = 0; w.cur_frames_ = 0; w.cur_start_lsn_ = 0; w.cur_end_lsn_ = 0;
  return w;
}

auto WalWriter::open_seq(std::uint64_t seq) -> std::expected<void, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  if (out_.is_open()) out_.close();
  seq_index_ = seq;
  std::ostringstream oss; oss << prefix_ << std::setw(8) << std::setfill('0') << seq << ".log";
  path_ = dir_ / oss.str();
  out_.open(path_, std::ios::binary | std::ios::out | std::ios::trunc);
  if (!out_.good()) return std::unexpected(error{error_code::io_failed, "open seq failed", "wal.io"});
  cur_bytes_ = cur_frames_ = 0; cur_start_lsn_ = cur_end_lsn_ = 0;
  return {};
}

auto WalWriter::maybe_rotate(std::size_t next_frame_bytes) -> std::expected<void, vesper::core::error> {
  if (max_file_bytes_ == 0) return {};
  if (!out_.is_open()) return open_seq(seq_index_ + 1);
  if (cur_bytes_ + next_frame_bytes > max_file_bytes_) {
    // Update manifest for the finished file
    Manifest mf; mf.entries.push_back({path_.filename().string(), seq_index_, cur_start_lsn_, cur_end_lsn_, cur_frames_, cur_bytes_});
    (void)save_manifest(dir_, mf);
    return open_seq(seq_index_ + 1);
  }
  return {};
}

auto WalWriter::append(std::uint64_t lsn, std::uint16_t type, std::span<const std::uint8_t> payload)
    -> std::expected<void, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  auto bytes = encode_frame(lsn, type, payload);
  if (!dir_.empty()) {
    auto rot = maybe_rotate(bytes.size()); if (!rot) return std::unexpected(rot.error());
    if (!out_.is_open()) { auto r = open_seq(seq_index_ + 1); if (!r) return std::unexpected(r.error()); }
  } else if (!out_.good()) {
    return std::unexpected(error{error_code::io_failed, "writer closed", "wal.io"});
  }
  out_.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!out_.good()) return std::unexpected(error{error_code::io_failed, "write failed", "wal.io"});
  cur_bytes_ += bytes.size();
  cur_frames_ += 1;
  if (cur_start_lsn_ == 0) cur_start_lsn_ = lsn;
  cur_end_lsn_ = lsn;
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

auto recover_scan_dir(const std::filesystem::path& dir, std::function<void(const WalFrame&)> on_frame)
    -> std::expected<RecoveryStats, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  RecoveryStats agg{};
  // Try manifest first
  Manifest m{};
  bool have_manifest = false;
  {
    auto mx = load_manifest(dir);
    if (mx) { m = *mx; have_manifest = true; }
  }
  std::vector<std::filesystem::path> files;
  if (have_manifest) {
    for (auto& e : m.entries) files.push_back(dir / e.file);
  } else {
    std::regex rx("^wal-([0-9]{8})\\.log$");
    std::vector<std::pair<std::uint64_t, std::filesystem::path>> tmp;
    for (auto& de : std::filesystem::directory_iterator(dir)) {
      if (!de.is_regular_file()) continue;
      auto name = de.path().filename().string();
      std::smatch mm; if (std::regex_match(name, mm, rx)) {
        tmp.emplace_back(static_cast<std::uint64_t>(std::stoull(mm[1].str())), de.path());
      }
    }
    std::sort(tmp.begin(), tmp.end(), [](auto& a, auto& b){ return a.first < b.first; });
    for (auto& kv : tmp) files.push_back(kv.second);
  }
  for (size_t i = 0; i < files.size(); ++i) {
    auto stats = recover_scan(files[i].string(), on_frame);
    if (!stats) {
      return std::unexpected(stats.error());
    }
    // If a non-last file ended early due to torn tail, that's a data_integrity error per spec.
    // Our recover_scan stops on torn tail without error; we detect by comparing file size vs bytes consumed.
    if (i + 1 < files.size()) {
      std::error_code ec; auto size = std::filesystem::file_size(files[i], ec);
      if (!ec && stats->bytes != size) {
        return std::unexpected(error{error_code::data_integrity, "torn middle file", "wal.io"});
      }
    }
    // Aggregate
    agg.frames += stats->frames;
    agg.bytes += stats->bytes;
    agg.last_lsn = stats->last_lsn;
    agg.lsn_monotonic = agg.lsn_monotonic && stats->lsn_monotonic;
    agg.lsn_violations += stats->lsn_violations;
    agg.min_len = (agg.min_len == 0) ? stats->min_len : std::min(agg.min_len, stats->min_len);
    agg.max_len = std::max(agg.max_len, stats->max_len);
    for (size_t t = 0; t < agg.type_counts.size(); ++t) agg.type_counts[t] += stats->type_counts[t];
  }
  return agg;
}


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

