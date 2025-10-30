#include "vesper/wal/io.hpp"
#include "vesper/wal/manifest.hpp"
#include "vesper/wal/snapshot.hpp"

#include <vector>
#include <cstring>
#include <regex>
#include <iomanip>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <optional>




#include <algorithm>

#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif


// Internal OS-level sync helpers. Errors are propagated via std::expected.
static auto fsync_file_path(const std::filesystem::path& p) -> std::expected<void, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
#if defined(__linux__) || defined(__APPLE__)
  int fd = ::open(p.string().c_str(), O_RDONLY);
  if (fd < 0) {
    return std::vesper_unexpected(error{error_code::io_failed, "fsync open failed", "wal.io"});
  }
  int rc = ::fsync(fd);
  (void)::close(fd);
  if (rc != 0) {
    return std::vesper_unexpected(error{error_code::io_failed, "fsync failed", "wal.io"});
  }
#elif defined(_WIN32)
  HANDLE h = ::CreateFileW(p.wstring().c_str(), GENERIC_READ,
                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    DWORD err = ::GetLastError();
    // Best-effort: sharing violations/access denied can occur while the file is open; treat as success.
    if (err == ERROR_SHARING_VIOLATION || err == ERROR_ACCESS_DENIED) { return {}; }
    return std::vesper_unexpected(error{error_code::io_failed, "fsync open failed", "wal.io"});
  }
  BOOL ok = ::FlushFileBuffers(h);
  DWORD fberr = ok ? 0 : ::GetLastError();
  ::CloseHandle(h);
  if (!ok) {
    // Best-effort: some FS/handles may not support FlushFileBuffers in this context; treat as success.
    if (fberr == ERROR_INVALID_HANDLE || fberr == ERROR_ACCESS_DENIED || fberr == ERROR_SHARING_VIOLATION) { return {}; }
    return std::vesper_unexpected(error{error_code::io_failed, "FlushFileBuffers failed", "wal.io"});
  }
#endif
  return {};
}

static auto fsync_dir_path(const std::filesystem::path& dir) -> std::expected<void, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
#if defined(__linux__) || defined(__APPLE__)
  int fd = ::open(dir.string().c_str(), O_RDONLY);
  if (fd < 0) {
    return {}; // best-effort for directory metadata
  }
  (void)::fsync(fd);
  (void)::close(fd);
#elif defined(_WIN32)
  HANDLE h = ::CreateFileW(dir.wstring().c_str(), GENERIC_READ,
                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
  if (h != INVALID_HANDLE_VALUE) {
    (void)::FlushFileBuffers(h);
    ::CloseHandle(h);
  }
#endif
  return {};
}



namespace {
static inline bool type_enabled(std::uint32_t mask, std::uint16_t t){ return (mask & (1u << t)) != 0; }
}

namespace vesper::wal {

WalWriter::~WalWriter(){ if (out_.is_open()) out_.close(); }
WalWriter::WalWriter(WalWriter&& o) noexcept
  : path_(std::move(o.path_)),
    dir_(std::move(o.dir_)),
    prefix_(std::move(o.prefix_)),
    max_file_bytes_(o.max_file_bytes_),
    strict_lsn_monotonic_(o.strict_lsn_monotonic_),
    fsync_on_rotation_(o.fsync_on_rotation_),
    fsync_on_flush_(o.fsync_on_flush_),
    seq_index_(o.seq_index_),
    cur_bytes_(o.cur_bytes_),
    cur_frames_(o.cur_frames_),
    cur_start_lsn_(o.cur_start_lsn_),
    cur_end_lsn_(o.cur_end_lsn_),
    prev_lsn_(o.prev_lsn_),
    have_prev_(o.have_prev_),
    stats_(o.stats_),
    out_(std::move(o.out_)) {}
WalWriter& WalWriter::operator=(WalWriter&& o) noexcept {
  if (this != &o) {
    if (out_.is_open()) out_.close();
    path_ = std::move(o.path_);
    dir_ = std::move(o.dir_);
    prefix_ = std::move(o.prefix_);
    max_file_bytes_ = o.max_file_bytes_;
    strict_lsn_monotonic_ = o.strict_lsn_monotonic_;
    fsync_on_rotation_ = o.fsync_on_rotation_;
    fsync_on_flush_ = o.fsync_on_flush_;
    seq_index_ = o.seq_index_;
    cur_bytes_ = o.cur_bytes_;
    cur_frames_ = o.cur_frames_;
    cur_start_lsn_ = o.cur_start_lsn_;
    cur_end_lsn_ = o.cur_end_lsn_;
    prev_lsn_ = o.prev_lsn_;
    have_prev_ = o.have_prev_;
    stats_ = o.stats_;
    out_ = std::move(o.out_);
  }
  return *this;
}

auto WalWriter::open(std::string_view p, bool create_if_missing)
    -> std::expected<WalWriter, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  WalWriter w;
  w.path_ = std::filesystem::path(p);
  std::ios::openmode mode = std::ios::binary | std::ios::out | std::ios::app;
  if (create_if_missing) {
    std::ofstream touch(w.path_, std::ios::binary | std::ios::app);
    if (!touch.good()) {
      return std::vesper_unexpected(error{error_code::io_failed, "open failed", "wal.io"});
    }
  }
  w.out_.open(w.path_, mode);
  if (!w.out_.good()) {
    return std::vesper_unexpected(error{error_code::io_failed, "open failed", "wal.io"});
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
  w.strict_lsn_monotonic_ = opts.strict_lsn_monotonic;
  // map durability profile to knobs if provided
  if (opts.durability_profile.has_value()) {
    switch (*opts.durability_profile) {
      case DurabilityProfile::None: w.fsync_on_rotation_ = false; w.fsync_on_flush_ = false; break;
      case DurabilityProfile::Rotation: w.fsync_on_rotation_ = true; w.fsync_on_flush_ = false; break;
      case DurabilityProfile::Flush: w.fsync_on_rotation_ = false; w.fsync_on_flush_ = true; break;
      case DurabilityProfile::RotationAndFlush: w.fsync_on_rotation_ = true; w.fsync_on_flush_ = true; break;
    }
  } else {
    w.fsync_on_rotation_ = opts.fsync_on_rotation;
    w.fsync_on_flush_ = opts.fsync_on_flush;
  }
  if (!std::filesystem::exists(w.dir_)) {
    std::error_code ec; std::filesystem::create_directories(w.dir_, ec);
    if (ec) return std::vesper_unexpected(error{error_code::io_failed, "mkdir failed", "wal.io"});
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
  w.prev_lsn_ = 0; w.have_prev_ = false;
  w.stats_ = {};
  return w;
}


namespace {
static void upsert_manifest(const std::filesystem::path& dir, const vesper::wal::ManifestEntry& e){
  using namespace vesper::wal;
  Manifest m{};
  if (auto mx = load_manifest(dir); mx) { m = *mx; }
  // replace any existing entry for same seq or filename
  m.entries.erase(std::remove_if(m.entries.begin(), m.entries.end(), [&](const ManifestEntry& x){
    return x.seq == e.seq || x.file == e.file;
  }), m.entries.end());
  m.entries.push_back(e);
  std::sort(m.entries.begin(), m.entries.end(), [](const ManifestEntry& a, const ManifestEntry& b){ return a.seq < b.seq; });
  (void)save_manifest(dir, m);
}
}

auto WalWriter::open_seq(std::uint64_t seq) -> std::expected<void, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  if (out_.is_open()) {
    out_.flush();
    // Close before attempting OS-level sync on Windows to avoid sharing violations
    out_.close();
    if (fsync_on_rotation_) {
      if (auto r = fsync_file_path(path_); !r) { return std::vesper_unexpected(r.error()); }
      stats_.syncs++;
    }
    stats_.rotations++;
  }
  seq_index_ = seq;
  std::ostringstream oss; oss << prefix_ << std::setw(8) << std::setfill('0') << seq << ".log";
  path_ = dir_ / oss.str();
  out_.open(path_, std::ios::binary | std::ios::out | std::ios::trunc);
  if (!out_.good()) return std::vesper_unexpected(error{error_code::io_failed, "open seq failed", "wal.io"});
  if (fsync_on_rotation_ && !dir_.empty()) {
    if (auto r = fsync_dir_path(dir_); !r) { return std::vesper_unexpected(r.error()); }
  }
  cur_bytes_ = cur_frames_ = 0; cur_start_lsn_ = cur_end_lsn_ = 0;
  return {};
}

auto WalWriter::maybe_rotate(std::size_t next_frame_bytes) -> std::expected<void, vesper::core::error> {
  if (max_file_bytes_ == 0) return {};
  if (!out_.is_open()) return open_seq(seq_index_ + 1);
  if (cur_bytes_ + next_frame_bytes > max_file_bytes_) {

    // Update manifest for the finished file
    ManifestEntry e{path_.filename().string(), seq_index_, cur_start_lsn_, cur_start_lsn_, cur_end_lsn_, cur_frames_, cur_bytes_};
    upsert_manifest(dir_, e);
    return open_seq(seq_index_ + 1);
  }
  return {};
}

auto WalWriter::append(std::uint64_t lsn, std::uint16_t type, std::span<const std::uint8_t> payload)
    -> std::expected<void, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  if (strict_lsn_monotonic_ && (type==1 || type==2)) {
    if (have_prev_ && lsn <= prev_lsn_) {
      return std::vesper_unexpected(error{error_code::precondition_failed, "non-monotonic LSN", "wal.io"});
    }
    prev_lsn_ = lsn; have_prev_ = true;
  }
  auto enc = encode_frame_expected(lsn, type, payload);
  if (!enc) return std::vesper_unexpected(enc.error());
  auto bytes = std::move(*enc);
  if (!dir_.empty()) {
    auto rot = maybe_rotate(bytes.size()); if (!rot) return std::vesper_unexpected(rot.error());
    if (!out_.is_open()) { auto r = open_seq(seq_index_ + 1); if (!r) return std::vesper_unexpected(r.error()); }
  } else if (!out_.good()) {
    return std::vesper_unexpected(error{error_code::io_failed, "writer closed", "wal.io"});
  }
  out_.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!out_.good()) return std::vesper_unexpected(error{error_code::io_failed, "write failed", "wal.io"});
  cur_bytes_ += bytes.size();
  cur_frames_ += 1; stats_.frames++;
  if (cur_start_lsn_ == 0) cur_start_lsn_ = lsn;
  cur_end_lsn_ = lsn;
  return {};
}

auto WalWriter::publish_snapshot(std::uint64_t last_lsn) -> std::expected<void, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  if (dir_.empty()) {
    return std::vesper_unexpected(error{error_code::precondition_failed, "not in rotation mode", "wal.io"});
  }
  return save_snapshot(dir_, Snapshot{last_lsn});
}


auto WalWriter::flush(bool sync) -> std::expected<void, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  if (!out_.good()) return std::vesper_unexpected(error{error_code::io_failed, "writer closed", "wal.io"});
  out_.flush(); stats_.flushes++;
  if (sync || fsync_on_flush_) {
  #if defined(_WIN32)
    // Attempt to sync via a separate handle; if sharing prevents it, fall back to close-sync-reopen.
    auto try_sync = [&]() -> std::expected<void, vesper::core::error> {
      if (auto r = fsync_file_path(path_); r) return r;
      // Fallback: close, sync, reopen in append mode
      out_.close();
      if (auto r2 = fsync_file_path(path_); !r2) return std::vesper_unexpected(r2.error());
      out_.open(path_, std::ios::binary | std::ios::out | std::ios::app);
      if (!out_.good()) return std::vesper_unexpected(error{error_code::io_failed, "reopen after sync failed", "wal.io"});
      return {};
    };
    if (auto s = try_sync(); !s) return std::vesper_unexpected(s.error());
  #else
    if (auto r = fsync_file_path(path_); !r) { return std::vesper_unexpected(r.error()); }
  #endif
    stats_.syncs++;
  }
  // If rotating mode and we have an open file, upsert a manifest entry snapshot
  if (!dir_.empty() && out_.is_open() && cur_frames_ > 0) {
    ManifestEntry e{path_.filename().string(), seq_index_, cur_start_lsn_, cur_start_lsn_, cur_end_lsn_, cur_frames_, cur_bytes_};
    upsert_manifest(dir_, e);
  }
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
  if (!in.good()) return std::vesper_unexpected(error{error_code::not_found, "open failed", "wal.io"});

  std::uint64_t prev_lsn = 0;
  bool have_prev = false;

  while (true) {
    // Peek header
    std::vector<std::uint8_t> hdr;
    if (!read_exact(in, hdr, WAL_HEADER_SIZE)) {
      break; // EOF or partial header -> stop without error (torn tail)
    }
    // Validate header magic and LEN before allocating
    std::uint32_t magic; std::memcpy(&magic, hdr.data(), 4);
    if (magic != WAL_MAGIC) { break; }
    std::uint32_t len; std::memcpy(&len, hdr.data() + 4, 4);
    // Total frame length must include header + CRC (≥ WAL_HEADER_SIZE + 4)
    if (len < WAL_HEADER_SIZE + 4) { break; }
    // Enforce an upper bound to prevent OOM/DoS
    static constexpr std::uint32_t MAX_FRAME_LEN = 32u * 1024u * 1024u; // 32 MiB
    if (len > MAX_FRAME_LEN) { break; }
    // Ensure remaining file contains the declared bytes
    std::streampos pos = in.tellg();
    if (pos == std::streampos(-1)) { break; }
    {
      std::error_code fec{};
      const auto file_sz = std::filesystem::file_size(std::filesystem::path(p), fec);
      if (!fec) {
        const std::uint64_t remaining = (file_sz >= static_cast<std::uint64_t>(pos)) ? (file_sz - static_cast<std::uint64_t>(pos)) : 0ull;
        const std::uint64_t rest_len = static_cast<std::uint64_t>(len) - WAL_HEADER_SIZE;
        if (rest_len > remaining) { break; }
      }
    }

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

// Accepting-callback variant with early-stop/error propagation and per-file stats (delivered-only)
auto recover_scan(std::string_view p, std::function<std::expected<DeliverDecision, vesper::core::error>(const WalFrame&)> on_frame)
    -> std::expected<RecoveryStats, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  RecoveryStats stats{};
  std::ifstream in(std::filesystem::path(p), std::ios::binary);
  if (!in.good()) return std::vesper_unexpected(error{error_code::not_found, "open failed", "wal.io"});

  std::uint64_t prev_lsn = 0;
  bool have_prev = false;

  while (true) {
    std::vector<std::uint8_t> hdr;
    if (!read_exact(in, hdr, WAL_HEADER_SIZE)) { break; }
    std::uint32_t magic; std::memcpy(&magic, hdr.data(), 4);
    if (magic != WAL_MAGIC) { break; }
    std::uint32_t len; std::memcpy(&len, hdr.data() + 4, 4);
    if (len < WAL_HEADER_SIZE + 4) { break; }
    static constexpr std::uint32_t MAX_FRAME_LEN = 32u * 1024u * 1024u;
    if (len > MAX_FRAME_LEN) { break; }
    std::streampos pos = in.tellg(); if (pos == std::streampos(-1)) { break; }
    {
      std::error_code fec{};
      const auto file_sz = std::filesystem::file_size(std::filesystem::path(p), fec);
      if (!fec) {
        const std::uint64_t remaining = (file_sz >= static_cast<std::uint64_t>(pos)) ? (file_sz - static_cast<std::uint64_t>(pos)) : 0ull;
        const std::uint64_t rest_len = static_cast<std::uint64_t>(len) - WAL_HEADER_SIZE;
        if (rest_len > remaining) { break; }
      }
    }
    std::vector<std::uint8_t> frame = hdr;
    std::vector<std::uint8_t> rest;
    if (!read_exact(in, rest, len - WAL_HEADER_SIZE)) { break; }
    frame.insert(frame.end(), rest.begin(), rest.end());
    if (!verify_crc32c(frame)) { break; }
    auto dec = decode_frame(frame); if (!dec) { break; }

    // Invoke callback; decide delivery/stop semantics
    auto r = on_frame(*dec);
    if (!r) return std::vesper_unexpected(r.error());

    switch (*r) {
      case DeliverDecision::Skip:
        // not delivered; continue without counting
        continue;
      case DeliverDecision::SkipAndStop:
        // not delivered; stop without counting
        return stats;
      case DeliverDecision::DeliverAndContinue:
      case DeliverDecision::DeliverAndStop: {
        // Update stats for delivered frame
        const auto t = dec->type;
        if (t < stats.type_counts.size()) stats.type_counts[t]++;
        stats.frames += 1;
        stats.bytes += frame.size();
        stats.last_lsn = dec->lsn;
        if (stats.min_len == 0 || dec->len < stats.min_len) stats.min_len = dec->len;
        if (dec->len > stats.max_len) stats.max_len = dec->len;
        if (t == 1 || t == 2) {
          if (have_prev && dec->lsn <= prev_lsn) {
            stats.lsn_monotonic = false; stats.lsn_violations += 1;
          }
          prev_lsn = dec->lsn; have_prev = true;
        }
        if (*r == DeliverDecision::DeliverAndStop) { return stats; }
        break;
      }
    }
  }
  return stats;
}

auto recover_scan_dir(const std::filesystem::path& dir, std::uint32_t type_mask, std::function<void(const WalFrame&)> on_frame)
    -> std::expected<RecoveryStats, vesper::core::error> {
  return recover_scan_dir(
    dir,
    type_mask,
    std::function<std::expected<DeliverDecision, vesper::core::error>(const WalFrame&)>(
      [&](const WalFrame& f) -> std::expected<DeliverDecision, vesper::core::error> {
        on_frame(f);
        return std::expected<DeliverDecision, vesper::core::error>{DeliverDecision::DeliverAndContinue};
      }));
}


auto recover_scan_dir(const std::filesystem::path& dir, std::function<void(const WalFrame&)> on_frame)
    -> std::expected<RecoveryStats, vesper::core::error> {
  // Route through accepting-callback variant so stats reflect only accepted frames
  return recover_scan_dir(
    dir,
    std::function<std::expected<DeliverDecision, vesper::core::error>(const WalFrame&)>(
      [&](const WalFrame& f) -> std::expected<DeliverDecision, vesper::core::error> {
        on_frame(f);
        return std::expected<DeliverDecision, vesper::core::error>{DeliverDecision::DeliverAndContinue};
      }));
}



auto recover_scan_dir(const std::filesystem::path& dir, const DeliveryLimits& limits, std::function<void(const WalFrame&)> on_frame)
    -> std::expected<RecoveryStats, vesper::core::error> {
  return recover_scan_dir(
    dir,
    limits,
    std::function<std::expected<DeliverDecision, vesper::core::error>(const WalFrame&)>(
      [&](const WalFrame& f) -> std::expected<DeliverDecision, vesper::core::error> {
        on_frame(f);
        return std::expected<DeliverDecision, vesper::core::error>{DeliverDecision::DeliverAndContinue};
      }));
}
// Accepting-callback directory scan: counts only accepted frames; supports early stop/error
auto recover_scan_dir(const std::filesystem::path& dir,
    std::function<std::expected<DeliverDecision, vesper::core::error>(const WalFrame&)> on_frame)
    -> std::expected<RecoveryStats, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  RecoveryStats agg{};
  // Try manifest first
  Manifest m{}; bool have_manifest = false;
  if (auto mx = load_manifest(dir); mx) { m = *mx; have_manifest = true; }
  // Snapshot cutoff (if present)
  std::uint64_t cutoff_lsn = 0;
  if (std::filesystem::exists(dir / "wal.snapshot")) {
    if (auto sx = load_snapshot(dir); sx) cutoff_lsn = sx->last_lsn;
  }
  // Build file list
  std::vector<std::filesystem::path> files;
  const bool have_manifest_entries = have_manifest && !m.entries.empty();
  if (have_manifest_entries) {
    std::regex rx("^wal-([0-9]{8})\\.log$");
    std::unordered_map<std::uint64_t, std::filesystem::path> by_seq;
    for (auto& de : std::filesystem::directory_iterator(dir)) {
      if (!de.is_regular_file()) continue;
      auto name = de.path().filename().string();
      std::smatch mm; if (std::regex_match(name, mm, rx)) {
        by_seq.emplace(static_cast<std::uint64_t>(std::stoull(mm[1].str())), de.path());
      }
    }
    for (const auto& e : m.entries) {
      std::smatch mm; if (std::regex_match(e.file, mm, rx)) {
        by_seq.emplace(static_cast<std::uint64_t>(std::stoull(mm[1].str())), dir / e.file);
      }
    }
    std::vector<std::pair<std::uint64_t, std::filesystem::path>> tmp; tmp.reserve(by_seq.size());
    for (auto& kv : by_seq) tmp.emplace_back(kv.first, kv.second);
    std::sort(tmp.begin(), tmp.end(), [](auto& a, auto& b){ return a.first < b.first || (a.first == b.first && a.second < b.second); });
    for (auto& kv : tmp) files.push_back(kv.second);
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
    std::sort(tmp.begin(), tmp.end(), [](auto& a, auto& b){ return a.first < b.first || (a.first == b.first && a.second < b.second); });
    for (auto& kv : tmp) files.push_back(kv.second);
  }
  // Cross-file monotonicity for TYPE {1,2}
  std::uint64_t prev_lsn_global12 = 0; bool have_prev_global12 = false;
  for (size_t i = 0; i < files.size(); ++i) {
    bool skip_deliver = false;
    if (have_manifest_entries) {
      const auto fname = files[i].filename().string();
      auto it = std::find_if(m.entries.begin(), m.entries.end(), [&](const ManifestEntry& me){ return me.file == fname; });
      if (it != m.entries.end() && it->end_lsn <= cutoff_lsn) skip_deliver = true;
    }
    // First pass: torn detection (noop)
    auto noop = [](const WalFrame&){};
    auto stats = recover_scan(files[i].string(), noop);
    if (!stats) return std::vesper_unexpected(stats.error());
    if (i + 1 < files.size()) {
      std::error_code ec; auto size = std::filesystem::file_size(files[i], ec);
      if (!ec && stats->bytes != size) {
        return std::vesper_unexpected(error{error_code::data_integrity, "torn middle file", "wal.io"});
      }
    }
    if (skip_deliver) continue;

    bool early_stop = false;
    std::optional<std::uint64_t> first_acc12{};
    std::optional<std::uint64_t> last_acc12{};
    auto per_file_cb = [&](const WalFrame& f) -> std::expected<DeliverDecision, error> {
      if (f.lsn <= cutoff_lsn) return std::expected<DeliverDecision, error>{DeliverDecision::Skip};
      auto r = on_frame(f);
      if (!r) return std::vesper_unexpected(r.error());
      if (*r == DeliverDecision::DeliverAndStop) {
        early_stop = true;
        if (f.type == 1 || f.type == 2) {
          if (!first_acc12.has_value()) first_acc12 = f.lsn;
          last_acc12 = f.lsn;
        }
        return std::expected<DeliverDecision, error>{DeliverDecision::DeliverAndStop};
      }
      if (*r == DeliverDecision::DeliverAndContinue) {
        if (f.type == 1 || f.type == 2) {
          if (!first_acc12.has_value()) first_acc12 = f.lsn;
          last_acc12 = f.lsn;
        }
        return std::expected<DeliverDecision, error>{DeliverDecision::DeliverAndContinue};
      }
      // Propagate Skip/SkipAndStop without updating accepted-12 bounds
      if (*r == DeliverDecision::SkipAndStop) { early_stop = true; }
      return r;
    };
    auto s = recover_scan(
      files[i].string(),
      std::function<std::expected<DeliverDecision, error>(const WalFrame&)>(per_file_cb));
    if (!s) return std::vesper_unexpected(s.error());

    // Aggregate accepted stats
    const auto& fs = *s;
    agg.frames += fs.frames;
    agg.bytes += fs.bytes;
    if (fs.last_lsn != 0) agg.last_lsn = fs.last_lsn;
    agg.min_len = (agg.min_len == 0) ? fs.min_len : std::min(agg.min_len, fs.min_len);
    agg.max_len = std::max(agg.max_len, fs.max_len);
    for (size_t t = 0; t < agg.type_counts.size(); ++t) agg.type_counts[t] += fs.type_counts[t];
    if (!fs.lsn_monotonic) agg.lsn_monotonic = false;
    agg.lsn_violations += fs.lsn_violations;

    // Cross-file boundary monotonicity
    if (first_acc12.has_value() && have_prev_global12) {
      if (*first_acc12 <= prev_lsn_global12) { agg.lsn_monotonic = false; agg.lsn_violations += 1; }
    }
    if (last_acc12.has_value()) { prev_lsn_global12 = *last_acc12; have_prev_global12 = true; }

    if (early_stop) return agg;
  }
  return agg;
}

// Accepting-callback + type mask wrapper
auto recover_scan_dir(const std::filesystem::path& dir, std::uint32_t type_mask,
    std::function<std::expected<DeliverDecision, vesper::core::error>(const WalFrame&)> on_frame)
    -> std::expected<RecoveryStats, vesper::core::error> {
  return recover_scan_dir(
    dir,
    std::function<std::expected<DeliverDecision, vesper::core::error>(const WalFrame&)>(
      [&](const WalFrame& f) -> std::expected<DeliverDecision, vesper::core::error> {
        if (!type_enabled(type_mask, f.type)) return std::expected<DeliverDecision, vesper::core::error>{DeliverDecision::Skip};
        return on_frame(f);
      }));
}

// Accepting-callback + DeliveryLimits wrapper
auto recover_scan_dir(const std::filesystem::path& dir, const DeliveryLimits& limits,
    std::function<std::expected<DeliverDecision, vesper::core::error>(const WalFrame&)> on_frame)
    -> std::expected<RecoveryStats, vesper::core::error> {
  std::size_t delivered_f = 0; std::size_t delivered_b = 0;
  const std::uint64_t cutoff = limits.cutoff_lsn;
  const std::uint32_t mask = limits.type_mask;
  const std::size_t maxf = limits.max_frames;
  const std::size_t maxb = limits.max_bytes;
  return recover_scan_dir(
    dir,
    std::function<std::expected<DeliverDecision, vesper::core::error>(const WalFrame&)>(
      [&](const WalFrame& f) -> std::expected<DeliverDecision, vesper::core::error> {
        if (cutoff > 0 && f.lsn <= cutoff) return std::expected<DeliverDecision, vesper::core::error>{DeliverDecision::Skip};
        if (!type_enabled(mask, f.type)) return std::expected<DeliverDecision, vesper::core::error>{DeliverDecision::Skip};
        if (maxf > 0 && delivered_f >= maxf) return std::expected<DeliverDecision, vesper::core::error>{DeliverDecision::SkipAndStop};
        if (maxb > 0 && (delivered_b + f.payload.size()) > maxb) return std::expected<DeliverDecision, vesper::core::error>{DeliverDecision::SkipAndStop};
        auto r = on_frame(f);
        if (!r) return std::vesper_unexpected(r.error());
        if (*r == DeliverDecision::DeliverAndStop) { delivered_f++; delivered_b += f.payload.size(); return std::expected<DeliverDecision, vesper::core::error>{DeliverDecision::DeliverAndStop}; }
        if (*r == DeliverDecision::DeliverAndContinue) { delivered_f++; delivered_b += f.payload.size(); return std::expected<DeliverDecision, vesper::core::error>{DeliverDecision::DeliverAndContinue}; }
        // If inner decided to Skip/SkipAndStop, propagate as-is (no counters increment)
        return r;
      }));
}


} // namespace vesper::wal

