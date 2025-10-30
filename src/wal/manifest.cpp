#include "vesper/wal/manifest.hpp"
#include "vesper/wal/io.hpp" // recover_scan, WalFrame


#include <fstream>
#include <sstream>
#include <charconv>
#include <cctype>

#include <set>
#include <unordered_map>
#include <regex>

#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#endif
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#ifndef VESPER_ENABLE_ATOMIC_RENAME
#if defined(_WIN32) || defined(__linux__) || defined(__APPLE__)
#define VESPER_ENABLE_ATOMIC_RENAME 1
#else
#define VESPER_ENABLE_ATOMIC_RENAME 0
#endif
#endif


namespace vesper::wal {

auto load_manifest(const std::filesystem::path& dir)
    -> std::expected<Manifest, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  Manifest m{};
  auto p = dir / "wal.manifest";
  std::ifstream in(p);
  if (!in.good()) {
    return std::vesper_unexpected(error{error_code::not_found, "manifest open failed", "wal.manifest"});
  }
  std::string header; std::getline(in, header);
  if (header != std::string("vesper-wal-manifest v1")) {
    return std::vesper_unexpected(error{error_code::data_integrity, "bad manifest header", "wal.manifest"});
  }

  auto has_ctrl = [](const std::string& s){
    for (unsigned char c : s) { if (c < 0x20 || c == 0x7F) return true; }
    return false;
  };
  auto is_valid_file = [](const std::string& v){
    if (v.empty()) return false;
    for (unsigned char c : v) { if (c < 0x20 || c == 0x7F) return false; }
    // Reject separators and traversal
    if (v.find('/') != std::string::npos || v.find('\\') != std::string::npos) return false;
    if (v.rfind("..", 0) == 0) return false;
    if (!v.empty() && (v[0] == '/' || v[0] == '\\')) return false; // absolute/UNC
    // Windows drive like C:...
    if (v.size() >= 2 && std::isalpha(static_cast<unsigned char>(v[0])) && v[1] == ':') return false;
    static const std::regex rx("^wal-([0-9]{8})\\.log$");
    return std::regex_match(v, rx);
  };
  auto parse_u64 = [](const std::string& s, std::uint64_t& out){
    const char* beg = s.data(); const char* end = beg + s.size();
    unsigned long long tmp = 0;
    auto [ptr, ec] = std::from_chars(beg, end, tmp, 10);
    if (ec != std::errc() || ptr != end) return false;
    out = static_cast<std::uint64_t>(tmp);
    return true;
  };

  std::string line; std::size_t line_no = 1; // header already consumed
  while (std::getline(in, line)) {
    ++line_no;
    if (line.empty()) continue;

    ManifestEntry e{};
    bool have_file=false, have_seq=false, have_start=false, have_end=false, have_frames=false, have_bytes=false, have_first=false;

    std::istringstream iss(line);
    std::string kv;
    while (iss >> kv) {
      auto eq = kv.find('='); if (eq == std::string::npos) continue;
      auto k = kv.substr(0, eq);
      auto v = kv.substr(eq + 1);
      if (k == "file") {
        if (!is_valid_file(v)) {
          return std::vesper_unexpected(error{error_code::data_integrity, std::string("manifest parse error at line ") + std::to_string(line_no) + ": invalid filename", "wal.manifest"});
        }
        e.file = v; have_file = true;
      } else if (k == "seq") {
        have_seq = parse_u64(v, e.seq);
        if (!have_seq) return std::vesper_unexpected(error{error_code::data_integrity, std::string("manifest parse error at line ") + std::to_string(line_no) + ": invalid seq=\"" + v + "\"", "wal.manifest"});
      } else if (k == "start_lsn") {
        have_start = parse_u64(v, e.start_lsn);
        if (!have_start) return std::vesper_unexpected(error{error_code::data_integrity, std::string("manifest parse error at line ") + std::to_string(line_no) + ": invalid start_lsn=\"" + v + "\"", "wal.manifest"});
      } else if (k == "first_lsn") {
        have_first = parse_u64(v, e.first_lsn);
        if (!have_first) return std::vesper_unexpected(error{error_code::data_integrity, std::string("manifest parse error at line ") + std::to_string(line_no) + ": invalid first_lsn=\"" + v + "\"", "wal.manifest"});
      } else if (k == "end_lsn") {
        have_end = parse_u64(v, e.end_lsn);
        if (!have_end) return std::vesper_unexpected(error{error_code::data_integrity, std::string("manifest parse error at line ") + std::to_string(line_no) + ": invalid end_lsn=\"" + v + "\"", "wal.manifest"});
      } else if (k == "frames") {
        have_frames = parse_u64(v, e.frames);
        if (!have_frames) return std::vesper_unexpected(error{error_code::data_integrity, std::string("manifest parse error at line ") + std::to_string(line_no) + ": invalid frames=\"" + v + "\"", "wal.manifest"});
      } else if (k == "bytes") {
        have_bytes = parse_u64(v, e.bytes);
        if (!have_bytes) return std::vesper_unexpected(error{error_code::data_integrity, std::string("manifest parse error at line ") + std::to_string(line_no) + ": invalid bytes=\"" + v + "\"", "wal.manifest"});
      }
    }

    if (!have_file || !have_seq || !have_start || !have_end || !have_frames || !have_bytes) {
      return std::vesper_unexpected(error{error_code::data_integrity, std::string("manifest parse error at line ") + std::to_string(line_no) + ": missing required field(s)", "wal.manifest"});
    }
    if (!have_first) e.first_lsn = e.start_lsn; // backward compat
    m.entries.push_back(std::move(e));
  }
  return m;
}

auto save_manifest_atomic(const std::filesystem::path& dir, const Manifest& m)
    -> std::expected<void, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  const auto p = dir / "wal.manifest";
  auto tmp = dir / "wal.manifest.tmp";

  // Build manifest content into a single buffer
  std::string content;
  content.reserve(64 + m.entries.size() * 96);
  content.append("vesper-wal-manifest v1\n");
  for (const auto& e : m.entries) {
    content.append("file=").append(e.file)
           .append(" seq=").append(std::to_string(e.seq))
           .append(" start_lsn=").append(std::to_string(e.start_lsn))
           .append(" first_lsn=").append(std::to_string(e.first_lsn))
           .append(" end_lsn=").append(std::to_string(e.end_lsn))
           .append(" frames=").append(std::to_string(e.frames))
           .append(" bytes=").append(std::to_string(e.bytes))
           .append("\n");
  }

  // Remove leftover wal.manifest.tmp from a previous crashed save. Ignore ENOENT/ERROR_FILE_NOT_FOUND.
  { std::error_code ec; std::filesystem::remove(tmp, ec); }

  // 1) Write tmp and fsync file
#if defined(__linux__) || defined(__APPLE__)
  int fd = ::open(tmp.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
  if (fd < 0) {
    return std::vesper_unexpected(error{error_code::io_failed, "manifest tmp open failed", "wal.manifest"});
  }
  ssize_t written = 0; const char* data = content.data(); ssize_t to_write = static_cast<ssize_t>(content.size());
  while (to_write > 0) {
    ssize_t n = ::write(fd, data + written, static_cast<size_t>(to_write));
    if (n < 0) { ::close(fd); return std::vesper_unexpected(error{error_code::io_failed, "manifest tmp write failed", "wal.manifest"}); }
    written += n; to_write -= n;
  }
  (void)::fsync(fd);
  (void)::close(fd);
#elif defined(_WIN32)
  // Wide-char paths on Windows; fall back to a unique tmp name if the default tmp is locked.
  std::wstring wtmp = tmp.wstring();
  HANDLE h = ::CreateFileW(wtmp.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    DWORD err = ::GetLastError();
    if (err == ERROR_SHARING_VIOLATION || err == ERROR_ACCESS_DENIED) {
      std::wstring alt = std::wstring(L"wal.manifest.tmp.") + std::to_wstring(::GetCurrentProcessId()) + L"." + std::to_wstring(::GetTickCount64());
      tmp = dir / alt;
      wtmp = tmp.wstring();
      h = ::CreateFileW(wtmp.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    }
    if (h == INVALID_HANDLE_VALUE) {
      return std::vesper_unexpected(error{error_code::io_failed, "manifest tmp open failed", "wal.manifest"});
    }
  }
  DWORD written = 0; BOOL ok = ::WriteFile(h, content.data(), static_cast<DWORD>(content.size()), &written, nullptr);
  if (!ok || written != content.size()) { ::CloseHandle(h); return std::vesper_unexpected(error{error_code::io_failed, "manifest tmp write failed", "wal.manifest"}); }
  ::FlushFileBuffers(h);
  ::CloseHandle(h);
#else
  // Fallback: using ofstream (no fsync); atomicity is preserved by the rename below where supported.
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out.good()) return std::vesper_unexpected(error{error_code::io_failed, "manifest tmp write failed", "wal.manifest"});
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    out.flush();
  }
#endif

  // 2) Atomic replace
#if VESPER_ENABLE_ATOMIC_RENAME
  std::error_code ec;
#ifndef _WIN32
  std::filesystem::rename(tmp, p, ec);
  if (ec) {
    return std::vesper_unexpected(error{error_code::io_failed, "manifest rename failed", "wal.manifest"});
  }
#else
  const auto wdst = p.wstring();
  // Prefer ReplaceFileW to preserve attributes/ACLs; fallback to MoveFileExW if needed.
  if (!::ReplaceFileW(wdst.c_str(), tmp.wstring().c_str(), nullptr, 0, nullptr, nullptr)) {
    if (!::MoveFileExW(tmp.wstring().c_str(), wdst.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
      std::error_code ec2; std::filesystem::remove(tmp, ec2);
      return std::vesper_unexpected(error{error_code::io_failed, "manifest replace failed", "wal.manifest"});
    }
  }
#endif
#else
  // Fallback: non-atomic overwrite (legacy)
  {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    if (!out.good()) return std::vesper_unexpected(error{error_code::io_failed, "manifest write failed", "wal.manifest"});
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    out.flush();
  }
#endif

  // POSIX: fsync the parent directory to persist the rename. Windows: FlushFileBuffers on the final file handle.
#if defined(__linux__) || defined(__APPLE__)
  int dfd = ::open(dir.c_str(), O_RDONLY);
  if (dfd >= 0) { (void)::fsync(dfd); (void)::close(dfd); }
#elif defined(_WIN32)
  HANDLE h2 = ::CreateFileW(p.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h2 != INVALID_HANDLE_VALUE) { ::FlushFileBuffers(h2); ::CloseHandle(h2); }
#endif

  // Attempt to remove the tmp path (harmless if it doesn't exist). Ignore errors.
  { std::error_code ec3; std::filesystem::remove(tmp, ec3); }
  return {};
}

auto save_manifest(const std::filesystem::path& dir, const Manifest& m)
    -> std::expected<void, vesper::core::error> {
  return save_manifest_atomic(dir, m);
}

auto validate_manifest(const std::filesystem::path& dir)
    -> std::expected<ManifestValidation, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  ManifestValidation v{};
  auto p = dir / "wal.manifest";
  std::ifstream in(p);
  if (!in.good()) {
    v.ok = false; v.issues.push_back({ManifestIssueCode::Empty, Severity::Error, "", 0, "manifest missing"});
    return v;
  }
  std::string header; std::getline(in, header);
  if (header != std::string("vesper-wal-manifest v1")) {
    v.ok = false; v.issues.push_back({ManifestIssueCode::BadHeader, Severity::Error, "", 0, "bad header"});
    return v;
  }
  std::vector<ManifestEntry> es; std::string line;
  std::size_t line_no = 1; // header consumed
  auto has_ctrl = [](const std::string& s){ for (unsigned char c : s) { if (c < 0x20 || c == 0x7F) return true; } return false; };
  auto is_valid_file = [&](const std::string& v){
    if (v.empty() || has_ctrl(v)) return false;
    if (v.find('/') != std::string::npos || v.find('\\') != std::string::npos) return false;
    if (v.rfind("..", 0) == 0) return false;
    if (!v.empty() && (v[0] == '/' || v[0] == '\\')) return false;
    if (v.size() >= 2 && std::isalpha(static_cast<unsigned char>(v[0])) && v[1] == ':') return false;
    static const std::regex rx("^wal-([0-9]{8})\\.log$");
    return std::regex_match(v, rx);
  };
  while (std::getline(in, line)){
    ++line_no;
    if (line.empty()) continue;
    ManifestEntry e{};
    bool ok_line = true; bool ok_file = true;
    std::istringstream iss(line);
    std::string kv;
    while (iss >> kv){
      auto eq = kv.find('='); if (eq==std::string::npos) continue; auto k=kv.substr(0,eq); auto val=kv.substr(eq+1);
      if(k=="file") { e.file=val; ok_file = is_valid_file(e.file); }
      else {
        auto parse_u64 = [](const std::string& s, std::uint64_t& out){ const char* b=s.data(); const char* e2=b+s.size(); unsigned long long tmp=0; auto [p,ec]=std::from_chars(b,e2,tmp,10); if(ec!=std::errc()||p!=e2) return false; out=static_cast<std::uint64_t>(tmp); return true; };
        if(k=="seq") { ok_line &= parse_u64(val, e.seq); }
        else if(k=="start_lsn") { ok_line &= parse_u64(val, e.start_lsn); }
        else if(k=="first_lsn") { ok_line &= parse_u64(val, e.first_lsn); }
        else if(k=="end_lsn") { ok_line &= parse_u64(val, e.end_lsn); }
        else if(k=="frames") { ok_line &= parse_u64(val, e.frames); }
        else if(k=="bytes") { ok_line &= parse_u64(val, e.bytes); }
      }
    }
    if (!ok_file) { v.ok=false; v.issues.push_back({ManifestIssueCode::BadHeader, Severity::Error, e.file, e.seq, "invalid filename"}); continue; }
    if (!ok_line) { v.ok=false; v.issues.push_back({ManifestIssueCode::BadHeader, Severity::Error, e.file, e.seq, std::string("invalid numeric at line ")+std::to_string(line_no)}); continue; }
    es.push_back(std::move(e));
  }
  if (es.empty()) { v.ok=false; v.issues.push_back({ManifestIssueCode::Empty, Severity::Error, "", 0, "no entries"}); }
  // Check ascending seq and duplicates
  std::uint64_t prev_seq=0; std::set<std::string> seen_files; std::unordered_map<std::uint64_t, std::string> seen_seq_first_file;
  for (size_t i=0;i<es.size();++i){
    if (i>0 && es[i].seq < prev_seq) { v.ok=false; v.issues.push_back({ManifestIssueCode::OutOfOrderSeq, Severity::Error, es[i].file, es[i].seq, "out of order"}); }
    if (!seen_files.insert(es[i].file).second) { v.ok=false; v.issues.push_back({ManifestIssueCode::DuplicateFile, Severity::Error, es[i].file, es[i].seq, "duplicate file"}); }
    // Duplicate seq detection (across different files)
    auto it = seen_seq_first_file.find(es[i].seq);
    if (it != seen_seq_first_file.end()) {
      if (it->second != es[i].file) {
        v.ok = false;
        v.issues.push_back({ManifestIssueCode::DuplicateSeq, Severity::Error, es[i].file, es[i].seq, std::string("duplicate seq: also in ") + it->second});
      }
    } else {
      seen_seq_first_file.emplace(es[i].seq, es[i].file);
    }
    prev_seq = es[i].seq;
  }
  // Intra-entry LSN invariants
  for (auto& e : es) {
    if (e.start_lsn > e.end_lsn) { v.ok=false; v.issues.push_back({ManifestIssueCode::LsnInvalid, Severity::Error, e.file, e.seq, "start_lsn > end_lsn"}); }
    if (e.first_lsn < e.start_lsn) { v.ok=false; v.issues.push_back({ManifestIssueCode::LsnInvalid, Severity::Error, e.file, e.seq, "first_lsn < start_lsn"}); }
    if (e.first_lsn > e.end_lsn) { v.ok=false; v.issues.push_back({ManifestIssueCode::LsnInvalid, Severity::Error, e.file, e.seq, "first_lsn > end_lsn"}); }
  }
  // Cross-entry LSN ordering (assumes entries are intended to be in seq order)
  for (size_t i=1;i<es.size();++i){
    auto& prev = es[i-1]; auto& cur = es[i];
    if (cur.start_lsn <= prev.end_lsn) {
      v.ok=false; v.issues.push_back({ManifestIssueCode::LsnOverlap, Severity::Error, cur.file, cur.seq,
        std::string("overlap with prev ")+prev.file+" seq="+std::to_string(prev.seq)+
        " ["+std::to_string(prev.start_lsn)+","+std::to_string(prev.end_lsn)+"] vs ["+
        std::to_string(cur.start_lsn)+","+std::to_string(cur.end_lsn)+"]"});
    }
    if (cur.end_lsn < prev.end_lsn) {
      v.ok=false; v.issues.push_back({ManifestIssueCode::LsnOrder, Severity::Error, cur.file, cur.seq,
        std::string("end_lsn decreased vs prev end_lsn ")+std::to_string(prev.end_lsn)});
    }
    if (cur.start_lsn > prev.end_lsn + 1) {
      v.issues.push_back({ManifestIssueCode::LsnGap, Severity::Warning, cur.file, cur.seq,
        std::string("gap between prev end_lsn ")+std::to_string(prev.end_lsn)+" and cur start_lsn "+std::to_string(cur.start_lsn)});
    }
  }
  // Seq gap
  for (size_t i=1;i<es.size();++i){ if (es[i].seq != es[i-1].seq + 1) { v.issues.push_back({ManifestIssueCode::SeqGap, Severity::Warning, es[i].file, es[i].seq, "gap in seq"}); } }
  // Missing files
  for (auto& e : es){ if (!std::filesystem::exists(dir / e.file)) { v.ok=false; v.issues.push_back({ManifestIssueCode::MissingFileOnDisk, Severity::Error, e.file, e.seq, "missing on disk"}); } }
  // Extra files on disk
  std::set<std::string> mf; for (auto& e: es) mf.insert(e.file);
  std::regex rx("^wal-([0-9]{8})\\.log$");
  for (auto& de : std::filesystem::directory_iterator(dir)){
    if (!de.is_regular_file()) continue; auto name = de.path().filename().string();
    if (std::regex_match(name, rx)) {
      if (!mf.count(name)) v.issues.push_back({ManifestIssueCode::ExtraFileOnDisk, Severity::Warning, name, 0, "extra on disk"});
    }
  }
  return v;
}

auto enforce_manifest_order(const std::filesystem::path& dir)
    -> std::expected<void, vesper::core::error> {
  auto mx = load_manifest(dir); if (!mx) return std::vesper_unexpected(mx.error());
  Manifest m = *mx; std::sort(m.entries.begin(), m.entries.end(), [](auto&a, auto&b){ return a.seq < b.seq; });
  return save_manifest(dir, m);
}


} // namespace vesper::wal


#include <regex>
#include <utility>
#include <algorithm>
#include <vector>

namespace {
using Pair = std::pair<std::uint64_t, std::filesystem::path>;
static std::vector<Pair> list_sorted(const std::filesystem::path& dir, const std::string& prefix){
  std::vector<Pair> v;
  std::regex rx(std::string("^") + prefix + "([0-9]{8})\\.log$");
  for (auto& de : std::filesystem::directory_iterator(dir)){
    if (!de.is_regular_file()) continue; auto name=de.path().filename().string();
    std::smatch m; if (std::regex_match(name, m, rx) && m.size()==2){
      try { auto seq = static_cast<std::uint64_t>(std::stoull(m[1].str())); v.emplace_back(seq, de.path()); } catch(...) {}
    }
  }
  std::sort(v.begin(), v.end(), [](auto&a, auto&b){ return a.first < b.first || (a.first == b.first && a.second < b.second); });
  return v;
}
}

namespace vesper::wal {


// Internal implementation: shared logic for strict vs lenient rebuild
static auto rebuild_manifest_impl(const std::filesystem::path& dir, RebuildMode mode)
    -> std::expected<LenientRebuildResult, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  LenientRebuildResult out{};
  auto files = list_sorted(dir, "wal-");
  std::uint64_t prev_end_lsn = 0; std::string prev_file; std::uint64_t prev_seq = 0; bool have_prev=false;

  for (auto& kv : files){
    const auto& path = kv.second; const auto filename = path.filename().string(); const std::uint64_t seq = kv.first;

    // Scan this file and accumulate stats
    std::size_t frames=0; std::size_t bytes=0; std::uint64_t first_lsn=0; std::uint64_t last_lsn=0;
    auto st = recover_scan(path.string(), [&](const WalFrame& f){
      frames++; bytes += f.len; if (first_lsn==0) first_lsn = f.lsn; last_lsn = f.lsn;
    });

    if (!st) {
      if (mode == RebuildMode::Strict) {
        return std::vesper_unexpected(st.error());
      } else {
        out.issues.push_back(RebuildIssue{filename, seq, st.error().code, std::string("scan failed"), std::nullopt, std::nullopt, std::nullopt, std::nullopt});
        continue; // skip this file
      }
    }

    ManifestEntry e{}; e.file = filename; e.seq = seq; e.start_lsn = first_lsn; e.first_lsn = first_lsn; e.end_lsn = last_lsn; e.frames = frames; e.bytes = bytes;

    // Intra-entry validation: start_lsn <= first_lsn <= end_lsn
    auto emit_intra_issue = [&](const char* msg){
      if (mode == RebuildMode::Strict) {
        return std::optional<error>{ error{error_code::data_integrity,
          std::string("rebuild_manifest: ") + msg + " for " + e.file +
          " seq=" + std::to_string(e.seq), "wal.manifest"} };
      } else {
        out.issues.push_back(RebuildIssue{ e.file, e.seq, error_code::data_integrity, msg,
          e.start_lsn, e.first_lsn, e.end_lsn, std::nullopt});
        return std::optional<error>{};
      }
    };

    if (e.start_lsn > e.end_lsn) { if (auto er = emit_intra_issue("start_lsn > end_lsn")) return std::vesper_unexpected(*er); else { continue; } }
    if (e.first_lsn < e.start_lsn) { if (auto er = emit_intra_issue("first_lsn < start_lsn")) return std::vesper_unexpected(*er); else { continue; } }
    if (e.first_lsn > e.end_lsn) { if (auto er = emit_intra_issue("first_lsn > end_lsn")) return std::vesper_unexpected(*er); else { continue; } }

    // Cross-entry validation against previous included entry
    if (have_prev) {
      if (e.start_lsn <= prev_end_lsn) {
        if (mode == RebuildMode::Strict) {
          return std::vesper_unexpected(error{error_code::data_integrity,
            std::string("rebuild_manifest: LSN overlap: prev ") + prev_file +
            " seq=" + std::to_string(prev_seq) +
            " end_lsn=" + std::to_string(prev_end_lsn) +
            ", curr " + e.file + " seq=" + std::to_string(e.seq) +
            " start_lsn=" + std::to_string(e.start_lsn), "wal.manifest"});
        } else {
          out.issues.push_back(RebuildIssue{ e.file, e.seq, error_code::data_integrity, "overlap with previous",
            e.start_lsn, e.first_lsn, e.end_lsn, prev_end_lsn});
          continue; // skip overlapping entry
        }
      }
      if (e.end_lsn < prev_end_lsn) {
        if (mode == RebuildMode::Strict) {
          return std::vesper_unexpected(error{error_code::data_integrity,
            std::string("rebuild_manifest: end_lsn decreased vs prev: prev end=") +
            std::to_string(prev_end_lsn) + ", curr end=" + std::to_string(e.end_lsn), "wal.manifest"});
        } else {
          out.issues.push_back(RebuildIssue{ e.file, e.seq, error_code::data_integrity, "end_lsn decreased vs previous",
            e.start_lsn, e.first_lsn, e.end_lsn, prev_end_lsn});
          continue; // skip out-of-order end_lsn
        }
      }
    }

    // Accept entry
    out.manifest.entries.push_back(e);
    prev_end_lsn = e.end_lsn; prev_file = e.file; prev_seq = e.seq; have_prev = true;
  }
  return out;
}

auto rebuild_manifest(const std::filesystem::path& dir)
    -> std::expected<Manifest, vesper::core::error> {
  auto r = rebuild_manifest_impl(dir, RebuildMode::Strict);
  if (!r) return std::vesper_unexpected(r.error());
  return r->manifest;
}

auto rebuild_manifest_lenient(const std::filesystem::path& dir)
    -> std::expected<LenientRebuildResult, vesper::core::error> {
  return rebuild_manifest_impl(dir, RebuildMode::Lenient);
}

} // namespace vesper::wal
