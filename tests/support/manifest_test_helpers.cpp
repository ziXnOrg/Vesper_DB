#include "tests/support/manifest_test_helpers.hpp"

#include <fstream>
#include <algorithm>
#include <stdexcept>

namespace manifest_test_helpers {

static constexpr const char* kHeader = "vesper-wal-manifest v1";

std::vector<std::string> read_manifest_entries(const std::filesystem::path& manifest_path) {
  std::ifstream in(manifest_path);
  if (!in.good()) throw std::runtime_error("failed to open manifest for read: " + manifest_path.string());
  std::string header; std::getline(in, header);
  if (header != kHeader) throw std::runtime_error("bad manifest header: " + header);
  std::vector<std::string> lines; std::string line;
  while (std::getline(in, line)) lines.push_back(line);
  return lines;
}

void write_manifest_entries(const std::filesystem::path& manifest_path, std::span<const std::string> lines) {
  std::ofstream out(manifest_path, std::ios::binary | std::ios::trunc);
  if (!out.good()) throw std::runtime_error("failed to open manifest for write: " + manifest_path.string());
  out << kHeader << "\n";
  for (const auto& ln : lines) out << ln << "\n";
}

std::vector<std::string> entries_reversed(std::span<const std::string> lines) {
  std::vector<std::string> v(lines.begin(), lines.end());
  std::reverse(v.begin(), v.end());
  return v;
}

std::vector<std::string> entries_without_filename(std::span<const std::string> lines, const std::string& filename) {
  std::vector<std::string> v; v.reserve(lines.size());
  for (const auto& ln : lines) if (ln.find(filename) == std::string::npos) v.push_back(ln);
  return v;
}

std::vector<std::string> entries_with_duplicated_filename(std::span<const std::string> lines, const std::string& filename) {
  std::vector<std::string> v(lines.begin(), lines.end());
  auto it = std::find_if(v.begin(), v.end(), [&](const std::string& ln){ return ln.find(filename) != std::string::npos; });
  if (it != v.end()) v.push_back(*it);
  return v;
}

std::vector<std::pair<std::uint64_t, std::filesystem::path>> list_wal_files_sorted(const std::filesystem::path& dir) {
  std::vector<std::pair<std::uint64_t, std::filesystem::path>> v;
  for (auto& de : std::filesystem::directory_iterator(dir)){
    if (!de.is_regular_file()) continue;
    auto name = de.path().filename().string();
    if (name.rfind("wal-", 0)==0 && name.size() >= 4+8) {
      try { auto seq = static_cast<std::uint64_t>(std::stoull(name.substr(4,8))); v.emplace_back(seq, de.path()); } catch(...) {}
    }
  }
  std::sort(v.begin(), v.end(), [](auto&a, auto&b){ return a.first < b.first; });
  return v;
}

} // namespace manifest_test_helpers

