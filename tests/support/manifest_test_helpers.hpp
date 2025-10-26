#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace manifest_test_helpers {

// Reads wal.manifest, validates header equals exactly "vesper-wal-manifest v1",
// and returns the entry lines (without the header). Throws std::runtime_error on I/O or bad header.
std::vector<std::string> read_manifest_entries(const std::filesystem::path& manifest_path);

// Writes wal.manifest by truncating and emitting the exact header followed by the given entry lines.
// Throws std::runtime_error on I/O errors.
void write_manifest_entries(const std::filesystem::path& manifest_path, std::span<const std::string> lines);

// Convenience transforms (pure functions)
std::vector<std::string> entries_reversed(std::span<const std::string> lines);
std::vector<std::string> entries_without_filename(std::span<const std::string> lines, const std::string& filename);
std::vector<std::string> entries_with_duplicated_filename(std::span<const std::string> lines, const std::string& filename);

// Enumerate wal-*.log files in ascending sequence order (sequence parsed from wal-XXXXXXXX.*)
std::vector<std::pair<std::uint64_t, std::filesystem::path>> list_wal_files_sorted(const std::filesystem::path& dir);

} // namespace manifest_test_helpers

