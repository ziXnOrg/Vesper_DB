#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <vesper/wal/frame.hpp>

#include <filesystem>
#include <fstream>
#include <vector>
#include <cstring>

using namespace vesper;
using namespace vesper::wal;

namespace {
namespace fs = std::filesystem;

static void write_header_only(const fs::path& file,
                              std::uint32_t magic,
                              std::uint32_t len,
                              std::uint16_t type = 1,
                              std::uint16_t reserved = 0,
                              std::uint64_t lsn = 1) {
  std::vector<std::uint8_t> hdr(WAL_HEADER_SIZE);
  std::uint8_t* p = hdr.data();
  std::memcpy(p, &magic, 4); p += 4;
  std::memcpy(p, &len, 4); p += 4;
  std::memcpy(p, &type, 2); p += 2;
  std::memcpy(p, &reserved, 2); p += 2;
  std::memcpy(p, &lsn, 8); p += 8;
  std::ofstream out(file, std::ios::binary);
  REQUIRE(out.good());
  out.write(reinterpret_cast<const char*>(hdr.data()), static_cast<std::streamsize>(hdr.size()));
  REQUIRE(out.good());
}

} // namespace

TEST_CASE("recover_scan guards: invalid magic with huge LEN", "[wal][recover][guards]") {
  fs::path dir = fs::temp_directory_path() / "vesper_wal_guard_invalid_magic";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  auto f = dir / "wal-00000001.log";

  const std::uint32_t huge_len = 0x50000000u; // ~1.34 GB
  write_header_only(f, /*magic=*/0u, /*len=*/huge_len);

  std::size_t delivered = 0;
  auto stats = recover_scan(f.string(), [&](const WalFrame&){ delivered++; });
  REQUIRE(stats.has_value());
  REQUIRE(stats->frames == 0);
  REQUIRE(delivered == 0);

  fs::remove_all(dir, ec);
}

TEST_CASE("recover_scan guards: LEN exceeding MAX_FRAME_LEN (32MiB)", "[wal][recover][guards]") {
  fs::path dir = fs::temp_directory_path() / "vesper_wal_guard_max_len";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  auto f = dir / "wal-00000001.log";

  const std::uint32_t max32 = 32u * 1024u * 1024u;
  const std::uint32_t len_over = max32 + 1u;
  write_header_only(f, /*magic=*/WAL_MAGIC, /*len=*/len_over);

  std::size_t delivered = 0;
  auto stats = recover_scan(f.string(), [&](const WalFrame&){ delivered++; });
  REQUIRE(stats.has_value());
  REQUIRE(stats->frames == 0);
  REQUIRE(delivered == 0);

  fs::remove_all(dir, ec);
}

TEST_CASE("recover_scan guards: LEN exceeds remaining file bytes", "[wal][recover][guards]") {
  fs::path dir = fs::temp_directory_path() / "vesper_wal_guard_remaining";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  auto f = dir / "wal-00000001.log";

  // Header-only file; declare a small payload+CRC that is not present
  const std::uint32_t declared = static_cast<std::uint32_t>(WAL_HEADER_SIZE + 4 + 128);
  write_header_only(f, /*magic=*/WAL_MAGIC, /*len=*/declared);

  std::size_t delivered = 0;
  auto stats = recover_scan(f.string(), [&](const WalFrame&){ delivered++; });
  REQUIRE(stats.has_value());
  REQUIRE(stats->frames == 0);
  REQUIRE(delivered == 0);

  fs::remove_all(dir, ec);
}

