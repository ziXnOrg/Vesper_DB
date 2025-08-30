#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "vesper/wal/manifest.hpp"

// Fuzzer: feeds arbitrary bytes as wal.manifest and exercises load/validate paths.
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  using namespace vesper::wal;
  try {
    // Create a unique temp subdir for this run
    static std::atomic<uint64_t> counter{0};
    const uint64_t id = ++counter;
    std::filesystem::path dir = std::filesystem::temp_directory_path() / ("vesper_wal_manifest_fuzz_" + std::to_string(id));
    std::filesystem::create_directories(dir);

    // Write input bytes as wal.manifest
    std::filesystem::path mf = dir / "wal.manifest";
    std::ofstream out(mf, std::ios::binary);
    if (out.good()) {
      out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
      out.close();
    }

    // Exercise load and validate paths
    (void)load_manifest(dir);
    (void)validate_manifest(dir);
    (void)enforce_manifest_order(dir);

    // Cleanup (best-effort)
    std::error_code ec;
    std::filesystem::remove(mf, ec);
    std::filesystem::remove(dir, ec);
  } catch (...) {
    // Fuzzers must not throw across the boundary
  }
  return 0;
}

