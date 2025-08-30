#include <cstdint>
#include <cstddef>
#include <vector>
#include <span>

#include "vesper/wal/frame.hpp"

// Fuzzer: feeds arbitrary bytes into decode_frame/verify_crc32c to hit edge paths.
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  using namespace vesper::wal;
  try {
    std::span<const std::uint8_t> bytes{data, size};
    (void)verify_crc32c(bytes);
    (void)decode_frame(bytes);
  } catch (...) {
    // Never propagate exceptions out of the fuzzer
  }
  return 0;
}

