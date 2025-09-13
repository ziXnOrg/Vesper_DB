#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include <cstdint>
#include <iostream>
#include <cstdlib>

#include "vesper/index/pq_fastscan.hpp"
#include "vesper/core/cpu_features.hpp"
#include "vesper/core/platform_utils.hpp"

using namespace vesper::index;

namespace {

static std::size_t get_codes_per_probe() {
    auto env = vesper::core::safe_getenv("VESPER_ADC_CODES_PER_PROBE");
    if (env && !env->empty()) {
        char* endp = nullptr;
        long v = std::strtol(env->c_str(), &endp, 10);
        if (endp != env->c_str() && v > 0) return static_cast<std::size_t>(v);
    }
    return 2048; // default
}

std::vector<float> make_codebooks(std::uint32_t m, std::uint32_t nbits, std::uint32_t dsub) {
    const std::size_t ksub = 1u << nbits;
    std::vector<float> cb(static_cast<std::size_t>(m) * ksub * dsub);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : cb) v = dist(rng);
    return cb;
}

std::vector<PqCodeBlock> make_blocks(std::uint32_t m, std::size_t total_codes, std::uint32_t block_size) {
    const std::size_t n_blocks = (total_codes + block_size - 1) / block_size;
    std::vector<PqCodeBlock> blocks;
    blocks.reserve(n_blocks);
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> code_dist(0, 255);

    PqCodeBlock blk(m, block_size);
    for (std::size_t i = 0; i < total_codes; ++i) {
        std::vector<std::uint8_t> code(m);
        for (std::uint32_t j = 0; j < m; ++j) code[j] = static_cast<std::uint8_t>(code_dist(rng));
        if (!blk.add_code(code.data())) {
            blocks.push_back(blk);
            blk.clear();
            (void)blk.add_code(code.data());
        }
    }
    if (blk.size()) blocks.push_back(blk);
    return blocks;
}

} // namespace

static void BenchADC_Scalar(benchmark::State& state) {
    const std::uint32_t dim = static_cast<std::uint32_t>(state.range(0));
    const std::uint32_t m   = static_cast<std::uint32_t>(state.range(1));
    const std::uint32_t nbits = 8;
    const std::uint32_t dsub = dim / m;
    const std::uint32_t block_size = 32;
    const std::uint32_t nprobe = (state.range(2) > 0) ? static_cast<std::uint32_t>(state.range(2)) : 1;
    const std::size_t codes_per_probe = get_codes_per_probe();
    const std::size_t codes = codes_per_probe * nprobe;

    FastScanPqConfig cfg{ .m = m, .nbits = nbits, .block_size = block_size, .use_avx512 = false };
    FastScanPq pq(cfg);

    auto cb = make_codebooks(m, nbits, dsub);
    pq.import_pretrained(dsub, std::span<const float>(cb.data(), cb.size()));

    auto blocks = make_blocks(m, codes, block_size);

    std::vector<float> query(dim);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : query) v = dist(rng);

    std::vector<float> out(blocks.size() * block_size, 0.0f);

    for (auto _ : state) {
        benchmark::DoNotOptimize(out.data());
        pq.compute_distances(query.data(), blocks, out.data());
        benchmark::ClobberMemory();
    }

    state.counters["impl"] = 0; // scalar
}

#if defined(__AVX2__)
static void BenchADC_AVX2(benchmark::State& state) {
    const std::uint32_t dim = static_cast<std::uint32_t>(state.range(0));
    const std::uint32_t m   = static_cast<std::uint32_t>(state.range(1));
    const std::uint32_t nbits = 8;
    const std::uint32_t dsub = dim / m;
    const std::uint32_t block_size = 32;
    const std::uint32_t nprobe = (state.range(2) > 0) ? static_cast<std::uint32_t>(state.range(2)) : 1;
    const std::size_t codes_per_probe = get_codes_per_probe();
    const std::size_t codes = codes_per_probe * nprobe;

    FastScanPqConfig cfg{ .m = m, .nbits = nbits, .block_size = block_size, .use_avx512 = false };
    FastScanPq pq(cfg);

    auto cb = make_codebooks(m, nbits, dsub);
    pq.import_pretrained(dsub, std::span<const float>(cb.data(), cb.size()));

    auto blocks = make_blocks(m, codes, block_size);

    std::vector<float> query(dim);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : query) v = dist(rng);

    std::vector<float> out(blocks.size() * block_size, 0.0f);

    for (auto _ : state) {
        benchmark::DoNotOptimize(out.data());
        pq.compute_distances_avx2(query.data(), blocks, out.data());
        benchmark::ClobberMemory();
    }

    state.counters["impl"] = 2; // avx2
}
#endif

#if defined(__AVX512F__)
static void BenchADC_AVX512(benchmark::State& state) {
    if (!vesper::core::cpu_supports_avx512_runtime()) {
        state.SkipWithError("AVX-512 not supported at runtime");
        return;
    }

    const std::uint32_t dim = static_cast<std::uint32_t>(state.range(0));
    const std::uint32_t m   = static_cast<std::uint32_t>(state.range(1));
    const std::uint32_t nbits = 8;
    const std::uint32_t dsub = dim / m;
    const std::uint32_t block_size = 32;
    const std::uint32_t nprobe = (state.range(2) > 0) ? static_cast<std::uint32_t>(state.range(2)) : 1;
    const std::size_t codes_per_probe = get_codes_per_probe();
    const std::size_t codes = codes_per_probe * nprobe;

    FastScanPqConfig cfg{ .m = m, .nbits = nbits, .block_size = block_size, .use_avx512 = true };
    FastScanPq pq(cfg);

    auto cb = make_codebooks(m, nbits, dsub);
    pq.import_pretrained(dsub, std::span<const float>(cb.data(), cb.size()));

    auto blocks = make_blocks(m, codes, block_size);

    std::vector<float> query(dim);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : query) v = dist(rng);

    std::vector<float> out(blocks.size() * block_size, 0.0f);

    for (auto _ : state) {
        benchmark::DoNotOptimize(out.data());
        pq.compute_distances_avx512(query.data(), blocks, out.data());
        benchmark::ClobberMemory();
    }

    state.counters["impl"] = 3; // avx512
}
#endif

// Params: dim, m
BENCHMARK(BenchADC_Scalar)
    ->Args({64, 8, 1})->Args({64, 8, 4})->Args({64, 8, 16})->Args({64, 8, 32})
    ->Args({96, 12, 1})->Args({96, 12, 4})->Args({96, 12, 16})->Args({96, 12, 32})
    ->Args({128, 16, 1})->Args({128, 16, 4})->Args({128, 16, 16})->Args({128, 16, 32})
    ->Args({192, 16, 1})->Args({192, 16, 4})->Args({192, 16, 16})->Args({192, 16, 32})
    ->Args({256, 16, 1})->Args({256, 16, 4})->Args({256, 16, 16})->Args({256, 16, 32})
    ->Unit(benchmark::kMillisecond);
#if defined(__AVX2__)
BENCHMARK(BenchADC_AVX2)
    ->Args({64, 8, 1})->Args({64, 8, 4})->Args({64, 8, 16})->Args({64, 8, 32})
    ->Args({96, 12, 1})->Args({96, 12, 4})->Args({96, 12, 16})->Args({96, 12, 32})
    ->Args({128, 16, 1})->Args({128, 16, 4})->Args({128, 16, 16})->Args({128, 16, 32})
    ->Args({192, 16, 1})->Args({192, 16, 4})->Args({192, 16, 16})->Args({192, 16, 32})
    ->Args({256, 16, 1})->Args({256, 16, 4})->Args({256, 16, 16})->Args({256, 16, 32})
    ->Unit(benchmark::kMillisecond);
#endif
#if defined(__AVX512F__)
BENCHMARK(BenchADC_AVX512)
    ->Args({64, 8, 1})->Args({64, 8, 4})->Args({64, 8, 16})->Args({64, 8, 32})
    ->Args({96, 12, 1})->Args({96, 12, 4})->Args({96, 12, 16})->Args({96, 12, 32})
    ->Args({128, 16, 1})->Args({128, 16, 4})->Args({128, 16, 16})->Args({128, 16, 32})
    ->Args({192, 16, 1})->Args({192, 16, 4})->Args({192, 16, 16})->Args({192, 16, 32})
    ->Args({256, 16, 1})->Args({256, 16, 4})->Args({256, 16, 16})->Args({256, 16, 32})
    ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_MAIN();

