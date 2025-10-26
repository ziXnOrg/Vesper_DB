#include <benchmark/benchmark.h>

#include "vesper/index/kmeans_elkan.hpp"
#include "vesper/index/kmeans.hpp"

#include <random>
#include <vector>
#include <algorithm>

using namespace vesper::index;

namespace {

std::vector<float> make_gaussian(std::size_t n, std::size_t dim, std::uint32_t seed = 12345) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> data(n * dim);
    for (auto &x : data) x = nd(rng);
    return data;
}

struct QualityMetrics {
    double inertia{0.0};
    std::uint32_t iterations{0};
};

QualityMetrics compute_quality(const float* data, std::size_t n, std::size_t dim,
                               KmeansElkan::Config::InitMethod method,
                               std::uint32_t k, std::uint32_t seed) {
    KmeansElkan elkan;
    KmeansElkan::Config cfg{};
    cfg.k = k;
    cfg.max_iter = 25;
    cfg.epsilon = 1e-4f;
    cfg.seed = seed;
    cfg.use_parallel = true;
    cfg.verbose = false;
    cfg.init_method = method;
    auto res = elkan.cluster(data, n, dim, cfg);
    QualityMetrics qm{};
    if (res) {
        qm.inertia = res->inertia;
        qm.iterations = res->iterations;
    }
    return qm;
}

} // namespace

// Measure k-means++ initialization time and attach quality counters (from a paused cluster run)
static void BM_KMeansInit_PlusPlus(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const std::size_t dim = 128;
    const std::uint32_t k = 256;
    const std::uint32_t seed = 4242;

    auto data = make_gaussian(n, dim, seed);

    // Pre-compute quality once (not timed)
    QualityMetrics qm{};
    bool quality_done = false;

    for (auto _ : state) {
        if (!quality_done) {
            state.PauseTiming();
            qm = compute_quality(data.data(), n, dim, KmeansElkan::Config::InitMethod::KMeansPlusPlus, k, seed);
            state.ResumeTiming();
            quality_done = true;
        }

        auto centers = kmeans_plusplus_init(data.data(), n, dim, k, seed);
        benchmark::DoNotOptimize(centers.data());
        benchmark::ClobberMemory();
    }

    state.counters["final_inertia"] = qm.inertia;
    state.counters["iterations"] = static_cast<double>(qm.iterations);
}

// Measure k-means|| initialization time and attach quality counters (from a paused cluster run)
static void BM_KMeansInit_Parallel(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const std::size_t dim = 128;
    const std::uint32_t k = 256;
    const std::uint32_t seed = 4242;
    const std::uint32_t rounds = 5;
    const std::uint32_t oversampling = 0; // 0 => 2*k

    auto data = make_gaussian(n, dim, seed);

    // Pre-compute quality once (not timed)
    QualityMetrics qm{};
    bool quality_done = false;

    for (auto _ : state) {
        if (!quality_done) {
            state.PauseTiming();
            qm = compute_quality(data.data(), n, dim, KmeansElkan::Config::InitMethod::KMeansParallel, k, seed);
            state.ResumeTiming();
            quality_done = true;
        }

        auto centers = kmeans_parallel_init(data.data(), n, dim, k, rounds, oversampling, seed);
        benchmark::DoNotOptimize(centers.data());
        benchmark::ClobberMemory();
    }

    state.counters["final_inertia"] = qm.inertia;
    state.counters["iterations"] = static_cast<double>(qm.iterations);
}

// N in {1k, 5k, 10k, 50k, 100k}
BENCHMARK(BM_KMeansInit_PlusPlus)->Arg(1000)->Arg(5000)->Arg(10000)->Arg(50000)->Arg(100000)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_KMeansInit_Parallel)->Arg(1000)->Arg(5000)->Arg(10000)->Arg(50000)->Arg(100000)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();

