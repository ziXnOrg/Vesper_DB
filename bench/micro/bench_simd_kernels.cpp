#include <benchmark/benchmark.h>
#include "vesper/kernels/dispatch.hpp"
#include "vesper/kernels/distance.hpp"
#include <random>
#include <vector>
#include <string>

using namespace vesper::kernels;

// Helper to generate random vectors
static std::vector<float> generate_random_vector(std::size_t size, std::uint32_t seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> vec(size);
    for (auto& v : vec) {
        v = dist(gen);
    }
    return vec;
}

// Benchmark different dimensions
static const std::vector<int> dimensions = {128, 256, 512, 768, 1024, 1536};

// L2 distance benchmarks
static void BM_L2_Scalar(benchmark::State& state) {
    const int dim = state.range(0);
    auto a = generate_random_vector(dim, 1);
    auto b = generate_random_vector(dim, 2);
    
    const auto& ops = select_backend("scalar");
    
    for (auto _ : state) {
        float result = ops.l2_sq(a, b);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
    state.SetLabel("scalar");
}

static void BM_L2_AVX2(benchmark::State& state) {
    const int dim = state.range(0);
    auto a = generate_random_vector(dim, 1);
    auto b = generate_random_vector(dim, 2);
    
    const auto& ops = select_backend("avx2");
    if (ops.l2_sq == select_backend("scalar").l2_sq) {
        state.SkipWithError("AVX2 not available");
        return;
    }
    
    for (auto _ : state) {
        float result = ops.l2_sq(a, b);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
    state.SetLabel("avx2");
}

static void BM_L2_AVX512(benchmark::State& state) {
    const int dim = state.range(0);
    auto a = generate_random_vector(dim, 1);
    auto b = generate_random_vector(dim, 2);
    
    const auto& ops = select_backend("avx512");
    if (ops.l2_sq == select_backend("scalar").l2_sq) {
        state.SkipWithError("AVX-512 not available");
        return;
    }
    
    for (auto _ : state) {
        float result = ops.l2_sq(a, b);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
    state.SetLabel("avx512");
}

static void BM_L2_Auto(benchmark::State& state) {
    const int dim = state.range(0);
    auto a = generate_random_vector(dim, 1);
    auto b = generate_random_vector(dim, 2);
    
    const auto& ops = select_backend_auto();
    
    for (auto _ : state) {
        float result = ops.l2_sq(a, b);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
    state.SetLabel("auto");
}

// Inner product benchmarks
static void BM_InnerProduct_Scalar(benchmark::State& state) {
    const int dim = state.range(0);
    auto a = generate_random_vector(dim, 1);
    auto b = generate_random_vector(dim, 2);
    
    const auto& ops = select_backend("scalar");
    
    for (auto _ : state) {
        float result = ops.inner_product(a, b);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
}

static void BM_InnerProduct_AVX2(benchmark::State& state) {
    const int dim = state.range(0);
    auto a = generate_random_vector(dim, 1);
    auto b = generate_random_vector(dim, 2);
    
    const auto& ops = select_backend("avx2");
    if (ops.inner_product == select_backend("scalar").inner_product) {
        state.SkipWithError("AVX2 not available");
        return;
    }
    
    for (auto _ : state) {
        float result = ops.inner_product(a, b);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
}

static void BM_InnerProduct_AVX512(benchmark::State& state) {
    const int dim = state.range(0);
    auto a = generate_random_vector(dim, 1);
    auto b = generate_random_vector(dim, 2);
    
    const auto& ops = select_backend("avx512");
    if (ops.inner_product == select_backend("scalar").inner_product) {
        state.SkipWithError("AVX-512 not available");
        return;
    }
    
    for (auto _ : state) {
        float result = ops.inner_product(a, b);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
}

// Cosine similarity benchmarks
static void BM_Cosine_Scalar(benchmark::State& state) {
    const int dim = state.range(0);
    auto a = generate_random_vector(dim, 1);
    auto b = generate_random_vector(dim, 2);
    
    // Normalize to avoid division by zero
    float norm_a = 0, norm_b = 0;
    for (std::size_t i = 0; i < dim; ++i) {
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    for (std::size_t i = 0; i < dim; ++i) {
        a[i] /= norm_a;
        b[i] /= norm_b;
    }
    
    const auto& ops = select_backend("scalar");
    
    for (auto _ : state) {
        float result = ops.cosine_similarity(a, b);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
}

static void BM_Cosine_AVX2(benchmark::State& state) {
    const int dim = state.range(0);
    auto a = generate_random_vector(dim, 1);
    auto b = generate_random_vector(dim, 2);
    
    // Normalize
    float norm_a = 0, norm_b = 0;
    for (std::size_t i = 0; i < dim; ++i) {
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    for (std::size_t i = 0; i < dim; ++i) {
        a[i] /= norm_a;
        b[i] /= norm_b;
    }
    
    const auto& ops = select_backend("avx2");
    if (ops.cosine_similarity == select_backend("scalar").cosine_similarity) {
        state.SkipWithError("AVX2 not available");
        return;
    }
    
    for (auto _ : state) {
        float result = ops.cosine_similarity(a, b);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
}

static void BM_Cosine_AVX512(benchmark::State& state) {
    const int dim = state.range(0);
    auto a = generate_random_vector(dim, 1);
    auto b = generate_random_vector(dim, 2);
    
    // Normalize
    float norm_a = 0, norm_b = 0;
    for (std::size_t i = 0; i < dim; ++i) {
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    for (std::size_t i = 0; i < dim; ++i) {
        a[i] /= norm_a;
        b[i] /= norm_b;
    }
    
    const auto& ops = select_backend("avx512");
    if (ops.cosine_similarity == select_backend("scalar").cosine_similarity) {
        state.SkipWithError("AVX-512 not available");
        return;
    }
    
    for (auto _ : state) {
        float result = ops.cosine_similarity(a, b);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
}

// Batch processing benchmark
static void BM_BatchL2_Auto(benchmark::State& state) {
    const int dim = state.range(0);
    const int batch_size = 100;
    
    std::vector<std::vector<float>> queries;
    std::vector<std::vector<float>> database;
    
    for (int i = 0; i < batch_size; ++i) {
        queries.push_back(generate_random_vector(dim, i));
        database.push_back(generate_random_vector(dim, i + 1000));
    }
    
    const auto& ops = select_backend_auto();
    
    for (auto _ : state) {
        float total = 0;
        for (int i = 0; i < batch_size; ++i) {
            float result = ops.l2_sq(queries[i], database[i]);
            total += result;
        }
        benchmark::DoNotOptimize(total);
    }
    
    state.SetBytesProcessed(state.iterations() * batch_size * dim * sizeof(float) * 2);
    state.SetItemsProcessed(state.iterations() * batch_size);
}

// Register benchmarks for all dimensions
#define REGISTER_BENCHMARK_DIM(BM_NAME) \
    for (int dim : dimensions) { \
        benchmark::RegisterBenchmark(#BM_NAME, BM_NAME)->Arg(dim); \
    }

// Register all benchmarks
void RegisterSIMDBenchmarks() {
    REGISTER_BENCHMARK_DIM(BM_L2_Scalar)
    REGISTER_BENCHMARK_DIM(BM_L2_AVX2)
    REGISTER_BENCHMARK_DIM(BM_L2_AVX512)
    REGISTER_BENCHMARK_DIM(BM_L2_Auto)
    
    REGISTER_BENCHMARK_DIM(BM_InnerProduct_Scalar)
    REGISTER_BENCHMARK_DIM(BM_InnerProduct_AVX2)
    REGISTER_BENCHMARK_DIM(BM_InnerProduct_AVX512)
    
    REGISTER_BENCHMARK_DIM(BM_Cosine_Scalar)
    REGISTER_BENCHMARK_DIM(BM_Cosine_AVX2)
    REGISTER_BENCHMARK_DIM(BM_Cosine_AVX512)
    
    REGISTER_BENCHMARK_DIM(BM_BatchL2_Auto)
}

// Auto-register benchmarks
namespace {
    struct BenchmarkRegistrar {
        BenchmarkRegistrar() {
            RegisterSIMDBenchmarks();
        }
    } registrar;
}

BENCHMARK_MAIN();