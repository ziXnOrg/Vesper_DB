#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include <cstring>

#include "vesper/kernels/dispatch.hpp"
#include "vesper/kernels/backends/scalar.hpp"

#ifdef __x86_64__
#include "vesper/kernels/backends/avx2.hpp"
#include "vesper/kernels/backends/avx512.hpp"
#endif

namespace {

/** \brief Generate random vectors for benchmarking.
 *  Uses fixed seed for reproducibility across runs.
 */
class VectorFixture : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) override {
        const std::size_t dim = state.range(0);
        const std::uint32_t seed = 42;
        
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        a_.resize(dim);
        b_.resize(dim);
        
        for (std::size_t i = 0; i < dim; ++i) {
            a_[i] = dist(gen);
            b_[i] = dist(gen);
        }
        
        // Normalize for cosine benchmarks
        float norm_a = 0.0f, norm_b = 0.0f;
        for (std::size_t i = 0; i < dim; ++i) {
            norm_a += a_[i] * a_[i];
            norm_b += b_[i] * b_[i];
        }
        norm_a = std::sqrt(norm_a);
        norm_b = std::sqrt(norm_b);
        
        a_normalized_.resize(dim);
        b_normalized_.resize(dim);
        for (std::size_t i = 0; i < dim; ++i) {
            a_normalized_[i] = a_[i] / norm_a;
            b_normalized_[i] = b_[i] / norm_b;
        }
    }
    
protected:
    std::vector<float> a_;
    std::vector<float> b_;
    std::vector<float> a_normalized_;
    std::vector<float> b_normalized_;
};

// L2 Distance Benchmarks
BENCHMARK_DEFINE_F(VectorFixture, L2_Scalar)(benchmark::State& state) {
    const auto& ops = vesper::kernels::get_scalar_ops();
    for (auto _ : state) {
        benchmark::DoNotOptimize(ops.l2_sq(a_, b_));
    }
    state.SetBytesProcessed(state.iterations() * a_.size() * sizeof(float) * 2);
    state.SetItemsProcessed(state.iterations() * a_.size());
}

#ifdef __x86_64__
BENCHMARK_DEFINE_F(VectorFixture, L2_AVX2)(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(vesper::kernels::avx2_l2_sq(a_, b_));
    }
    state.SetBytesProcessed(state.iterations() * a_.size() * sizeof(float) * 2);
    state.SetItemsProcessed(state.iterations() * a_.size());
}

BENCHMARK_DEFINE_F(VectorFixture, L2_AVX512)(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(vesper::kernels::avx512_l2_sq(a_, b_));
    }
    state.SetBytesProcessed(state.iterations() * a_.size() * sizeof(float) * 2);
    state.SetItemsProcessed(state.iterations() * a_.size());
}
#endif

// Inner Product Benchmarks
BENCHMARK_DEFINE_F(VectorFixture, IP_Scalar)(benchmark::State& state) {
    const auto& ops = vesper::kernels::get_scalar_ops();
    for (auto _ : state) {
        benchmark::DoNotOptimize(ops.inner_product(a_, b_));
    }
    state.SetBytesProcessed(state.iterations() * a_.size() * sizeof(float) * 2);
    state.SetItemsProcessed(state.iterations() * a_.size());
}

#ifdef __x86_64__
BENCHMARK_DEFINE_F(VectorFixture, IP_AVX2)(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(vesper::kernels::avx2_inner_product(a_, b_));
    }
    state.SetBytesProcessed(state.iterations() * a_.size() * sizeof(float) * 2);
    state.SetItemsProcessed(state.iterations() * a_.size());
}

BENCHMARK_DEFINE_F(VectorFixture, IP_AVX512)(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(vesper::kernels::avx512_inner_product(a_, b_));
    }
    state.SetBytesProcessed(state.iterations() * a_.size() * sizeof(float) * 2);
    state.SetItemsProcessed(state.iterations() * a_.size());
}
#endif

// Cosine Similarity Benchmarks
BENCHMARK_DEFINE_F(VectorFixture, Cosine_Scalar)(benchmark::State& state) {
    const auto& ops = vesper::kernels::get_scalar_ops();
    for (auto _ : state) {
        benchmark::DoNotOptimize(ops.cosine_similarity(a_, b_));
    }
    state.SetBytesProcessed(state.iterations() * a_.size() * sizeof(float) * 2);
    state.SetItemsProcessed(state.iterations() * a_.size());
}

#ifdef __x86_64__
BENCHMARK_DEFINE_F(VectorFixture, Cosine_AVX2)(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(vesper::kernels::avx2_cosine_similarity(a_, b_));
    }
    state.SetBytesProcessed(state.iterations() * a_.size() * sizeof(float) * 2);
    state.SetItemsProcessed(state.iterations() * a_.size());
}

BENCHMARK_DEFINE_F(VectorFixture, Cosine_AVX512)(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(vesper::kernels::avx512_cosine_similarity(a_, b_));
    }
    state.SetBytesProcessed(state.iterations() * a_.size() * sizeof(float) * 2);
    state.SetItemsProcessed(state.iterations() * a_.size());
}
#endif

// Auto-dispatch benchmark
BENCHMARK_DEFINE_F(VectorFixture, AutoDispatch_L2)(benchmark::State& state) {
    const auto& ops = vesper::kernels::select_backend_auto();
    for (auto _ : state) {
        benchmark::DoNotOptimize(ops.l2_sq(a_, b_));
    }
    state.SetBytesProcessed(state.iterations() * a_.size() * sizeof(float) * 2);
    state.SetItemsProcessed(state.iterations() * a_.size());
}

} // anonymous namespace

// Register benchmarks with typical dimensions
// Small dimensions (< cache line)
BENCHMARK_REGISTER_F(VectorFixture, L2_Scalar)->Arg(8)->Arg(16);
BENCHMARK_REGISTER_F(VectorFixture, IP_Scalar)->Arg(8)->Arg(16);
BENCHMARK_REGISTER_F(VectorFixture, Cosine_Scalar)->Arg(8)->Arg(16);

#ifdef __x86_64__
BENCHMARK_REGISTER_F(VectorFixture, L2_AVX2)->Arg(8)->Arg(16);
BENCHMARK_REGISTER_F(VectorFixture, L2_AVX512)->Arg(8)->Arg(16);
BENCHMARK_REGISTER_F(VectorFixture, IP_AVX2)->Arg(8)->Arg(16);
BENCHMARK_REGISTER_F(VectorFixture, IP_AVX512)->Arg(8)->Arg(16);
BENCHMARK_REGISTER_F(VectorFixture, Cosine_AVX2)->Arg(8)->Arg(16);
BENCHMARK_REGISTER_F(VectorFixture, Cosine_AVX512)->Arg(8)->Arg(16);
#endif

// Medium dimensions (typical embeddings)
BENCHMARK_REGISTER_F(VectorFixture, L2_Scalar)->Arg(128)->Arg(256)->Arg(384);
BENCHMARK_REGISTER_F(VectorFixture, IP_Scalar)->Arg(128)->Arg(256)->Arg(384);
BENCHMARK_REGISTER_F(VectorFixture, Cosine_Scalar)->Arg(128)->Arg(256)->Arg(384);

#ifdef __x86_64__
BENCHMARK_REGISTER_F(VectorFixture, L2_AVX2)->Arg(128)->Arg(256)->Arg(384);
BENCHMARK_REGISTER_F(VectorFixture, L2_AVX512)->Arg(128)->Arg(256)->Arg(384);
BENCHMARK_REGISTER_F(VectorFixture, IP_AVX2)->Arg(128)->Arg(256)->Arg(384);
BENCHMARK_REGISTER_F(VectorFixture, IP_AVX512)->Arg(128)->Arg(256)->Arg(384);
BENCHMARK_REGISTER_F(VectorFixture, Cosine_AVX2)->Arg(128)->Arg(256)->Arg(384);
BENCHMARK_REGISTER_F(VectorFixture, Cosine_AVX512)->Arg(128)->Arg(256)->Arg(384);
#endif

// Large dimensions (LLM embeddings)
BENCHMARK_REGISTER_F(VectorFixture, L2_Scalar)->Arg(768)->Arg(1024)->Arg(1536);
BENCHMARK_REGISTER_F(VectorFixture, IP_Scalar)->Arg(768)->Arg(1024)->Arg(1536);
BENCHMARK_REGISTER_F(VectorFixture, Cosine_Scalar)->Arg(768)->Arg(1024)->Arg(1536);

#ifdef __x86_64__
BENCHMARK_REGISTER_F(VectorFixture, L2_AVX2)->Arg(768)->Arg(1024)->Arg(1536);
BENCHMARK_REGISTER_F(VectorFixture, L2_AVX512)->Arg(768)->Arg(1024)->Arg(1536);
BENCHMARK_REGISTER_F(VectorFixture, IP_AVX2)->Arg(768)->Arg(1024)->Arg(1536);
BENCHMARK_REGISTER_F(VectorFixture, IP_AVX512)->Arg(768)->Arg(1024)->Arg(1536);
BENCHMARK_REGISTER_F(VectorFixture, Cosine_AVX2)->Arg(768)->Arg(1024)->Arg(1536);
BENCHMARK_REGISTER_F(VectorFixture, Cosine_AVX512)->Arg(768)->Arg(1024)->Arg(1536);
#endif

// Auto-dispatch across all dimensions
BENCHMARK_REGISTER_F(VectorFixture, AutoDispatch_L2)
    ->Arg(8)->Arg(16)->Arg(128)->Arg(256)->Arg(384)->Arg(768)->Arg(1024)->Arg(1536);

BENCHMARK_MAIN();