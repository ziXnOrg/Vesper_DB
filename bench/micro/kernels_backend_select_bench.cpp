#include <benchmark/benchmark.h>
#include <vesper/kernels/dispatch.hpp>
#include <vesper/kernels/backends/scalar.hpp>
#include <vesper/kernels/distance.hpp>
#include <vector>

using namespace vesper::kernels;

static void Prep(std::vector<float>& a, std::vector<float>& b, int n){ a.resize(n); b.resize(n); for(int i=0;i<n;++i){ a[i]=i*0.25f; b[i]=(n-1-i)*0.5f; } }

static void BenchL2_Scalar_vs_StubAVX2(benchmark::State& state){
  int n = static_cast<int>(state.range(0));
  std::vector<float> a, b; Prep(a,b,n);
  const auto& scalar = select_backend("scalar");
  const auto& avx2 = select_backend("stub-avx2");
  for (auto _ : state) {
    benchmark::DoNotOptimize(scalar.l2_sq(a,b));
    benchmark::DoNotOptimize(avx2.l2_sq(a,b));
    benchmark::DoNotOptimize(l2_sq(a,b)); // direct
  }
}
BENCHMARK(BenchL2_Scalar_vs_StubAVX2)->Arg(64);

static void BenchIP_Scalar_vs_StubAVX2(benchmark::State& state){
  int n = static_cast<int>(state.range(0));
  std::vector<float> a, b; Prep(a,b,n);
  const auto& scalar = select_backend("scalar");
  const auto& avx2 = select_backend("stub-avx2");
  for (auto _ : state) {
    benchmark::DoNotOptimize(scalar.inner_product(a,b));
    benchmark::DoNotOptimize(avx2.inner_product(a,b));
    benchmark::DoNotOptimize(inner_product(a,b));
  }
}
BENCHMARK(BenchIP_Scalar_vs_StubAVX2)->Arg(128);

static void BenchCos_Scalar_vs_StubNEON(benchmark::State& state){
  int n = static_cast<int>(state.range(0));
  std::vector<float> a, b; Prep(a,b,n);
  const auto& scalar = select_backend("scalar");
  const auto& neon = select_backend("stub-neon");
  for (auto _ : state) {
    benchmark::DoNotOptimize(scalar.cosine_similarity(a,b));
    benchmark::DoNotOptimize(neon.cosine_similarity(a,b));
    benchmark::DoNotOptimize(cosine_similarity(a,b));
  }
}
BENCHMARK(BenchCos_Scalar_vs_StubNEON)->Arg(256);

BENCHMARK_MAIN();

