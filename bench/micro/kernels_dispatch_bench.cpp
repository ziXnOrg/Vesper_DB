#include <benchmark/benchmark.h>
#include <vesper/kernels/dispatch.hpp>
#include <vesper/kernels/backends/scalar.hpp>
#include <vesper/kernels/distance.hpp>
#include <vector>

using namespace vesper::kernels;

static void BenchL2_Dispatch64(benchmark::State& state){
  std::vector<float> a(64), b(64);
  for (int i=0;i<64;++i){ a[i]=i*0.25f; b[i]=(63-i)*0.5f; }
  const auto& ops = select_backend("scalar");
  for (auto _ : state) { benchmark::DoNotOptimize(ops.l2_sq(a,b)); }
}
BENCHMARK(BenchL2_Dispatch64);

static void BenchL2_Direct64(benchmark::State& state){
  std::vector<float> a(64), b(64);
  for (int i=0;i<64;++i){ a[i]=i*0.25f; b[i]=(63-i)*0.5f; }
  for (auto _ : state) { benchmark::DoNotOptimize(l2_sq(a,b)); }
}
BENCHMARK(BenchL2_Direct64);

static void BenchIP_Dispatch128(benchmark::State& state){
  std::vector<float> a(128), b(128);
  for (int i=0;i<128;++i){ a[i]=i*0.1f; b[i]=(127-i)*0.2f; }
  const auto& ops = select_backend("scalar");
  for (auto _ : state) { benchmark::DoNotOptimize(ops.inner_product(a,b)); }
}
BENCHMARK(BenchIP_Dispatch128);

static void BenchIP_Direct128(benchmark::State& state){
  std::vector<float> a(128), b(128);
  for (int i=0;i<128;++i){ a[i]=i*0.1f; b[i]=(127-i)*0.2f; }
  for (auto _ : state) { benchmark::DoNotOptimize(inner_product(a,b)); }
}
BENCHMARK(BenchIP_Direct128);

static void BenchCos_Dispatch256(benchmark::State& state){
  std::vector<float> a(256), b(256);
  for (int i=0;i<256;++i){ a[i]=i*0.05f; b[i]=(255-i)*0.05f; }
  const auto& ops = select_backend("scalar");
  for (auto _ : state) { benchmark::DoNotOptimize(ops.cosine_similarity(a,b)); }
}
BENCHMARK(BenchCos_Dispatch256);

static void BenchCos_Direct256(benchmark::State& state){
  std::vector<float> a(256), b(256);
  for (int i=0;i<256;++i){ a[i]=i*0.05f; b[i]=(255-i)*0.05f; }
  for (auto _ : state) { benchmark::DoNotOptimize(cosine_similarity(a,b)); }
}
BENCHMARK(BenchCos_Direct256);

BENCHMARK_MAIN();

