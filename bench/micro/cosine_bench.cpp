#include <benchmark/benchmark.h>
#include <vesper/kernels/distance.hpp>
#include <vector>

using namespace vesper::kernels;

static void BenchCosine64(benchmark::State& state){
  std::vector<float> a(64), b(64);
  for (int i=0;i<64;++i){ a[i]=i*0.5f; b[i]=(63-i)*0.25f; }
  for (auto _ : state) {
    benchmark::DoNotOptimize(cosine_similarity(a, b));
  }
}
BENCHMARK(BenchCosine64);

static void BenchCosine128(benchmark::State& state){
  std::vector<float> a(128), b(128);
  for (int i=0;i<128;++i){ a[i]=i*0.25f; b[i]=(127-i)*0.125f; }
  for (auto _ : state) {
    benchmark::DoNotOptimize(cosine_similarity(a, b));
  }
}
BENCHMARK(BenchCosine128);

static void BenchCosine256(benchmark::State& state){
  std::vector<float> a(256), b(256);
  for (int i=0;i<256;++i){ a[i]=i*0.1f; b[i]=(255-i)*0.05f; }
  for (auto _ : state) {
    benchmark::DoNotOptimize(cosine_similarity(a, b));
  }
}
BENCHMARK(BenchCosine256);

BENCHMARK_MAIN();

