#include <benchmark/benchmark.h>
#include <vesper/kernels/distance.hpp>
#include <vector>

using namespace vesper::kernels;

static void BenchL2Sq64(benchmark::State& state){
  std::vector<float> a(64), b(64);
  for (int i=0;i<64;++i){ a[i]=i*1.0f; b[i]=(63-i)*1.0f; }
  for (auto _ : state) {
    benchmark::DoNotOptimize(l2_sq(a, b));
  }
}
BENCHMARK(BenchL2Sq64);

static void BenchIP128(benchmark::State& state){
  std::vector<float> a(128), b(128);
  for (int i=0;i<128;++i){ a[i]=i*0.5f; b[i]=(127-i)*0.25f; }
  for (auto _ : state) {
    benchmark::DoNotOptimize(inner_product(a, b));
  }
}
BENCHMARK(BenchIP128);

BENCHMARK_MAIN();

