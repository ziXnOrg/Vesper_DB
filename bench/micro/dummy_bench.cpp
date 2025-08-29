#include <benchmark/benchmark.h>
#include <vesper/collection.hpp>

static void BM_CreateSearchParams(benchmark::State& state) {
  for (auto _ : state) {
    vesper::search_params p{};
    benchmark::DoNotOptimize(p);
  }
}
BENCHMARK(BM_CreateSearchParams);

BENCHMARK_MAIN();

