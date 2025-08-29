#include <benchmark/benchmark.h>
#include "tests/support/replayer_payload.hpp"
#include <vector>

using namespace test_support;

static void BenchApplyUpsert64(benchmark::State& state){
  std::vector<float> v(64); for (int i=0;i<64;++i) v[i]=i*0.5f;
  auto payload = make_upsert(123, v, {{"k","v"}});
  ToyIndex idx; idx.reserve(1);
  for (auto _ : state) {
    apply_frame_payload(payload, idx);
    benchmark::DoNotOptimize(idx.size());
  }
}
BENCHMARK(BenchApplyUpsert64);

static void BenchApplyUpsert128(benchmark::State& state){
  std::vector<float> v(128); for (int i=0;i<128;++i) v[i]=i*0.25f;
  auto payload = make_upsert(456, v, {{"a","b"}});
  ToyIndex idx; idx.reserve(1);
  for (auto _ : state) {
    apply_frame_payload(payload, idx);
    benchmark::DoNotOptimize(idx.size());
  }
}
BENCHMARK(BenchApplyUpsert128);

BENCHMARK_MAIN();

