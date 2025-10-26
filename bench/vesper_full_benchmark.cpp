#include <benchmark/benchmark.h>

#include "vesper/index/index_manager.hpp"
#include <random>

using namespace vesper;
using namespace vesper::index;

namespace {
std::vector<float> gen(std::size_t n, std::size_t d, std::uint32_t seed=42){
  std::vector<float> v(n*d); std::mt19937 g(seed); std::normal_distribution<float> dist(0,1);
  for(auto &x: v) x = dist(g); return v;
}
}

struct BenchState {
  std::size_t dim{64};
  std::size_t n{100000};
  std::size_t nq{1000};
  std::vector<float> base;
  std::vector<float> queries;
  std::unique_ptr<IndexManager> mgr;

  void setup(){
    base = gen(n, dim, 123);
    queries = gen(nq, dim, 321);
    mgr = std::make_unique<IndexManager>(dim);
    IndexBuildConfig cfg; cfg.strategy = SelectionStrategy::Auto;
    auto r = mgr->build(base.data(), n, cfg);
    if (!r) throw std::runtime_error("index build failed");
  }
};

static void BM_Search(benchmark::State& state) {
  BenchState bs; bs.setup();
  QueryConfig q; q.k = static_cast<std::uint32_t>(state.range(0));
  std::size_t i = 0;
  for (auto _ : state) {
    auto r = bs.mgr->search(&bs.queries[(i++ % bs.nq)*bs.dim], q);
    benchmark::DoNotOptimize(r);
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_Search)->Arg(10)->Arg(100);

BENCHMARK_MAIN();

