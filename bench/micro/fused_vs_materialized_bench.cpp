#include <benchmark/benchmark.h>
#include <vesper/kernels/batch_distances.hpp>
#include <vesper/index/aligned_buffer.hpp>
#include <random>
#include <vector>
#include <cstdint>

using namespace vesper;
using namespace vesper::kernels;
using vesper::index::AlignedCentroidBuffer;

namespace {

std::mt19937 rng(12345);
float frand() {
  static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  return dist(rng);
}

void make_data(std::size_t nQ, std::uint32_t nC, std::size_t dim,
               std::vector<float>& queries, AlignedCentroidBuffer& C) {
  queries.resize(nQ * dim);
  for (auto& v : queries) v = frand();
  for (std::uint32_t c=0;c<nC;++c) {
    std::vector<float> row(dim);
    for (std::size_t d=0; d<dim; ++d) row[d] = frand();
    C.set_centroid(c, row);
  }
}

enum class OpId : int { L2=0, IP=1, CosDist=2, CosSim=3 };

static void BM_Fused(benchmark::State& state) {
  const std::uint32_t k   = static_cast<std::uint32_t>(state.range(0));
  const std::size_t   dim = static_cast<std::size_t>(state.range(1));
  const std::uint32_t nC  = static_cast<std::uint32_t>(state.range(2));
  const OpId op_id        = static_cast<OpId>(state.range(3));
  const std::size_t nQ    = 64;

  AlignedCentroidBuffer C(nC, dim);
  std::vector<float> queries;
  make_data(nQ, nC, dim, queries, C);
  std::vector<std::uint32_t> idx(nQ * k);
  std::vector<float> vals(nQ * k);

  DistanceOp op = DistanceOp::L2;
  switch (op_id) {
    case OpId::L2:      op = DistanceOp::L2; break;
    case OpId::IP:      op = DistanceOp::InnerProduct; break;
    case OpId::CosDist: op = DistanceOp::CosineDistance; break;
    case OpId::CosSim:  op = DistanceOp::CosineSimilarity; break;
  }

  for (auto _ : state) {
    find_nearest_centroids_batch_fused(op, queries.data(), nQ, C, k, idx.data(), vals.data());
    benchmark::DoNotOptimize(idx.data());
    benchmark::DoNotOptimize(vals.data());
  }

  state.counters["bytes_out"] = benchmark::Counter(double(nQ) * k * (sizeof(float)+sizeof(std::uint32_t)), benchmark::Counter::kDefaults, benchmark::Counter::OneK::kIs1024);
}

static void BM_Materialized(benchmark::State& state) {
  const std::uint32_t k   = static_cast<std::uint32_t>(state.range(0));
  const std::size_t   dim = static_cast<std::size_t>(state.range(1));
  const std::uint32_t nC  = static_cast<std::uint32_t>(state.range(2));
  const OpId op_id        = static_cast<OpId>(state.range(3));
  const std::size_t nQ    = 64;

  AlignedCentroidBuffer C(nC, dim);
  std::vector<float> queries;
  make_data(nQ, nC, dim, queries, C);
  std::vector<std::uint32_t> idx(nQ * k);
  std::vector<float> vals(nQ * k);
  std::vector<float> all(nQ * nC);

  DistanceOp op = DistanceOp::L2;
  switch (op_id) {
    case OpId::L2:      op = DistanceOp::L2; break;
    case OpId::IP:      op = DistanceOp::InnerProduct; break;
    case OpId::CosDist: op = DistanceOp::CosineDistance; break;
    case OpId::CosSim:  op = DistanceOp::CosineSimilarity; break;
  }

  for (auto _ : state) {
    distance_matrix(op, queries.data(), nQ, C, dim, all.data());
    for (std::size_t q=0; q<nQ; ++q) {
      std::vector<std::uint32_t> order(nC);
      std::iota(order.begin(), order.end(), 0);
      const bool sim = (op == DistanceOp::InnerProduct || op == DistanceOp::CosineSimilarity);
      auto* row = all.data() + q * nC;
      if (!sim) {
        std::partial_sort(order.begin(), order.begin()+k, order.end(), [&](auto a, auto b){ return row[a] < row[b]; });
      } else {
        std::partial_sort(order.begin(), order.begin()+k, order.end(), [&](auto a, auto b){ return row[a] > row[b]; });
      }
      for (std::uint32_t i=0;i<k;++i){ idx[q*k+i]=order[i]; vals[q*k+i]=row[order[i]]; }
    }
    benchmark::DoNotOptimize(idx.data());
    benchmark::DoNotOptimize(vals.data());
  }

  state.counters["bytes_mat"] = benchmark::Counter(double(nQ) * nC * sizeof(float), benchmark::Counter::kDefaults, benchmark::Counter::OneK::kIs1024);
}

void register_all() {
  const int ks[] = {1,10,50,100};
  const int dims[] = {64,128,256,512};
  const int ncs[] = {1000,10000};
  const int ops[] = {int(OpId::L2), int(OpId::IP), int(OpId::CosDist), int(OpId::CosSim)};
  for (int k : ks) for (int d : dims) for (int nc : ncs) for (int op : ops) {
    BENCHMARK(BM_Fused)->Args({k,d,nc,op});
    BENCHMARK(BM_Materialized)->Args({k,d,nc,op});
  }
}

int x = (register_all(), 0);

} // namespace



BENCHMARK_MAIN();
