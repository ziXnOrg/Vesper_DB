#include <catch2/catch_all.hpp>
#include <vesper/kernels/batch_distances.hpp>
#include <vesper/index/aligned_buffer.hpp>
#include <vector>

using namespace vesper;
using namespace vesper::kernels;
using vesper::index::AlignedCentroidBuffer;

TEST_CASE("materialized_topk stable tie-breaking: L2 distances", "[kernels][topk][ties][materialized]"){
  const std::size_t dim = 2; const std::uint32_t k = 3;
  AlignedCentroidBuffer C(k, dim);
  // Two centroids equidistant from the query: (1,0) and (-1,0) w.r.t. (0,0)
  C.set_centroid(0, std::vector<float>{ 1.f, 0.f});
  C.set_centroid(1, std::vector<float>{-1.f, 0.f});
  C.set_centroid(2, std::vector<float>{ 0.f, 2.f}); // farther
  std::vector<float> q{0.f, 0.f};
  std::vector<std::uint32_t> idx(2); std::vector<float> vals(2);
  find_nearest_centroids_batch(q.data(), C, 1, 2, idx.data(), vals.data());
  // Expect indices in ascending order for equal distances
  REQUIRE(idx[0] == 0);
  REQUIRE(idx[1] == 1);
}

