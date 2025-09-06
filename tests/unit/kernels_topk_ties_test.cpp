#include <catch2/catch_all.hpp>
#include <vesper/kernels/batch_distances.hpp>
#include <vesper/index/aligned_buffer.hpp>
#include <vector>

using namespace vesper;
using namespace vesper::kernels;
using vesper::index::AlignedCentroidBuffer;

TEST_CASE("fused_topk stable tie-breaking: L2 distances", "[kernels][topk][ties][l2]"){
  const std::size_t dim = 2; const std::uint32_t k = 3;
  AlignedCentroidBuffer C(k, dim);
  // Two centroids equidistant from the query: (1,0) and (-1,0) w.r.t. (0,0)
  C.set_centroid(0, std::vector<float>{ 1.f, 0.f});
  C.set_centroid(1, std::vector<float>{-1.f, 0.f});
  C.set_centroid(2, std::vector<float>{ 0.f, 2.f}); // farther
  std::vector<float> q{0.f, 0.f};
  std::vector<std::uint32_t> idx(2); std::vector<float> vals(2);
  find_nearest_centroids_batch_fused(DistanceOp::L2, q.data(), 1, C, 2, idx.data(), vals.data());
  // Expect indices in ascending order for equal distances
  REQUIRE(idx[0] == 0);
  REQUIRE(idx[1] == 1);
}

TEST_CASE("fused_topk stable tie-breaking: InnerProduct", "[kernels][topk][ties][ip]"){
  const std::size_t dim = 2; const std::uint32_t k = 3;
  AlignedCentroidBuffer C(k, dim);
  // Duplicate centroid vectors to force equal inner products
  C.set_centroid(0, std::vector<float>{1.f, 0.f});
  C.set_centroid(1, std::vector<float>{1.f, 0.f});
  C.set_centroid(2, std::vector<float>{0.f, 1.f});
  std::vector<float> q{1.f, 0.f};
  std::vector<std::uint32_t> idx(2); std::vector<float> vals(2);
  find_nearest_centroids_batch_fused(DistanceOp::InnerProduct, q.data(), 1, C, 2, idx.data(), vals.data());
  // Equal scores tie-break by smaller index
  REQUIRE(idx[0] == 0);
  REQUIRE(idx[1] == 1);
}

TEST_CASE("fused_topk stable tie-breaking: Cosine ops", "[kernels][topk][ties][cosine]"){
  const std::size_t dim = 3; const std::uint32_t k = 3;
  AlignedCentroidBuffer C(k, dim);
  // Two identical direction centroids for zero cosine distance / max similarity
  C.set_centroid(0, std::vector<float>{1.f, 0.f, 0.f});
  C.set_centroid(1, std::vector<float>{1.f, 0.f, 0.f});
  C.set_centroid(2, std::vector<float>{0.f, 1.f, 0.f});
  std::vector<float> q{1.f, 0.f, 0.f};

  // CosineSimilarity: descending, ties by smaller index
  {
    std::vector<std::uint32_t> idx(2); std::vector<float> vals(2);
    find_nearest_centroids_batch_fused(DistanceOp::CosineSimilarity, q.data(), 1, C, 2, idx.data(), vals.data());
    REQUIRE(idx[0] == 0);
    REQUIRE(idx[1] == 1);
  }
  // CosineDistance: ascending, ties by smaller index
  {
    std::vector<std::uint32_t> idx(2); std::vector<float> vals(2);
    find_nearest_centroids_batch_fused(DistanceOp::CosineDistance, q.data(), 1, C, 2, idx.data(), vals.data());
    REQUIRE(idx[0] == 0);
    REQUIRE(idx[1] == 1);
  }
}

