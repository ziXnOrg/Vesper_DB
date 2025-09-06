#include <catch2/catch_all.hpp>
#include <vesper/kernels/batch_distances.hpp>
#include <vesper/index/aligned_buffer.hpp>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>

using namespace vesper;
using namespace vesper::kernels;
using vesper::index::AlignedCentroidBuffer;
using Catch::Approx;

TEST_CASE("fused_topk_l2 matches baseline partial_sort", "[kernels][topk][l2]"){
  const std::size_t dim = 3;
  const std::uint32_t k = 5;
  AlignedCentroidBuffer C(k, dim);
  // Centroids: simple separated points
  C.set_centroid(0, std::vector<float>{0.f, 0.f, 0.f});
  C.set_centroid(1, std::vector<float>{1.f, 0.f, 0.f});
  C.set_centroid(2, std::vector<float>{0.f, 1.f, 0.f});
  C.set_centroid(3, std::vector<float>{0.f, 0.f, 1.f});
  C.set_centroid(4, std::vector<float>{1.f, 1.f, 1.f});

  // Two queries
  std::vector<float> queries{
    0.1f, 0.0f, 0.0f,  // close to centroid 0/1
    0.9f, 0.9f, 0.9f   // close to centroid 4
  };
  const std::size_t nQ = 2;
  const std::uint32_t topk = 2;

  std::vector<std::uint32_t> idx_fused(nQ * topk);
  std::vector<float> dist_fused(nQ * topk);

  find_nearest_centroids_batch_fused(DistanceOp::L2,
    queries.data(), nQ, C, topk,
    idx_fused.data(), dist_fused.data());

  // Baseline: materialize all L2 distances and partial_sort per query
  for (std::size_t q=0; q<nQ; ++q){
    std::vector<float> all(k);
    const float* qv = queries.data() + q*dim;
    for (std::uint32_t c=0;c<k;++c){
      float d = 0.f;
      auto cv = C.get_centroid(c);
      for (std::size_t d_i=0; d_i<dim; ++d_i){ float diff = qv[d_i] - cv[d_i]; d += diff*diff; }
      all[c] = d;
    }
    // Build baseline candidates and sort with epsilon+index tiebreaker
    std::vector<std::pair<float,std::uint32_t>> cand; cand.reserve(k);
    for (std::uint32_t c=0;c<k;++c) cand.emplace_back(all[c], c);
    std::sort(cand.begin(), cand.end(), [](auto& a, auto& b){
      if (std::fabs(a.first - b.first) > 1e-6f) return a.first < b.first;
      return a.second < b.second;
    });

    // Compare: indices equal and distances ascending within tolerance
    for (std::uint32_t i=0;i<topk;++i){
      REQUIRE(idx_fused[q*topk + i] == cand[i].second);
    }
    REQUIRE(dist_fused[q*topk + 0] <= dist_fused[q*topk + 1] + 1e-7f);

    // Compare distances approximately (allow tiny float error)
    for (std::uint32_t i=0;i<topk;++i){
      REQUIRE(dist_fused[q*topk + i] == Approx(cand[i].first).margin(1e-5f));
    }
  }
}



TEST_CASE("fused_topk_inner_product descending order and accuracy", "[kernels][topk][ip]"){
  const std::size_t dim = 4; // non-multiple-of-8 edge
  const std::uint32_t k = 5;
  AlignedCentroidBuffer C(k, dim);
  // Simple basis-like centroids
  C.set_centroid(0, std::vector<float>{1.f, 0.f, 0.f, 0.f});
  C.set_centroid(1, std::vector<float>{0.f, 1.f, 0.f, 0.f});
  C.set_centroid(2, std::vector<float>{0.f, 0.f, 1.f, 0.f});
  C.set_centroid(3, std::vector<float>{0.f, 0.f, 0.f, 1.f});
  C.set_centroid(4, std::vector<float>{1.f, 1.f, 1.f, 1.f});

  std::vector<float> queries{
    0.9f, 0.0f, 0.0f, 0.0f,
    0.2f, 0.2f, 0.2f, 0.2f
  };
  const std::size_t nQ = 2;
  const std::uint32_t topk = k;

  std::vector<std::uint32_t> idx(nQ*topk);
  std::vector<float> scores(nQ*topk);
  find_nearest_centroids_batch_fused(DistanceOp::InnerProduct, queries.data(), nQ, C, topk, idx.data(), scores.data());

  // For q0, centroid 0 has highest IP, q1: centroid 4 has highest IP
  REQUIRE(scores[0*topk + 0] >= scores[0*topk + 1] - 1e-7f);
  REQUIRE(scores[1*topk + 0] >= scores[1*topk + 1] - 1e-7f);
}

TEST_CASE("fused_topk_cosine_similarity descending order and accuracy", "[kernels][topk][cosine_sim]"){
  const std::size_t dim = 3;
  const std::uint32_t k = 4;
  AlignedCentroidBuffer C(k, dim);
  C.set_centroid(0, std::vector<float>{1.f, 0.f, 0.f});
  C.set_centroid(1, std::vector<float>{0.f, 1.f, 0.f});
  C.set_centroid(2, std::vector<float>{0.f, 0.f, 1.f});
  C.set_centroid(3, std::vector<float>{1.f, 1.f, 0.f});

  std::vector<float> queries{
    1.f, 0.f, 0.f,
    0.f, 1.f, 0.f
  };
  const std::size_t nQ = 2;
  const std::uint32_t topk = 2;

  std::vector<std::uint32_t> idx(nQ*topk);
  std::vector<float> scores(nQ*topk);
  find_nearest_centroids_batch_fused(DistanceOp::CosineSimilarity, queries.data(), nQ, C, topk, idx.data(), scores.data());

  // Should be descending per query
  for (std::size_t q=0;q<nQ;++q){
    REQUIRE(scores[q*topk + 0] >= scores[q*topk + 1] - 1e-7f);
    REQUIRE(scores[q*topk + 0] <= 1.f + 1e-6f);
  }
}

TEST_CASE("fused_topk_cosine_distance ascending order and accuracy", "[kernels][topk][cosine_dist]"){
  const std::size_t dim = 3;
  const std::uint32_t k = 3;
  AlignedCentroidBuffer C(k, dim);
  C.set_centroid(0, std::vector<float>{1.f, 0.f, 0.f});
  C.set_centroid(1, std::vector<float>{0.f, 1.f, 0.f});
  C.set_centroid(2, std::vector<float>{1.f, 1.f, 0.f});

  std::vector<float> queries{
    1.f, 0.f, 0.f
  };
  const std::size_t nQ = 1;
  const std::uint32_t topk = k;

  std::vector<std::uint32_t> idx(nQ*topk);
  std::vector<float> dists(nQ*topk);
  find_nearest_centroids_batch_fused(DistanceOp::CosineDistance, queries.data(), nQ, C, topk, idx.data(), dists.data());

  // Ascending distances
  for (std::uint32_t i=1;i<topk;++i){
    REQUIRE(dists[i-1] <= dists[i] + 1e-7f);
  }
  // Closest should be centroid 0 (distance ~0)
  REQUIRE(idx[0] == 0);
  REQUIRE(dists[0] == Approx(0.f).margin(1e-5f));
}

TEST_CASE("fused_topk edge cases: k=1, k=all, empty queries, non-multiple-of-8 dims", "[kernels][topk][edges]"){
  // k=1
  {
    const std::size_t dim = 5; // non-multiple-of-8
    const std::uint32_t k = 3;
    AlignedCentroidBuffer C(k, dim);
    C.set_centroid(0, std::vector<float>{0,0,0,0,0});
    C.set_centroid(1, std::vector<float>{1,0,0,0,0});
    C.set_centroid(2, std::vector<float>{0,1,0,0,0});
    std::vector<float> q{0.9f,0,0,0,0};
    std::uint32_t topk=1; std::size_t nQ=1;
    std::vector<std::uint32_t> idx(nQ*topk); std::vector<float> out(nQ*topk);
    find_nearest_centroids_batch_fused(DistanceOp::L2, q.data(), nQ, C, topk, idx.data(), out.data());
    REQUIRE(idx[0] == 1);
  }
  // k=all
  {
    const std::size_t dim = 3; const std::uint32_t k = 4;
    AlignedCentroidBuffer C(k, dim);
    C.set_centroid(0, std::vector<float>{0,0,0});
    C.set_centroid(1, std::vector<float>{1,0,0});
    C.set_centroid(2, std::vector<float>{0,1,0});
    C.set_centroid(3, std::vector<float>{0,0,1});
    std::vector<float> q{0.2f,0.0f,0.0f};
    std::uint32_t topk=k; std::size_t nQ=1;
    std::vector<std::uint32_t> idx(nQ*topk); std::vector<float> out(nQ*topk);
    find_nearest_centroids_batch_fused(DistanceOp::L2, q.data(), nQ, C, topk, idx.data(), out.data());
    for (std::uint32_t i=1;i<topk;++i) REQUIRE(out[i-1] <= out[i] + 1e-7f);
  }
  // empty queries
  {
    const std::size_t dim = 3; const std::uint32_t k = 2;
    AlignedCentroidBuffer C(k, dim);
    C.set_centroid(0, std::vector<float>{0,0,0});
    C.set_centroid(1, std::vector<float>{1,0,0});
    std::vector<float> q; // empty
    std::uint32_t topk=1; std::size_t nQ=0;
    std::vector<std::uint32_t> idx; std::vector<float> out;
    // Should not crash
    find_nearest_centroids_batch_fused(DistanceOp::L2, q.data(), nQ, C, topk, idx.data(), out.data());
  }
}
