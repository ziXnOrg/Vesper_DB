#include <catch2/catch_test_macros.hpp>
#include <vesper/index/ivf_pq.hpp>

using namespace vesper::index;

TEST_CASE("IvfPqTrainParams defaults: coarse assigner and ANN toggles", "[ivfpq][defaults]") {
    IvfPqTrainParams p{};

    // Coarse assigner defaults
    REQUIRE(p.coarse_assigner == CoarseAssigner::KDTree);
    REQUIRE(p.use_centroid_ann == true);

    // HNSW ANN defaults (only relevant when coarse_assigner==HNSW)
    REQUIRE(p.centroid_ann_ef_search == 96);
    REQUIRE(p.centroid_ann_ef_construction == 200);
    REQUIRE(p.centroid_ann_M == 16);
    REQUIRE(p.centroid_ann_refine_k == 96);

    // KD-tree tuning defaults
    REQUIRE(p.kd_leaf_size == 256);
    REQUIRE(p.kd_batch_assign == true);
    REQUIRE(p.kd_split == KdSplitHeuristic::Variance);
}

