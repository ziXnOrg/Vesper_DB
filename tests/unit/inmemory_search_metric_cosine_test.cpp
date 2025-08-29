#include <catch2/catch_all.hpp>
#include <vesper/collection.hpp>
#include <vesper/filter_expr.hpp>

using namespace vesper;

TEST_CASE("search metric selection: cosine vs l2/ip", "[search][metric]") {
  auto coll = collection::open("/tmp/metric_cosine");
  REQUIRE(coll.has_value());

  // Define vectors: v1 parallel to q, v2 orthogonal to q, v3 anti-parallel to q
  float q[2]{1.0f, 0.0f};
  float v1[2]{2.0f, 0.0f};   // cos=1, l2 small, ip positive
  float v2[2]{0.0f, 3.0f};   // cos=0, l2 larger, ip 0
  float v3[2]{-4.0f, 0.0f};  // cos=-1, l2 large, ip negative

  REQUIRE(coll->insert(1, v1, 2).has_value());
  REQUIRE(coll->insert(2, v2, 2).has_value());
  REQUIRE(coll->insert(3, v3, 2).has_value());

  search_params p; p.k = 3; p.target_recall = 1.0f;

  // Cosine: expect id order [1,2,3] since 1.0 > 0 > -1, but we sort by distance (1-cos) ascending: [1,2,3]
  p.metric = "cosine";
  {
    auto res = coll->search(q, 2, p, nullptr);
    REQUIRE(res.has_value());
    REQUIRE(res->size() == 3);
    REQUIRE((*res)[0].id == 1);
    REQUIRE((*res)[1].id == 2);
    REQUIRE((*res)[2].id == 3);
  }

  // L2: nearest is v1 (distance 1), then v2 (sqrt(10)), then v3 (5)
  p.metric = "l2";
  {
    auto res = coll->search(q, 2, p, nullptr);
    REQUIRE(res.has_value());
    REQUIRE(res->size() == 3);
    REQUIRE((*res)[0].id == 1);
    REQUIRE((*res)[1].id == 2);
    REQUIRE((*res)[2].id == 3);
  }

  // IP: we use -inner_product as distance so highest IP ranked first: v1 (2), v2 (0), v3 (-4)
  p.metric = "ip";
  {
    auto res = coll->search(q, 2, p, nullptr);
    REQUIRE(res.has_value());
    REQUIRE(res->size() == 3);
    REQUIRE((*res)[0].id == 1);
    REQUIRE((*res)[1].id == 2);
    REQUIRE((*res)[2].id == 3);
  }
}

