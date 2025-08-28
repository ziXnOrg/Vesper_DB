#include <catch2/catch_all.hpp>
#include <vesper/collection.hpp>
#include <vesper/filter_expr.hpp>

using namespace vesper;

TEST_CASE("in-memory search honors filter_expr", "[search][filter]") {
  auto c = collection::open("/tmp/test");
  REQUIRE(c.has_value());

  float v1[2]{0.0f, 0.0f};
  float v2[2]{1.0f, 1.0f};
  float q[2]{0.0f, 0.0f};

  REQUIRE(c->insert(1, v1, 2).has_value());
  REQUIRE(c->insert(2, v2, 2).has_value());

  // Build filters: color == red selects only id 1
  filter_expr t_red{ term{"color","red"} };
  // No tags set yet => nobody matches
  {
    search_params p; p.k = 10; p.metric = "l2";
    auto res = c->search(q, 2, p, &t_red);
    REQUIRE(res.has_value());
    REQUIRE(res->empty());
  }
}

