#include <vesper/collection.hpp>
#include <catch2/catch_all.hpp>

TEST_CASE("no-op open returns collection", "[smoke]") {
  auto c = vesper::collection::open("/tmp/vesper_coll_smoke");
  REQUIRE(c.has_value());
}

