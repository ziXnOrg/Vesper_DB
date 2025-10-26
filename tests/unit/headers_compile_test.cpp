#include <vesper/collection.hpp>
#include <vesper/filter_expr.hpp>
#include <vesper/error.hpp>
#include <catch2/catch_all.hpp>

TEST_CASE("headers compile and basic types exist", "[headers]") {
  vesper::search_params p{};
  REQUIRE(p.k == 10);
}

