#include <catch2/catch_all.hpp>
#include <vesper/filter_eval.hpp>
#include <vesper/filter_expr.hpp>

using namespace vesper;

static filter_expr make_term(const std::string& f, const std::string& v){ return filter_expr{ filter_expr::and_t{ { filter_expr{ term{f,v} } } } }; }

TEST_CASE("filter eval basic semantics", "[filter]"){
  filter_eval::tags_t tags1{{"color","red"}, {"shape","circle"}};
  filter_eval::nums_t nums1{{"price", 9.99}};
  filter_eval::tags_t tags2{{"color","blue"}};
  filter_eval::nums_t nums2{{"price", 5.0}};

  filter_expr t_red{ term{"color","red"} };
  filter_expr t_blue{ term{"color","blue"} };
  filter_expr r_mid{ range{"price", 6.0, 10.0} };
  filter_expr both{ filter_expr::and_t{ {t_red, r_mid} } };
  filter_expr any{ filter_expr::or_t{ {t_red, t_blue} } };
  filter_expr none{ filter_expr::not_t{ {t_red, t_blue} } };
  filter_expr empty_and{ filter_expr::and_t{ { } } };
  filter_expr empty_or{ filter_expr::or_t{ { } } };
  filter_expr empty_not{ filter_expr::not_t{ { } } };

  REQUIRE(filter_eval::matches(t_red, tags1, nums1));
  REQUIRE_FALSE(filter_eval::matches(t_red, tags2, nums2));
  REQUIRE(filter_eval::matches(r_mid, tags1, nums1));
  REQUIRE_FALSE(filter_eval::matches(r_mid, tags2, nums2));
  REQUIRE(filter_eval::matches(both, tags1, nums1));
  REQUIRE(filter_eval::matches(any, tags1, nums1));
  REQUIRE_FALSE(filter_eval::matches(none, tags1, nums1));
  REQUIRE(filter_eval::matches(empty_and, tags2, nums2));
  REQUIRE_FALSE(filter_eval::matches(empty_or, tags2, nums2));
  REQUIRE(filter_eval::matches(empty_not, tags2, nums2));
}

