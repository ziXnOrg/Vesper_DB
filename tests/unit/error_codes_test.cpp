#include <vesper/error.hpp>
#include <catch2/catch_all.hpp>

TEST_CASE("error codes stable subset", "[errors]") {
  using vesper::core::error_code;
  REQUIRE(static_cast<unsigned>(error_code::ok) == 0u);
  REQUIRE(static_cast<unsigned>(error_code::io_failed) == 1001u);
  REQUIRE(static_cast<unsigned>(error_code::internal) == 9001u);
}

