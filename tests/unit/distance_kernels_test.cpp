#include <catch2/catch_all.hpp>
#include <vesper/kernels/distance.hpp>

using namespace vesper::kernels;

TEST_CASE("distance kernels known values", "[kernels][distance]") {
using Catch::Approx;

  float a1[1]{3.0f}; float b1[1]{-1.0f};
  REQUIRE(l2_sq(a1, b1) == Approx(16.0f));
  REQUIRE(inner_product(a1, b1) == Approx(-3.0f));

  float a2[2]{1.0f,2.0f}; float b2[2]{4.0f,6.0f};
  REQUIRE(l2_sq(a2, b2) == Approx((1-4)*(1-4) + (2-6)*(2-6)));
  REQUIRE(inner_product(a2, b2) == Approx(1*4 + 2*6));

  float a16[16]; float b16[16];
  for (int i=0;i<16;++i){ a16[i]=i*1.0f; b16[i]= (16-i)*1.0f; }
  REQUIRE(l2_sq(a16, b16) >= 0.0f);
  REQUIRE(inner_product(a16, a16) >= inner_product(a16, b16));
}

TEST_CASE("distance kernels properties", "[kernels][distance]") {
  using Catch::Approx;
  float a[4]{0,0,0,0};
  REQUIRE(l2_sq(a, a) == Approx(0.0f));
  float x[2]{2.0f, -5.0f}; float y[2]{-5.0f, 2.0f};
  REQUIRE(inner_product(x, y) == Approx(inner_product(y, x)));
}

