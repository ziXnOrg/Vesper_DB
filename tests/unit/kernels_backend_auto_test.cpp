#include <catch2/catch_all.hpp>
#include <vesper/kernels/dispatch.hpp>
#include <vesper/kernels/backends/scalar.hpp>
#include <vesper/kernels/distance.hpp>
#include <random>
#include <algorithm>

using namespace vesper::kernels;
using Catch::Approx;


static void make_nonzero(std::vector<float>& v, std::mt19937& rng){
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  do { for (auto& x : v) x = dist(rng); } while (std::all_of(v.begin(), v.end(), [](float x){return x==0.0f;}));
}

TEST_CASE("auto backend selects scalar and matches outputs", "[kernels][dispatch][auto]") {
  const auto& auto_ops = select_backend_auto();
  const auto& scalar_ops = select_backend("scalar");
  REQUIRE(&auto_ops == &scalar_ops);

  std::seed_seq seed{101,103,107}; std::mt19937 rng(seed);
  for (int d : {1,3,16,128}) {
    std::vector<float> a(d), b(d);
    make_nonzero(a, rng); make_nonzero(b, rng);
    REQUIRE(auto_ops.l2_sq(a,b) == Approx(l2_sq(a,b)).margin(1e-6f));
    REQUIRE(auto_ops.inner_product(a,b) == Approx(inner_product(a,b)).margin(1e-6f));
    REQUIRE(auto_ops.cosine_similarity(a,b) == Approx(cosine_similarity(a,b)).margin(1e-6f));
    REQUIRE(auto_ops.cosine_distance(a,b) == Approx(cosine_distance(a,b)).margin(1e-6f));
  }
}

