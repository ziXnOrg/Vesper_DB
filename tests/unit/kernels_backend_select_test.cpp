#include <catch2/catch_all.hpp>
#include <vesper/kernels/dispatch.hpp>
#include <vesper/kernels/backends/scalar.hpp>
#include <random>
#include <algorithm>

using namespace vesper::kernels;
using Catch::Approx;


static void make_nonzero(std::vector<float>& v, std::mt19937& rng){
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  do { for (auto& x : v) x = dist(rng); } while (std::all_of(v.begin(), v.end(), [](float x){return x==0.0f;}));
}

TEST_CASE("backend selection returns stable references and matches scalar outputs", "[kernels][dispatch]") {
  const auto& s1 = select_backend("scalar");
  const auto& s2 = select_backend("scalar");
  REQUIRE(&s1 == &s2);

  const auto& a1 = select_backend("stub-avx2");
  const auto& a2 = select_backend("stub-avx2");
  REQUIRE(&a1 == &a2);

  const auto& n1 = select_backend("stub-neon");
  const auto& n2 = select_backend("stub-neon");
  REQUIRE(&n1 == &n2);

  const auto& u = select_backend("does-not-exist");
  REQUIRE(&u == &s1); // fallback to scalar

  std::seed_seq seed{31,59,27}; std::mt19937 rng(seed);
  for (int d : {1,3,16,128}) {
    std::vector<float> a(d), b(d);
    make_nonzero(a, rng); make_nonzero(b, rng);

    auto check_backend = [&](const KernelOps& ops){
      REQUIRE(ops.l2_sq(a,b) == Approx(l2_sq(a,b)).margin(1e-6f));
      REQUIRE(ops.inner_product(a,b) == Approx(inner_product(a,b)).margin(1e-6f));
      REQUIRE(ops.cosine_similarity(a,b) == Approx(cosine_similarity(a,b)).margin(1e-6f));
      REQUIRE(ops.cosine_distance(a,b) == Approx(cosine_distance(a,b)).margin(1e-6f));
    };

    check_backend(s1);
    check_backend(a1);
    check_backend(n1);
  }
}

