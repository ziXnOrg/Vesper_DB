#include <catch2/catch_all.hpp>
#include <vesper/kernels/distance.hpp>
#include <random>

using namespace vesper::kernels;

static void make_nonzero(std::vector<float>& v, std::mt19937& rng){
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  do {
    for (auto& x : v) x = dist(rng);
  } while (std::all_of(v.begin(), v.end(), [](float x){return x==0.0f;}));
}

TEST_CASE("cosine known values and properties", "[kernels][cosine]") {
  // Known small cases
  {
    float a1[1]{2.0f}, b1[1]{-2.0f};
    REQUIRE(cosine_similarity(a1, b1) == Approx(-1.0f).margin(1e-6f));
  }
  {
    float a2[2]{1.0f, 0.0f}, b2[2]{0.0f, 1.0f};
    REQUIRE(cosine_similarity(a2, b2) == Approx(0.0f).margin(1e-6f));
  }
  {
    float a3[3]{1.0f, 2.0f, 3.0f};
    float b3[3]{4.0f, 5.0f, 6.0f};
    float dot = 1*4 + 2*5 + 3*6; float na = std::sqrt(1+4+9); float nb= std::sqrt(16+25+36);
    REQUIRE(cosine_similarity(a3,b3) == Approx(dot/(na*nb)).margin(1e-6f));
  }

  // Property-style checks with deterministic RNG
  std::seed_seq seed{17,23,43}; std::mt19937 rng(seed);
  for (int d : {1,3,16,128}) {
    std::vector<float> a(d), b(d), ak(d), bl(d), an(d), bn(d);
    make_nonzero(a, rng); make_nonzero(b, rng);

    // Symmetry
    REQUIRE(cosine_similarity(a, b) == Approx(cosine_similarity(b, a)).margin(1e-6f));
    // Bounded
    auto c = cosine_similarity(a, b);
    REQUIRE(c <= 1.0f + 1e-6f);
    REQUIRE(c >= -1.0f - 1e-6f);

    // Scale invariance
    float k = 2.5f, l = 0.3f;
    for (int i=0;i<d;++i){ ak[i]=k*a[i]; bl[i]=l*b[i]; }
    REQUIRE(cosine_similarity(ak, bl) == Approx(c).margin(1e-6f));

    // Normalization consistency: IP(â, b̂) == cos(a,b)
    float na = std::sqrt(inner_product(a, a));
    float nb = std::sqrt(inner_product(b, b));
    for (int i=0;i<d;++i){ an[i]=a[i]/na; bn[i]=b[i]/nb; }
    REQUIRE(inner_product(an, bn) == Approx(c).margin(1e-6f));

    // Cosine distance = 1 - cosine
    REQUIRE(cosine_distance(a, b) == Approx(1.0f - c).margin(1e-6f));
  }
}

