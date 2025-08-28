#include <catch2/catch_all.hpp>

// Simple property-style test: L2(a,a) == 0, symmetry of dot product
static float l2(const std::vector<float>& a, const std::vector<float>& b) {
  float s = 0.0f; for (size_t i=0;i<a.size();++i){ float d=a[i]-b[i]; s+=d*d;} return s;
}
static float dot(const std::vector<float>& a, const std::vector<float>& b) {
  float s = 0.0f; for (size_t i=0;i<a.size();++i){ s+=a[i]*b[i]; } return s;
}

TEST_CASE("vector invariants", "[property]") {
  std::seed_seq seed{17,23,43}; std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f,1.0f);
  for (int t=0;t<100;++t){
    size_t d = 16;
    std::vector<float> a(d), b(d);
    for (size_t i=0;i<d;++i){ a[i]=dist(rng); b[i]=dist(rng);}    
    REQUIRE(l2(a,a) == Approx(0.0f).margin(1e-6f));
    REQUIRE(dot(a,b) == Approx(dot(b,a)).margin(1e-6f));
  }
}

