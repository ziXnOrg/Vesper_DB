#include <catch2/catch_all.hpp>
#include <vesper/collection.hpp>
#include <vesper/kernels/distance.hpp>

using namespace vesper;
using Catch::Approx;


static auto make_collection() {
  auto col = collection::open(""); REQUIRE(col.has_value());
  collection c = std::move(*col);
  float v0[2]{1.0f, 2.0f}; REQUIRE(c.insert(100, v0, 2).has_value());
  float v1[2]{0.0f, 1.0f}; REQUIRE(c.insert(101, v1, 2).has_value());
  float v2[2]{5.0f, 6.0f}; REQUIRE(c.insert(102, v2, 2).has_value());
  return c;
}

static std::vector<search_result> expected_results(const std::vector<float>& qv, const std::string& metric) {
  std::vector<std::pair<std::uint64_t, std::vector<float>>> docs{
    {100, {1.0f, 2.0f}}, {101, {0.0f, 1.0f}}, {102, {5.0f, 6.0f}}
  };
  std::vector<search_result> out;
  for (auto& d : docs) {
    float s = 0.0f;
    if (metric == "ip") s = -kernels::inner_product(qv, d.second);
    else if (metric == "cosine") s = 1.0f - kernels::cosine_similarity(qv, d.second);
    else s = kernels::l2_sq(qv, d.second);
    out.push_back({d.first, s});
  }
  std::sort(out.begin(), out.end(), [](auto& a, auto& b){ if (a.score == b.score) return a.id < b.id; return a.score < b.score; });
  return out;
}

TEST_CASE("dispatcher integration equivalence for l2/ip/cosine", "[search][dispatch]") {
  auto c = make_collection();
  for (auto metric : {std::string("l2"), std::string("ip"), std::string("cosine")}) {
    std::vector<float> qv{1.0f, 1.0f};
    auto exp = expected_results(qv, metric);

    search_params p; p.k = 3; p.metric = metric;
    auto got = c.search(qv.data(), qv.size(), p, nullptr);
    REQUIRE(got.has_value());
    REQUIRE(got->size() == exp.size());
    for (size_t i=0;i<exp.size();++i){
      REQUIRE((*got)[i].id == exp[i].id);
      REQUIRE((*got)[i].score == Approx(exp[i].score).margin(1e-6));
    }
  }
}

