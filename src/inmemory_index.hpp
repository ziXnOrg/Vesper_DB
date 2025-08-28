#pragma once
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <cmath>

namespace vesper::detail {

struct inmem_index {
  std::unordered_map<std::uint64_t, std::vector<float>> store;

  static float l2(const std::vector<float>& a, const std::vector<float>& b){
    float s=0.0f; for (size_t i=0;i<a.size();++i){ float d=a[i]-b[i]; s+=d*d; } return s;
  }
};

} // namespace vesper::detail

