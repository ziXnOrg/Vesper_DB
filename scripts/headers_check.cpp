#include "vesper/collection.hpp"
#include "vesper/filter_expr.hpp"
#include "vesper/error.hpp"
#include "vesper/kernels/distance.hpp"
#include "vesper/kernels/dispatch.hpp"
#include "vesper/kernels/backends/scalar.hpp"
#include <iostream>

int main() {
  float a[1]{0.0f}; (void)vesper::kernels::l2_sq(a,a);
  auto& ops = vesper::kernels::select_backend(); (void)ops;
  std::cout << "Headers compile" << std::endl;
  return 0;
}

