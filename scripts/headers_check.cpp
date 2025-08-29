#include "vesper/collection.hpp"
#include "vesper/filter_expr.hpp"
#include "vesper/error.hpp"
#include "vesper/kernels/distance.hpp"
#include <iostream>

int main() {
  // Touch symbols to avoid ODR/unused warnings in header-only compilation
  float a[1]{0.0f}; (void)vesper::kernels::l2_sq(a,a);
  std::cout << "Headers compile" << std::endl;
  return 0;
}

