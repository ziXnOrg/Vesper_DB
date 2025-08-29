#include "vesper/collection.hpp"
#include "vesper/filter_expr.hpp"
#include "vesper/error.hpp"
#include "vesper/kernels/distance.hpp"
#include "vesper/kernels/dispatch.hpp"
#include "vesper/kernels/backends/scalar.hpp"
#include "vesper/kernels/backends/stub_avx2.hpp"
#include "vesper/kernels/backends/stub_neon.hpp"
#include <iostream>

int main() {
  float a[1]{0.0f}; (void)vesper::kernels::l2_sq(a,a);
  auto& ops = vesper::kernels::select_backend(); (void)ops;
  auto& ops_auto = vesper::kernels::select_backend_auto(); (void)ops_auto;
  auto& avx = vesper::kernels::get_stub_avx2_ops(); (void)avx;
  auto& neo = vesper::kernels::get_stub_neon_ops(); (void)neo;
  std::cout << "Headers compile" << std::endl;
  return 0;
}

