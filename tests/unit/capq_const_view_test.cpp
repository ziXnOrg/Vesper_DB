// Copyright (c) 2025 Colin Macritchie / Ripple Group, LLC

#include <catch2/catch_test_macros.hpp>
#include <type_traits>

#include "vesper/index/capq.hpp"

using namespace vesper::index;

TEST_CASE("CAPQ const view is read-only and shape-valid", "[capq][const]") {
  CapqSoAStorage storage(3, CapqHammingBits::B256);
  // Non-const view remains mutable
  auto v_mut = storage.view();
  static_assert(std::is_same_v<decltype(v_mut), CapqSoAView>, "mutable view type");
  REQUIRE(v_mut.hamming_words.size() == 3 * v_mut.words_per_vector());

  // Const view returns const spans
  const CapqSoAStorage cstore(2, CapqHammingBits::B256);
  auto v_const = cstore.view();
  // Types: element types are const-qualified
  static_assert(std::is_const_v<std::remove_pointer_t<decltype(v_const.hamming_words.data())>>);
  static_assert(std::is_const_v<std::remove_pointer_t<decltype(v_const.q4_packed.data())>>);
  static_assert(std::is_const_v<std::remove_pointer_t<decltype(v_const.q8.data())>>);
  static_assert(std::is_const_v<std::remove_pointer_t<decltype(v_const.residual_energy.data())>>);

  // Validate shapes via overload
  auto ok = validate_capq_view(v_const);
  REQUIRE(ok.has_value());
}

