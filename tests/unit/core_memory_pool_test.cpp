// Copyright (c) 2025 Colin Macritchie / Ripple Group, LLC

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <limits>

#include "vesper/core/memory_pool.hpp"

using namespace vesper::core;

TEST_CASE("MemoryArena returns nullptr for huge/overflowing requests", "[memory_pool][overflow]") {
  MemoryArena arena(1024);

  // Request that would overflow alignment math if not guarded
  void* p = arena.allocate((std::numeric_limits<std::size_t>::max)());
  REQUIRE(p == nullptr);
  REQUIRE(arena.used() == 0);
}

TEST_CASE("MemoryArena rejects invalid alignment (0 and non-power-of-two)", "[memory_pool][alignment]") {
  MemoryArena arena(1024);

  void* p0 = arena.allocate(16, 0);     // invalid: zero alignment
  void* p3 = arena.allocate(16, 3);     // invalid: not power-of-two
  REQUIRE(p0 == nullptr);
  REQUIRE(p3 == nullptr);
  REQUIRE(arena.used() == 0);
}

TEST_CASE("MemoryArena exhaustion returns nullptr without wrap", "[memory_pool][exhaustion]") {
  MemoryArena arena(1024);
  // Default ALIGNMENT is 64; 512 aligns to 512
  void* a = arena.allocate(512);
  void* b = arena.allocate(512);
  void* c = arena.allocate(1); // would need 64 more bytes, none left
  REQUIRE(a != nullptr);
  REQUIRE(b != nullptr);
  REQUIRE(c == nullptr);
  REQUIRE(arena.used() == 1024);
}

TEST_CASE("MemoryArena honors alignment and updates used by aligned size", "[memory_pool][normal]") {
  MemoryArena arena(1024);
  const std::size_t align = 128;

  void* a = arena.allocate(1, align); // rounds to 128 (may include leading pad to satisfy alignment)
  REQUIRE(a != nullptr);
  REQUIRE((reinterpret_cast<std::uintptr_t>(a) % align) == 0);
  const auto used1 = arena.used();
  REQUIRE(used1 >= align);

  void* b = arena.allocate(1, align); // next 128-byte chunk
  REQUIRE(b != nullptr);
  REQUIRE((reinterpret_cast<std::uintptr_t>(b) % align) == 0);
  REQUIRE(reinterpret_cast<std::uintptr_t>(b) - reinterpret_cast<std::uintptr_t>(a) == 128);
  const auto used2 = arena.used();
  REQUIRE(used2 - used1 == 128);
}


TEST_CASE("PooledVector works within PoolScope", "[memory_pool][scope]") {
  PoolScope scope;
  auto v = make_pooled_vector<int>(0);
  v.reserve(128);
  v.push_back(42);
  v.push_back(43);
  REQUIRE(v.size() == 2);
  REQUIRE(v[0] == 42);
  REQUIRE(v[1] == 43);
}


TEST_CASE("ThreadLocalPool prewarm succeeds for default size", "[memory_pool][noexcept_hot]") {
  auto res = ThreadLocalPool::prewarm(MemoryArena::DEFAULT_SIZE);
  REQUIRE(res.has_value());
}

TEST_CASE("TempBuffer::try_create returns error on OOM-sized request", "[memory_pool][noexcept_hot]") {
  // Intentionally huge count to force failure
  const std::size_t huge = (std::numeric_limits<std::size_t>::max)() / sizeof(int);
  auto tb = TempBuffer<int>::try_create(huge);
  REQUIRE_FALSE(tb.has_value());
}

TEST_CASE("Presized pooled vector performs no allocations in hot region", "[memory_pool][noexcept_hot]") {
#ifndef NDEBUG
  PoolScope scope;
  auto& pool = ThreadLocalPool::instance();
  auto v = make_pooled_vector<int>(0);
  v.resize(1024); // pre-size before hot loop
  pool.debug_begin_hot_region();
  for (int i = 0; i < 1024; ++i) {
    v[i] = i;
  }
  auto allocs = pool.debug_end_hot_region();
  REQUIRE(allocs == 0);
#else
  SUCCEED(); // instrumentation is debug-only
#endif
}
