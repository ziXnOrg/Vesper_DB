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

