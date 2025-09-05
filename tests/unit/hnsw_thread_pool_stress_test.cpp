#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <random>
#include <thread>

#include "vesper/index/hnsw_thread_pool.hpp"

using namespace std::chrono_literals;

namespace {

TEST(ThreadPoolStress, ChurnAndWaitAll) {
  using vesper::index::HnswThreadPool;

  const std::size_t nt = std::max(2u, std::thread::hardware_concurrency() / 2);
  HnswThreadPool pool(nt);

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 9);

  std::atomic<std::uint64_t> planned{0};
  std::atomic<std::uint64_t> executed{0};

  auto work = [&](auto&& self, int depth) -> void {
    executed.fetch_add(1, std::memory_order_relaxed);
    // small compute to vary timing
    volatile float acc = 0.f;
    for (int i = 0; i < 128; ++i) acc += (i * 0.5f);
    (void)acc;

    if (depth < 3) {
      if (dist(rng) == 0) { // ~10% chance to spawn
        planned.fetch_add(1, std::memory_order_relaxed);
        (void)pool.submit([&self, depth]{ self(self, depth + 1); });
      }
    }
  };

  constexpr int rounds = 3;
  constexpr int base_tasks = 2000;
  for (int r = 0; r < rounds; ++r) {
    planned.store(0, std::memory_order_relaxed);
    executed.store(0, std::memory_order_relaxed);

    planned.fetch_add(base_tasks, std::memory_order_relaxed);
    for (int i = 0; i < base_tasks; ++i) {
      (void)pool.submit([&work]{ work(work, 0); });
    }

    // wait for all tasks (including spawned ones)
    // To avoid infinite wait in case of a bug, add a guard timeout loop
    auto start = std::chrono::steady_clock::now();
    while (true) {
      pool.wait_all();
      if (executed.load(std::memory_order_relaxed) == planned.load(std::memory_order_relaxed)) break;
      if (std::chrono::steady_clock::now() - start > 10s) break; // guard
      std::this_thread::sleep_for(5ms);
    }

    EXPECT_EQ(executed.load(), planned.load()) << "Round " << r << " mismatch";
  }
}

TEST(ThreadPoolStress, ManyFuturesMix) {
  using vesper::index::HnswThreadPool;

  HnswThreadPool pool(std::max(2u, std::thread::hardware_concurrency() / 2));

  constexpr int N = 1000;
  std::vector<std::future<void>> futs;
  futs.reserve(N);

  std::atomic<int> acc{0};
  for (int i = 0; i < N; ++i) {
    futs.emplace_back(pool.submit([&acc]{ acc.fetch_add(1, std::memory_order_relaxed); }));
  }

  // Also add non-future tasks interleaved
  for (int i = 0; i < N; ++i) {
    (void)pool.submit([&acc]{ acc.fetch_add(1, std::memory_order_relaxed); });
  }

  for (auto& f : futs) f.wait();
  pool.wait_all();

  EXPECT_EQ(acc.load(), 2 * N);
}

} // namespace

