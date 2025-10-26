#include <benchmark/benchmark.h>
#include "vesper/memory/numa_allocator.hpp"
#include <vector>
#include <cstring>
#include <thread>
#include <random>

using namespace vesper::memory;

// Benchmark standard allocator as baseline
static void BM_StandardAlloc(benchmark::State& state) {
    const std::size_t size = state.range(0);
    
    for (auto _ : state) {
        void* ptr = std::aligned_alloc(64, size);
        benchmark::DoNotOptimize(ptr);
        
        // Touch first and last cache line
        if (ptr) {
            std::memset(ptr, 0, 64);
            std::memset(static_cast<char*>(ptr) + size - 64, 0, 64);
        }
        
        std::free(ptr);
    }
    
    state.SetBytesProcessed(state.iterations() * size);
}

// Benchmark NUMA local allocation
static void BM_NumaLocalAlloc(benchmark::State& state) {
    const std::size_t size = state.range(0);
    
    NumaConfig config;
    config.policy = NumaPolicy::LOCAL;
    
    auto allocator = NumaAllocator::create(config);
    if (!allocator) {
        state.SkipWithError("NUMA allocator creation failed");
        return;
    }
    
    auto& alloc = *allocator.value();
    
    for (auto _ : state) {
        auto ptr = alloc.allocate(size);
        benchmark::DoNotOptimize(ptr);
        
        if (ptr.has_value() && ptr.value()) {
            // Touch first and last cache line
            std::memset(ptr.value(), 0, 64);
            std::memset(static_cast<char*>(ptr.value()) + size - 64, 0, 64);
            
            alloc.deallocate(ptr.value(), size);
        }
    }
    
    state.SetBytesProcessed(state.iterations() * size);
}

// Benchmark NUMA interleaved allocation
static void BM_NumaInterleaveAlloc(benchmark::State& state) {
    const std::size_t size = state.range(0);
    
    NumaConfig config;
    config.policy = NumaPolicy::INTERLEAVE;
    
    auto allocator = NumaAllocator::create(config);
    if (!allocator) {
        state.SkipWithError("NUMA allocator creation failed");
        return;
    }
    
    auto& alloc = *allocator.value();
    
    for (auto _ : state) {
        auto ptr = alloc.allocate(size);
        benchmark::DoNotOptimize(ptr);
        
        if (ptr.has_value() && ptr.value()) {
            // Touch pages to fault them in
            alloc.touch_pages(ptr.value(), size);
            alloc.deallocate(ptr.value(), size);
        }
    }
    
    state.SetBytesProcessed(state.iterations() * size);
}

// Benchmark huge page allocation
static void BM_NumaHugePageAlloc(benchmark::State& state) {
    const std::size_t size = state.range(0);
    
    NumaConfig config;
    config.policy = NumaPolicy::LOCAL;
    config.use_huge_pages = true;
    
    auto allocator = NumaAllocator::create(config);
    if (!allocator) {
        state.SkipWithError("NUMA allocator creation failed");
        return;
    }
    
    auto& alloc = *allocator.value();
    
    for (auto _ : state) {
        auto ptr = alloc.allocate(size);
        benchmark::DoNotOptimize(ptr);
        
        if (ptr.has_value() && ptr.value()) {
            // Touch pages
            alloc.touch_pages(ptr.value(), size);
            alloc.deallocate(ptr.value(), size);
        }
    }
    
    state.SetBytesProcessed(state.iterations() * size);
}

// Benchmark memory access patterns
static void BM_SequentialAccess_Standard(benchmark::State& state) {
    const std::size_t size = state.range(0);
    const std::size_t stride = 64; // Cache line size
    
    void* ptr = std::aligned_alloc(64, size);
    if (!ptr) {
        state.SkipWithError("Allocation failed");
        return;
    }
    
    auto data = static_cast<volatile char*>(ptr);
    
    for (auto _ : state) {
        for (std::size_t i = 0; i < size; i += stride) {
            data[i] = static_cast<char>(i);
        }
        benchmark::ClobberMemory();
    }
    
    std::free(ptr);
    state.SetBytesProcessed(state.iterations() * size);
}

static void BM_SequentialAccess_NUMA(benchmark::State& state) {
    const std::size_t size = state.range(0);
    const std::size_t stride = 64; // Cache line size
    
    auto allocator = NumaAllocator::create();
    if (!allocator) {
        state.SkipWithError("NUMA allocator creation failed");
        return;
    }
    
    auto ptr = allocator.value()->allocate(size);
    if (!ptr || !ptr.value()) {
        state.SkipWithError("Allocation failed");
        return;
    }
    
    auto data = static_cast<volatile char*>(ptr.value());
    
    for (auto _ : state) {
        for (std::size_t i = 0; i < size; i += stride) {
            data[i] = static_cast<char>(i);
        }
        benchmark::ClobberMemory();
    }
    
    allocator.value()->deallocate(ptr.value(), size);
    state.SetBytesProcessed(state.iterations() * size);
}

static void BM_RandomAccess_Standard(benchmark::State& state) {
    const std::size_t size = state.range(0);
    const std::size_t num_accesses = size / 64; // One per cache line
    
    void* ptr = std::aligned_alloc(64, size);
    if (!ptr) {
        state.SkipWithError("Allocation failed");
        return;
    }
    
    auto data = static_cast<volatile char*>(ptr);
    
    // Generate random access pattern
    std::vector<std::size_t> indices(num_accesses);
    std::mt19937 gen(42);
    for (std::size_t i = 0; i < num_accesses; ++i) {
        indices[i] = (i * 64) % size;
    }
    std::shuffle(indices.begin(), indices.end(), gen);
    
    for (auto _ : state) {
        for (auto idx : indices) {
            data[idx] = static_cast<char>(idx);
        }
        benchmark::ClobberMemory();
    }
    
    std::free(ptr);
    state.SetBytesProcessed(state.iterations() * num_accesses * 64);
}

static void BM_RandomAccess_NUMA(benchmark::State& state) {
    const std::size_t size = state.range(0);
    const std::size_t num_accesses = size / 64; // One per cache line
    
    auto allocator = NumaAllocator::create();
    if (!allocator) {
        state.SkipWithError("NUMA allocator creation failed");
        return;
    }
    
    auto ptr = allocator.value()->allocate(size);
    if (!ptr || !ptr.value()) {
        state.SkipWithError("Allocation failed");
        return;
    }
    
    auto data = static_cast<volatile char*>(ptr.value());
    
    // Generate random access pattern
    std::vector<std::size_t> indices(num_accesses);
    std::mt19937 gen(42);
    for (std::size_t i = 0; i < num_accesses; ++i) {
        indices[i] = (i * 64) % size;
    }
    std::shuffle(indices.begin(), indices.end(), gen);
    
    for (auto _ : state) {
        for (auto idx : indices) {
            data[idx] = static_cast<char>(idx);
        }
        benchmark::ClobberMemory();
    }
    
    allocator.value()->deallocate(ptr.value(), size);
    state.SetBytesProcessed(state.iterations() * num_accesses * 64);
}

// Multi-threaded allocation benchmark
static void BM_MultithreadedAlloc_Standard(benchmark::State& state) {
    const std::size_t size = 4096;
    const int num_threads = state.range(0);
    
    for (auto _ : state) {
        std::vector<std::thread> threads;
        
        auto worker = [size]() {
            for (int i = 0; i < 100; ++i) {
                void* ptr = std::aligned_alloc(64, size);
                if (ptr) {
                    std::memset(ptr, 0, size);
                    std::free(ptr);
                }
            }
        };
        
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back(worker);
        }
        
        for (auto& t : threads) {
            t.join();
        }
    }
    
    state.SetItemsProcessed(state.iterations() * num_threads * 100);
}

static void BM_MultithreadedAlloc_NUMA(benchmark::State& state) {
    const std::size_t size = 4096;
    const int num_threads = state.range(0);
    
    for (auto _ : state) {
        std::vector<std::thread> threads;
        
        auto worker = [size]() {
            auto allocator = NumaAllocatorPool::get_local();
            if (!allocator) return;
            
            for (int i = 0; i < 100; ++i) {
                auto ptr = allocator.value()->allocate(size);
                if (ptr.has_value() && ptr.value()) {
                    std::memset(ptr.value(), 0, size);
                    allocator.value()->deallocate(ptr.value(), size);
                }
            }
        };
        
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back(worker);
        }
        
        for (auto& t : threads) {
            t.join();
        }
    }
    
    state.SetItemsProcessed(state.iterations() * num_threads * 100);
}

// Register benchmarks
BENCHMARK(BM_StandardAlloc)->Range(64, 1 << 20);
BENCHMARK(BM_NumaLocalAlloc)->Range(64, 1 << 20);
BENCHMARK(BM_NumaInterleaveAlloc)->Range(64, 1 << 20);
BENCHMARK(BM_NumaHugePageAlloc)->Range(1 << 21, 1 << 24); // 2MB to 16MB

BENCHMARK(BM_SequentialAccess_Standard)->Range(1 << 10, 1 << 20);
BENCHMARK(BM_SequentialAccess_NUMA)->Range(1 << 10, 1 << 20);
BENCHMARK(BM_RandomAccess_Standard)->Range(1 << 10, 1 << 20);
BENCHMARK(BM_RandomAccess_NUMA)->Range(1 << 10, 1 << 20);

BENCHMARK(BM_MultithreadedAlloc_Standard)->Range(1, 16);
BENCHMARK(BM_MultithreadedAlloc_NUMA)->Range(1, 16);

BENCHMARK_MAIN();