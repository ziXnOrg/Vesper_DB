#include <catch2/catch_test_macros.hpp>
#include "vesper/memory/numa_allocator.hpp"
#include <cstring>
#include <thread>
#include <vector>

using namespace vesper::memory;

TEST_CASE("NumaTopology detection", "[numa]") {
    auto topology = NumaTopology::detect();
    REQUIRE(topology.has_value());
    
    auto& topo = *topology.value();
    
    SECTION("Basic properties") {
        REQUIRE(topo.num_nodes() >= 1);
        REQUIRE(topo.total_memory() > 0);
        
        auto node0 = topo.get_node(0);
        REQUIRE(node0 != nullptr);
        REQUIRE(node0->memory_size > 0);
        REQUIRE(!node0->cpus.empty());
    }
    
    SECTION("Current node") {
        auto current = topo.current_node();
        REQUIRE(current < topo.num_nodes());
    }
    
    SECTION("Distance matrix") {
        for (std::uint32_t i = 0; i < topo.num_nodes(); ++i) {
            // Local distance should be smallest
            auto local_dist = topo.distance(i, i);
            REQUIRE(local_dist <= 20);
            
            for (std::uint32_t j = 0; j < topo.num_nodes(); ++j) {
                auto dist = topo.distance(i, j);
                REQUIRE(dist > 0);
                REQUIRE(dist <= 100);
                
                // Distance should be symmetric
                REQUIRE(dist == topo.distance(j, i));
            }
        }
    }
}

TEST_CASE("NumaAllocator basic allocation", "[numa]") {
    auto allocator = NumaAllocator::create();
    REQUIRE(allocator.has_value());
    
    auto& alloc = *allocator.value();
    
    SECTION("Simple allocation") {
        const std::size_t size = 1024;
        auto ptr = alloc.allocate(size);
        REQUIRE(ptr.has_value());
        REQUIRE(ptr.value() != nullptr);
        
        // Write to memory to verify it's accessible
        std::memset(ptr.value(), 0x42, size);
        
        // Check first and last byte
        auto bytes = static_cast<unsigned char*>(ptr.value());
        REQUIRE(bytes[0] == 0x42);
        REQUIRE(bytes[size - 1] == 0x42);
        
        alloc.deallocate(ptr.value(), size);
    }
    
    SECTION("Aligned allocation") {
        const std::size_t size = 1024;
        const std::size_t alignment = 256;
        
        auto ptr = alloc.allocate_aligned(size, alignment);
        REQUIRE(ptr.has_value());
        REQUIRE(ptr.value() != nullptr);
        
        // Check alignment
        auto addr = reinterpret_cast<std::uintptr_t>(ptr.value());
        REQUIRE((addr % alignment) == 0);
        
        alloc.deallocate(ptr.value(), size);
    }
    
    SECTION("Zero-size allocation") {
        auto ptr = alloc.allocate(0);
        REQUIRE(ptr.has_value());
        // Zero-size allocation may return nullptr or valid pointer
        
        if (ptr.value() != nullptr) {
            alloc.deallocate(ptr.value(), 0);
        }
    }
    
    SECTION("Statistics") {
        auto stats_before = alloc.get_stats();
        
        const std::size_t size = 4096;
        auto ptr = alloc.allocate(size);
        REQUIRE(ptr.has_value());
        
        auto stats_after = alloc.get_stats();
        REQUIRE(stats_after.total_allocated >= stats_before.total_allocated + size);
        REQUIRE(stats_after.current_usage >= stats_before.current_usage + size);
        REQUIRE(stats_after.peak_usage >= stats_after.current_usage);
        
        alloc.deallocate(ptr.value(), size);
        
        auto stats_final = alloc.get_stats();
        REQUIRE(stats_final.total_deallocated >= stats_after.total_deallocated + size);
        REQUIRE(stats_final.current_usage == stats_before.current_usage);
    }
}

TEST_CASE("NumaAllocator policies", "[numa]") {
    SECTION("Local policy") {
        NumaConfig config;
        config.policy = NumaPolicy::LOCAL;
        
        auto allocator = NumaAllocator::create(config);
        REQUIRE(allocator.has_value());
        
        auto ptr = allocator.value()->allocate(1024);
        REQUIRE(ptr.has_value());
        
        allocator.value()->deallocate(ptr.value(), 1024);
    }
    
    SECTION("Interleave policy") {
        NumaConfig config;
        config.policy = NumaPolicy::INTERLEAVE;
        
        auto allocator = NumaAllocator::create(config);
        REQUIRE(allocator.has_value());
        
        // Allocate multiple pages to test interleaving
        const std::size_t page_size = 4096;
        const std::size_t num_pages = 16;
        auto ptr = allocator.value()->allocate(page_size * num_pages);
        REQUIRE(ptr.has_value());
        
        // Touch pages to fault them in
        allocator.value()->touch_pages(ptr.value(), page_size * num_pages);
        
        allocator.value()->deallocate(ptr.value(), page_size * num_pages);
    }
}

TEST_CASE("NumaAllocator node-specific allocation", "[numa]") {
    auto topology = NumaTopology::detect();
    REQUIRE(topology.has_value());
    
    auto allocator = NumaAllocator::create();
    REQUIRE(allocator.has_value());
    
    auto& alloc = *allocator.value();
    
    for (std::uint32_t node = 0; node < topology.value()->num_nodes(); ++node) {
        auto ptr = alloc.allocate_on_node(1024, node);
        REQUIRE(ptr.has_value());
        REQUIRE(ptr.value() != nullptr);
        
        // Write to verify accessibility
        std::memset(ptr.value(), 0, 1024);
        
        alloc.deallocate(ptr.value(), 1024);
    }
}

TEST_CASE("StlNumaAllocator", "[numa]") {
    auto allocator = NumaAllocator::create();
    REQUIRE(allocator.has_value());
    
    StlNumaAllocator<int> stl_alloc(allocator.value().get());
    
    SECTION("Vector with NUMA allocator") {
        std::vector<int, StlNumaAllocator<int>> vec(stl_alloc);
        
        for (int i = 0; i < 1000; ++i) {
            vec.push_back(i);
        }
        
        REQUIRE(vec.size() == 1000);
        
        for (int i = 0; i < 1000; ++i) {
            REQUIRE(vec[i] == i);
        }
    }
    
    SECTION("Rebind") {
        StlNumaAllocator<int> int_alloc(allocator.value().get());
        StlNumaAllocator<double> double_alloc(int_alloc);
        
        std::vector<double, StlNumaAllocator<double>> vec(double_alloc);
        vec.resize(100);
        
        REQUIRE(vec.size() == 100);
    }
}

TEST_CASE("ThreadAffinity", "[numa]") {
    SECTION("Current CPU") {
        auto cpu = ThreadAffinity::current_cpu();
        REQUIRE(cpu < std::thread::hardware_concurrency());
    }
    
    SECTION("Bind to node") {
        auto topology = NumaTopology::detect();
        REQUIRE(topology.has_value());
        
        if (topology.value()->num_nodes() > 1) {
            auto result = ThreadAffinity::bind_to_node(0);
            // May fail on systems without NUMA support
            if (result.has_value()) {
                auto cpu = ThreadAffinity::current_cpu();
                auto node = topology.value()->get_node(0);
                
                bool cpu_on_node = false;
                for (auto node_cpu : node->cpus) {
                    if (cpu == node_cpu) {
                        cpu_on_node = true;
                        break;
                    }
                }
                REQUIRE(cpu_on_node);
            }
        }
        
        // Reset affinity
        ThreadAffinity::reset();
    }
    
    SECTION("Bind to CPUs") {
        auto available_cpus = std::thread::hardware_concurrency();
        if (available_cpus >= 2) {
            std::vector<std::uint32_t> cpus = {0, 1};
            auto result = ThreadAffinity::bind_to_cpus(cpus);
            
            if (result.has_value()) {
                auto current = ThreadAffinity::current_cpu();
                REQUIRE((current == 0 || current == 1));
            }
        }
        
        ThreadAffinity::reset();
    }
}

TEST_CASE("NumaAllocatorPool", "[numa]") {
    SECTION("Get local allocator") {
        auto alloc1 = NumaAllocatorPool::get_local();
        REQUIRE(alloc1.has_value());
        
        auto alloc2 = NumaAllocatorPool::get_local();
        REQUIRE(alloc2.has_value());
        
        // Should return same instance
        REQUIRE(alloc1.value() == alloc2.value());
    }
    
    SECTION("Get per-node allocators") {
        auto topology = NumaTopology::detect();
        REQUIRE(topology.has_value());
        
        std::vector<NumaAllocator*> allocators;
        for (std::uint32_t node = 0; node < topology.value()->num_nodes(); ++node) {
            auto alloc = NumaAllocatorPool::get_for_node(node);
            REQUIRE(alloc.has_value());
            allocators.push_back(alloc.value());
        }
        
        // Each node should have unique allocator
        for (std::size_t i = 0; i < allocators.size(); ++i) {
            for (std::size_t j = i + 1; j < allocators.size(); ++j) {
                REQUIRE(allocators[i] != allocators[j]);
            }
        }
    }
    
    SECTION("Reset pool") {
        auto alloc = NumaAllocatorPool::get_local();
        REQUIRE(alloc.has_value());
        
        NumaAllocatorPool::reset();
        
        auto new_alloc = NumaAllocatorPool::get_local();
        REQUIRE(new_alloc.has_value());
        REQUIRE(new_alloc.value() != alloc.value());
    }
}