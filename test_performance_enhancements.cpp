// Test for LRU Cache and Bitmap Filter enhancements
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <unordered_map>
#include "vesper/cache/lru_cache.hpp"
#include "vesper/filter/bitmap_filter.hpp"
#include "vesper/filter_expr.hpp"

using namespace vesper;
using namespace std::chrono;

void test_lru_cache() {
    std::cout << "\n=== Testing LRU Cache ===" << std::endl;
    
    // Create cache with 10MB capacity
    cache::ShardedLruCache<int, std::vector<float>> cache(
        10 * 1024 * 1024,  // 10MB
        16,  // shards
        std::chrono::seconds(60)  // TTL
    );
    
    // Test data
    const size_t vec_size = 128;
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    // Insert test vectors
    std::cout << "Inserting 1000 vectors..." << std::endl;
    for (int i = 0; i < 1000; ++i) {
        std::vector<float> vec(vec_size);
        for (auto& v : vec) {
            v = dist(gen);
        }
        cache.put(i, vec, vec_size * sizeof(float));
    }
    
    // Test cache hits
    std::cout << "Testing cache hits..." << std::endl;
    int hits = 0;
    auto start = high_resolution_clock::now();
    
    for (int i = 0; i < 10000; ++i) {
        int key = gen() % 1000;
        auto result = cache.get(key);
        if (result.has_value()) {
            hits++;
        }
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    
    // Print statistics
    auto stats = cache.stats();
    std::cout << "Cache Statistics:" << std::endl;
    std::cout << "  Hit rate: " << (stats.hit_rate() * 100) << "%" << std::endl;
    std::cout << "  Hits: " << stats.hits << std::endl;
    std::cout << "  Misses: " << stats.misses << std::endl;
    std::cout << "  Evictions: " << stats.evictions << std::endl;
    std::cout << "  Memory used: " << (stats.bytes_used / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  Avg lookup time: " << (duration.count() / 10000.0) << " μs" << std::endl;
    
    if (stats.hit_rate() > 0.8) {
        std::cout << "✓ Cache test PASSED" << std::endl;
    } else {
        std::cout << "✗ Cache test FAILED (hit rate too low)" << std::endl;
    }
}

void test_bitmap_filter() {
    std::cout << "\n=== Testing Bitmap Filter ===" << std::endl;
    
    const size_t num_vectors = 10000;
    filter::BitmapFilter filter(num_vectors);
    
    // Add metadata for vectors
    std::cout << "Adding metadata for " << num_vectors << " vectors..." << std::endl;
    for (uint32_t i = 0; i < num_vectors; ++i) {
        std::unordered_map<std::string, std::string> tags;
        std::unordered_map<std::string, double> nums;
        
        // Add category tags
        tags["category"] = (i % 3 == 0) ? "A" : (i % 3 == 1) ? "B" : "C";
        tags["type"] = (i < 5000) ? "train" : "test";
        
        // Add numeric attributes
        nums["score"] = static_cast<double>(i) / 100.0;
        nums["confidence"] = 0.5 + (i % 100) / 200.0;
        
        auto result = filter.add_metadata(i, tags, nums);
        if (!result) {
            std::cerr << "Failed to add metadata for vector " << i << std::endl;
        }
    }
    
    // Test 1: Simple term filter
    std::cout << "\nTest 1: Filter by category='A'" << std::endl;
    filter_expr expr1;
    expr1.node = term{"category", "A"};
    
    auto start = high_resolution_clock::now();
    auto result1 = filter.apply_filter(expr1);
    auto end = high_resolution_clock::now();
    
    if (result1.has_value()) {
        std::cout << "  Found " << result1->size() << " matches" << std::endl;
        std::cout << "  Time: " << duration_cast<microseconds>(end - start).count() << " μs" << std::endl;
        std::cout << "  Selectivity: " << filter.estimate_selectivity(expr1) << std::endl;
        
        // Verify correctness
        bool correct = true;
        for (auto id : *result1) {
            if (id % 3 != 0) {
                correct = false;
                break;
            }
        }
        std::cout << (correct ? "  ✓ Correctness verified" : "  ✗ Incorrect results") << std::endl;
    }
    
    // Test 2: Range filter
    std::cout << "\nTest 2: Filter by score between 10 and 20" << std::endl;
    filter_expr expr2;
    expr2.node = range{"score", 10.0, 20.0};
    
    start = high_resolution_clock::now();
    auto result2 = filter.apply_filter(expr2);
    end = high_resolution_clock::now();
    
    if (result2.has_value()) {
        std::cout << "  Found " << result2->size() << " matches" << std::endl;
        std::cout << "  Time: " << duration_cast<microseconds>(end - start).count() << " μs" << std::endl;
        std::cout << "  Selectivity: " << filter.estimate_selectivity(expr2) << std::endl;
    }
    
    // Test 3: Complex filter (AND operation)
    std::cout << "\nTest 3: Filter by category='B' AND type='train'" << std::endl;
    filter_expr expr3;
    filter_expr::and_t and_expr;
    and_expr.children.push_back(filter_expr{term{"category", "B"}});
    and_expr.children.push_back(filter_expr{term{"type", "train"}});
    expr3.node = and_expr;
    
    start = high_resolution_clock::now();
    auto result3 = filter.apply_filter(expr3);
    end = high_resolution_clock::now();
    
    if (result3.has_value()) {
        std::cout << "  Found " << result3->size() << " matches" << std::endl;
        std::cout << "  Time: " << duration_cast<microseconds>(end - start).count() << " μs" << std::endl;
        std::cout << "  Selectivity: " << filter.estimate_selectivity(expr3) << std::endl;
        
        // Verify correctness
        bool correct = true;
        for (auto id : *result3) {
            if (id % 3 != 1 || id >= 5000) {
                correct = false;
                break;
            }
        }
        std::cout << (correct ? "  ✓ Correctness verified" : "  ✗ Incorrect results") << std::endl;
    }
    
    std::cout << "\n✓ Bitmap filter tests completed" << std::endl;
}

void test_cache_performance() {
    std::cout << "\n=== Cache Performance Benchmark ===" << std::endl;
    
    const size_t num_items = 100000;
    const size_t num_queries = 1000000;
    const size_t item_size = 1024;  // 1KB per item
    
    // Test with different cache sizes
    std::vector<size_t> cache_sizes = {10, 50, 100, 500};  // MB
    
    for (size_t cache_mb : cache_sizes) {
        std::cout << "\nCache size: " << cache_mb << " MB" << std::endl;
        
        cache::ShardedLruCache<uint32_t, std::vector<uint8_t>> cache(
            cache_mb * 1024 * 1024,
            16  // shards
        );
        
        // Fill cache
        std::mt19937 gen(42);
        for (uint32_t i = 0; i < num_items; ++i) {
            std::vector<uint8_t> data(item_size, i % 256);
            cache.put(i, data, item_size);
        }
        
        // Benchmark lookups
        std::uniform_int_distribution<uint32_t> dist(0, num_items - 1);
        
        auto start = high_resolution_clock::now();
        for (size_t i = 0; i < num_queries; ++i) {
            uint32_t key = dist(gen);
            auto result = cache.get(key);
        }
        auto end = high_resolution_clock::now();
        
        auto duration = duration_cast<milliseconds>(end - start);
        auto stats = cache.stats();
        
        std::cout << "  Queries: " << num_queries << std::endl;
        std::cout << "  Total time: " << duration.count() << " ms" << std::endl;
        std::cout << "  Throughput: " << (num_queries / (duration.count() / 1000.0)) << " QPS" << std::endl;
        std::cout << "  Hit rate: " << (stats.hit_rate() * 100) << "%" << std::endl;
    }
}

int main() {
    std::cout << "Testing Vesper Performance Enhancements" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        test_lru_cache();
        test_bitmap_filter();
        test_cache_performance();
        
        std::cout << "\n✅ All performance enhancement tests completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}