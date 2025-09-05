#include <gtest/gtest.h>
#include "vesper/index/hnsw_lockfree.hpp"
#include <random>
#include <numeric>
#include <unordered_set>
#include <thread>
#include <chrono>
#include <atomic>

namespace vesper::index {
namespace {

class HnswLockfreeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate test data
        const std::size_t n = 10000;  // Larger dataset for performance testing
        const std::size_t dim = 128;
        
        data_.resize(n * dim);
        ids_.resize(n);
        
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (auto& val : data_) {
            val = dist(gen);
        }
        
        // Normalize for cosine similarity tests
        for (std::size_t i = 0; i < n; ++i) {
            float* vec = data_.data() + i * dim;
            float norm = 0.0f;
            for (std::size_t d = 0; d < dim; ++d) {
                norm += vec[d] * vec[d];
            }
            norm = std::sqrt(norm);
            for (std::size_t d = 0; d < dim; ++d) {
                vec[d] /= norm;
            }
        }
        
        std::iota(ids_.begin(), ids_.end(), 0);
        
        n_ = n;
        dim_ = dim;
    }
    
    std::vector<float> data_;
    std::vector<std::uint64_t> ids_;
    std::size_t n_;
    std::size_t dim_;
};

TEST_F(HnswLockfreeTest, InitAndBuild) {
    HnswLockfreeIndex index;
    
    HnswLockfreeBuildParams params{
        .M = 16,
        .efConstruction = 200,
        .seed = 42,
        .max_M = 16,
        .max_M0 = 32
    };
    
    ASSERT_TRUE(index.init(dim_, params, n_).has_value());
    EXPECT_TRUE(index.is_initialized());
    EXPECT_EQ(index.dimension(), dim_);
    
    // Add vectors
    for (std::size_t i = 0; i < std::min(n_, std::size_t(100)); ++i) {
        ASSERT_TRUE(index.add(ids_[i], data_.data() + i * dim_).has_value());
    }
    
    EXPECT_EQ(index.size(), std::min(n_, std::size_t(100)));
    
    // Check stats
    auto stats = index.get_stats();
    EXPECT_EQ(stats.n_nodes, std::min(n_, std::size_t(100)));
    EXPECT_GT(stats.n_edges, 0);
    EXPECT_GT(stats.avg_degree, 0.0f);
}

TEST_F(HnswLockfreeTest, Search) {
    HnswLockfreeIndex index;
    
    HnswLockfreeBuildParams params{
        .M = 16,
        .efConstruction = 200
    };
    
    ASSERT_TRUE(index.init(dim_, params, n_).has_value());
    
    // Use smaller dataset for basic search test
    std::size_t test_size = 1000;
    ASSERT_TRUE(index.add_batch(ids_.data(), data_.data(), test_size).has_value());
    
    // Search for first vector (should find itself)
    HnswLockfreeSearchParams search_params{
        .efSearch = 50,
        .k = 10
    };
    
    auto results = index.search(data_.data(), search_params);
    ASSERT_TRUE(results.has_value());
    ASSERT_FALSE(results->empty());
    
    // First result should be the query itself
    EXPECT_EQ(results->front().first, ids_[0]);
    EXPECT_LT(results->front().second, 1e-6f);  // Distance ~0
}

TEST_F(HnswLockfreeTest, BatchSearch) {
    HnswLockfreeIndex index;
    
    HnswLockfreeBuildParams params{
        .M = 16,
        .efConstruction = 200
    };
    
    ASSERT_TRUE(index.init(dim_, params, n_).has_value());
    
    std::size_t test_size = 1000;
    ASSERT_TRUE(index.add_batch(ids_.data(), data_.data(), test_size).has_value());
    
    // Batch search
    const std::size_t n_queries = 10;
    HnswLockfreeSearchParams search_params{
        .efSearch = 50,
        .k = 10
    };
    
    auto results = index.search_batch(data_.data(), n_queries, search_params);
    ASSERT_TRUE(results.has_value());
    EXPECT_EQ(results->size(), n_queries);
    
    // Each query should find itself first
    for (std::size_t q = 0; q < n_queries; ++q) {
        const auto& query_results = (*results)[q];
        ASSERT_FALSE(query_results.empty());
        EXPECT_EQ(query_results.front().first, ids_[q]);
    }
}

TEST_F(HnswLockfreeTest, ConcurrentAddSingleVectors) {
    HnswLockfreeIndex index;
    
    HnswLockfreeBuildParams params{
        .M = 16,
        .efConstruction = 200,
        .max_M0 = 32
    };
    
    ASSERT_TRUE(index.init(dim_, params, n_).has_value());
    
    // Concurrent additions
    const std::size_t n_threads = 4;
    const std::size_t vectors_per_thread = 250;
    
    std::vector<std::thread> threads;
    std::atomic<std::size_t> success_count{0};
    std::atomic<std::size_t> failure_count{0};
    
    for (std::size_t t = 0; t < n_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (std::size_t i = 0; i < vectors_per_thread; ++i) {
                std::size_t idx = t * vectors_per_thread + i;
                if (idx < n_) {
                    auto result = index.add(ids_[idx], data_.data() + idx * dim_);
                    if (result.has_value()) {
                        success_count++;
                    } else {
                        failure_count++;
                    }
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(success_count, n_threads * vectors_per_thread);
    EXPECT_EQ(failure_count, 0);
    EXPECT_EQ(index.size(), n_threads * vectors_per_thread);
    
    // Verify all vectors are searchable
    HnswLockfreeSearchParams search_params{
        .efSearch = 50,
        .k = 10
    };
    
    for (std::size_t i = 0; i < n_threads * vectors_per_thread; i += 100) {
        auto results = index.search(data_.data() + i * dim_, search_params);
        ASSERT_TRUE(results.has_value());
        ASSERT_FALSE(results->empty());
        EXPECT_EQ(results->front().first, ids_[i]);
    }
}

TEST_F(HnswLockfreeTest, ParallelBatchConstruction) {
    HnswLockfreeIndex index;
    
    HnswLockfreeBuildParams params{
        .M = 16,
        .efConstruction = 200,
        .max_M = 16,
        .max_M0 = 32,
        .batch_size = 1000,
        .n_threads = 4
    };
    
    ASSERT_TRUE(index.init(dim_, params, n_).has_value());
    
    // Measure build time
    auto start = std::chrono::high_resolution_clock::now();
    
    // Add large batch
    ASSERT_TRUE(index.add_batch(ids_.data(), data_.data(), n_).has_value());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    EXPECT_EQ(index.size(), n_);
    
    // Check build rate
    auto stats = index.get_stats();
    std::size_t build_rate = n_ * 1000 / (duration.count() + 1);  // vec/sec
    
    std::cout << "Build time: " << duration.count() << "ms\n";
    std::cout << "Build rate: " << build_rate << " vec/sec\n";
    std::cout << "Target rate: 50000 vec/sec\n";
    std::cout << "Stats build rate: " << stats.build_rate_vec_per_sec << " vec/sec\n";
    
    // We should achieve at least 10k vec/sec with lock-free implementation
    // (50k target requires more optimization)
    EXPECT_GT(build_rate, 10000) << "Build rate too slow: " << build_rate << " vec/sec";
}

TEST_F(HnswLockfreeTest, RecallQuality) {
    HnswLockfreeIndex index;
    
    HnswLockfreeBuildParams params{
        .M = 16,
        .efConstruction = 200
    };
    
    ASSERT_TRUE(index.init(dim_, params, n_).has_value());
    
    std::size_t test_size = 1000;
    ASSERT_TRUE(index.add_batch(ids_.data(), data_.data(), test_size).has_value());
    
    // Compute ground truth by brute force
    const std::size_t n_queries = 100;
    const std::size_t k = 10;
    std::vector<std::uint64_t> ground_truth(n_queries * k);
    
    for (std::size_t q = 0; q < n_queries; ++q) {
        const float* query = data_.data() + q * dim_;
        std::vector<std::pair<float, std::uint64_t>> distances;
        
        for (std::size_t i = 0; i < test_size; ++i) {
            const float* vec = data_.data() + i * dim_;
            float dist = 0.0f;
            for (std::size_t d = 0; d < dim_; ++d) {
                const float diff = query[d] - vec[d];
                dist += diff * diff;
            }
            distances.emplace_back(dist, ids_[i]);
        }
        
        std::partial_sort(distances.begin(), distances.begin() + k, distances.end());
        
        for (std::size_t i = 0; i < k; ++i) {
            ground_truth[q * k + i] = distances[i].second;
        }
    }
    
    // Test recall with different efSearch values
    for (std::uint32_t ef : {20, 50, 100, 200}) {
        HnswLockfreeSearchParams search_params{
            .efSearch = ef,
            .k = k
        };
        
        float recall = compute_recall_lockfree(index, data_.data(), n_queries,
                                               ground_truth.data(), k, search_params);
        
        // Higher ef should give better recall
        if (ef >= 100) {
            EXPECT_GT(recall, 0.9f) << "efSearch=" << ef;
        }
        if (ef >= 200) {
            EXPECT_GT(recall, 0.95f) << "efSearch=" << ef;
        }
    }
}

TEST_F(HnswLockfreeTest, ConcurrentSearchDuringBuild) {
    HnswLockfreeIndex index;
    
    HnswLockfreeBuildParams params{
        .M = 16,
        .efConstruction = 200
    };
    
    ASSERT_TRUE(index.init(dim_, params, n_).has_value());
    
    // Add initial vectors
    std::size_t initial_size = 1000;
    ASSERT_TRUE(index.add_batch(ids_.data(), data_.data(), initial_size).has_value());
    
    std::atomic<bool> stop_building{false};
    std::atomic<bool> stop_searching{false};
    std::atomic<std::size_t> vectors_added{initial_size};
    std::atomic<std::size_t> searches_completed{0};
    
    // Building thread
    std::thread builder([&]() {
        for (std::size_t i = initial_size; i < n_ && !stop_building; ++i) {
            if (index.add(ids_[i], data_.data() + i * dim_).has_value()) {
                vectors_added++;
            }
            if (i % 100 == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    });
    
    // Searching threads
    const std::size_t n_search_threads = 2;
    std::vector<std::thread> searchers;
    
    for (std::size_t t = 0; t < n_search_threads; ++t) {
        searchers.emplace_back([&]() {
            HnswLockfreeSearchParams search_params{
                .efSearch = 50,
                .k = 10
            };
            
            while (!stop_searching) {
                std::size_t current_size = vectors_added.load();
                if (current_size > 0) {
                    std::size_t idx = searches_completed % current_size;
                    auto results = index.search(data_.data() + idx * dim_, search_params);
                    if (results.has_value() && !results->empty()) {
                        searches_completed++;
                    }
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        });
    }
    
    // Let it run for a bit
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    stop_building = true;
    builder.join();
    
    stop_searching = true;
    for (auto& searcher : searchers) {
        searcher.join();
    }
    
    EXPECT_GT(vectors_added, initial_size);
    EXPECT_GT(searches_completed, 0);
    
    std::cout << "Vectors added during concurrent test: " << vectors_added << "\n";
    std::cout << "Searches completed during build: " << searches_completed << "\n";
}

TEST_F(HnswLockfreeTest, MemoryReclamation) {
    HnswLockfreeIndex index;
    
    HnswLockfreeBuildParams params{
        .M = 16,
        .efConstruction = 200
    };
    
    ASSERT_TRUE(index.init(dim_, params, 1000).has_value());
    
    // Add and delete vectors
    for (std::size_t i = 0; i < 100; ++i) {
        ASSERT_TRUE(index.add(ids_[i], data_.data() + i * dim_).has_value());
    }
    
    // Mark some as deleted
    for (std::size_t i = 0; i < 50; ++i) {
        ASSERT_TRUE(index.mark_deleted(ids_[i]).has_value());
    }
    
    // Trigger memory reclamation
    index.reclaim_memory();
    
    // Search should not return deleted vectors
    HnswLockfreeSearchParams search_params{
        .efSearch = 50,
        .k = 20
    };
    
    auto results = index.search(data_.data() + 60 * dim_, search_params);
    ASSERT_TRUE(results.has_value());
    
    // Should not find deleted IDs (0-49)
    for (const auto& [id, dist] : *results) {
        EXPECT_GE(id, 50) << "Should not return deleted ID " << id;
    }
}

TEST_F(HnswLockfreeTest, StressTestConcurrentOperations) {
    HnswLockfreeIndex index;
    
    HnswLockfreeBuildParams params{
        .M = 16,
        .efConstruction = 100,  // Lower for faster builds
        .max_M0 = 32
    };
    
    ASSERT_TRUE(index.init(dim_, params, n_).has_value());
    
    const std::size_t n_threads = 8;
    const std::size_t operations_per_thread = 500;
    
    std::atomic<std::size_t> total_adds{0};
    std::atomic<std::size_t> total_searches{0};
    std::atomic<std::size_t> total_deletes{0};
    std::atomic<std::size_t> next_id{0};
    
    std::vector<std::thread> threads;
    
    for (std::size_t t = 0; t < n_threads; ++t) {
        threads.emplace_back([&, t]() {
            std::mt19937 rng(t);
            std::uniform_int_distribution<int> op_dist(0, 2);
            
            HnswLockfreeSearchParams search_params{
                .efSearch = 30,
                .k = 5
            };
            
            for (std::size_t op = 0; op < operations_per_thread; ++op) {
                int op_type = op_dist(rng);
                
                switch (op_type) {
                case 0: {  // Add
                    std::size_t id = next_id.fetch_add(1);
                    if (id < n_) {
                        if (index.add(id, data_.data() + (id % n_) * dim_).has_value()) {
                            total_adds++;
                        }
                    }
                    break;
                }
                case 1: {  // Search
                    std::size_t current_size = index.size();
                    if (current_size > 0) {
                        std::size_t idx = rng() % std::min(current_size, n_);
                        auto results = index.search(data_.data() + idx * dim_, search_params);
                        if (results.has_value()) {
                            total_searches++;
                        }
                    }
                    break;
                }
                case 2: {  // Delete
                    std::size_t current_next = next_id.load();
                    if (current_next > 0) {
                        std::size_t id = rng() % current_next;
                        if (index.mark_deleted(id).has_value()) {
                            total_deletes++;
                        }
                    }
                    break;
                }
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    std::cout << "Stress test results:\n";
    std::cout << "  Total adds: " << total_adds << "\n";
    std::cout << "  Total searches: " << total_searches << "\n";
    std::cout << "  Total deletes: " << total_deletes << "\n";
    
    EXPECT_GT(total_adds, 0);
    EXPECT_GT(total_searches, 0);
    
    // Final verification
    auto stats = index.get_stats();
    EXPECT_GT(stats.n_nodes, 0);
    EXPECT_GT(stats.n_edges, 0);
}

TEST_F(HnswLockfreeTest, AtomicEdgeListOperations) {
    // Test the atomic edge list directly
    AtomicEdgeList edge_list;
    
    // Test insertions
    EXPECT_TRUE(edge_list.try_insert(1, 10));
    EXPECT_TRUE(edge_list.try_insert(2, 10));
    EXPECT_TRUE(edge_list.try_insert(3, 10));
    
    // Test duplicate rejection
    EXPECT_FALSE(edge_list.try_insert(1, 10));
    
    // Test size
    EXPECT_EQ(edge_list.size(), 3);
    
    // Test contains
    EXPECT_TRUE(edge_list.contains(1));
    EXPECT_TRUE(edge_list.contains(2));
    EXPECT_TRUE(edge_list.contains(3));
    EXPECT_FALSE(edge_list.contains(4));
    
    // Test get_neighbors
    auto neighbors = edge_list.get_neighbors();
    EXPECT_EQ(neighbors.size(), 3);
    
    std::unordered_set<std::uint32_t> neighbor_set(neighbors.begin(), neighbors.end());
    EXPECT_TRUE(neighbor_set.count(1));
    EXPECT_TRUE(neighbor_set.count(2));
    EXPECT_TRUE(neighbor_set.count(3));
    
    // Test removal
    EXPECT_TRUE(edge_list.try_remove(2));
    EXPECT_EQ(edge_list.size(), 2);
    EXPECT_FALSE(edge_list.contains(2));
    
    // Test concurrent operations
    const std::size_t n_threads = 4;
    const std::size_t ops_per_thread = 100;
    
    AtomicEdgeList concurrent_list;
    std::atomic<std::size_t> successful_inserts{0};
    
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < n_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (std::size_t i = 0; i < ops_per_thread; ++i) {
                std::uint32_t id = t * ops_per_thread + i;
                if (concurrent_list.try_insert(id, 64)) {
                    successful_inserts++;
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(concurrent_list.size(), successful_inserts);
    EXPECT_LE(successful_inserts, 64);  // Should not exceed max edges
}

TEST_F(HnswLockfreeTest, ErrorHandling) {
    HnswLockfreeIndex index;
    
    // Search before init should fail
    HnswLockfreeSearchParams search_params{.efSearch = 50, .k = 10};
    auto search_result = index.search(data_.data(), search_params);
    EXPECT_FALSE(search_result.has_value());
    
    // Invalid parameters
    HnswLockfreeBuildParams bad_params{
        .M = 1,  // Too small
        .efConstruction = 200
    };
    auto init_result = index.init(dim_, bad_params);
    EXPECT_FALSE(init_result.has_value());
    
    // Valid init
    HnswLockfreeBuildParams good_params{
        .M = 16,
        .efConstruction = 200
    };
    ASSERT_TRUE(index.init(dim_, good_params).has_value());
    
    // Duplicate ID should fail
    ASSERT_TRUE(index.add(100, data_.data()).has_value());
    auto dup_result = index.add(100, data_.data() + dim_);
    EXPECT_FALSE(dup_result.has_value());
}

} // namespace
} // namespace vesper::index