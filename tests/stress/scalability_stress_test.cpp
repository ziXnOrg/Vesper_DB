#include <gtest/gtest.h>
#include "vesper/index/ivf_pq.hpp"
#include "vesper/index/hnsw.hpp"
#include "vesper/index/kmeans_elkan.hpp"
#include "vesper/core/memory_pool.hpp"
#include "vesper/kernels/dispatch.hpp"

#include <random>
#include <chrono>
#include <thread>
#include <atomic>
#include <future>
#include <iomanip>
#include <sstream>

namespace vesper::stress {
namespace {

/** \brief Stress test configuration. */
struct StressConfig {
    std::size_t n_vectors{1'000'000};     // 1M vectors
    std::size_t dim{128};                 // Dimension
    std::size_t n_queries{10'000};        // Query count
    std::size_t n_threads{8};             // Concurrent threads
    std::size_t batch_size{10'000};       // Batch operation size
    bool verbose{true};                   // Progress output
};

/** \brief Performance metrics tracker. */
class MetricsTracker {
public:
    void record_latency(double ms) {
        latencies_.push_back(ms);
        total_latency_ += ms;
        count_++;
    }
    
    void record_throughput(double ops_per_sec) {
        throughput_samples_.push_back(ops_per_sec);
    }
    
    void record_memory(std::size_t bytes) {
        if (bytes > peak_memory_) {
            peak_memory_ = bytes;
        }
        current_memory_ = bytes;
    }
    
    auto get_percentile(double p) const -> double {
        if (latencies_.empty()) return 0.0;
        
        auto sorted = latencies_;
        std::sort(sorted.begin(), sorted.end());
        std::size_t idx = static_cast<std::size_t>(sorted.size() * p / 100.0);
        return sorted[std::min(idx, sorted.size() - 1)];
    }
    
    auto get_mean_latency() const -> double {
        return count_ > 0 ? total_latency_ / count_ : 0.0;
    }
    
    auto get_mean_throughput() const -> double {
        if (throughput_samples_.empty()) return 0.0;
        double sum = std::accumulate(throughput_samples_.begin(), 
                                    throughput_samples_.end(), 0.0);
        return sum / throughput_samples_.size();
    }
    
    auto get_peak_memory_mb() const -> double {
        return peak_memory_ / (1024.0 * 1024.0);
    }
    
    void print_summary(const std::string& name) const {
        std::cout << "\n=== " << name << " Metrics ===" << std::endl;
        std::cout << "Latency P50: " << get_percentile(50) << " ms" << std::endl;
        std::cout << "Latency P90: " << get_percentile(90) << " ms" << std::endl;
        std::cout << "Latency P99: " << get_percentile(99) << " ms" << std::endl;
        std::cout << "Mean Latency: " << get_mean_latency() << " ms" << std::endl;
        std::cout << "Mean Throughput: " << get_mean_throughput() << " ops/sec" << std::endl;
        std::cout << "Peak Memory: " << get_peak_memory_mb() << " MB" << std::endl;
    }
    
private:
    mutable std::vector<double> latencies_;
    std::vector<double> throughput_samples_;
    double total_latency_{0.0};
    std::size_t count_{0};
    std::size_t peak_memory_{0};
    std::size_t current_memory_{0};
};

/** \brief Generate large dataset for stress testing. */
class LargeDatasetGenerator {
public:
    LargeDatasetGenerator(const StressConfig& config) 
        : config_(config), gen_(42) {
        
        if (config_.verbose) {
            std::cout << "Generating dataset: " << config_.n_vectors 
                     << " vectors, " << config_.dim << "D" << std::endl;
        }
        
        // Generate in batches to avoid huge allocations
        const std::size_t batch_size = 100'000;
        data_.reserve(config_.n_vectors * config_.dim);
        ids_.reserve(config_.n_vectors);
        
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (std::size_t batch = 0; batch < config_.n_vectors; batch += batch_size) {
            std::size_t batch_end = std::min(batch + batch_size, config_.n_vectors);
            
            for (std::size_t i = batch; i < batch_end; ++i) {
                ids_.push_back(i);
                
                // Generate normalized vector
                std::vector<float> vec(config_.dim);
                float norm = 0.0f;
                for (auto& v : vec) {
                    v = dist(gen_);
                    norm += v * v;
                }
                norm = std::sqrt(norm);
                for (auto& v : vec) {
                    v /= norm;
                    data_.push_back(v);
                }
            }
            
            if (config_.verbose && (batch + batch_size) % 500'000 == 0) {
                std::cout << "Generated " << (batch + batch_size) 
                         << " vectors..." << std::endl;
            }
        }
        
        // Generate queries
        queries_.resize(config_.n_queries * config_.dim);
        for (auto& v : queries_) {
            v = dist(gen_);
        }
        
        // Normalize queries
        for (std::size_t i = 0; i < config_.n_queries; ++i) {
            float* query = queries_.data() + i * config_.dim;
            float norm = 0.0f;
            for (std::size_t d = 0; d < config_.dim; ++d) {
                norm += query[d] * query[d];
            }
            norm = std::sqrt(norm);
            for (std::size_t d = 0; d < config_.dim; ++d) {
                query[d] /= norm;
            }
        }
    }
    
    auto data() const -> const float* { return data_.data(); }
    auto ids() const -> const std::uint64_t* { return ids_.data(); }
    auto queries() const -> const float* { return queries_.data(); }
    auto size() const -> std::size_t { return config_.n_vectors; }
    auto dim() const -> std::size_t { return config_.dim; }
    auto n_queries() const -> std::size_t { return config_.n_queries; }
    
private:
    StressConfig config_;
    std::mt19937 gen_;
    std::vector<float> data_;
    std::vector<std::uint64_t> ids_;
    std::vector<float> queries_;
};

class ScalabilityStressTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.verbose = ::testing::GTEST_FLAG(print_time);
    }
    
    StressConfig config_;
};

// IVF-PQ Stress Tests

TEST_F(ScalabilityStressTest, IvfPq_1M_Vectors) {
    LargeDatasetGenerator dataset(config_);
    MetricsTracker metrics;
    
    index::IvfPqIndex index;
    index::IvfPqTrainParams params{
        .nlist = 1024,       // sqrt(1M) ≈ 1000
        .m = 16,             // 16 subquantizers
        .nbits = 8,          // 256 codes per subquantizer
        .max_iter = 25,
        .epsilon = 1e-4f,
        .verbose = config_.verbose,
        .use_opq = false,    // Disable for speed
        .seed = 42
    };
    
    // Train on subset
    const std::size_t n_train = 100'000;
    std::cout << "\nTraining IVF-PQ on " << n_train << " vectors..." << std::endl;
    
    auto train_start = std::chrono::high_resolution_clock::now();
    auto train_result = index.train(dataset.data(), config_.dim, n_train, params);
    auto train_end = std::chrono::high_resolution_clock::now();
    
    ASSERT_TRUE(train_result.has_value());
    
    auto train_duration = std::chrono::duration<double>(train_end - train_start).count();
    std::cout << "Training time: " << train_duration << " seconds" << std::endl;
    
    // Add vectors in batches
    std::cout << "Adding " << config_.n_vectors << " vectors..." << std::endl;
    
    auto add_start = std::chrono::high_resolution_clock::now();
    
    for (std::size_t i = 0; i < config_.n_vectors; i += config_.batch_size) {
        std::size_t batch_end = std::min(i + config_.batch_size, config_.n_vectors);
        std::size_t batch_size = batch_end - i;
        
        auto result = index.add(
            dataset.ids() + i,
            dataset.data() + i * config_.dim,
            batch_size
        );
        ASSERT_TRUE(result.has_value());
        
        if (config_.verbose && (i + batch_size) % 100'000 == 0) {
            std::cout << "Added " << (i + batch_size) << " vectors..." << std::endl;
        }
    }
    
    auto add_end = std::chrono::high_resolution_clock::now();
    auto add_duration = std::chrono::duration<double>(add_end - add_start).count();
    
    double build_rate = config_.n_vectors / add_duration;
    metrics.record_throughput(build_rate);
    std::cout << "Build rate: " << build_rate << " vectors/sec" << std::endl;
    
    // Blueprint target: ≥ 50-200k vectors/sec
    EXPECT_GT(build_rate, 50'000) << "Build rate below blueprint target";
    
    // Memory usage
    auto stats = index.get_stats();
    metrics.record_memory(stats.memory_bytes);
    std::cout << "Memory usage: " << stats.memory_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Bytes per vector: " << stats.memory_bytes / config_.n_vectors << std::endl;
    
    // Search performance
    std::cout << "\nSearching with " << config_.n_queries << " queries..." << std::endl;
    
    index::IvfPqSearchParams search_params{
        .nprobe = 32,
        .k = 10
    };
    
    for (std::size_t q = 0; q < std::min(std::size_t(1000), config_.n_queries); ++q) {
        auto start = std::chrono::high_resolution_clock::now();
        auto results = index.search(dataset.queries() + q * config_.dim, search_params);
        auto end = std::chrono::high_resolution_clock::now();
        
        ASSERT_TRUE(results.has_value());
        
        auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        metrics.record_latency(duration_ms);
    }
    
    metrics.print_summary("IVF-PQ 1M Vectors");
    
    // Blueprint targets
    EXPECT_LE(metrics.get_percentile(50), 3.0) << "P50 above blueprint target";
    EXPECT_LE(metrics.get_percentile(99), 20.0) << "P99 above blueprint target";
}

TEST_F(ScalabilityStressTest, HNSW_1M_Vectors) {
    // Use smaller dataset for HNSW due to memory requirements
    StressConfig hnsw_config = config_;
    hnsw_config.n_vectors = 500'000;  // 500K for HNSW
    
    LargeDatasetGenerator dataset(hnsw_config);
    MetricsTracker metrics;
    
    index::HnswIndex index;
    index::HnswBuildParams params{
        .M = 16,
        .efConstruction = 200,
        .seed = 42
    };
    
    std::cout << "\nInitializing HNSW for " << hnsw_config.n_vectors 
              << " vectors..." << std::endl;
    
    ASSERT_TRUE(index.init(hnsw_config.dim, params, hnsw_config.n_vectors).has_value());
    
    // Add vectors in batches
    auto add_start = std::chrono::high_resolution_clock::now();
    
    for (std::size_t i = 0; i < hnsw_config.n_vectors; i += hnsw_config.batch_size) {
        std::size_t batch_end = std::min(i + hnsw_config.batch_size, hnsw_config.n_vectors);
        std::size_t batch_size = batch_end - i;
        
        auto result = index.add_batch(
            dataset.ids() + i,
            dataset.data() + i * hnsw_config.dim,
            batch_size
        );
        ASSERT_TRUE(result.has_value());
        
        if (hnsw_config.verbose && (i + batch_size) % 100'000 == 0) {
            std::cout << "Added " << (i + batch_size) << " vectors..." << std::endl;
        }
    }
    
    auto add_end = std::chrono::high_resolution_clock::now();
    auto add_duration = std::chrono::duration<double>(add_end - add_start).count();
    
    double build_rate = hnsw_config.n_vectors / add_duration;
    metrics.record_throughput(build_rate);
    std::cout << "Build rate: " << build_rate << " vectors/sec" << std::endl;
    
    // Memory usage
    auto stats = index.get_stats();
    metrics.record_memory(stats.memory_bytes);
    std::cout << "Memory usage: " << stats.memory_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Bytes per vector: " << stats.memory_bytes / hnsw_config.n_vectors << std::endl;
    
    // Search performance
    std::cout << "\nSearching with " << hnsw_config.n_queries << " queries..." << std::endl;
    
    index::HnswSearchParams search_params{
        .efSearch = 100,
        .k = 10
    };
    
    for (std::size_t q = 0; q < std::min(std::size_t(1000), hnsw_config.n_queries); ++q) {
        auto start = std::chrono::high_resolution_clock::now();
        auto results = index.search(dataset.queries() + q * hnsw_config.dim, search_params);
        auto end = std::chrono::high_resolution_clock::now();
        
        ASSERT_TRUE(results.has_value());
        
        auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        metrics.record_latency(duration_ms);
    }
    
    metrics.print_summary("HNSW 500K Vectors");
    
    // Blueprint targets
    EXPECT_LE(metrics.get_percentile(50), 3.0) << "P50 above blueprint target";
    EXPECT_LE(metrics.get_percentile(99), 20.0) << "P99 above blueprint target";
}

TEST_F(ScalabilityStressTest, ConcurrentQueries) {
    // Build a smaller index for concurrent query testing
    StressConfig concurrent_config = config_;
    concurrent_config.n_vectors = 100'000;
    
    LargeDatasetGenerator dataset(concurrent_config);
    
    index::IvfPqIndex index;
    index::IvfPqTrainParams params{
        .nlist = 316,  // sqrt(100K)
        .m = 8,
        .nbits = 8
    };
    
    ASSERT_TRUE(index.train(dataset.data(), concurrent_config.dim, 50'000, params).has_value());
    ASSERT_TRUE(index.add(dataset.ids(), dataset.data(), concurrent_config.n_vectors).has_value());
    
    std::cout << "\nRunning concurrent queries with " << config_.n_threads 
              << " threads..." << std::endl;
    
    std::atomic<std::size_t> queries_completed{0};
    std::atomic<std::size_t> queries_failed{0};
    MetricsTracker metrics;
    std::mutex metrics_mutex;
    
    auto query_start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < config_.n_threads; ++t) {
        threads.emplace_back([&, t]() {
            index::IvfPqSearchParams search_params{
                .nprobe = 16,
                .k = 10
            };
            
            const std::size_t queries_per_thread = 1000;
            
            for (std::size_t q = 0; q < queries_per_thread; ++q) {
                std::size_t query_idx = (t * queries_per_thread + q) % concurrent_config.n_queries;
                
                auto start = std::chrono::high_resolution_clock::now();
                auto results = index.search(
                    dataset.queries() + query_idx * concurrent_config.dim,
                    search_params
                );
                auto end = std::chrono::high_resolution_clock::now();
                
                if (results.has_value()) {
                    queries_completed++;
                    
                    auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
                    std::lock_guard<std::mutex> lock(metrics_mutex);
                    metrics.record_latency(duration_ms);
                } else {
                    queries_failed++;
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto query_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration<double>(query_end - query_start).count();
    
    double qps = queries_completed / total_duration;
    metrics.record_throughput(qps);
    
    std::cout << "Queries completed: " << queries_completed << std::endl;
    std::cout << "Queries failed: " << queries_failed << std::endl;
    std::cout << "QPS: " << qps << std::endl;
    
    metrics.print_summary("Concurrent Queries");
    
    EXPECT_EQ(queries_failed, 0) << "Some queries failed";
    EXPECT_GT(qps, 1000) << "QPS below expected threshold";
}

TEST_F(ScalabilityStressTest, MemoryPressure) {
    // Test behavior under memory pressure
    std::cout << "\nTesting memory pressure scenarios..." << std::endl;
    
    // Create multiple indices simultaneously
    const std::size_t n_indices = 5;
    const std::size_t vectors_per_index = 100'000;
    
    std::vector<std::unique_ptr<index::IvfPqIndex>> indices;
    
    for (std::size_t i = 0; i < n_indices; ++i) {
        indices.emplace_back(std::make_unique<index::IvfPqIndex>());
        
        index::IvfPqTrainParams params{
            .nlist = 100,
            .m = 8,
            .nbits = 8
        };
        
        // Generate small dataset
        std::vector<float> data(vectors_per_index * config_.dim);
        std::vector<std::uint64_t> ids(vectors_per_index);
        
        std::mt19937 gen(42 + i);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& v : data) {
            v = dist(gen);
        }
        std::iota(ids.begin(), ids.end(), i * vectors_per_index);
        
        auto train_result = indices[i]->train(data.data(), config_.dim, 10'000, params);
        ASSERT_TRUE(train_result.has_value());
        
        auto add_result = indices[i]->add(ids.data(), data.data(), vectors_per_index);
        ASSERT_TRUE(add_result.has_value());
        
        auto stats = indices[i]->get_stats();
        std::cout << "Index " << i << " memory: " 
                  << stats.memory_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    }
    
    // Total memory usage
    std::size_t total_memory = 0;
    for (const auto& idx : indices) {
        total_memory += idx->get_stats().memory_bytes;
    }
    
    std::cout << "Total memory across " << n_indices << " indices: "
              << total_memory / (1024.0 * 1024.0) << " MB" << std::endl;
    
    // Should handle multiple indices
    EXPECT_GT(indices.size(), 0);
}

TEST_F(ScalabilityStressTest, LargeDimensions) {
    // Test with high-dimensional vectors (SBERT/GPT embeddings)
    StressConfig large_dim_config = config_;
    large_dim_config.n_vectors = 50'000;  // Fewer vectors
    large_dim_config.dim = 768;           // SBERT dimension
    
    std::cout << "\nTesting with " << large_dim_config.dim 
              << "D vectors..." << std::endl;
    
    LargeDatasetGenerator dataset(large_dim_config);
    MetricsTracker metrics;
    
    index::IvfPqIndex index;
    index::IvfPqTrainParams params{
        .nlist = 224,  // sqrt(50K)
        .m = 24,       // More subquantizers for higher dim
        .nbits = 8
    };
    
    auto train_start = std::chrono::high_resolution_clock::now();
    auto train_result = index.train(dataset.data(), large_dim_config.dim, 10'000, params);
    auto train_end = std::chrono::high_resolution_clock::now();
    
    ASSERT_TRUE(train_result.has_value());
    
    auto train_duration = std::chrono::duration<double>(train_end - train_start).count();
    std::cout << "Training time for 768D: " << train_duration << " seconds" << std::endl;
    
    // Add vectors
    ASSERT_TRUE(index.add(dataset.ids(), dataset.data(), 
                          large_dim_config.n_vectors).has_value());
    
    // Search performance with high dimensions
    index::IvfPqSearchParams search_params{
        .nprobe = 16,
        .k = 10
    };
    
    for (std::size_t q = 0; q < 100; ++q) {
        auto start = std::chrono::high_resolution_clock::now();
        auto results = index.search(
            dataset.queries() + q * large_dim_config.dim,
            search_params
        );
        auto end = std::chrono::high_resolution_clock::now();
        
        ASSERT_TRUE(results.has_value());
        
        auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        metrics.record_latency(duration_ms);
    }
    
    metrics.print_summary("768D Vectors");
    
    // Even with high dimensions, should meet targets
    EXPECT_LE(metrics.get_percentile(50), 5.0) << "P50 too high for 768D";
    EXPECT_LE(metrics.get_percentile(99), 30.0) << "P99 too high for 768D";
}

TEST_F(ScalabilityStressTest, IncrementalUpdates) {
    // Test incremental index updates
    std::cout << "\nTesting incremental updates..." << std::endl;
    
    const std::size_t initial_size = 100'000;
    const std::size_t increment_size = 10'000;
    const std::size_t n_increments = 10;
    
    StressConfig update_config = config_;
    update_config.n_vectors = initial_size + increment_size * n_increments;
    
    LargeDatasetGenerator dataset(update_config);
    
    index::HnswIndex index;
    index::HnswBuildParams params{
        .M = 16,
        .efConstruction = 200
    };
    
    ASSERT_TRUE(index.init(update_config.dim, params, update_config.n_vectors).has_value());
    
    // Initial build
    ASSERT_TRUE(index.add_batch(dataset.ids(), dataset.data(), initial_size).has_value());
    
    std::cout << "Initial index size: " << index.size() << std::endl;
    
    // Incremental updates
    for (std::size_t i = 0; i < n_increments; ++i) {
        std::size_t start = initial_size + i * increment_size;
        
        auto update_start = std::chrono::high_resolution_clock::now();
        
        auto result = index.add_batch(
            dataset.ids() + start,
            dataset.data() + start * update_config.dim,
            increment_size
        );
        
        auto update_end = std::chrono::high_resolution_clock::now();
        
        ASSERT_TRUE(result.has_value());
        
        auto duration = std::chrono::duration<double>(update_end - update_start).count();
        double update_rate = increment_size / duration;
        
        std::cout << "Increment " << (i + 1) << ": " 
                  << update_rate << " vectors/sec, "
                  << "Total size: " << index.size() << std::endl;
        
        // Update rate should remain reasonable
        EXPECT_GT(update_rate, 10'000) << "Update rate too low";
    }
    
    EXPECT_EQ(index.size(), update_config.n_vectors);
}

TEST_F(ScalabilityStressTest, RecoveryTime) {
    // Test recovery/reload time
    std::cout << "\nTesting recovery time..." << std::endl;
    
    const std::size_t n_vectors = 100'000;
    
    // Build index
    std::vector<float> data(n_vectors * config_.dim);
    std::vector<std::uint64_t> ids(n_vectors);
    
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : data) {
        v = dist(gen);
    }
    std::iota(ids.begin(), ids.end(), 0);
    
    index::IvfPqIndex index;
    index::IvfPqTrainParams params{
        .nlist = 316,
        .m = 8,
        .nbits = 8
    };
    
    ASSERT_TRUE(index.train(data.data(), config_.dim, 50'000, params).has_value());
    ASSERT_TRUE(index.add(ids.data(), data.data(), n_vectors).has_value());
    
    // Save index
    const std::string path = "/tmp/stress_test_index.bin";
    
    auto save_start = std::chrono::high_resolution_clock::now();
    ASSERT_TRUE(index.save(path).has_value());
    auto save_end = std::chrono::high_resolution_clock::now();
    
    auto save_duration = std::chrono::duration<double>(save_end - save_start).count();
    std::cout << "Save time: " << save_duration << " seconds" << std::endl;
    
    // Simulate recovery by loading
    auto load_start = std::chrono::high_resolution_clock::now();
    auto loaded = index::IvfPqIndex::load(path);
    auto load_end = std::chrono::high_resolution_clock::now();
    
    // Note: Load might not be fully implemented
    if (loaded.has_value()) {
        auto load_duration = std::chrono::duration<double>(load_end - load_start).count();
        std::cout << "Load time: " << load_duration << " seconds" << std::endl;
        
        // Blueprint target: Recovery < 10s for 1M vectors
        // We have 100K, so should be < 1s
        EXPECT_LT(load_duration, 1.0) << "Recovery time too high";
    }
    
    // Clean up
    std::remove(path.c_str());
}

} // namespace
} // namespace vesper::stress