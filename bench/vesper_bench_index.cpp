#include <benchmark/benchmark.h>
#include "vesper/index/ivf_pq.hpp"
#include "vesper/index/hnsw.hpp"
#include "vesper/kernels/dispatch.hpp"

#include <random>
#include <numeric>
#include <chrono>
#include <fstream>
#include <vector>
#include <algorithm>

namespace vesper::bench {

/** \brief Generate synthetic dataset. */
class DatasetGenerator {
public:
    DatasetGenerator(std::size_t n, std::size_t dim, std::uint32_t seed = 42)
        : n_(n), dim_(dim), data_(n * dim), ids_(n) {
        
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (auto& val : data_) {
            val = dist(gen);
        }
        
        // Normalize vectors for cosine similarity
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
    }
    
    auto data() const -> const float* { return data_.data(); }
    auto ids() const -> const std::uint64_t* { return ids_.data(); }
    auto size() const -> std::size_t { return n_; }
    auto dim() const -> std::size_t { return dim_; }
    
    /** \brief Get queries (subset of data). */
    auto queries(std::size_t n_queries) const -> std::vector<float> {
        std::vector<float> q(n_queries * dim_);
        std::copy(data_.begin(), data_.begin() + n_queries * dim_, q.begin());
        return q;
    }
    
    /** \brief Compute ground truth by brute force. */
    auto ground_truth(const float* queries, std::size_t n_queries, std::size_t k) const 
        -> std::vector<std::uint64_t> {
        
        std::vector<std::uint64_t> gt(n_queries * k);
        
        #pragma omp parallel for
        for (int q = 0; q < static_cast<int>(n_queries); ++q) {
            const float* query = queries + static_cast<std::size_t>(q) * dim_;
            std::vector<std::pair<float, std::uint64_t>> distances;
            
            for (std::size_t i = 0; i < n_; ++i) {
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
                gt[static_cast<std::size_t>(q) * k + i] = distances[i].second;
            }
        }
        
        return gt;
    }
    
private:
    std::size_t n_;
    std::size_t dim_;
    std::vector<float> data_;
    std::vector<std::uint64_t> ids_;
};

/** \brief Measure latency percentiles. */
class LatencyTracker {
public:
    void record(double ms) {
        latencies_.push_back(ms);
    }
    
    auto percentile(double p) -> double {
        if (latencies_.empty()) return 0.0;
        
        std::sort(latencies_.begin(), latencies_.end());
        const std::size_t idx = static_cast<std::size_t>(latencies_.size() * p / 100.0);
        return latencies_[std::min(idx, latencies_.size() - 1)];
    }
    
    auto mean() -> double {
        if (latencies_.empty()) return 0.0;
        return std::accumulate(latencies_.begin(), latencies_.end(), 0.0) / latencies_.size();
    }
    
    void clear() {
        latencies_.clear();
    }
    
private:
    std::vector<double> latencies_;
};

// IVF-PQ Benchmarks

static void BM_IvfPq_Build(benchmark::State& state) {
    const std::size_t n = state.range(0);
    const std::size_t dim = state.range(1);
    const std::uint32_t nlist = state.range(2);
    
    DatasetGenerator dataset(n, dim);
    
    for (auto _ : state) {
        index::IvfPqIndex index;
        index::IvfPqTrainParams params{
            .nlist = nlist,
            .m = 8,
            .nbits = 8,
            .max_iter = 25,
            .epsilon = 1e-4f,
            .verbose = false,
            .use_opq = false
        };
        
        auto start = std::chrono::high_resolution_clock::now();
        
        auto train_result = index.train(dataset.data(), dim, n / 2, params);
        if (train_result.has_value()) {
            index.add(dataset.ids(), dataset.data(), n);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end - start);
        
        state.SetIterationTime(duration.count());
        
        // Report metrics
        state.counters["vectors_per_sec"] = n / duration.count();
        state.counters["build_time_ms"] = duration.count() * 1000;
    }
}

static void BM_IvfPq_Search(benchmark::State& state) {
    const std::size_t n = 100000;
    const std::size_t dim = state.range(0);
    const std::uint32_t nprobe = state.range(1);
    const std::uint32_t k = 10;
    
    DatasetGenerator dataset(n, dim);
    
    // Build index
    index::IvfPqIndex index;
    index::IvfPqTrainParams params{
        .nlist = 1000,
        .m = 8,
        .nbits = 8
    };
    
    index.train(dataset.data(), dim, n / 2, params);
    index.add(dataset.ids(), dataset.data(), n);
    
    // Prepare queries
    const std::size_t n_queries = 1000;
    auto queries = dataset.queries(n_queries);
    
    LatencyTracker tracker;
    
    for (auto _ : state) {
        index::IvfPqSearchParams search_params{
            .nprobe = nprobe,
            .k = k
        };
        
        auto start = std::chrono::high_resolution_clock::now();
        auto results = index.search(queries.data(), search_params);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        tracker.record(duration_ms);
    }
    
    state.counters["p50_ms"] = tracker.percentile(50);
    state.counters["p90_ms"] = tracker.percentile(90);
    state.counters["p99_ms"] = tracker.percentile(99);
    state.counters["mean_ms"] = tracker.mean();
}

// HNSW Benchmarks

static void BM_Hnsw_Build(benchmark::State& state) {
    const std::size_t n = state.range(0);
    const std::size_t dim = state.range(1);
    const std::uint32_t M = state.range(2);
    
    DatasetGenerator dataset(n, dim);
    
    for (auto _ : state) {
        index::HnswIndex index;
        index::HnswBuildParams params{
            .M = M,
            .efConstruction = 200,
            .seed = 42
        };
        
        auto start = std::chrono::high_resolution_clock::now();
        
        index.init(dim, params, n);
        index.add_batch(dataset.ids(), dataset.data(), n);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end - start);
        
        state.SetIterationTime(duration.count());
        
        state.counters["vectors_per_sec"] = n / duration.count();
        state.counters["build_time_ms"] = duration.count() * 1000;
        
        auto stats = index.get_stats();
        state.counters["memory_mb"] = stats.memory_bytes / (1024.0 * 1024.0);
    }
}

static void BM_Hnsw_Search(benchmark::State& state) {
    const std::size_t n = 100000;
    const std::size_t dim = state.range(0);
    const std::uint32_t ef = state.range(1);
    const std::uint32_t k = 10;
    
    DatasetGenerator dataset(n, dim);
    
    // Build index
    index::HnswIndex index;
    index::HnswBuildParams params{
        .M = 16,
        .efConstruction = 200
    };
    
    index.init(dim, params, n);
    index.add_batch(dataset.ids(), dataset.data(), n);
    
    // Prepare queries
    const std::size_t n_queries = 1000;
    auto queries = dataset.queries(n_queries);
    
    LatencyTracker tracker;
    
    for (auto _ : state) {
        index::HnswSearchParams search_params{
            .efSearch = ef,
            .k = k
        };
        
        auto start = std::chrono::high_resolution_clock::now();
        auto results = index.search(queries.data(), search_params);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        tracker.record(duration_ms);
    }
    
    state.counters["p50_ms"] = tracker.percentile(50);
    state.counters["p90_ms"] = tracker.percentile(90);
    state.counters["p99_ms"] = tracker.percentile(99);
    state.counters["mean_ms"] = tracker.mean();
}

// Recall Benchmarks

static void BM_Recall_Comparison(benchmark::State& state) {
    const std::size_t n = 10000;
    const std::size_t dim = 128;
    const std::size_t n_queries = 100;
    const std::size_t k = 10;
    
    DatasetGenerator dataset(n, dim);
    auto queries = dataset.queries(n_queries);
    auto ground_truth = dataset.ground_truth(queries.data(), n_queries, k);
    
    // Build IVF-PQ
    index::IvfPqIndex ivf_pq;
    index::IvfPqTrainParams ivf_params{
        .nlist = 100,
        .m = 8,
        .nbits = 8
    };
    ivf_pq.train(dataset.data(), dim, n / 2, ivf_params);
    ivf_pq.add(dataset.ids(), dataset.data(), n);
    
    // Build HNSW
    index::HnswIndex hnsw;
    index::HnswBuildParams hnsw_params{
        .M = 16,
        .efConstruction = 200
    };
    hnsw.init(dim, hnsw_params, n);
    hnsw.add_batch(dataset.ids(), dataset.data(), n);
    
    for (auto _ : state) {
        // Test IVF-PQ recall
        index::IvfPqSearchParams ivf_search{.nprobe = 8, .k = k};
        float ivf_recall = index::compute_recall(ivf_pq, queries.data(), n_queries,
                                                 ground_truth.data(), k, ivf_search);
        
        // Test HNSW recall
        index::HnswSearchParams hnsw_search{.efSearch = 50, .k = k};
        float hnsw_recall = index::compute_recall(hnsw, queries.data(), n_queries,
                                                  ground_truth.data(), k, hnsw_search);
        
        state.counters["ivf_pq_recall"] = ivf_recall;
        state.counters["hnsw_recall"] = hnsw_recall;
    }
}

// Blueprint Target Validation

static void BM_Blueprint_Targets(benchmark::State& state) {
    // Test against blueprint targets:
    // P50 ≤ 1-3ms, P99 ≤ 10-20ms, Recall@10 ≈ 0.95
    
    const std::size_t n = 1000000;  // 1M vectors
    const std::size_t dim = state.range(0);  // 128, 768, 1536
    const std::size_t n_queries = 1000;
    const std::size_t k = 10;
    
    DatasetGenerator dataset(n, dim);
    auto queries = dataset.queries(n_queries);
    auto ground_truth = dataset.ground_truth(queries.data(), std::min(n_queries, std::size_t(100)), k);
    
    // Build HNSW (for hot segments)
    index::HnswIndex hnsw;
    index::HnswBuildParams params{
        .M = 16,
        .efConstruction = 200
    };
    
    auto init_start = std::chrono::high_resolution_clock::now();
    hnsw.init(dim, params, n);
    hnsw.add_batch(dataset.ids(), dataset.data(), n);
    auto init_end = std::chrono::high_resolution_clock::now();
    
    auto build_time = std::chrono::duration<double>(init_end - init_start).count();
    state.counters["build_vectors_per_sec"] = n / build_time;
    
    LatencyTracker tracker;
    std::size_t total_recalled = 0;
    
    for (auto _ : state) {
        index::HnswSearchParams search_params{
            .efSearch = 100,  // Tuned for recall ~0.95
            .k = k
        };
        
        for (std::size_t q = 0; q < std::min(n_queries, std::size_t(100)); ++q) {
            auto start = std::chrono::high_resolution_clock::now();
            auto results = hnsw.search(queries.data() + q * dim, search_params);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
            tracker.record(duration_ms);
            
            // Check recall
            if (results.has_value() && q < 100) {
                const std::uint64_t* gt = ground_truth.data() + q * k;
                for (const auto& [id, dist] : *results) {
                    for (std::size_t i = 0; i < k; ++i) {
                        if (id == gt[i]) {
                            total_recalled++;
                            break;
                        }
                    }
                }
            }
        }
    }
    
    const float recall = static_cast<float>(total_recalled) / (100 * k);
    
    state.counters["p50_ms"] = tracker.percentile(50);
    state.counters["p99_ms"] = tracker.percentile(99);
    state.counters["recall@10"] = recall;
    
    // Check against targets
    state.counters["meets_p50_target"] = (tracker.percentile(50) <= 3.0) ? 1.0 : 0.0;
    state.counters["meets_p99_target"] = (tracker.percentile(99) <= 20.0) ? 1.0 : 0.0;
    state.counters["meets_recall_target"] = (recall >= 0.90) ? 1.0 : 0.0;
}

// Register benchmarks

// IVF-PQ benchmarks
BENCHMARK(BM_IvfPq_Build)
    ->Args({10000, 128, 100})    // 10K vectors, 128D, 100 lists
    ->Args({100000, 128, 1000})  // 100K vectors, 128D, 1000 lists
    ->Args({100000, 768, 1000})  // 100K vectors, 768D (SBERT)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_IvfPq_Search)
    ->Args({128, 1})    // 128D, nprobe=1
    ->Args({128, 8})    // 128D, nprobe=8
    ->Args({128, 16})   // 128D, nprobe=16
    ->Args({768, 8})    // 768D, nprobe=8
    ->Unit(benchmark::kMillisecond);

// HNSW benchmarks
BENCHMARK(BM_Hnsw_Build)
    ->Args({10000, 128, 16})    // 10K vectors, 128D, M=16
    ->Args({100000, 128, 16})   // 100K vectors, 128D, M=16
    ->Args({100000, 768, 16})   // 100K vectors, 768D
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Hnsw_Search)
    ->Args({128, 20})   // 128D, ef=20
    ->Args({128, 50})   // 128D, ef=50
    ->Args({128, 100})  // 128D, ef=100
    ->Args({768, 50})   // 768D, ef=50
    ->Unit(benchmark::kMillisecond);

// Comparison benchmarks
BENCHMARK(BM_Recall_Comparison)
    ->Unit(benchmark::kMillisecond);

// Blueprint validation
BENCHMARK(BM_Blueprint_Targets)
    ->Args({128})   // SIFT-like
    ->Args({768})   // SBERT
    ->Args({1536})  // Large embeddings
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);

} // namespace vesper::bench

BENCHMARK_MAIN();