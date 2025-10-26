/**
 * Debug version of reranking test with more diagnostics
 */

#include "vesper/index/ivf_pq.hpp"
#include <iostream>
#include <random>
#include <vector>
#include <iomanip>
#include <chrono>
#include <set>
#include <numeric>
#include <algorithm>

using namespace vesper::index;

// Generate synthetic dataset
void generate_dataset(std::vector<float>& data, std::size_t n, std::size_t dim, std::uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    data.resize(n * dim);
    for (auto& v : data) {
        v = dist(gen);
    }
}

// Compute exact k-NN for ground truth
std::vector<std::uint64_t> compute_ground_truth(
    const float* query,
    const float* data,
    const std::uint64_t* ids,
    std::size_t n,
    std::size_t dim,
    std::size_t k
) {
    std::vector<std::pair<float, std::uint64_t>> dists;
    dists.reserve(n);
    
    for (std::size_t i = 0; i < n; ++i) {
        float dist = 0.0f;
        for (std::size_t d = 0; d < dim; ++d) {
            float diff = query[d] - data[i * dim + d];
            dist += diff * diff;
        }
        dists.emplace_back(dist, ids[i]);
    }
    
    std::partial_sort(dists.begin(), dists.begin() + k, dists.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
    
    std::vector<std::uint64_t> result;
    result.reserve(k);
    for (std::size_t i = 0; i < k; ++i) {
        result.push_back(dists[i].second);
    }
    return result;
}

// Compute recall@k
float compute_recall(
    const std::vector<std::pair<std::uint64_t, float>>& retrieved,
    const std::vector<std::uint64_t>& ground_truth,
    std::size_t k
) {
    std::set<std::uint64_t> gt_set(ground_truth.begin(), ground_truth.begin() + k);
    
    std::size_t found = 0;
    for (std::size_t i = 0; i < std::min(retrieved.size(), k); ++i) {
        if (gt_set.count(retrieved[i].first)) {
            found++;
        }
    }
    
    return static_cast<float>(found) / k;
}

int main() {
    // Test configuration - smaller for debugging
    const std::size_t dim = 128;
    const std::size_t n_train = 5000;
    const std::size_t n_base = 10000;
    const std::size_t n_query = 10;  // Just 10 queries for debugging
    const std::uint32_t nlist = 100;
    const std::uint32_t m = 8;  
    const std::uint32_t nprobe = 20;
    const std::uint32_t k = 10;
    
    std::cout << "=== IVF-PQ Reranking Debug Test ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Dimension: " << dim << std::endl;
    std::cout << "  Training size: " << n_train << std::endl;
    std::cout << "  Base size: " << n_base << std::endl;
    std::cout << "  Queries: " << n_query << std::endl;
    std::cout << "  nlist: " << nlist << std::endl;
    std::cout << "  m: " << m << std::endl;
    std::cout << "  nprobe: " << nprobe << std::endl;
    std::cout << "  k: " << k << std::endl;
    std::cout << std::endl;
    
    // Generate data
    std::cout << "Generating synthetic dataset..." << std::endl;
    std::vector<float> train_data;
    generate_dataset(train_data, n_train, dim, 42);
    
    std::vector<float> base_data;
    generate_dataset(base_data, n_base, dim, 123);
    
    std::vector<float> query_data;
    generate_dataset(query_data, n_query, dim, 456);
    
    std::vector<std::uint64_t> ids(n_base);
    std::iota(ids.begin(), ids.end(), 0);
    
    // Train index
    std::cout << "Training IVF-PQ index..." << std::endl;
    IvfPqIndex index;
    
    IvfPqTrainParams train_params;
    train_params.nlist = nlist;
    train_params.m = m;
    train_params.nbits = 8;
    train_params.verbose = false;
    train_params.coarse_assigner = CoarseAssigner::KDTree;
    train_params.kd_leaf_size = 64;
    
    auto train_result = index.train(train_data.data(), dim, n_train, train_params);
    if (!train_result) {
        std::cerr << "Training failed: " << train_result.error().message << std::endl;
        return 1;
    }
    
    // Add base vectors
    std::cout << "Adding " << n_base << " vectors..." << std::endl;
    auto add_result = index.add(ids.data(), base_data.data(), n_base);
    if (!add_result) {
        std::cerr << "Add failed: " << add_result.error().message << std::endl;
        return 1;
    }
    
    // Test query-by-query with detailed output
    for (std::size_t q = 0; q < n_query; ++q) {
        std::cout << "\n=== Query " << q << " ===" << std::endl;
        const float* query = query_data.data() + q * dim;
        
        // Compute ground truth
        auto ground_truth = compute_ground_truth(query, base_data.data(), ids.data(), 
                                                 n_base, dim, k);
        
        std::cout << "Ground truth top-" << k << ": ";
        for (auto id : ground_truth) {
            std::cout << id << " ";
        }
        std::cout << std::endl;
        
        // Search without reranking
        IvfPqSearchParams search_params_no_rerank;
        search_params_no_rerank.nprobe = nprobe;
        search_params_no_rerank.k = k;
        search_params_no_rerank.use_exact_rerank = false;
        
        auto results_no_rerank = index.search(query, search_params_no_rerank);
        if (!results_no_rerank) {
            std::cerr << "Search failed: " << results_no_rerank.error().message << std::endl;
            return 1;
        }
        
        std::cout << "Without rerank: ";
        for (const auto& [id, dist] : results_no_rerank.value()) {
            std::cout << id << "(" << std::fixed << std::setprecision(2) << dist << ") ";
        }
        std::cout << std::endl;
        
        float recall_no_rerank = compute_recall(results_no_rerank.value(), ground_truth, k);
        std::cout << "Recall without rerank: " << recall_no_rerank << std::endl;
        
        // Search WITH reranking - fetch more candidates
        IvfPqSearchParams search_params_rerank;
        search_params_rerank.nprobe = nprobe;
        search_params_rerank.k = k;
        search_params_rerank.use_exact_rerank = true;
        search_params_rerank.rerank_k = k * 10;  // Rerank top-100 candidates
        
        auto results_rerank = index.search(query, search_params_rerank);
        if (!results_rerank) {
            std::cerr << "Search with rerank failed: " << results_rerank.error().message << std::endl;
            return 1;
        }
        
        std::cout << "With rerank:    ";
        for (const auto& [id, dist] : results_rerank.value()) {
            std::cout << id << "(" << std::fixed << std::setprecision(2) << dist << ") ";
        }
        std::cout << std::endl;
        
        float recall_rerank = compute_recall(results_rerank.value(), ground_truth, k);
        std::cout << "Recall with rerank: " << recall_rerank << std::endl;
        
        // Check if the candidates pool contains ground truth
        search_params_rerank.k = 100;  // Get more candidates to analyze
        search_params_rerank.use_exact_rerank = false;
        auto large_pool = index.search(query, search_params_rerank);
        if (large_pool) {
            std::set<std::uint64_t> pool_ids;
            for (const auto& [id, _] : large_pool.value()) {
                pool_ids.insert(id);
            }
            
            std::size_t gt_in_pool = 0;
            for (auto gt_id : ground_truth) {
                if (pool_ids.count(gt_id)) {
                    gt_in_pool++;
                }
            }
            std::cout << "Ground truth in top-100 pool: " << gt_in_pool << "/" << k << std::endl;
        }
    }
    
    return 0;
}