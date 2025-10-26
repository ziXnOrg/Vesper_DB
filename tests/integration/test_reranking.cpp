/**
 * Test program to verify exact reranking improves recall
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
    // Test configuration
    const std::size_t dim = 128;
    const std::size_t n_train = 10000;
    const std::size_t n_base = 50000;
    const std::size_t n_query = 100;
    const std::uint32_t nlist = 256;
    const std::uint32_t m = 8;  // Reduced from 16 for less aggressive compression
    const std::uint32_t nprobe = 64;  // Increased from 32 to search more clusters
    const std::uint32_t k = 10;
    
    std::cout << "=== IVF-PQ Exact Reranking Test ===" << std::endl;
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
    
    // Test without reranking
    std::cout << "\n=== Testing WITHOUT reranking ===" << std::endl;
    IvfPqSearchParams search_params_no_rerank;
    search_params_no_rerank.nprobe = nprobe;
    search_params_no_rerank.k = k;
    search_params_no_rerank.use_exact_rerank = false;
    
    float total_recall_no_rerank = 0.0f;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (std::size_t q = 0; q < n_query; ++q) {
        const float* query = query_data.data() + q * dim;
        
        // Compute ground truth
        auto ground_truth = compute_ground_truth(query, base_data.data(), ids.data(), 
                                                 n_base, dim, k);
        
        // Search
        auto results = index.search(query, search_params_no_rerank);
        if (!results) {
            std::cerr << "Search failed: " << results.error().message << std::endl;
            return 1;
        }
        
        // Compute recall
        float recall = compute_recall(results.value(), ground_truth, k);
        total_recall_no_rerank += recall;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_no_rerank = std::chrono::duration<double, std::milli>(end - start).count();
    
    float avg_recall_no_rerank = total_recall_no_rerank / n_query;
    std::cout << "Average Recall@" << k << ": " << std::fixed << std::setprecision(4) 
              << avg_recall_no_rerank << std::endl;
    std::cout << "Total time: " << duration_no_rerank << " ms" << std::endl;
    std::cout << "Avg query time: " << duration_no_rerank / n_query << " ms" << std::endl;
    
    // Test WITH reranking
    std::cout << "\n=== Testing WITH exact reranking ===" << std::endl;
    IvfPqSearchParams search_params_rerank;
    search_params_rerank.nprobe = nprobe;
    search_params_rerank.k = k;
    search_params_rerank.use_exact_rerank = true;
    search_params_rerank.rerank_k = k * 5;  // Rerank top-50 candidates
    
    float total_recall_rerank = 0.0f;
    start = std::chrono::high_resolution_clock::now();
    
    for (std::size_t q = 0; q < n_query; ++q) {
        const float* query = query_data.data() + q * dim;
        
        // Compute ground truth
        auto ground_truth = compute_ground_truth(query, base_data.data(), ids.data(), 
                                                 n_base, dim, k);
        
        // Search with reranking
        auto results = index.search(query, search_params_rerank);
        if (!results) {
            std::cerr << "Search failed: " << results.error().message << std::endl;
            return 1;
        }
        
        // Compute recall
        float recall = compute_recall(results.value(), ground_truth, k);
        total_recall_rerank += recall;
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto duration_rerank = std::chrono::duration<double, std::milli>(end - start).count();
    
    float avg_recall_rerank = total_recall_rerank / n_query;
    std::cout << "Average Recall@" << k << ": " << std::fixed << std::setprecision(4) 
              << avg_recall_rerank << std::endl;
    std::cout << "Total time: " << duration_rerank << " ms" << std::endl;
    std::cout << "Avg query time: " << duration_rerank / n_query << " ms" << std::endl;
    
    // Summary
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Recall improvement: " << std::fixed << std::setprecision(1)
              << ((avg_recall_rerank - avg_recall_no_rerank) / avg_recall_no_rerank * 100) 
              << "%" << std::endl;
    std::cout << "Time overhead: " << std::fixed << std::setprecision(1)
              << ((duration_rerank - duration_no_rerank) / duration_no_rerank * 100) 
              << "%" << std::endl;
    
    if (avg_recall_rerank > avg_recall_no_rerank) {
        std::cout << "\n✅ Reranking successfully improved recall!" << std::endl;
    } else {
        std::cout << "\n❌ Reranking did not improve recall (unexpected)" << std::endl;
        return 1;
    }
    
    return 0;
}