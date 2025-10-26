/**
 * IVF-PQ Auto-tuning Utility
 * 
 * Automatically selects optimal IVF-PQ parameters based on:
 * - Dataset size and dimensionality
 * - Target recall level
 * - Latency constraints
 */

#include "vesper/index/ivf_pq.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <set>

using namespace vesper::index;

struct TuningConfig {
    std::size_t n_vectors = 50000;
    std::size_t dimension = 128;
    std::size_t n_queries = 1000;
    float target_recall = 0.7f;  // 70% recall target
    float max_latency_ms = 5.0f;  // 5ms max latency
    bool use_opq = false;
    bool use_rerank = false;
    std::string dataset_path;  // Optional: use real dataset
    bool verbose = false;
};

struct ParameterSet {
    std::uint32_t nlist;
    std::uint32_t m;
    std::uint32_t nprobe;
    bool use_opq;
    bool use_rerank;
    std::uint32_t rerank_k;
    
    float recall;
    float latency_ms;
    float memory_mb;
    float score;  // Combined metric
};

// Generate synthetic vectors
std::vector<float> generate_vectors(std::size_t n, std::size_t dim, std::uint32_t seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<float> data(n * dim);
    for (auto& v : data) {
        v = dist(gen);
    }
    
    // Normalize vectors for cosine similarity
    for (std::size_t i = 0; i < n; ++i) {
        float norm = 0.0f;
        for (std::size_t d = 0; d < dim; ++d) {
            float v = data[i * dim + d];
            norm += v * v;
        }
        norm = std::sqrt(norm);
        if (norm > 0) {
            for (std::size_t d = 0; d < dim; ++d) {
                data[i * dim + d] /= norm;
            }
        }
    }
    
    return data;
}

// Compute exact k-NN for ground truth
std::vector<std::vector<std::uint64_t>> compute_ground_truth(
    const float* queries,
    const float* base,
    std::size_t n_queries,
    std::size_t n_base,
    std::size_t dim,
    std::size_t k
) {
    std::vector<std::vector<std::uint64_t>> gt(n_queries);
    
    for (std::size_t q = 0; q < n_queries; ++q) {
        const float* query = queries + q * dim;
        std::vector<std::pair<float, std::uint64_t>> dists;
        dists.reserve(n_base);
        
        for (std::size_t i = 0; i < n_base; ++i) {
            float dist = 0.0f;
            for (std::size_t d = 0; d < dim; ++d) {
                float diff = query[d] - base[i * dim + d];
                dist += diff * diff;
            }
            dists.emplace_back(dist, i);
        }
        
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        
        gt[q].reserve(k);
        for (std::size_t i = 0; i < k; ++i) {
            gt[q].push_back(dists[i].second);
        }
    }
    
    return gt;
}

// Compute recall@k
float compute_recall(
    const std::vector<std::vector<std::pair<std::uint64_t, float>>>& results,
    const std::vector<std::vector<std::uint64_t>>& ground_truth,
    std::size_t k
) {
    std::size_t total_found = 0;
    std::size_t total_count = 0;
    
    for (std::size_t q = 0; q < results.size(); ++q) {
        std::set<std::uint64_t> gt_set(ground_truth[q].begin(), 
                                       ground_truth[q].begin() + std::min(k, ground_truth[q].size()));
        
        for (std::size_t i = 0; i < std::min(k, results[q].size()); ++i) {
            if (gt_set.count(results[q][i].first)) {
                total_found++;
            }
        }
        total_count += k;
    }
    
    return static_cast<float>(total_found) / total_count;
}

// Automatic parameter selection based on dataset characteristics
ParameterSet auto_select_params(std::size_t n_vectors, std::size_t dim, float target_recall) {
    ParameterSet params;
    
    // Select nlist based on dataset size
    if (n_vectors < 100000) {
        params.nlist = static_cast<std::uint32_t>(std::sqrt(n_vectors));
    } else if (n_vectors < 1000000) {
        params.nlist = static_cast<std::uint32_t>(std::sqrt(n_vectors) * 0.75);
    } else {
        params.nlist = static_cast<std::uint32_t>(std::sqrt(n_vectors) * 0.5);
    }
    
    // Ensure nlist is reasonable
    params.nlist = std::max<std::uint32_t>(256, std::min<std::uint32_t>(params.nlist, 4096));
    
    // Select m based on dimension and target recall
    if (target_recall >= 0.9f) {
        // High recall: less compression
        params.m = static_cast<std::uint32_t>(dim / 8);
    } else if (target_recall >= 0.7f) {
        // Balanced
        params.m = static_cast<std::uint32_t>(dim / 4);
    } else {
        // Speed-optimized
        params.m = static_cast<std::uint32_t>(dim / 2);
    }
    
    // Find valid m that divides dimension
    while (dim % params.m != 0 && params.m > 1) {
        params.m--;
    }
    if (params.m == 0) params.m = 1;
    
    // Select nprobe based on target recall
    float coverage_ratio = 0.05f;  // Default 5%
    if (target_recall >= 0.9f) {
        coverage_ratio = 0.15f;  // 15% for high recall
    } else if (target_recall >= 0.7f) {
        coverage_ratio = 0.08f;  // 8% for balanced
    }
    
    params.nprobe = static_cast<std::uint32_t>(params.nlist * coverage_ratio);
    params.nprobe = std::max<std::uint32_t>(1, std::min<std::uint32_t>(params.nprobe, params.nlist));
    
    // Enable OPQ for higher dimensions
    params.use_opq = (dim >= 256) || (target_recall >= 0.8f);
    
    // Enable reranking for high recall targets
    params.use_rerank = (target_recall >= 0.8f);
    params.rerank_k = params.use_rerank ? 50 : 0;
    
    return params;
}

// Evaluate a parameter set
ParameterSet evaluate_params(
    const ParameterSet& params,
    const float* train_data,
    const float* base_data,
    const float* query_data,
    const std::vector<std::uint64_t>& ids,
    const std::vector<std::vector<std::uint64_t>>& ground_truth,
    const TuningConfig& config
) {
    ParameterSet result = params;
    
    // Train index
    IvfPqIndex index;
    IvfPqTrainParams train_params;
    train_params.nlist = params.nlist;
    train_params.m = params.m;
    train_params.nbits = 8;
    train_params.use_opq = params.use_opq;
    train_params.verbose = false;
    train_params.coarse_assigner = CoarseAssigner::KDTree;
    train_params.kd_leaf_size = 64;
    
    auto train_result = index.train(train_data, config.dimension, 
                                    std::min(config.n_vectors, std::size_t(50000)), 
                                    train_params);
    
    if (!train_result) {
        result.recall = 0.0f;
        result.latency_ms = 999999.0f;
        return result;
    }
    
    // Add vectors
    auto add_result = index.add(ids.data(), base_data, config.n_vectors);
    if (!add_result) {
        result.recall = 0.0f;
        result.latency_ms = 999999.0f;
        return result;
    }
    
    // Search and measure performance
    IvfPqSearchParams search_params;
    search_params.nprobe = params.nprobe;
    search_params.k = 10;
    search_params.use_exact_rerank = params.use_rerank;
    search_params.rerank_k = params.rerank_k;
    
    std::vector<std::vector<std::pair<std::uint64_t, float>>> results;
    results.reserve(config.n_queries);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (std::size_t q = 0; q < config.n_queries; ++q) {
        auto query_result = index.search(query_data + q * config.dimension, search_params);
        if (query_result) {
            results.push_back(query_result.value());
        } else {
            results.push_back({});
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    // Calculate metrics
    result.recall = compute_recall(results, ground_truth, 10);
    result.latency_ms = std::chrono::duration<float, std::milli>(end - start).count() / config.n_queries;
    
    auto stats = index.get_stats();
    result.memory_mb = stats.memory_bytes / (1024.0f * 1024.0f);
    
    // Combined score (weighted by importance)
    float recall_score = result.recall / config.target_recall;  // Normalize to target
    float latency_score = config.max_latency_ms / result.latency_ms;  // Inverse for lower is better
    result.score = recall_score * 0.7f + latency_score * 0.3f;  // 70% weight on recall
    
    return result;
}

// Grid search over parameter space
std::vector<ParameterSet> grid_search(const TuningConfig& config) {
    // Generate test data
    auto train_data = generate_vectors(std::min(config.n_vectors, std::size_t(50000)), 
                                       config.dimension, 42);
    auto base_data = generate_vectors(config.n_vectors, config.dimension, 123);
    auto query_data = generate_vectors(config.n_queries, config.dimension, 456);
    
    std::vector<std::uint64_t> ids(config.n_vectors);
    for (std::size_t i = 0; i < config.n_vectors; ++i) {
        ids[i] = i;
    }
    
    // Compute ground truth
    auto ground_truth = compute_ground_truth(query_data.data(), base_data.data(),
                                            config.n_queries, config.n_vectors,
                                            config.dimension, 10);
    
    // Define search space
    std::vector<std::uint32_t> nlist_values;
    std::vector<std::uint32_t> m_values;
    std::vector<std::uint32_t> nprobe_values;
    
    // Adaptive search space based on dataset size
    std::uint32_t base_nlist = static_cast<std::uint32_t>(std::sqrt(config.n_vectors));
    nlist_values = {base_nlist / 2, base_nlist, base_nlist * 2};
    
    // Find valid m values
    for (std::uint32_t m : {4, 8, 16, 32}) {
        if (config.dimension % m == 0 && m <= config.dimension / 2) {
            m_values.push_back(m);
        }
    }
    
    // Test results
    std::vector<ParameterSet> results;
    
    for (auto nlist : nlist_values) {
        // Adjust nlist to reasonable bounds
        nlist = std::max<std::uint32_t>(256, std::min<std::uint32_t>(nlist, 4096));
        
        // Adaptive nprobe based on nlist
        nprobe_values = {
            std::max<std::uint32_t>(1, nlist / 32),
            std::max<std::uint32_t>(1, nlist / 16),
            std::max<std::uint32_t>(1, nlist / 8),
            std::max<std::uint32_t>(1, nlist / 4)
        };
        
        for (auto m : m_values) {
            for (auto nprobe : nprobe_values) {
                ParameterSet params;
                params.nlist = nlist;
                params.m = m;
                params.nprobe = nprobe;
                params.use_opq = config.use_opq;
                params.use_rerank = config.use_rerank;
                params.rerank_k = config.use_rerank ? 50 : 0;
                
                if (config.verbose) {
                    std::cout << "Testing: nlist=" << nlist 
                             << " m=" << m 
                             << " nprobe=" << nprobe << "... ";
                }
                
                auto result = evaluate_params(params, train_data.data(), base_data.data(),
                                            query_data.data(), ids, ground_truth, config);
                
                if (config.verbose) {
                    std::cout << "recall=" << std::fixed << std::setprecision(3) << result.recall
                             << " latency=" << result.latency_ms << "ms" << std::endl;
                }
                
                results.push_back(result);
            }
        }
    }
    
    // Sort by score
    std::sort(results.begin(), results.end(), 
              [](const auto& a, const auto& b) { return a.score > b.score; });
    
    return results;
}

void print_recommendations(const std::vector<ParameterSet>& results, const TuningConfig& config) {
    std::cout << "\n=== IVF-PQ Auto-Tuning Results ===" << std::endl;
    std::cout << "Dataset: " << config.n_vectors << " vectors, " 
              << config.dimension << " dimensions" << std::endl;
    std::cout << "Target recall: " << config.target_recall << std::endl;
    std::cout << "Max latency: " << config.max_latency_ms << " ms" << std::endl;
    
    std::cout << "\n=== Top 5 Configurations ===" << std::endl;
    std::cout << std::setw(8) << "nlist" 
              << std::setw(6) << "m" 
              << std::setw(8) << "nprobe"
              << std::setw(8) << "Recall"
              << std::setw(10) << "Latency"
              << std::setw(10) << "Memory"
              << std::setw(8) << "Score" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (std::size_t i = 0; i < std::min<std::size_t>(5, results.size()); ++i) {
        const auto& r = results[i];
        std::cout << std::setw(8) << r.nlist
                  << std::setw(6) << r.m
                  << std::setw(8) << r.nprobe
                  << std::setw(8) << std::fixed << std::setprecision(3) << r.recall
                  << std::setw(10) << std::setprecision(2) << r.latency_ms << "ms"
                  << std::setw(10) << std::setprecision(1) << r.memory_mb << "MB"
                  << std::setw(8) << std::setprecision(3) << r.score << std::endl;
    }
    
    // Find best configuration meeting constraints
    const ParameterSet* best = nullptr;
    for (const auto& r : results) {
        if (r.recall >= config.target_recall && r.latency_ms <= config.max_latency_ms) {
            best = &r;
            break;
        }
    }
    
    if (best) {
        std::cout << "\n=== Recommended Configuration ===" << std::endl;
        std::cout << "nlist = " << best->nlist << std::endl;
        std::cout << "m = " << best->m << std::endl;
        std::cout << "nprobe = " << best->nprobe << std::endl;
        std::cout << "use_opq = " << (best->use_opq ? "true" : "false") << std::endl;
        std::cout << "use_exact_rerank = " << (best->use_rerank ? "true" : "false") << std::endl;
        if (best->use_rerank) {
            std::cout << "rerank_k = " << best->rerank_k << std::endl;
        }
        std::cout << "\nExpected performance:" << std::endl;
        std::cout << "  Recall@10: " << std::fixed << std::setprecision(1) 
                  << (best->recall * 100) << "%" << std::endl;
        std::cout << "  Latency: " << std::setprecision(2) << best->latency_ms << " ms" << std::endl;
        std::cout << "  Memory: " << std::setprecision(1) << best->memory_mb << " MB" << std::endl;
    } else {
        std::cout << "\n⚠️  No configuration found meeting constraints!" << std::endl;
        std::cout << "Consider relaxing target recall or latency requirements." << std::endl;
    }
}

int main(int argc, char** argv) {
    TuningConfig config;
    
    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--n" && i + 1 < argc) {
            config.n_vectors = std::stoull(argv[++i]);
        } else if (arg == "--dim" && i + 1 < argc) {
            config.dimension = std::stoull(argv[++i]);
        } else if (arg == "--queries" && i + 1 < argc) {
            config.n_queries = std::stoull(argv[++i]);
        } else if (arg == "--target_recall" && i + 1 < argc) {
            config.target_recall = std::stof(argv[++i]);
        } else if (arg == "--max_latency" && i + 1 < argc) {
            config.max_latency_ms = std::stof(argv[++i]);
        } else if (arg == "--use_opq") {
            config.use_opq = true;
        } else if (arg == "--use_rerank") {
            config.use_rerank = true;
        } else if (arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "--help") {
            std::cout << "IVF-PQ Auto-Tuning Utility" << std::endl;
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --n N               Number of vectors (default: 50000)" << std::endl;
            std::cout << "  --dim D             Vector dimension (default: 128)" << std::endl;
            std::cout << "  --queries Q         Number of test queries (default: 1000)" << std::endl;
            std::cout << "  --target_recall R   Target recall@10 (default: 0.7)" << std::endl;
            std::cout << "  --max_latency L     Max latency in ms (default: 5.0)" << std::endl;
            std::cout << "  --use_opq           Enable OPQ rotation" << std::endl;
            std::cout << "  --use_rerank        Enable exact reranking" << std::endl;
            std::cout << "  --verbose           Show detailed progress" << std::endl;
            return 0;
        }
    }
    
    std::cout << "IVF-PQ Auto-Tuning for Vesper" << std::endl;
    std::cout << "==============================" << std::endl;
    
    // First try automatic parameter selection
    std::cout << "\n1. Trying automatic parameter selection..." << std::endl;
    auto auto_params = auto_select_params(config.n_vectors, config.dimension, config.target_recall);
    
    std::cout << "Auto-selected parameters:" << std::endl;
    std::cout << "  nlist = " << auto_params.nlist << std::endl;
    std::cout << "  m = " << auto_params.m << std::endl;
    std::cout << "  nprobe = " << auto_params.nprobe << std::endl;
    std::cout << "  use_opq = " << (auto_params.use_opq ? "true" : "false") << std::endl;
    std::cout << "  use_rerank = " << (auto_params.use_rerank ? "true" : "false") << std::endl;
    
    // Run grid search
    std::cout << "\n2. Running parameter grid search..." << std::endl;
    std::cout << "This may take a few minutes..." << std::endl;
    
    auto results = grid_search(config);
    
    // Print recommendations
    print_recommendations(results, config);
    
    return 0;
}