# IVF-PQ Recall Optimization Guide

## Overview

This guide provides strategies for optimizing recall in Vesper's IVF-PQ implementation. Low recall is a common challenge with Product Quantization due to compression losses. This document covers parameter tuning, implementation improvements, and best practices.

## Understanding the Recall Problem

### Current State
- **Observed Recall**: 7.9% on SIFT-128 dataset with default parameters
- **Root Causes**:
  1. Aggressive compression (m=16, 8 bits per subquantizer)
  2. Insufficient nprobe relative to nlist
  3. No exact reranking of candidates
  4. Suboptimal OPQ usage

### Target Recall Levels
- **Basic**: 50-60% - Suitable for approximate similarity search
- **Good**: 70-80% - Balanced accuracy/speed for most applications  
- **High**: 90%+ - Near-exact search with performance trade-offs

## Parameter Optimization Strategies

### 1. Automatic Parameter Selection

#### Dataset Size-Based Configuration

```cpp
// For dataset size N
struct OptimalParams {
    // Balanced configuration
    static IvfPqTrainParams balanced(size_t N, size_t dim) {
        IvfPqTrainParams params;
        params.nlist = static_cast<uint32_t>(std::sqrt(N));
        params.m = std::min(32u, static_cast<uint32_t>(dim / 4));
        params.nbits = 8;
        params.use_opq = (dim >= 256);  // Enable for high-dim
        params.coarse_assigner = CoarseAssigner::KDTree;
        params.kd_leaf_size = 64;  // Our optimized value
        return params;
    }
    
    // High recall configuration
    static IvfPqTrainParams high_recall(size_t N, size_t dim) {
        IvfPqTrainParams params;
        params.nlist = static_cast<uint32_t>(std::sqrt(N) / 2);  // Fewer, larger clusters
        params.m = std::min(64u, static_cast<uint32_t>(dim / 2));  // Less compression
        params.nbits = 8;
        params.use_opq = true;  // Always use OPQ
        params.opq_iter = 20;    // More iterations
        return params;
    }
    
    // Speed-optimized configuration
    static IvfPqTrainParams fast(size_t N, size_t dim) {
        IvfPqTrainParams params;
        params.nlist = std::min(4096u, static_cast<uint32_t>(2 * std::sqrt(N)));
        params.m = std::max(8u, static_cast<uint32_t>(dim / 8));  // High compression
        params.nbits = 8;
        params.use_opq = false;  // Skip for speed
        return params;
    }
};
```

#### Search Parameter Optimization

```cpp
struct OptimalSearchParams {
    static IvfPqSearchParams for_recall(float target_recall, uint32_t nlist) {
        IvfPqSearchParams params;
        
        if (target_recall >= 0.9) {
            // High recall: search 10-20% of lists
            params.nprobe = static_cast<uint32_t>(nlist * 0.15);
            params.cand_k = params.k * 10;  // Large rerank pool
            params.use_exact_rerank = true;
            params.rerank_k = params.k * 5;
        } else if (target_recall >= 0.7) {
            // Balanced: search 5-10% of lists
            params.nprobe = static_cast<uint32_t>(nlist * 0.08);
            params.cand_k = params.k * 5;
            params.use_exact_rerank = true;
            params.rerank_k = params.k * 3;
        } else {
            // Fast: search 2-5% of lists
            params.nprobe = static_cast<uint32_t>(nlist * 0.03);
            params.cand_k = params.k * 2;
            params.use_exact_rerank = false;
        }
        
        return params;
    }
};
```

### 2. Parameter Tuning Guidelines

#### nlist (Number of Clusters)
- **Formula**: `nlist = sqrt(N)` where N is dataset size
- **Range**: 256 to 65536
- **Trade-off**: More clusters = better selectivity but higher memory
- **Recommendations**:
  - < 100k vectors: nlist = 256-1024
  - 100k-1M vectors: nlist = 1024-4096  
  - 1M-10M vectors: nlist = 4096-16384
  - > 10M vectors: nlist = 16384-65536

#### m (Number of Subquantizers)
- **Formula**: `m = dimension / 4` for balanced, `dimension / 8` for speed
- **Constraint**: dimension must be divisible by m
- **Trade-off**: Higher m = better accuracy but more memory
- **Recommendations**:
  - Low dimensions (< 128): m = 8-16
  - Medium dimensions (128-512): m = 16-32
  - High dimensions (> 512): m = 32-64

#### nprobe (Search Parameter)
- **Formula**: `nprobe = nlist * coverage_ratio`
- **Coverage ratio**: 0.01-0.20 (1%-20% of lists)
- **Dynamic adjustment**: Start low, increase until recall target met
- **Recommendations**:
  - Initial: nprobe = sqrt(nlist)
  - Balanced: nprobe = nlist * 0.05
  - High recall: nprobe = nlist * 0.10-0.15

### 3. OPQ (Optimized Product Quantization)

#### When to Use OPQ
- Always for dimensions >= 256
- When recall requirements > 70%
- When training time is not critical
- For datasets with complex distributions

#### OPQ Configuration
```cpp
IvfPqTrainParams params;
params.use_opq = true;
params.opq_iter = 10;          // Default, increase for better quality
params.opq_sample_n = 50000;   // Sample size for OPQ training
params.opq_init = OpqInit::PCA; // Better than Identity for structured data
```

## Implementation Improvements

### 1. Exact Reranking (Critical for Recall)

The most effective way to improve recall is to implement exact reranking:

```cpp
// Pseudo-code for reranking implementation
auto search_with_rerank(const float* query, const IvfPqSearchParams& params) {
    // Step 1: Get more candidates than needed
    uint32_t fetch_k = params.use_exact_rerank ? 
        (params.rerank_k > 0 ? params.rerank_k : params.k * 3) : params.k;
    
    // Step 2: Search with larger k
    auto candidates = search_internal(query, fetch_k, params.nprobe);
    
    // Step 3: Recompute exact distances for top candidates
    if (params.use_exact_rerank) {
        for (auto& [id, approx_dist] : candidates) {
            auto vector = reconstruct(id);  // Or fetch original
            float exact_dist = compute_distance(query, vector);
            approx_dist = exact_dist;
        }
        
        // Step 4: Re-sort by exact distances
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) { 
                      return a.second < b.second; 
                  });
    }
    
    // Step 5: Return top-k
    candidates.resize(std::min(candidates.size(), size_t(params.k)));
    return candidates;
}
```

### 2. Multi-Index Strategy

For very high recall requirements, consider multi-index approach:

```cpp
// Build multiple indices with different random rotations
class MultiIvfPq {
    std::vector<IvfPqIndex> indices;
    
    auto search(const float* query, uint32_t k) {
        std::map<uint64_t, float> candidate_scores;
        
        // Query all indices
        for (const auto& index : indices) {
            auto results = index.search(query, params);
            for (const auto& [id, dist] : results) {
                // Take minimum distance across indices
                if (candidate_scores.find(id) == candidate_scores.end() ||
                    dist < candidate_scores[id]) {
                    candidate_scores[id] = dist;
                }
            }
        }
        
        // Convert to sorted vector
        // ...
    }
};
```

### 3. Adaptive nprobe Selection

Dynamically adjust nprobe based on query difficulty:

```cpp
uint32_t adaptive_nprobe(const float* query, const IvfPqIndex& index) {
    // Compute query's distance to nearest centroids
    auto centroid_dists = index.compute_centroid_distances(query);
    
    // If nearest centroids are far, search more lists
    float min_dist = centroid_dists[0].second;
    float median_dist = centroid_dists[centroid_dists.size()/2].second;
    float ratio = min_dist / median_dist;
    
    if (ratio > 0.9) {
        // Query is far from all centroids, search more
        return base_nprobe * 2;
    } else if (ratio < 0.5) {
        // Query is very close to some centroids
        return base_nprobe / 2;
    }
    
    return base_nprobe;
}
```

## Benchmarking and Validation

### Recall Measurement

```cpp
float measure_recall(const IvfPqIndex& index,
                     const float* queries, size_t nq,
                     const uint64_t* ground_truth, size_t k,
                     const IvfPqSearchParams& params) {
    size_t total_found = 0;
    
    for (size_t i = 0; i < nq; ++i) {
        auto results = index.search(queries + i * dim, params);
        std::set<uint64_t> result_set;
        for (const auto& [id, _] : results) {
            result_set.insert(id);
        }
        
        for (size_t j = 0; j < k; ++j) {
            uint64_t gt_id = ground_truth[i * k + j];
            if (result_set.count(gt_id)) {
                total_found++;
            }
        }
    }
    
    return float(total_found) / (nq * k);
}
```

### Parameter Sweep

```cpp
void optimize_parameters(const Dataset& data) {
    std::vector<uint32_t> nlist_values = {256, 512, 1024, 2048, 4096};
    std::vector<uint32_t> m_values = {8, 16, 32, 64};
    std::vector<uint32_t> nprobe_values = {16, 32, 64, 128, 256};
    
    for (auto nlist : nlist_values) {
        for (auto m : m_values) {
            if (data.dim % m != 0) continue;
            
            // Train index
            IvfPqTrainParams train_params;
            train_params.nlist = nlist;
            train_params.m = m;
            // ...
            
            IvfPqIndex index;
            index.train(data.train_vectors, data.dim, data.n_train, train_params);
            index.add(data.ids, data.vectors, data.n);
            
            // Test different nprobe values
            for (auto nprobe : nprobe_values) {
                IvfPqSearchParams search_params;
                search_params.nprobe = nprobe;
                search_params.k = 10;
                
                auto recall = measure_recall(index, data.queries, data.nq,
                                            data.ground_truth, 10, search_params);
                
                std::cout << "nlist=" << nlist 
                         << " m=" << m 
                         << " nprobe=" << nprobe
                         << " recall=" << recall << std::endl;
            }
        }
    }
}
```

## Quick Reference

### Problem-Solution Matrix

| Problem | Solution | Expected Improvement |
|---------|----------|---------------------|
| Low recall (<20%) | Enable exact reranking | +30-50% recall |
| Still low after reranking | Increase nprobe to 10-15% of nlist | +10-20% recall |
| High compression loss | Reduce m (fewer subquantizers) | +10-15% recall |
| Complex data distribution | Enable OPQ | +10-15% recall |
| Uneven cluster sizes | Reduce nlist | +5-10% recall |

### Configuration Templates

#### High Recall (90%+)
```cpp
// Training
params.nlist = sqrt(N) / 2;
params.m = dim / 2;
params.use_opq = true;
params.opq_iter = 20;

// Search
search.nprobe = nlist * 0.15;
search.use_exact_rerank = true;
search.rerank_k = k * 10;
```

#### Balanced (70-80%)
```cpp
// Training  
params.nlist = sqrt(N);
params.m = dim / 4;
params.use_opq = (dim >= 256);

// Search
search.nprobe = nlist * 0.08;
search.use_exact_rerank = true;
search.rerank_k = k * 5;
```

#### Speed-Optimized (50-60%)
```cpp
// Training
params.nlist = 2 * sqrt(N);
params.m = dim / 8;
params.use_opq = false;

// Search
search.nprobe = nlist * 0.03;
search.use_exact_rerank = false;
```

## Conclusion

Improving IVF-PQ recall requires a combination of:
1. **Exact reranking** - The single most effective improvement
2. **Proper parameter tuning** - Based on dataset characteristics
3. **OPQ for complex data** - Especially for high dimensions
4. **Sufficient nprobe** - 5-15% of nlist for good recall

Start with exact reranking and gradually tune other parameters based on your specific recall/latency requirements.