// ivfpq_diagnostic_test.cpp - Diagnose IVF-PQ recall issues
#include <vesper/index/ivf_pq.hpp>
#include <vesper/kernels/distance.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <numeric>

using namespace vesper;

// Helper to generate synthetic data
std::vector<float> generate_vectors(std::size_t n, std::size_t dim, std::uint32_t seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    
    std::vector<float> data(n * dim);
    for (auto& v : data) {
        v = dis(gen);
    }
    
    // Normalize vectors
    for (std::size_t i = 0; i < n; ++i) {
        float norm = 0.0f;
        for (std::size_t d = 0; d < dim; ++d) {
            float val = data[i * dim + d];
            norm += val * val;
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

// Compute exact L2 distance
float compute_l2_distance(const float* a, const float* b, std::size_t dim) {
    float dist = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

TEST_CASE("IVF-PQ Diagnostic: PQ Codebook Quality", "[ivfpq][diagnostic]") {
    const std::size_t dim = 128;
    const std::size_t n_train = 10000;
    const std::size_t n_test = 100;
    
    // Generate normalized training data
    auto train_data = generate_vectors(n_train, dim, 42);
    auto test_data = generate_vectors(n_test, dim, 43);
    
    // Generate IDs
    std::vector<std::uint64_t> train_ids(n_train);
    std::iota(train_ids.begin(), train_ids.end(), 0);
    
    SECTION("Test 1: PQ Reconstruction Error") {
        // Train with simple parameters
        index::IvfPqTrainParams params;
        params.nlist = 256;
        params.m = 16;
        params.nbits = 8;
        params.use_opq = false;
        params.verbose = true;
        
        index::IvfPqIndex index;
        auto train_result = index.train(train_data.data(), dim, n_train, params);
        REQUIRE(train_result.has_value());
        
        // Add vectors
        auto add_result = index.add(train_ids.data(), train_data.data(), n_train);
        REQUIRE(add_result.has_value());
        
        // Test reconstruction quality
        std::cout << "\n=== PQ Reconstruction Test ===\n";
        
        // Sample some vectors and check reconstruction error
        float total_reconstruction_error = 0.0f;
        const std::size_t n_sample = std::min<std::size_t>(100, n_train);
        
        for (std::size_t i = 0; i < n_sample; ++i) {
            // Get original vector
            const float* original = train_data.data() + i * dim;
            
            // Search for this exact vector with k=1
            index::IvfPqSearchParams search_params;
            search_params.k = 1;
            search_params.nprobe = params.nlist / 4;  // Search 25% of lists
            
            auto results = index.search(original, search_params);
            REQUIRE(results.has_value());
            REQUIRE(!results->empty());
            
            // The top result should be the vector itself
            auto [found_id, approx_dist] = results->front();
            
            if (found_id == i) {
                // Perfect match - distance should be near 0
                total_reconstruction_error += approx_dist;
                if (approx_dist > 0.1f) {
                    std::cout << "  Vector " << i << ": self-distance = " << approx_dist 
                             << " (should be ~0)\n";
                }
            } else {
                std::cout << "  WARNING: Vector " << i << " not found as top result! "
                         << "Found ID " << found_id << " instead\n";
            }
        }
        
        float avg_reconstruction_error = total_reconstruction_error / n_sample;
        std::cout << "Average reconstruction error: " << avg_reconstruction_error << "\n";
        
        // Reconstruction error should be reasonable
        CHECK(avg_reconstruction_error < 1.0f);
    }
    
    SECTION("Test 2: Distance Computation Consistency") {
        // Train index
        index::IvfPqTrainParams params;
        params.nlist = 256;
        params.m = 16;
        params.nbits = 8;
        params.use_opq = false;
        
        index::IvfPqIndex index;
        auto train_result = index.train(train_data.data(), dim, n_train, params);
        REQUIRE(train_result.has_value());
        
        auto add_result = index.add(train_ids.data(), train_data.data(), n_train);
        REQUIRE(add_result.has_value());
        
        std::cout << "\n=== Distance Computation Test ===\n";
        
        // For each test query, compare approximate vs exact distances
        float total_distance_error = 0.0f;
        const std::size_t n_queries = 10;
        
        for (std::size_t q = 0; q < n_queries; ++q) {
            const float* query = test_data.data() + q * dim;
            
            // Search with high nprobe for better coverage
            index::IvfPqSearchParams search_params;
            search_params.k = 10;
            search_params.nprobe = params.nlist / 2;  // Search 50% of lists
            
            auto results = index.search(query, search_params);
            REQUIRE(results.has_value());
            
            // Compute exact distances for comparison
            for (const auto& [id, approx_dist] : *results) {
                const float* vec = train_data.data() + id * dim;
                float exact_dist = compute_l2_distance(query, vec, dim);
                float error = std::abs(approx_dist - exact_dist);
                total_distance_error += error;
                
                if (error > exact_dist * 0.5f) {  // Error > 50% of exact distance
                    std::cout << "  Query " << q << ", ID " << id << ": "
                             << "approx=" << approx_dist << ", exact=" << exact_dist
                             << ", error=" << error << " (" 
                             << (error/exact_dist * 100) << "%)\n";
                }
            }
        }
        
        float avg_distance_error = total_distance_error / (n_queries * 10);
        std::cout << "Average distance error: " << avg_distance_error << "\n";
        
        // Distance approximation should be reasonable
        CHECK(avg_distance_error < 2.0f);
    }
    
    SECTION("Test 3: Residual Computation Check") {
        // This tests if residuals are being computed correctly
        index::IvfPqTrainParams params;
        params.nlist = 256;
        params.m = 16;
        params.nbits = 8;
        params.use_opq = false;
        params.verbose = true;
        
        index::IvfPqIndex index;
        auto train_result = index.train(train_data.data(), dim, n_train, params);
        REQUIRE(train_result.has_value());
        
        std::cout << "\n=== Residual Computation Test ===\n";
        
        // Add a single vector and verify its encoding
        std::vector<float> single_vec = generate_vectors(1, dim, 99);
        std::vector<std::uint64_t> single_id = {999999};
        
        auto add_result = index.add(single_id.data(), single_vec.data(), 1);
        REQUIRE(add_result.has_value());
        
        // Search for this exact vector
        index::IvfPqSearchParams search_params;
        search_params.k = 1;
        search_params.nprobe = params.nlist;  // Search all lists
        
        auto results = index.search(single_vec.data(), search_params);
        REQUIRE(results.has_value());
        REQUIRE(!results->empty());
        
        auto [found_id, distance] = results->front();
        std::cout << "Single vector self-search: ID=" << found_id
                  << ", distance=" << distance << "\n";

        // Verify the retrieved ID
        CHECK(found_id == 999999);

        // ADC parity: returned distance should match exact L2 to reconstructed vector
        auto recon = index.reconstruct(999999);
        REQUIRE(recon.has_value());
        const auto& recon_vec = recon.value();
        float exact_self = compute_l2_distance(single_vec.data(), recon_vec.data(), dim);
        std::cout << "Reconstructed self-distance=" << exact_self << "\n";
        REQUIRE(exact_self >= 0.0f);
        CHECK_THAT(distance, Catch::Matchers::WithinRel(exact_self, 1e-4f));

        // Sanity bound: reconstruction error should be reasonable for m=16, nbits=8
        CHECK(exact_self < 1.0f);
    }

    SECTION("Test 4: Parameter Impact on Recall") {
        std::cout << "\n=== Parameter Impact Test ===\n";
        
        // Test different m values
        std::vector<std::uint32_t> m_values = {4, 8, 16, 32};
        
        for (auto m : m_values) {
            if (dim % m != 0) continue;
            
            index::IvfPqTrainParams params;
            params.nlist = 256;
            params.m = m;
            params.nbits = 8;
            params.use_opq = false;
            
            index::IvfPqIndex index;
            auto train_result = index.train(train_data.data(), dim, n_train, params);
            REQUIRE(train_result.has_value());
            
            auto add_result = index.add(train_ids.data(), train_data.data(), n_train);
            REQUIRE(add_result.has_value());
            
            // Test self-recall (each vector should find itself)
            std::size_t found_count = 0;
            const std::size_t n_test_recall = 100;
            
            for (std::size_t i = 0; i < n_test_recall; ++i) {
                const float* vec = train_data.data() + i * dim;
                
                index::IvfPqSearchParams search_params;
                search_params.k = 10;
                search_params.nprobe = params.nlist / 4;
                
                auto results = index.search(vec, search_params);
                REQUIRE(results.has_value());
                
                // Check if vector found itself in top-10
                for (const auto& [id, dist] : *results) {
                    if (id == i) {
                        found_count++;
                        break;
                    }
                }
            }
            
            float self_recall = static_cast<float>(found_count) / n_test_recall;
            std::cout << "m=" << m << ": self-recall@10 = " << self_recall << "\n";
            
            // Self-recall should be high (>80%)
            CHECK(self_recall > 0.8f);
        }
    }
}

TEST_CASE("IVF-PQ Diagnostic: Zero Vector Handling", "[ivfpq][diagnostic]") {
    const std::size_t dim = 128;
    const std::size_t n = 1000;
    
    // Create data with some zero vectors
    std::vector<float> data(n * dim, 0.0f);
    std::vector<std::uint64_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    
    // Add some non-zero vectors
    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    
    for (std::size_t i = 0; i < n/2; ++i) {
        for (std::size_t d = 0; d < dim; ++d) {
            data[i * dim + d] = dis(gen);
        }
    }
    // Second half remains zero vectors
    
    index::IvfPqTrainParams params;
    params.nlist = 64;
    params.m = 16;
    params.nbits = 8;
    params.use_opq = false;
    
    index::IvfPqIndex index;
    
    // Training should handle zero vectors gracefully
    auto train_result = index.train(data.data(), dim, n, params);
    CHECK(train_result.has_value());
    
    auto add_result = index.add(ids.data(), data.data(), n);
    CHECK(add_result.has_value());
    
    // Search with zero vector query
    std::vector<float> zero_query(dim, 0.0f);
    index::IvfPqSearchParams search_params;
    search_params.k = 10;
    search_params.nprobe = 16;
    
    auto results = index.search(zero_query.data(), search_params);
    CHECK(results.has_value());
    
    std::cout << "\nZero vector search returned " << results->size() << " results\n";
}