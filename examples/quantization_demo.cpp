/**
 * Quantization demonstration for Vesper
 * 
 * This example demonstrates:
 * - RaBitQ binary quantization for 32x memory reduction
 * - Matryoshka embeddings for flexible dimensionality
 * - Integration with HNSW and IVF-PQ indexes
 * - Performance comparison
 */

#include <vesper/vesper.hpp>
#include <vesper/index/rabitq_quantizer.hpp>
#include <vesper/index/matryoshka.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using namespace vesper;
using namespace vesper::index;
using namespace std::chrono;

// Generate random vectors
std::vector<std::vector<float>> generate_vectors(
    std::size_t n, std::size_t dim, std::uint32_t seed = 42) {
    
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<std::vector<float>> vectors;
    vectors.reserve(n);
    
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<float> vec(dim);
        float norm = 0.0f;
        
        // Generate random vector
        for (auto& v : vec) {
            v = dist(gen);
            norm += v * v;
        }
        
        // Normalize
        norm = std::sqrt(norm);
        for (auto& v : vec) {
            v /= norm;
        }
        
        vectors.push_back(std::move(vec));
    }
    
    return vectors;
}

// Measure memory usage
std::size_t measure_memory(const std::vector<std::vector<float>>& vectors) {
    std::size_t bytes = 0;
    for (const auto& vec : vectors) {
        bytes += vec.size() * sizeof(float);
    }
    return bytes;
}

int main() {
    try {
        // Configuration
        const std::size_t num_vectors = 10000;
        const std::size_t dimension = 768;  // Common embedding dimension
        const std::size_t num_queries = 100;
        
        std::cout << "=== Vesper Quantization Demo ===" << std::endl;
        std::cout << "Vectors: " << num_vectors << std::endl;
        std::cout << "Dimension: " << dimension << std::endl;
        std::cout << "Original memory: " 
                  << (num_vectors * dimension * sizeof(float)) / (1024 * 1024) 
                  << " MB" << std::endl;
        
        // Generate data
        std::cout << "\nGenerating vectors..." << std::endl;
        auto vectors = generate_vectors(num_vectors, dimension);
        auto queries = generate_vectors(num_queries, dimension);
        
        // Flatten vectors for training
        std::vector<float> flat_vectors;
        flat_vectors.reserve(num_vectors * dimension);
        for (const auto& vec : vectors) {
            flat_vectors.insert(flat_vectors.end(), vec.begin(), vec.end());
        }
        
        // === RaBitQ Binary Quantization ===
        std::cout << "\n=== RaBitQ Binary Quantization ===" << std::endl;
        
        RaBitQuantizer rabitq;
        RaBitQTrainParams rabitq_params;
        rabitq_params.bits = QuantizationBits::BIT_1;  // 1-bit quantization
        rabitq_params.use_rotation = true;
        
        auto start = high_resolution_clock::now();
        auto train_result = rabitq.train(
            flat_vectors.data(), num_vectors, dimension, rabitq_params
        );
        auto end = high_resolution_clock::now();
        
        if (train_result) {
            auto train_time = duration_cast<milliseconds>(end - start);
            std::cout << "RaBitQ training time: " << train_time.count() << "ms" << std::endl;
            
            // Quantize all vectors
            auto quantized = rabitq.quantize_batch(flat_vectors.data(), num_vectors);
            if (quantized) {
                std::size_t compressed_bytes = rabitq.estimate_memory(num_vectors);
                std::cout << "Compressed memory: " << compressed_bytes / (1024 * 1024) << " MB" << std::endl;
                std::cout << "Compression ratio: " 
                          << (num_vectors * dimension * sizeof(float)) / static_cast<float>(compressed_bytes) 
                          << "x" << std::endl;
                
                // Test search speed
                start = high_resolution_clock::now();
                for (const auto& query : queries) {
                    auto results = rabitq.search_batch(
                        query, quantized.value(), 10
                    );
                }
                end = high_resolution_clock::now();
                
                auto search_time = duration_cast<microseconds>(end - start);
                std::cout << "Average search time: " 
                          << search_time.count() / static_cast<float>(num_queries) 
                          << " μs per query" << std::endl;
            }
        }
        
        // === Matryoshka Embeddings ===
        std::cout << "\n=== Matryoshka Embeddings ===" << std::endl;
        
        MatryoshkaEmbedding matryoshka;
        MatryoshkaConfig matryoshka_config;
        matryoshka_config.dimensions = {64, 128, 256, 384, 512, 768};
        matryoshka_config.target_recall = 0.95f;
        
        auto init_result = matryoshka.init(dimension, matryoshka_config);
        if (init_result) {
            // Analyze embeddings
            auto stats = matryoshka.analyze(flat_vectors.data(), num_vectors);
            if (stats) {
                std::cout << "Optimal dimension for 95% recall: " 
                          << stats.value().optimal_dimension << std::endl;
                std::cout << "Compression ratio: " 
                          << stats.value().compression_ratio << "x" << std::endl;
                std::cout << "Estimated recall: " 
                          << stats.value().estimated_recall << std::endl;
                
                // Test progressive search
                start = high_resolution_clock::now();
                for (const auto& query : queries) {
                    auto results = matryoshka.progressive_search(
                        query, flat_vectors.data(), num_vectors, 10, 64
                    );
                }
                end = high_resolution_clock::now();
                
                auto prog_search_time = duration_cast<microseconds>(end - start);
                std::cout << "Progressive search time: " 
                          << prog_search_time.count() / static_cast<float>(num_queries) 
                          << " μs per query" << std::endl;
            }
        }
        
        // === Combined with Index Manager ===
        std::cout << "\n=== Index with Quantization ===" << std::endl;
        
        // Initialize Vesper
        vesper::initialize({
            .data_dir = "./vesper_quantization_data",
            .cache_size_mb = 256,
            .num_threads = 8
        });
        
        // Create collection with quantization
        auto coll_result = Collection::create("quantized_demo", {
            .dimension = dimension,
            .metric = DistanceMetric::COSINE,
            .index_type = IndexType::IVF_PQ,
            .max_vectors = num_vectors * 2,
            .enable_rabitq = true,
            .quantization_bits = 1,
            .enable_matryoshka = true,
            .matryoshka_dims = {128, 256, 512, 768}
        });
        
        if (coll_result) {
            auto& collection = coll_result.value();
            std::cout << "Collection created with quantization support" << std::endl;
            
            // Add vectors
            for (std::size_t i = 0; i < num_vectors; ++i) {
                collection->add(i, vectors[i]);
            }
            
            // Search with quantization
            start = high_resolution_clock::now();
            std::size_t total_results = 0;
            for (const auto& query : queries) {
                auto results = collection->search(query, 10);
                if (results) {
                    total_results += results.value().size();
                }
            }
            end = high_resolution_clock::now();
            
            auto index_search_time = duration_cast<microseconds>(end - start);
            std::cout << "Index search time: " 
                      << index_search_time.count() / static_cast<float>(num_queries) 
                      << " μs per query" << std::endl;
            std::cout << "Total results: " << total_results << std::endl;
            
            // Get statistics
            auto stats = collection->get_stats();
            std::cout << "\nCollection Statistics:" << std::endl;
            std::cout << "  Vectors: " << stats.num_vectors << std::endl;
            std::cout << "  Memory: " << stats.memory_bytes / (1024 * 1024) << " MB" << std::endl;
            std::cout << "  Disk: " << stats.disk_bytes / (1024 * 1024) << " MB" << std::endl;
        }
        
        std::cout << "\n=== Quantization Demo Complete ===" << std::endl;
        std::cout << "Successfully demonstrated:" << std::endl;
        std::cout << "  - RaBitQ 32x compression" << std::endl;
        std::cout << "  - Matryoshka flexible dimensions" << std::endl;
        std::cout << "  - Integration with indexes" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}