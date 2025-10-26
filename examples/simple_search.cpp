/**
 * Simple vector search example using Vesper
 * 
 * This example demonstrates:
 * - Creating a collection
 * - Adding vectors
 * - Performing similarity search
 * - Using filters
 */

#include <vesper/vesper.hpp>
#include <iostream>
#include <vector>
#include <random>

// Generate random vectors for demonstration
std::vector<float> generate_random_vector(std::size_t dim, std::uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> vec(dim);
    for (auto& v : vec) {
        v = dist(gen);
    }
    
    // Normalize for cosine similarity
    float norm = 0.0f;
    for (auto v : vec) {
        norm += v * v;
    }
    norm = std::sqrt(norm);
    for (auto& v : vec) {
        v /= norm;
    }
    
    return vec;
}

int main() {
    using namespace vesper;
    
    try {
        // Initialize Vesper
        vesper::initialize({
            .data_dir = "./vesper_data",
            .cache_size_mb = 256,
            .num_threads = 4
        });
        
        // Create a collection for 128-dimensional vectors
        const std::uint32_t dimension = 128;
        
        auto coll_result = Collection::create("demo_collection", {
            .dimension = dimension,
            .metric = DistanceMetric::COSINE,
            .index_type = IndexType::HNSW,
            .max_vectors = 10000
        });
        
        if (!coll_result) {
            std::cerr << "Failed to create collection: " 
                     << coll_result.error().message << std::endl;
            return 1;
        }
        
        auto& collection = coll_result.value();
        std::cout << "Collection created successfully!" << std::endl;
        
        // Add some vectors with metadata
        const std::size_t num_vectors = 1000;
        std::cout << "Adding " << num_vectors << " vectors..." << std::endl;
        
        for (std::size_t i = 0; i < num_vectors; ++i) {
            auto vec = generate_random_vector(dimension, i);
            
            // Add metadata
            VectorMetadata metadata;
            metadata.fields["id"] = static_cast<std::int64_t>(i);
            metadata.fields["category"] = (i % 3 == 0) ? "A" : (i % 3 == 1) ? "B" : "C";
            metadata.fields["score"] = static_cast<double>(i) / num_vectors;
            
            if (i % 10 == 0) {
                metadata.tags.push_back("special");
            }
            if (i % 5 == 0) {
                metadata.tags.push_back("featured");
            }
            
            auto result = collection->add_with_metadata(i, vec, metadata);
            if (!result) {
                std::cerr << "Failed to add vector " << i << ": " 
                         << result.error().message << std::endl;
            }
            
            if ((i + 1) % 100 == 0) {
                std::cout << "  Added " << (i + 1) << " vectors\r" << std::flush;
            }
        }
        std::cout << std::endl;
        
        // Flush to ensure persistence
        collection->flush();
        std::cout << "Vectors added and flushed to disk." << std::endl;
        
        // Perform a simple search
        std::cout << "\nPerforming similarity search..." << std::endl;
        auto query_vec = generate_random_vector(dimension, 9999);
        
        auto search_result = collection->search(query_vec, 10);
        if (!search_result) {
            std::cerr << "Search failed: " << search_result.error().message << std::endl;
            return 1;
        }
        
        std::cout << "Top 10 similar vectors:" << std::endl;
        for (const auto& result : search_result.value()) {
            std::cout << "  ID: " << result.id 
                     << ", Distance: " << result.distance 
                     << ", Similarity: " << (1.0f - result.distance) << std::endl;
        }
        
        // Search with filters
        std::cout << "\nSearching with filters (category='A' AND score > 0.5)..." << std::endl;
        
        Filter filter = Filter::And({
            Filter::Eq("category", "A"),
            Filter::Gt("score", 0.5)
        });
        
        SearchParams params{
            .k = 5,
            .filter = filter,
            .ef_search = 100
        };
        
        auto filtered_result = collection->search(query_vec, params);
        if (!filtered_result) {
            std::cerr << "Filtered search failed: " 
                     << filtered_result.error().message << std::endl;
            return 1;
        }
        
        std::cout << "Top 5 filtered results:" << std::endl;
        for (const auto& result : filtered_result.value()) {
            std::cout << "  ID: " << result.id 
                     << ", Distance: " << result.distance << std::endl;
        }
        
        // Search for vectors with specific tags
        std::cout << "\nSearching for 'special' tagged vectors..." << std::endl;
        
        Filter tag_filter = Filter::HasTag("special");
        SearchParams tag_params{
            .k = 5,
            .filter = tag_filter
        };
        
        auto tag_result = collection->search(query_vec, tag_params);
        if (tag_result) {
            std::cout << "Found " << tag_result.value().size() 
                     << " special vectors" << std::endl;
        }
        
        // Get collection statistics
        auto stats = collection->get_stats();
        std::cout << "\nCollection Statistics:" << std::endl;
        std::cout << "  Total vectors: " << stats.num_vectors << std::endl;
        std::cout << "  Segments: " << stats.num_segments << std::endl;
        std::cout << "  Disk usage: " << stats.disk_bytes << " bytes" << std::endl;
        std::cout << "  Memory usage: " << stats.memory_bytes << " bytes" << std::endl;
        
        std::cout << "\nExample completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}