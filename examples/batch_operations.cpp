/**
 * Batch operations example for Vesper
 * 
 * This example demonstrates:
 * - Efficient batch insertion
 * - Batch updates
 * - Batch deletion
 * - Progress tracking
 * - Error handling
 */

#include <vesper/vesper.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>

using namespace vesper;
using namespace std::chrono;

// Generate batch of random vectors
std::vector<std::vector<float>> generate_batch(
    std::size_t batch_size, 
    std::size_t dim, 
    std::uint32_t seed) {
    
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<std::vector<float>> batch;
    batch.reserve(batch_size);
    
    for (std::size_t i = 0; i < batch_size; ++i) {
        std::vector<float> vec(dim);
        for (auto& v : vec) {
            v = dist(gen);
        }
        
        // Normalize
        float norm = 0.0f;
        for (auto v : vec) {
            norm += v * v;
        }
        norm = std::sqrt(norm);
        for (auto& v : vec) {
            v /= norm;
        }
        
        batch.push_back(std::move(vec));
    }
    
    return batch;
}

// Progress reporter
class ProgressReporter {
public:
    ProgressReporter(std::size_t total) 
        : total_(total), processed_(0), start_time_(steady_clock::now()) {}
    
    void update(std::size_t count) {
        processed_ += count;
        
        auto now = steady_clock::now();
        auto elapsed = duration_cast<seconds>(now - start_time_).count();
        
        float progress = static_cast<float>(processed_) / total_ * 100.0f;
        float rate = elapsed > 0 ? static_cast<float>(processed_) / elapsed : 0;
        
        std::cout << "\rProgress: " << processed_ << "/" << total_ 
                  << " (" << std::fixed << std::setprecision(1) << progress << "%)"
                  << " - Rate: " << std::fixed << std::setprecision(0) << rate 
                  << " vectors/sec" << std::flush;
    }
    
    void finish() {
        auto elapsed = duration_cast<milliseconds>(
            steady_clock::now() - start_time_
        ).count();
        
        std::cout << "\nCompleted: " << processed_ << " vectors in " 
                  << elapsed << "ms" << std::endl;
    }

private:
    std::size_t total_;
    std::size_t processed_;
    steady_clock::time_point start_time_;
};

int main() {
    try {
        // Initialize Vesper
        vesper::initialize({
            .data_dir = "./vesper_batch_data",
            .cache_size_mb = 512,
            .num_threads = 8
        });
        
        // Create collection
        const std::uint32_t dimension = 256;
        const std::size_t total_vectors = 100000;
        const std::size_t batch_size = 1000;
        
        auto coll_result = Collection::create("batch_demo", {
            .dimension = dimension,
            .metric = DistanceMetric::L2,
            .index_type = IndexType::IVF_PQ,
            .max_vectors = total_vectors
        });
        
        if (!coll_result) {
            std::cerr << "Failed to create collection: " 
                     << coll_result.error().message << std::endl;
            return 1;
        }
        
        auto& collection = coll_result.value();
        std::cout << "Collection created for " << total_vectors 
                  << " vectors" << std::endl;
        
        // Batch insertion with progress tracking
        std::cout << "\n=== Batch Insertion ===" << std::endl;
        ProgressReporter insert_progress(total_vectors);
        
        std::size_t start_id = 0;
        std::size_t errors = 0;
        
        for (std::size_t batch_num = 0; 
             batch_num < total_vectors / batch_size; 
             ++batch_num) {
            
            // Generate batch
            auto vectors = generate_batch(batch_size, dimension, batch_num);
            
            // Generate IDs
            std::vector<std::uint64_t> ids;
            ids.reserve(batch_size);
            for (std::size_t i = 0; i < batch_size; ++i) {
                ids.push_back(start_id + i);
            }
            
            // Insert batch
            auto result = collection->add_batch(ids, vectors);
            
            if (!result) {
                std::cerr << "\nBatch " << batch_num << " failed: " 
                         << result.error().message << std::endl;
                errors++;
            }
            
            start_id += batch_size;
            insert_progress.update(batch_size);
            
            // Periodic flush for durability
            if (batch_num % 10 == 0) {
                collection->flush();
            }
        }
        
        insert_progress.finish();
        std::cout << "Insertion completed with " << errors << " errors" << std::endl;
        
        // Get statistics
        auto stats = collection->get_stats();
        std::cout << "Vectors in collection: " << stats.num_vectors << std::endl;
        
        // Batch updates
        std::cout << "\n=== Batch Updates ===" << std::endl;
        const std::size_t update_count = 10000;
        ProgressReporter update_progress(update_count);
        
        for (std::size_t i = 0; i < update_count; i += batch_size) {
            std::size_t current_batch = std::min(batch_size, update_count - i);
            
            // Generate new vectors
            auto new_vectors = generate_batch(current_batch, dimension, i + 1000);
            
            // Update existing vectors
            for (std::size_t j = 0; j < current_batch; ++j) {
                auto result = collection->update(i + j, new_vectors[j]);
                if (!result) {
                    errors++;
                }
            }
            
            update_progress.update(current_batch);
        }
        
        update_progress.finish();
        std::cout << "Updates completed with " << errors << " errors" << std::endl;
        
        // Batch search
        std::cout << "\n=== Batch Search ===" << std::endl;
        const std::size_t num_queries = 100;
        auto query_vectors = generate_batch(num_queries, dimension, 99999);
        
        auto search_start = high_resolution_clock::now();
        std::size_t total_results = 0;
        
        for (const auto& query : query_vectors) {
            auto results = collection->search(query, 10);
            if (results) {
                total_results += results.value().size();
            }
        }
        
        auto search_end = high_resolution_clock::now();
        auto search_time = duration_cast<milliseconds>(search_end - search_start);
        
        std::cout << "Searched " << num_queries << " queries in " 
                  << search_time.count() << "ms" << std::endl;
        std::cout << "Average latency: " 
                  << static_cast<float>(search_time.count()) / num_queries 
                  << "ms per query" << std::endl;
        std::cout << "Total results: " << total_results << std::endl;
        
        // Batch deletion
        std::cout << "\n=== Batch Deletion ===" << std::endl;
        const std::size_t delete_count = 5000;
        std::vector<std::uint64_t> ids_to_delete;
        
        for (std::size_t i = 0; i < delete_count; ++i) {
            ids_to_delete.push_back(i * 2); // Delete even IDs
        }
        
        auto delete_start = high_resolution_clock::now();
        auto delete_result = collection->remove_batch(ids_to_delete);
        auto delete_end = high_resolution_clock::now();
        
        if (delete_result) {
            auto delete_time = duration_cast<milliseconds>(delete_end - delete_start);
            std::cout << "Deleted " << delete_count << " vectors in " 
                     << delete_time.count() << "ms" << std::endl;
        } else {
            std::cerr << "Deletion failed: " << delete_result.error().message 
                     << std::endl;
        }
        
        // Compact after deletions
        std::cout << "\n=== Compaction ===" << std::endl;
        std::cout << "Starting compaction..." << std::endl;
        
        auto compact_start = high_resolution_clock::now();
        auto compact_result = collection->compact();
        auto compact_end = high_resolution_clock::now();
        
        if (compact_result) {
            auto compact_time = duration_cast<seconds>(compact_end - compact_start);
            std::cout << "Compaction completed in " << compact_time.count() 
                     << " seconds" << std::endl;
        }
        
        // Final statistics
        stats = collection->get_stats();
        std::cout << "\n=== Final Statistics ===" << std::endl;
        std::cout << "Vectors: " << stats.num_vectors << std::endl;
        std::cout << "Segments: " << stats.num_segments << std::endl;
        std::cout << "Disk usage: " << stats.disk_bytes / (1024 * 1024) 
                  << " MB" << std::endl;
        std::cout << "Memory usage: " << stats.memory_bytes / (1024 * 1024) 
                  << " MB" << std::endl;
        
        // Create snapshot
        std::cout << "\n=== Creating Snapshot ===" << std::endl;
        auto snapshot_result = collection->snapshot("batch_demo_snapshot");
        if (snapshot_result) {
            std::cout << "Snapshot created: " << snapshot_result.value() 
                     << std::endl;
        }
        
        std::cout << "\nBatch operations example completed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}