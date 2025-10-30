# Vesper API Reference

## Overview

Vesper is a crash-safe, embeddable vector search engine designed for CPU-only environments. This document provides comprehensive API documentation for integrating Vesper into your applications.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Collection API](#collection-api)
3. [Index API](#index-api)
4. [Query API](#query-api)
5. [Storage API](#storage-api)
6. [Error Handling](#error-handling)
7. [Configuration](#configuration)
8. [Examples](#examples)

## Core Concepts

### Vectors
- **Dimensions**: 64-4096 (typical: 128, 256, 512, 768, 1536)
- **Types**: `float32`, `float16`, `int8` (quantized)
- **Distance Metrics**: L2 (squared), Inner Product, Cosine

### Collections
A collection is a named set of vectors with associated metadata. Each collection can use different index types optimized for specific use cases.

### Index Types
- **IVF-PQ**: Inverted File with Product Quantization - memory efficient, disk-resident
- **HNSW**: Hierarchical Navigable Small World - high recall, low latency, RAM-resident
- **DiskGraph**: Vamana/DiskANN-style - billion-scale, SSD-resident

## Collection API

### Creating a Collection

```cpp
#include <vesper/collection.hpp>

using namespace vesper;

// Create collection with default settings
auto coll_result = Collection::create("my_collection", {
    .dimension = 768,
    .metric = DistanceMetric::L2,
    .index_type = IndexType::HNSW
});

if (!coll_result) {
    // Handle error
    std::cerr << "Error: " << coll_result.error().message << std::endl;
    return;
}

auto& collection = coll_result.value();
```

### Collection Configuration

```cpp
struct CollectionConfig {
    std::uint32_t dimension;           // Vector dimension
    DistanceMetric metric;              // L2, IP, or Cosine
    IndexType index_type;               // IVF_PQ, HNSW, or DISK_GRAPH

    // Optional parameters
    std::size_t max_vectors = 1000000; // Maximum vectors
    bool use_compression = false;       // Enable PQ compression
    std::uint32_t compression_bits = 8; // Bits per subquantizer
};
```

### Opening an Existing Collection

```cpp
auto coll_result = Collection::open("my_collection");
if (!coll_result) {
    // Handle error
}
auto& collection = coll_result.value();
```

## Index API

### Adding Vectors

```cpp
// Single vector insertion
std::vector<float> vector(768);  // Your embedding
std::uint64_t id = 12345;        // Unique ID

auto result = collection->add(id, vector);
if (!result) {
    // Handle error
}

// Batch insertion for better performance
std::vector<std::uint64_t> ids = {1, 2, 3, 4, 5};
std::vector<std::vector<float>> vectors = {...};  // 5 vectors

auto batch_result = collection->add_batch(ids, vectors);
if (!batch_result) {
    // Handle error
}
```

### Adding Vectors with Metadata

```cpp
struct VectorMetadata {
    std::unordered_map<std::string, std::variant<
        bool, int64_t, double, std::string
    >> fields;

    std::vector<std::string> tags;
    std::int64_t timestamp = 0;
};

VectorMetadata metadata;
metadata.fields["category"] = "document";
metadata.fields["score"] = 0.95;
metadata.tags = {"important", "reviewed"};

auto result = collection->add_with_metadata(id, vector, metadata);
```

### Updating Vectors

```cpp
// Update existing vector (upsert semantics)
auto result = collection->update(id, new_vector);

// Update metadata only
auto result = collection->update_metadata(id, new_metadata);
```

### Deleting Vectors

```cpp
// Delete single vector
auto result = collection->remove(id);

// Delete multiple vectors
std::vector<std::uint64_t> ids_to_delete = {1, 2, 3};
auto result = collection->remove_batch(ids_to_delete);
```

## Query API

### Basic Search

```cpp
std::vector<float> query_vector(768);  // Your query embedding
std::uint32_t k = 10;                  // Top-k results

auto results = collection->search(query_vector, k);
if (!results) {
    // Handle error
}

for (const auto& result : results.value()) {
    std::cout << "ID: " << result.id
              << " Distance: " << result.distance << std::endl;
}
```

### Search with Filters

```cpp
// Create filter expression
Filter filter = Filter::And({
    Filter::Eq("category", "document"),
    Filter::Gt("score", 0.8),
    Filter::In("tags", {"important", "reviewed"})
});

SearchParams params{
    .k = 10,
    .filter = filter,
    .ef_search = 100  // HNSW parameter
};

auto results = collection->search(query_vector, params);
```

### Filter Operators

```cpp
// Comparison operators
Filter::Eq("field", value)    // Equal
Filter::Ne("field", value)    // Not equal
Filter::Lt("field", value)    // Less than
Filter::Le("field", value)    // Less than or equal
Filter::Gt("field", value)    // Greater than
Filter::Ge("field", value)    // Greater than or equal

// Range operators
Filter::Between("field", min, max)
Filter::In("field", {val1, val2, val3})
Filter::NotIn("field", {val1, val2})

// String operators
Filter::StartsWith("field", "prefix")
Filter::Contains("field", "substring")

// Logical operators
Filter::And({filter1, filter2})
Filter::Or({filter1, filter2})
Filter::Not(filter)

// Tag operators
Filter::HasTag("tag_name")
Filter::HasAnyTag({"tag1", "tag2"})
Filter::HasAllTags({"tag1", "tag2"})
```

### Advanced Search Parameters

```cpp
struct SearchParams {
    std::uint32_t k = 10;              // Number of results
    Filter filter;                      // Optional filter

    // Index-specific parameters
    std::uint32_t ef_search = 50;      // HNSW: search beam width
    std::uint32_t nprobe = 10;         // IVF-PQ: cells to probe
    std::uint32_t beam_width = 100;    // DiskGraph: beam search width

    // Performance hints
    bool exact_rerank = false;         // Re-rank with exact distance
    std::uint32_t rerank_k = 0;        // Number to re-rank (0 = k*2)
};
```

## Storage API

### Persistence Operations

```cpp
// Force flush to disk
auto result = collection->flush();

// Create snapshot
auto snapshot_result = collection->snapshot("snapshot_name");
if (snapshot_result) {
    std::cout << "Snapshot created: " << snapshot_result.value() << std::endl;
}

// Compact segments (background operation)
auto compact_result = collection->compact();

// Get collection statistics
auto stats = collection->get_stats();
std::cout << "Vectors: " << stats.num_vectors << std::endl;
std::cout << "Segments: " << stats.num_segments << std::endl;
std::cout << "Disk usage: " << stats.disk_bytes << std::endl;
```

### WAL Management

```cpp
// Configure WAL settings
WalConfig wal_config{
    .max_size_mb = 1024,           // Maximum WAL size
    .sync_interval_ms = 100,       // Sync interval
    .compression = true,           // Enable compression
    .retention_days = 7            // Keep for 7 days
};

collection->configure_wal(wal_config);

// Manual checkpoint
collection->checkpoint();
```

#### WAL Manifest Utilities

Vesper provides utilities to rebuild and validate the WAL manifest from on-disk WAL segment files. There are two explicit modes:

- Strict rebuild (default): fail-fast on the first data-integrity violation; returns an error.
- Lenient rebuild (best-effort): skip corrupt or invalid files/entries and return a partial manifest alongside structured issues.

```cpp
#include <vesper/wal/manifest.hpp>

// Strict (fail-fast)
if (auto m = wal::rebuild_manifest(dir)) {
    wal::save_manifest(dir, *m);  // overwrite wal.manifest
} else {
    // Handle error: m.error().code == error_code::data_integrity, etc.
}

// Lenient (best-effort)
if (auto r = wal::rebuild_manifest_lenient(dir)) {
    // r->manifest contains only valid entries; r->issues documents skipped files/entries
    wal::save_manifest(dir, r->manifest);
    for (const auto& issue : r->issues) {
        std::cout << "issue: file=" << issue.file << " seq=" << issue.seq << "\n";
    }
}

// Validate manifest on disk (advisory). LSN gaps are warnings; invariants must hold.
if (auto v = wal::validate_manifest(dir)) {
    if (!v->ok) {
        for (auto& is : v->issues) {
            // Severity::Error indicates a non-conforming manifest
        }
    }
}
```

API summary:
- `auto wal::rebuild_manifest(const std::filesystem::path&) -> expected<Manifest, error>`
- `auto wal::rebuild_manifest_lenient(const std::filesystem::path&) -> expected<LenientRebuildResult, error>`
  - `struct LenientRebuildResult { Manifest manifest; std::vector<RebuildIssue> issues; }`
  - `struct RebuildIssue { std::string file; std::uint64_t seq; error_code code; std::string message; /* optional LSN fields */ }`
```

## Error Handling

Vesper uses `std::expected` for error handling:

```cpp
#include <vesper/core/error.hpp>

// All operations return expected<T, error>
auto result = collection->add(id, vector);

if (!result) {
    const auto& error = result.error();

    switch (error.code) {
        case error_code::duplicate_id:
            // Handle duplicate
            break;
        case error_code::dimension_mismatch:
            // Handle wrong dimension
            break;
        case error_code::out_of_memory:
            // Handle OOM
            break;
        default:
            std::cerr << "Error: " << error.message
                     << " (" << error.source << ")" << std::endl;
    }
}
```

### Error Codes

```cpp
enum class error_code {
    // General errors
    success = 0,
    unknown_error,
    invalid_argument,
    out_of_memory,
    io_error,

    // Collection errors
    collection_not_found,
    collection_exists,

    // Vector errors
    dimension_mismatch,
    duplicate_id,
    vector_not_found,

    // Index errors
    index_not_ready,
    index_build_failed,

    // Storage errors
    wal_error,
    snapshot_failed,
    corruption_detected,

    // Query errors
    invalid_filter,
    query_timeout
};
```

## Configuration

### Global Configuration

```cpp
#include <vesper/config.hpp>

VesperConfig config{
    .data_dir = "/path/to/data",
    .cache_size_mb = 1024,
    .num_threads = 8,
    .enable_metrics = true,
    .log_level = LogLevel::INFO
};

vesper::initialize(config);
```

### Index-Specific Configuration

#### HNSW Configuration

```cpp
HnswConfig hnsw_config{
    .M = 16,                    // Number of connections
    .ef_construction = 200,     // Construction beam width
    .max_M = 16,               // Max connections
    .seed = 42,                // Random seed
    .use_heuristic = true      // Use pruning heuristic
};
```

#### IVF-PQ Configuration

```cpp
IvfPqConfig ivf_config{
    .nlist = 1024,             // Number of inverted lists
    .nprobe = 10,              // Lists to search
    .m = 8,                    // Number of subquantizers
    .nbits = 8,                // Bits per subquantizer
    .use_opq = true,           // Use Optimized PQ
    .train_size = 100000       // Training set size
};
```

##### IVF-PQ coarse assigner selection and ANN toggles

Defaults:
- coarse_assigner = KDTree (exact, recommended)
- use_centroid_ann = true (enables non-brute assigners; for KDTree this enables the exact KD assignment path)

You can opt into HNSW or Projection assigners for experimentation; KDTree remains the recommended default for exactness.

```cpp
#include <vesper/index/ivf_pq.hpp>
using vesper::index::IvfPqTrainParams;
using vesper::index::CoarseAssigner;

IvfPqTrainParams tp;
// Core
tp.nlist = 2048;              // number of coarse centroids
tp.m = 16;                    // subquantizers
tp.nbits = 8;                 // bits per subquantizer

// Coarse assigner selection (default KDTree)
tp.coarse_assigner = CoarseAssigner::KDTree; // default

tp.use_centroid_ann = true;   // default: true

// HNSW-specific ANN toggles (only used when coarse_assigner==HNSW)
// Controls recall/latency tradeoff during assignment:
tp.centroid_ann_ef_search = 96;     // default: 96 (raise to 128–200 for higher quality)
// Controls HNSW construction quality when building centroid graph:
tp.centroid_ann_ef_construction = 200; // default: 200

// Optional sampled correctness validation vs brute-force assignment during add()
tp.validate_ann_assignment = false; // default: false (enable for debugging/acceptance)
tp.validate_ann_sample_rate = 0.01f; // default: 0.0f (fraction in [0,1])
```

Recommended acceptance configuration (Phase 1 exit criteria):
- nlist ≥ 1024 (e.g., 2048)
- ef_search = 200
- ef_construction = 200
- validate_ann_assignment = true
- validate_ann_sample_rate = 0.01

Use tools/ivfpq_add_bench (see docs/IVFPQ_Add_Bench.md) to measure add() throughput and verify ≥3x speedup vs brute-force with ≤1% sampled mismatch.



### IVF-PQ K-means Initialization (Elkan coarse quantizer)

For training the IVF coarse quantizer, Vesper exposes two initialization methods for Elkan's k-means:
- KMeansPlusPlus (default): high-quality sequential seeding; recommended for small/medium n (< 10k)
- KMeansParallel (k-means||): scalable parallel sampling per Bahmani et al. (VLDB'12); recommended for large n (> 10k)

Fields in IvfPqTrainParams (defaults preserve prior behavior):

```cpp
#include <vesper/index/ivf_pq.hpp>
using vesper::index::IvfPqTrainParams;

IvfPqTrainParams tp;
// ... core IVF params ...
// K-means init (Elkan) for coarse quantizer
tp.kmeans_init_method = vesper::index::KmeansElkan::Config::InitMethod::KMeansPlusPlus; // default
tp.kmeans_parallel_rounds = 5;          // sampling rounds (typical: 3–8)
tp.kmeans_parallel_oversampling = 0;    // oversampling factor l; 0 => 2*k (typical: k..4*k)
```

Example (enable k-means||):

```cpp
IvfPqTrainParams tp;
// ... set nlist, m, nbits, etc. ...
tp.kmeans_init_method = vesper::index::KmeansElkan::Config::InitMethod::KMeansParallel;
tp.kmeans_parallel_rounds = 5;          // tune 3–8 for quality/throughput
// keep oversampling default (2*k) or set explicitly: tp.kmeans_parallel_oversampling = 2*tp.nlist;
```

Quality: k-means|| produces comparable final inertia to k-means++ on well-separated data at much lower initialization time for large n.

Reference: Bahmani, Moseley, Vattani, Kumar, Vassilvitskii. "Scalable K-Means++" (VLDB 2012).

### IVF-PQ Exact Rerank Controls

Use QueryConfig to enable exact reranking and control the shortlist used before exact L2 rerank. The adaptive heuristic sizes the candidate pool as cand_k = max(k, alpha * k * log2(1+nprobe)).

```cpp
vesper::index::QueryConfig qc{
  .k = 10,
  .nprobe = 128,
  .use_exact_rerank = true,
  .rerank_k = 0,             // 0 = auto via heuristic
  .rerank_alpha = 2.0f,      // heuristic multiplier
  .rerank_cand_ceiling = 2000// hard cap (0 = no cap)
};
```

Environment overrides (for quick experiments): VESPER_USE_EXACT_RERANK, VESPER_RERANK_K, VESPER_RERANK_ALPHA, VESPER_RERANK_CEIL.

#### DiskGraph Configuration

```cpp
DiskGraphConfig disk_config{
    .R = 64,                   // Graph degree
    .L = 128,                  // Build search list size
    .alpha = 1.2f,             // Pruning parameter
    .cache_size_mb = 512,      // Graph cache size
    .use_pq = true,            // Use PQ compression
    .build_threads = 0         // 0 = auto
};
```

## Examples

### Complete Example: Document Search

```cpp
#include <vesper/vesper.hpp>
#include <iostream>
#include <vector>

int main() {
    using namespace vesper;

    // Initialize Vesper
    vesper::initialize({
        .data_dir = "./vesper_data",
        .cache_size_mb = 512
    });

    // Create or open collection
    auto coll_result = Collection::create("documents", {
        .dimension = 768,
        .metric = DistanceMetric::COSINE,
        .index_type = IndexType::HNSW
    });

    if (!coll_result) {
        std::cerr << "Failed to create collection: "
                  << coll_result.error().message << std::endl;
        return 1;
    }

    auto& collection = coll_result.value();

    // Add documents with metadata
    std::vector<Document> documents = load_documents();

    for (const auto& doc : documents) {
        VectorMetadata metadata;
        metadata.fields["title"] = doc.title;
        metadata.fields["content"] = doc.content;
        metadata.fields["date"] = doc.timestamp;
        metadata.tags = doc.tags;

        auto result = collection->add_with_metadata(
            doc.id,
            doc.embedding,
            metadata
        );

        if (!result) {
            std::cerr << "Failed to add document " << doc.id << std::endl;
        }
    }

    // Search with filters
    auto query_embedding = encode_query("machine learning");

    Filter filter = Filter::And({
        Filter::Ge("date", timestamp_30_days_ago()),
        Filter::HasAnyTag({"AI", "ML", "deep-learning"})
    });

    SearchParams params{
        .k = 10,
        .filter = filter,
        .ef_search = 100
    };

    auto results = collection->search(query_embedding, params);

    if (results) {
        for (const auto& result : results.value()) {
            std::cout << "Document ID: " << result.id
                     << " Score: " << (1.0 - result.distance)
                     << std::endl;
        }
    }

    // Clean shutdown
    collection->flush();

    return 0;
}
```

### Batch Processing Example

```cpp
// Efficient batch insertion with progress tracking
void batch_insert(Collection& collection,
                  const std::vector<Vector>& vectors,
                  std::size_t batch_size = 1000) {

    for (std::size_t i = 0; i < vectors.size(); i += batch_size) {
        std::size_t end = std::min(i + batch_size, vectors.size());

        std::vector<std::uint64_t> ids;
        std::vector<std::vector<float>> batch_vectors;

        for (std::size_t j = i; j < end; ++j) {
            ids.push_back(vectors[j].id);
            batch_vectors.push_back(vectors[j].data);
        }

        auto result = collection.add_batch(ids, batch_vectors);

        if (!result) {
            std::cerr << "Batch failed at " << i << std::endl;
            // Handle error or retry
        }

        // Progress
        std::cout << "Inserted " << end << "/" << vectors.size()
                  << " vectors\r" << std::flush;
    }

    // Ensure persistence
    collection.flush();
}
```

### Recovery Example

```cpp
// Crash recovery and verification
void recover_collection(const std::string& name) {
    auto coll_result = Collection::open(name);

    if (!coll_result) {
        if (coll_result.error().code == error_code::corruption_detected) {
            std::cout << "Corruption detected, attempting recovery..."
                      << std::endl;

            // Try to recover from snapshot
            auto recovery_result = Collection::recover_from_snapshot(
                name,
                "latest"
            );

            if (recovery_result) {
                std::cout << "Recovered from snapshot: "
                          << recovery_result.value() << std::endl;
                coll_result = Collection::open(name);
            }
        }
    }

    if (coll_result) {
        auto& collection = coll_result.value();
        auto stats = collection->get_stats();

        std::cout << "Collection recovered:" << std::endl;
        std::cout << "  Vectors: " << stats.num_vectors << std::endl;
        std::cout << "  Segments: " << stats.num_segments << std::endl;
        std::cout << "  WAL entries: " << stats.wal_entries << std::endl;

        // Verify integrity
        auto verify_result = collection->verify_integrity();
        if (verify_result) {
            std::cout << "Integrity check passed" << std::endl;
        }
    }
}
```

## Performance Tips

1. **Batch Operations**: Always prefer batch operations over individual inserts
2. **Index Selection**:
   - Use HNSW for < 10M vectors with high recall requirements
   - Use IVF-PQ for memory-constrained scenarios
   - Use DiskGraph for billion-scale with SSD storage
3. **Parameter Tuning**:
   - Increase `ef_search`/`nprobe` for better recall
   - Decrease for lower latency
   - Find the sweet spot for your use case
4. **Memory Management**:
   - Configure appropriate cache sizes
   - Use memory-mapped files for large datasets
   - Enable compression for memory savings
5. **Concurrency**:
   - Vesper is thread-safe for reads
   - Writes are serialized per collection
   - Use multiple collections for write parallelism

## Thread Safety

- All read operations are thread-safe
- Write operations are serialized per collection
- Snapshots provide consistent point-in-time views
- Use read-write locks for custom synchronization

## Migration Guide

### From v0.x to v1.0

```cpp
// Old API (v0.x)
auto collection = vesper::open_collection("name");
collection.insert(id, vector);

// New API (v1.0)
auto coll_result = vesper::Collection::open("name");
if (coll_result) {
    coll_result.value()->add(id, vector);
}
```

## Support

- GitHub Issues: https://github.com/vesper-arch/vesper/issues
- Documentation: https://vesper-arch.github.io/vesper
- Examples: https://github.com/vesper-arch/vesper/tree/main/examples
