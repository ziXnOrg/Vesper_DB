/** \file hybrid_search_bench.cpp
 *  \brief Performance benchmarks for hybrid sparse-dense search.
 */

#include <benchmark/benchmark.h>
#include "vesper/index/bm25.hpp"
#include "vesper/search/hybrid_searcher.hpp"
#include "vesper/search/fusion_algorithms.hpp"
#include "vesper/index/index_manager.hpp"
#include <random>
#include <string>
#include <vector>
#include <memory>

using namespace vesper;
using namespace vesper::index;
using namespace vesper::search;
using namespace vesper::search::fusion;

namespace {

// Generate random text documents
std::vector<std::string> generate_documents(std::size_t n_docs, 
                                           std::size_t avg_length,
                                           std::size_t vocab_size) {
    std::vector<std::string> docs;
    std::mt19937 gen(42);
    std::uniform_int_distribution<> word_dist(0, vocab_size - 1);
    std::normal_distribution<> len_dist(static_cast<double>(avg_length), avg_length / 4.0);
    
    for (std::size_t i = 0; i < n_docs; ++i) {
        std::string doc;
        int doc_len = std::max(1, static_cast<int>(len_dist(gen)));
        
        for (int j = 0; j < doc_len; ++j) {
            if (j > 0) doc += " ";
            doc += "word" + std::to_string(word_dist(gen));
        }
        docs.push_back(doc);
    }
    
    return docs;
}

// Generate random embeddings
std::vector<std::vector<float>> generate_embeddings(std::size_t n_vecs, std::size_t dim) {
    std::vector<std::vector<float>> embeddings;
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (std::size_t i = 0; i < n_vecs; ++i) {
        std::vector<float> vec(dim);
        for (auto& val : vec) {
            val = dist(gen);
        }
        
        // Normalize
        float norm = 0.0f;
        for (float val : vec) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        for (auto& val : vec) {
            val /= norm;
        }
        
        embeddings.push_back(vec);
    }
    
    return embeddings;
}

// Simple mock IndexManager for benchmarking
class BenchIndexManager : public IndexManager {
public:
    BenchIndexManager(std::size_t dim) : dim_(dim) {
        config_.dimension = dim;
    }
    
    auto add(const float* embedding, std::uint64_t id) 
        -> std::expected<void, core::error> override {
        embeddings_[id] = std::vector<float>(embedding, embedding + dim_);
        return {};
    }
    
    auto search(const float* query, const QueryConfig& config)
        -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error> override {
        
        std::vector<std::pair<std::uint64_t, float>> results;
        results.reserve(embeddings_.size());
        
        // Compute all distances
        for (const auto& [id, embedding] : embeddings_) {
            float score = 0.0f;
            for (std::size_t i = 0; i < dim_; ++i) {
                score += query[i] * embedding[i];
            }
            results.push_back({id, score});
        }
        
        // Partial sort for top-k
        if (results.size() > config.k) {
            std::partial_sort(results.begin(), results.begin() + config.k, results.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
            results.resize(config.k);
        } else {
            std::sort(results.begin(), results.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
        }
        
        return results;
    }
    
    auto get_config() const -> IndexConfig override {
        return config_;
    }
    
private:
    std::size_t dim_;
    IndexConfig config_;
    std::unordered_map<std::uint64_t, std::vector<float>> embeddings_;
};

} // namespace

// BM25 Benchmarks

static void BM_BM25_AddDocument(benchmark::State& state) {
    std::size_t n_docs = state.range(0);
    std::size_t avg_doc_len = 100;
    std::size_t vocab_size = 10000;
    
    auto docs = generate_documents(n_docs, avg_doc_len, vocab_size);
    
    for (auto _ : state) {
        BM25Index index;
        for (std::size_t i = 0; i < n_docs; ++i) {
            auto result = index.add_document(i, docs[i]);
            benchmark::DoNotOptimize(result);
        }
    }
    
    state.SetItemsProcessed(state.iterations() * n_docs);
}
BENCHMARK(BM_BM25_AddDocument)->Range(100, 10000);

static void BM_BM25_Search(benchmark::State& state) {
    std::size_t n_docs = state.range(0);
    std::size_t avg_doc_len = 100;
    std::size_t vocab_size = 10000;
    
    BM25Index index;
    auto docs = generate_documents(n_docs, avg_doc_len, vocab_size);
    
    for (std::size_t i = 0; i < n_docs; ++i) {
        index.add_document(i, docs[i]);
    }
    
    std::vector<std::string> queries = {"word42", "word100 word200", "word1 word2 word3"};
    std::size_t query_idx = 0;
    
    for (auto _ : state) {
        auto results = index.search(queries[query_idx % queries.size()], 10);
        benchmark::DoNotOptimize(results);
        query_idx++;
    }
    
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_BM25_Search)->Range(1000, 100000);

static void BM_BM25_BatchSearch(benchmark::State& state) {
    std::size_t n_docs = 10000;
    std::size_t batch_size = state.range(0);
    
    BM25Index index;
    auto docs = generate_documents(n_docs, 100, 10000);
    
    for (std::size_t i = 0; i < n_docs; ++i) {
        index.add_document(i, docs[i]);
    }
    
    std::vector<std::string> queries;
    for (std::size_t i = 0; i < batch_size; ++i) {
        queries.push_back("word" + std::to_string(i * 10));
    }
    
    for (auto _ : state) {
        auto results = index.batch_search(queries, 10);
        benchmark::DoNotOptimize(results);
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
}
BENCHMARK(BM_BM25_BatchSearch)->Range(1, 100);

// Fusion Algorithm Benchmarks

static void BM_RRF_Fusion(benchmark::State& state) {
    std::size_t n_results = state.range(0);
    
    // Create mock results
    std::vector<SearchResult> dense_results;
    std::vector<SearchResult> sparse_results;
    
    for (std::size_t i = 0; i < n_results; ++i) {
        dense_results.push_back({i, 1.0f - i * 0.01f, static_cast<std::uint32_t>(i + 1)});
        sparse_results.push_back({i + n_results/2, 1.0f - i * 0.01f, static_cast<std::uint32_t>(i + 1)});
    }
    
    ReciprocalRankFusion rrf(60.0f);
    
    for (auto _ : state) {
        auto fused = rrf.fuse(dense_results, sparse_results, n_results);
        benchmark::DoNotOptimize(fused);
    }
    
    state.SetItemsProcessed(state.iterations() * n_results * 2);
}
BENCHMARK(BM_RRF_Fusion)->Range(10, 1000);

static void BM_WeightedFusion(benchmark::State& state) {
    std::size_t n_results = state.range(0);
    
    std::vector<SearchResult> dense_results;
    std::vector<SearchResult> sparse_results;
    
    for (std::size_t i = 0; i < n_results; ++i) {
        dense_results.push_back({i, 1.0f - i * 0.01f, static_cast<std::uint32_t>(i + 1)});
        sparse_results.push_back({i + n_results/2, 1.0f - i * 0.01f, static_cast<std::uint32_t>(i + 1)});
    }
    
    WeightedFusion fusion(0.5f, 0.5f);
    
    for (auto _ : state) {
        auto fused = fusion.fuse(dense_results, sparse_results, n_results);
        benchmark::DoNotOptimize(fused);
    }
    
    state.SetItemsProcessed(state.iterations() * n_results * 2);
}
BENCHMARK(BM_WeightedFusion)->Range(10, 1000);

static void BM_LateInteraction(benchmark::State& state) {
    std::size_t n_tokens = state.range(0);
    std::size_t n_docs = 100;
    
    // Create mock token scores
    std::vector<std::vector<float>> dense_scores;
    std::vector<std::vector<float>> sparse_scores;
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (std::size_t i = 0; i < n_tokens; ++i) {
        std::vector<float> dense_token(n_docs);
        std::vector<float> sparse_token(n_docs);
        
        for (std::size_t j = 0; j < n_docs; ++j) {
            dense_token[j] = dist(gen);
            sparse_token[j] = dist(gen);
        }
        
        dense_scores.push_back(dense_token);
        sparse_scores.push_back(sparse_token);
    }
    
    LateInteractionFusion::Config config;
    config.use_max_sim = true;
    LateInteractionFusion fusion(config);
    
    for (auto _ : state) {
        auto fused = fusion.fuse_tokens(dense_scores, sparse_scores, 10);
        benchmark::DoNotOptimize(fused);
    }
    
    state.SetItemsProcessed(state.iterations() * n_tokens * n_docs);
}
BENCHMARK(BM_LateInteraction)->Range(1, 100);

// End-to-end Hybrid Search Benchmarks

static void BM_HybridSearch_Sequential(benchmark::State& state) {
    std::size_t n_docs = state.range(0);
    std::size_t dim = 128;
    
    // Setup indices
    auto dense_index = std::make_shared<BenchIndexManager>(dim);
    auto sparse_index = std::make_shared<BM25Index>();
    
    // Add documents
    auto text_docs = generate_documents(n_docs, 100, 10000);
    auto embeddings = generate_embeddings(n_docs, dim);
    
    for (std::size_t i = 0; i < n_docs; ++i) {
        sparse_index->add_document(i, text_docs[i]);
        dense_index->add(embeddings[i].data(), i);
    }
    
    HybridSearcher searcher(dense_index, sparse_index);
    
    // Create query
    HybridQuery query;
    query.text = "word100 word200";
    query.dense_embedding = embeddings[0];  // Use first embedding as query
    
    HybridSearchConfig config;
    config.query_strategy = QueryStrategy::DENSE_FIRST;
    config.k = 10;
    
    for (auto _ : state) {
        auto results = searcher.search(query, config);
        benchmark::DoNotOptimize(results);
    }
    
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_HybridSearch_Sequential)->Range(1000, 100000);

static void BM_HybridSearch_Parallel(benchmark::State& state) {
    std::size_t n_docs = state.range(0);
    std::size_t dim = 128;
    
    // Setup indices
    auto dense_index = std::make_shared<BenchIndexManager>(dim);
    auto sparse_index = std::make_shared<BM25Index>();
    
    // Add documents
    auto text_docs = generate_documents(n_docs, 100, 10000);
    auto embeddings = generate_embeddings(n_docs, dim);
    
    for (std::size_t i = 0; i < n_docs; ++i) {
        sparse_index->add_document(i, text_docs[i]);
        dense_index->add(embeddings[i].data(), i);
    }
    
    HybridSearcher searcher(dense_index, sparse_index);
    
    // Create query
    HybridQuery query;
    query.text = "word100 word200";
    query.dense_embedding = embeddings[0];
    
    HybridSearchConfig config;
    config.query_strategy = QueryStrategy::PARALLEL;
    config.k = 10;
    
    for (auto _ : state) {
        auto results = searcher.search(query, config);
        benchmark::DoNotOptimize(results);
    }
    
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_HybridSearch_Parallel)->Range(1000, 100000);

static void BM_HybridSearch_BatchQueries(benchmark::State& state) {
    std::size_t n_docs = 10000;
    std::size_t batch_size = state.range(0);
    std::size_t dim = 128;
    
    // Setup indices
    auto dense_index = std::make_shared<BenchIndexManager>(dim);
    auto sparse_index = std::make_shared<BM25Index>();
    
    // Add documents
    auto text_docs = generate_documents(n_docs, 100, 10000);
    auto embeddings = generate_embeddings(n_docs, dim);
    
    for (std::size_t i = 0; i < n_docs; ++i) {
        sparse_index->add_document(i, text_docs[i]);
        dense_index->add(embeddings[i].data(), i);
    }
    
    HybridSearcher searcher(dense_index, sparse_index);
    
    // Create batch of queries
    std::vector<HybridQuery> queries;
    for (std::size_t i = 0; i < batch_size; ++i) {
        HybridQuery query;
        query.text = "word" + std::to_string(i * 10);
        query.dense_embedding = embeddings[i % embeddings.size()];
        queries.push_back(query);
    }
    
    HybridSearchConfig config;
    config.k = 10;
    
    for (auto _ : state) {
        auto results = searcher.batch_search(queries, config);
        benchmark::DoNotOptimize(results);
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
}
BENCHMARK(BM_HybridSearch_BatchQueries)->Range(1, 100);

// Memory usage benchmark
static void BM_BM25_MemoryUsage(benchmark::State& state) {
    std::size_t n_docs = state.range(0);
    std::size_t avg_doc_len = 100;
    std::size_t vocab_size = 10000;
    
    auto docs = generate_documents(n_docs, avg_doc_len, vocab_size);
    
    for (auto _ : state) {
        state.PauseTiming();
        BM25Index index;
        state.ResumeTiming();
        
        for (std::size_t i = 0; i < n_docs; ++i) {
            index.add_document(i, docs[i]);
        }
        
        auto stats = index.get_stats();
        state.counters["num_docs"] = stats.num_documents;
        state.counters["avg_doc_len"] = stats.avg_doc_length;
    }
    
    state.SetItemsProcessed(state.iterations() * n_docs);
}
BENCHMARK(BM_BM25_MemoryUsage)->Range(100, 10000);

BENCHMARK_MAIN();