/** \file bm25_test.cpp
 *  \brief Unit tests for BM25 sparse search implementation.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "vesper/index/bm25.hpp"
#include <algorithm>
#include <random>
#include <string>
#include <vector>
#include <unordered_set>
#include <sstream>
#include <cstdio>
#include <fstream>


using namespace vesper;
using namespace vesper::index;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

namespace {

// Helper to create test documents
std::vector<std::string> create_test_documents() {
    return {
        "the quick brown fox jumps over the lazy dog",
        "a quick brown dog runs through the forest",
        "the lazy cat sleeps under the warm sun",
        "brown foxes are quick and clever animals",
        "dogs and cats are popular pets"
    };
}

// Helper to tokenize a document (simple whitespace tokenization)
std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string token;
    while (iss >> token) {
        // Simple normalization: lowercase
        std::transform(token.begin(), token.end(), token.begin(), ::tolower);
        tokens.push_back(token);
    }
    return tokens;
}

} // namespace

TEST_CASE("BM25Index construction and configuration", "[bm25]") {
    SECTION("Default construction") {
        BM25Index index;
        auto stats = index.get_stats();

        REQUIRE(stats.num_documents == 0);
        REQUIRE(index.is_initialized());
    }

    SECTION("Custom parameters") {
        BM25Params params;
        params.k1 = 1.5f;
        params.b = 0.8f;

        BM25Index index;
        auto result = index.init(params);
        REQUIRE(result.has_value());

        // Verify parameters took effect by checking behavior
        REQUIRE(index.is_initialized());
    }

    SECTION("Parameter validation") {
        BM25Params params;
        BM25Index index;

        // Negative k1 should fail
        params.k1 = -1.0f;
        auto result = index.init(params);
        REQUIRE(!result.has_value());

        // b outside [0,1] should fail
        params.k1 = 1.2f;
        params.b = 1.5f;
        result = index.init(params);
        REQUIRE(!result.has_value());

        // Valid params should succeed
        params.b = 0.75f;
        result = index.init(params);
        REQUIRE(result.has_value());
    }
}

TEST_CASE("BM25Index document indexing", "[bm25]") {
    BM25Index index;
    auto docs = create_test_documents();

    SECTION("Single document indexing") {
        auto result = index.add_document(0, docs[0]);
        REQUIRE(result.has_value());

        auto stats = index.get_stats();
        REQUIRE(stats.num_documents == 1);
    }

    SECTION("Multiple document indexing") {
        for (std::size_t i = 0; i < docs.size(); ++i) {
            auto result = index.add_document(i, docs[i]);
            REQUIRE(result.has_value());
        }

        auto stats = index.get_stats();
        REQUIRE(stats.num_documents == 5);
        REQUIRE(stats.avg_doc_length > 0);
    }

    SECTION("Duplicate document ID handling") {
        auto result1 = index.add_document(0, docs[0]);
        REQUIRE(result1.has_value());

        // Adding with same ID should update
        auto result2 = index.add_document(0, docs[1]);
        REQUIRE(result2.has_value());

        auto stats = index.get_stats();
        REQUIRE(stats.num_documents == 1);
    }

    SECTION("Empty document handling") {
        auto result = index.add_document(0, "");
        REQUIRE(result.has_value());

        auto stats = index.get_stats();
        REQUIRE(stats.num_documents == 1);
    }
}

TEST_CASE("BM25Index search operations", "[bm25]") {
    BM25Index index;
    auto docs = create_test_documents();

    // Index all documents
    for (std::size_t i = 0; i < docs.size(); ++i) {
        index.add_document(i, docs[i]);
    }

    SECTION("Single term search") {
        auto results = index.search("fox", 10);
        REQUIRE(results.has_value());
        REQUIRE(results->size() > 0);

        // Documents 0 and 3 contain "fox"
        std::unordered_set<std::uint64_t> expected_ids = {0, 3};
        for (const auto& [id, score] : *results) {
            REQUIRE(expected_ids.count(id) > 0);
            REQUIRE(score > 0.0f);
        }
    }

    SECTION("Multi-term search") {
        auto results = index.search("quick brown", 10);
        REQUIRE(results.has_value());
        REQUIRE(results->size() > 0);

        // Documents with both "quick" and "brown" should score higher
        auto top_result = (*results)[0];
        REQUIRE(top_result.second > 0.0f);
    }

    SECTION("Non-existent term search") {
        auto results = index.search("unicorn", 10);
        REQUIRE(results.has_value());
        REQUIRE(results->empty());
    }

    SECTION("Empty query handling") {
        auto results = index.search("", 10);
        REQUIRE(results.has_value());
        REQUIRE(results->empty());
    }

    SECTION("Top-k limiting") {
        auto results = index.search("the", 2);
        REQUIRE(results.has_value());
        REQUIRE(results->size() <= 2);
    }

    SECTION("Score ordering") {
        auto results = index.search("brown dog", 10);
        REQUIRE(results.has_value());

        if (results->size() > 1) {
            for (std::size_t i = 1; i < results->size(); ++i) {
                REQUIRE((*results)[i-1].second >= (*results)[i].second);
            }
        }
    }
}

TEST_CASE("BM25Index batch operations", "[bm25]") {
    BM25Index index;

    SECTION("Multiple document insertion") {
        auto docs = create_test_documents();

        // Add documents individually (no batch API)
        for (std::size_t i = 0; i < docs.size(); ++i) {
            auto result = index.add_document(i, docs[i]);
            REQUIRE(result.has_value());
        }

        auto stats = index.get_stats();
        REQUIRE(stats.num_documents == 5);
    }

    SECTION("Multiple searches") {
        // First add documents
        auto docs = create_test_documents();
        for (std::size_t i = 0; i < docs.size(); ++i) {
            index.add_document(i, docs[i]);
        }

        // Perform multiple searches (no batch API)
        std::vector<std::string> queries = {"fox", "dog", "cat"};
        for (const auto& query : queries) {
            auto results = index.search(query, 5);
            REQUIRE(results.has_value());
            REQUIRE(results->size() <= 5);
        }
    }
}

TEST_CASE("BM25Index with filters", "[bm25]") {
    BM25Index index;
    auto docs = create_test_documents();

    // Index all documents
    for (std::size_t i = 0; i < docs.size(); ++i) {
        index.add_document(i, docs[i]);
    }

    SECTION("Search with include filter") {
        roaring::Roaring filter;
        filter.add(0);
        filter.add(1);

        auto results = index.search("brown", 10, &filter);
        REQUIRE(results.has_value());

        // Only documents 0 and 1 should be in results
        for (const auto& [id, score] : *results) {
            REQUIRE((id == 0 || id == 1));
        }
    }

    SECTION("Search with exclude filter") {
        roaring::Roaring include_all;
        for (std::size_t i = 0; i < docs.size(); ++i) {
            include_all.add(i);
        }

        // Remove document 0
        include_all.remove(0);

        auto results = index.search("fox", 10, &include_all);
        REQUIRE(results.has_value());

        // Document 0 should not be in results
        for (const auto& [id, score] : *results) {
            REQUIRE(id != 0);
        }
    }
}

TEST_CASE("BM25Index serialization", "[bm25]") {
    SECTION("Save and load index") {
        BM25Index index;
        auto docs = create_test_documents();

        // Index documents
        for (std::size_t i = 0; i < docs.size(); ++i) {
            index.add_document(i, docs[i]);
        }

        // Save to file
        std::string temp_path = "test_bm25_index.bin";
        auto save_result = index.save(temp_path);
        REQUIRE(save_result.has_value());

        // Load into new index
        auto load_result = BM25Index::load(temp_path);
        REQUIRE(load_result.has_value());
        auto& loaded_index = load_result.value();

        // Verify loaded index has same stats
        auto original_stats = index.get_stats();
        auto loaded_stats = loaded_index.get_stats();

        REQUIRE(loaded_stats.num_documents == original_stats.num_documents);
        REQUIRE(loaded_stats.avg_doc_length == Catch::Approx(original_stats.avg_doc_length));

        // Verify search equivalence before and after roundtrip
        auto original_results = index.search("fox", 10);
        REQUIRE(original_results.has_value());
        auto loaded_results = loaded_index.search("fox", 10);
        REQUIRE(loaded_results.has_value());
        REQUIRE(loaded_results->size() == original_results->size());
        for (std::size_t i = 0; i < loaded_results->size(); ++i) {
            REQUIRE((*loaded_results)[i].first == (*original_results)[i].first);
            REQUIRE((*loaded_results)[i].second == Catch::Approx((*original_results)[i].second));
        }

        // Clean up
        std::remove(temp_path.c_str());
    }
}

TEST_CASE("BM25Index serialization edge cases", "[bm25]") {
    SECTION("Empty index roundtrip") {
        BM25Index index;
        std::string path = "bm25_empty.bin";
        REQUIRE(index.save(path).has_value());
        auto loaded = BM25Index::load(path);
        REQUIRE(loaded.has_value());
        REQUIRE(loaded->get_stats().num_documents == 0);
        std::remove(path.c_str());
    }

    SECTION("Corruption detected by checksum") {
        BM25Index index;
        auto docs = create_test_documents();
        for (size_t i=0;i<docs.size();++i) index.add_document(i, docs[i]);
        std::string path = "bm25_corrupt.bin";
        REQUIRE(index.save(path).has_value());
        // Flip a byte in the file body
        {
            std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
            REQUIRE(static_cast<bool>(f));
            f.seekp(12, std::ios::beg); // inside header
            char b = 0;
            f.write(&b, 1);
        }
        auto loaded = BM25Index::load(path);
        REQUIRE_FALSE(loaded.has_value());
        std::remove(path.c_str());
    }
}


TEST_CASE("BM25 scoring edge cases", "[bm25]") {
    BM25Index index;

    SECTION("Single document corpus") {
        index.add_document(0, "the quick brown fox");

        auto results = index.search("fox", 10);
        REQUIRE(results.has_value());
        REQUIRE(results->size() == 1);
        REQUIRE((*results)[0].first == 0);
        REQUIRE((*results)[0].second > 0.0f);
    }

    SECTION("Document with repeated terms") {
        index.add_document(0, "cat cat cat dog");

        auto results = index.search("cat", 10);
        REQUIRE(results.has_value());
        REQUIRE(results->size() == 1);

        // Term frequency should be properly counted
        auto score_cat = (*results)[0].second;

        auto results_dog = index.search("dog", 10);
        REQUIRE(results_dog.has_value());
        auto score_dog = (*results_dog)[0].second;

        // Cat appears 3 times, dog once - cat should score higher
        REQUIRE(score_cat > score_dog);
    }

    SECTION("Very long document normalization") {
        // Create a very long document
        std::string long_doc;
        for (int i = 0; i < 1000; ++i) {
            if (i > 0) long_doc += " ";
            long_doc += "word" + std::to_string(i % 100);
        }
        long_doc += " unique";
        index.add_document(0, long_doc);

        // Create a short document with same term
        index.add_document(1, "unique other");

        auto results = index.search("unique", 10);
        REQUIRE(results.has_value());
        REQUIRE(results->size() == 2);

        // Due to length normalization, scores should be relatively close
        auto long_score = (*results)[0].second;
        auto short_score = (*results)[1].second;

        // The absolute difference shouldn't be too extreme
        REQUIRE(std::abs(long_score - short_score) < 10.0f);
    }
}
