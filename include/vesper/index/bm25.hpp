#pragma once

/** \file bm25.hpp
 *  \brief BM25 inverted index for keyword search.
 *
 * BM25 (Best Matching 25) is a probabilistic relevance ranking function
 * used for information retrieval. This implementation provides:
 * - Inverted index with Roaring bitmap posting lists
 * - Configurable BM25 parameters (k1, b)
 * - Fast tokenization and term extraction
 * - Thread-safe search operations
 *
 * Thread-safety: Index building is single-threaded; search operations are thread-safe.
 * Memory: O(V + D*L) where V is vocabulary size, D is documents, L is avg doc length.
 */

#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <shared_mutex>
#include <optional>

#include "vesper/error.hpp"
#include "roaring.hh"

namespace vesper::index {

/** \brief BM25 scoring parameters. */
struct BM25Params {
    float k1{1.2f};              /**< Term frequency saturation parameter (typically 1.2-2.0) */
    float b{0.75f};              /**< Length normalization parameter (0.0-1.0) */
    bool lowercase{true};        /**< Convert terms to lowercase */
    bool remove_stopwords{true}; /**< Remove common stopwords */
    std::uint32_t min_term_length{2}; /**< Minimum term length to index */
    std::uint32_t max_term_length{50}; /**< Maximum term length to index */
};

/** \brief Sparse vector representation for BM25 scores. */
struct SparseVector {
    std::vector<std::uint32_t> indices;  /**< Term IDs (sorted) */
    std::vector<float> values;           /**< BM25 scores or TF-IDF weights */
    
    /** \brief Get number of non-zero elements. */
    auto nnz() const noexcept -> std::size_t { return indices.size(); }
    
    /** \brief Check if vector is empty. */
    auto empty() const noexcept -> bool { return indices.empty(); }
    
    /** \brief Compute dot product with another sparse vector. */
    auto dot(const SparseVector& other) const noexcept -> float;
    
    /** \brief L2 normalize the vector in-place. */
    auto normalize() -> void;
};

/** \brief Document statistics for BM25 scoring. */
struct DocumentStats {
    std::uint64_t doc_id{0};           /**< Document identifier */
    std::uint32_t length{0};           /**< Document length in tokens */
    SparseVector term_freqs;           /**< Term frequencies as sparse vector */
};

/** \brief BM25 index statistics. */
struct BM25Stats {
    std::size_t num_documents{0};      /**< Total documents indexed */
    std::size_t vocabulary_size{0};    /**< Unique terms in vocabulary */
    std::size_t total_tokens{0};       /**< Total tokens processed */
    float avg_doc_length{0.0f};        /**< Average document length */
    std::size_t memory_bytes{0};       /**< Memory usage in bytes */
};

/** \brief BM25 inverted index for keyword search.
 *
 * Example usage:
 * ```cpp
 * BM25Index index;
 * BM25Params params;
 * params.k1 = 1.5f;
 * params.b = 0.75f;
 * index.init(params);
 * 
 * // Index documents
 * index.add_document(1, "The quick brown fox jumps over the lazy dog");
 * index.add_document(2, "A fast red fox leaps over a sleepy cat");
 * 
 * // Search
 * auto results = index.search("quick fox", 10);
 * for (const auto& [doc_id, score] : results.value()) {
 *     std::cout << "Doc " << doc_id << ": " << score << "\n";
 * }
 * ```
 */
class BM25Index {
public:
    BM25Index();
    ~BM25Index();
    BM25Index(BM25Index&&) noexcept;
    BM25Index& operator=(BM25Index&&) noexcept;
    BM25Index(const BM25Index&) = delete;
    BM25Index& operator=(const BM25Index&) = delete;
    
    /** \brief Initialize index with parameters.
     *
     * \param params BM25 parameters
     * \param expected_docs Expected number of documents (for pre-allocation)
     * \return Success or error
     *
     * Preconditions: k1 > 0; 0 <= b <= 1
     * Complexity: O(1)
     */
    auto init(const BM25Params& params, std::size_t expected_docs = 0)
        -> std::expected<void, core::error>;
    
    /** \brief Add a document to the index.
     *
     * \param doc_id Document identifier
     * \param text Document text content
     * \param metadata Optional metadata for filtering
     * \return Success or error
     *
     * Preconditions: Index is initialized; doc_id is unique
     * Complexity: O(L) where L is document length
     * Thread-safety: NOT thread-safe; use external synchronization
     */
    auto add_document(std::uint64_t doc_id, std::string_view text,
                      const std::unordered_map<std::string, std::string>& metadata = {})
        -> std::expected<void, core::error>;
    
    /** \brief Batch add documents.
     *
     * \param doc_ids Document identifiers [n]
     * \param texts Document texts [n]
     * \param n Number of documents
     * \return Success or error
     *
     * Thread-safety: Internally parallelized where safe
     */
    auto add_batch(const std::uint64_t* doc_ids, 
                   const std::string* texts,
                   std::size_t n)
        -> std::expected<void, core::error>;
    
    /** \brief Search for documents matching query.
     *
     * \param query Query text
     * \param k Number of results to return
     * \param filter Optional document filter
     * \return Ranked list of (doc_id, score) pairs
     *
     * Preconditions: Index is non-empty
     * Complexity: O(k * log(N)) where N is matching documents
     * Thread-safety: Safe for concurrent calls
     */
    auto search(std::string_view query, std::uint32_t k,
                const roaring::Roaring* filter = nullptr) const
        -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error>;
    
    /** \brief Get sparse vector representation of text.
     *
     * \param text Input text
     * \return Sparse vector with BM25 weights
     *
     * This is useful for hybrid search where we need to combine
     * sparse and dense vectors.
     */
    auto encode_text(std::string_view text) const
        -> std::expected<SparseVector, core::error>;
    
    /** \brief Get sparse vector for existing document.
     *
     * \param doc_id Document identifier
     * \return Sparse vector representation
     */
    auto get_document_vector(std::uint64_t doc_id) const
        -> std::expected<SparseVector, core::error>;
    
    /** \brief Compute BM25 score between query and document.
     *
     * \param query_terms Query term frequencies
     * \param doc_id Document to score
     * \return BM25 score
     */
    auto score_document(const SparseVector& query_terms, std::uint64_t doc_id) const
        -> float;
    
    /** \brief Get index statistics. */
    auto get_stats() const noexcept -> BM25Stats;
    
    /** \brief Check if index is initialized. */
    auto is_initialized() const noexcept -> bool;
    
    /** \brief Get total number of indexed documents. */
    auto size() const noexcept -> std::size_t;
    
    /** \brief Clear all indexed data. */
    auto clear() -> void;
    
    /** \brief Save index to file.
     *
     * \param path Output file path
     * \return Success or error
     */
    auto save(const std::string& path) const
        -> std::expected<void, core::error>;
    
    /** \brief Load index from file.
     *
     * \param path Input file path
     * \return Loaded index or error
     */
    static auto load(const std::string& path)
        -> std::expected<BM25Index, core::error>;
    
    /** \brief Get vocabulary size. */
    auto vocabulary_size() const noexcept -> std::size_t;
    
    /** \brief Get average document length. */
    auto avg_doc_length() const noexcept -> float;
    
    /** \brief Update BM25 parameters.
     *
     * This allows tuning parameters without rebuilding the index.
     */
    auto update_params(const BM25Params& params) -> void;
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/** \brief Tokenizer for text processing.
 *
 * Simple whitespace tokenizer with optional preprocessing.
 */
class Tokenizer {
public:
    /** \brief Tokenization options. */
    struct Options {
        bool lowercase{true};
        bool remove_stopwords{true};
        bool remove_punctuation{true};
        std::uint32_t min_length{2};
        std::uint32_t max_length{50};
    };
    
    /** \brief Tokenize text into terms.
     *
     * \param text Input text
     * \param options Tokenization options
     * \return Vector of tokens
     */
    static auto tokenize(std::string_view text, const Options& options = {})
        -> std::vector<std::string>;
    
    /** \brief Check if word is a stopword.
     *
     * \param word Word to check
     * \return True if stopword
     */
    static auto is_stopword(std::string_view word) -> bool;
    
    /** \brief Normalize a term (lowercase, stemming, etc).
     *
     * \param term Input term
     * \return Normalized term
     */
    static auto normalize(std::string_view term) -> std::string;
};

} // namespace vesper::index