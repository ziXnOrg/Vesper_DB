#include "vesper/index/bm25.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <sstream>
#include <cctype>
#include <fstream>
#include <unordered_set>
#include <expected>
#include <filesystem>
#include <cstdint>
#include <cstring>



namespace vesper::index {

namespace {

// Common English stopwords
const std::unordered_set<std::string> STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for",
    "from", "has", "he", "in", "is", "it", "its", "of", "on", "that",
    "the", "to", "was", "will", "with", "the", "this", "these", "those",
    "i", "you", "we", "they", "them", "their", "what", "which", "who",
    "when", "where", "why", "how", "all", "would", "there", "could"
};

} // anonymous namespace

// SparseVector implementation

auto SparseVector::dot(const SparseVector& other) const noexcept -> float {
    float result = 0.0f;
    std::size_t i = 0, j = 0;

    // Merge-based dot product for sorted indices
    while (i < indices.size() && j < other.indices.size()) {
        if (indices[i] < other.indices[j]) {
            ++i;
        } else if (indices[i] > other.indices[j]) {
            ++j;
        } else {
            result += values[i] * other.values[j];
            ++i;
            ++j;
        }
    }

    return result;
}

auto SparseVector::normalize() -> void {
    float norm = 0.0f;
    for (float val : values) {
        norm += val * val;
    }

    if (norm > 0.0f) {
        norm = std::sqrt(norm);
        for (float& val : values) {
            val /= norm;
        }
    }
}

// Tokenizer implementation

auto Tokenizer::tokenize(std::string_view text, const Options& options)
    -> std::vector<std::string> {
    std::vector<std::string> tokens;
    std::string current_token;

    for (char c : text) {
        if (std::isspace(c) || (options.remove_punctuation && std::ispunct(c))) {
            if (!current_token.empty()) {
                if (current_token.length() >= options.min_length &&
                    current_token.length() <= options.max_length) {

                    if (options.lowercase) {
                        std::transform(current_token.begin(), current_token.end(),
                                     current_token.begin(), ::tolower);
                    }

                    if (!options.remove_stopwords || !is_stopword(current_token)) {
                        tokens.push_back(std::move(current_token));
                    }
                }
                current_token.clear();
            }
        } else {
            current_token.push_back(c);
        }
    }

    // Handle last token
    if (!current_token.empty()) {
        if (current_token.length() >= options.min_length &&
            current_token.length() <= options.max_length) {

            if (options.lowercase) {
                std::transform(current_token.begin(), current_token.end(),
                             current_token.begin(), ::tolower);
            }

            if (!options.remove_stopwords || !is_stopword(current_token)) {
                tokens.push_back(std::move(current_token));
            }
        }
    }

    return tokens;
}

auto Tokenizer::is_stopword(std::string_view word) -> bool {
    std::string lower_word(word);
    std::transform(lower_word.begin(), lower_word.end(),
                  lower_word.begin(), ::tolower);
    return STOPWORDS.count(lower_word) > 0;
}

auto Tokenizer::normalize(std::string_view term) -> std::string {
    std::string normalized(term);
    std::transform(normalized.begin(), normalized.end(),
                  normalized.begin(), ::tolower);
    // TODO: Add stemming support (Porter stemmer, etc.)
    return normalized;
}

// BM25Index::Impl class definition

class BM25Index::Impl {
public:
    Impl() = default;

    auto init(const BM25Params& params, std::size_t expected_docs)
        -> std::expected<void, core::error>;

    auto add_document(std::uint64_t doc_id, std::string_view text,
                      const std::unordered_map<std::string, std::string>& metadata)
        -> std::expected<void, core::error>;

    auto search(std::string_view query, std::uint32_t k,
                const roaring::Roaring* filter) const
        -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error>;

    auto encode_text(std::string_view text) const
        -> std::expected<SparseVector, core::error>;

    auto get_document_vector(std::uint64_t doc_id) const
        -> std::expected<SparseVector, core::error>;

    auto score_document(const SparseVector& query_terms, std::uint64_t doc_id) const
        -> float;

    auto get_stats() const noexcept -> BM25Stats;
    auto is_initialized() const noexcept -> bool { return initialized_; }
    auto size() const noexcept -> std::size_t { return doc_stats_.size(); }
    // Serialization helpers (defined later in this file)
    auto save_to_file(const std::string& path) const -> std::expected<void, core::error>;
    static auto load_from_file(const std::string& path) -> std::expected<std::unique_ptr<Impl>, core::error>;

    auto clear() -> void;
    auto vocabulary_size() const noexcept -> std::size_t { return term_to_id_.size(); }
    auto avg_doc_length() const noexcept -> float { return avg_doc_length_; }
    auto update_params(const BM25Params& params) -> void { params_ = params; }

private:
    // Term to term ID mapping
    std::unordered_map<std::string, std::uint32_t> term_to_id_;
    std::vector<std::string> id_to_term_;

    // Inverted index: term_id -> document IDs
    std::vector<roaring::Roaring> inverted_index_;

    // Document statistics
    std::unordered_map<std::uint64_t, DocumentStats> doc_stats_;

    // Document frequency for each term
    std::vector<std::uint32_t> doc_freqs_;

    // Global statistics
    float avg_doc_length_{0.0f};
    std::size_t total_tokens_{0};

    // Parameters
    BM25Params params_;
    bool initialized_{false};

    // Thread safety
    mutable std::shared_mutex mutex_;

    // Helper functions
    auto get_or_create_term_id(const std::string& term) -> std::uint32_t;
    auto compute_idf(std::uint32_t term_id) const -> float;
    auto tokenize_and_count(std::string_view text) const
        -> std::unordered_map<std::uint32_t, std::uint32_t>;
};

// BM25Index::Impl implementation

auto BM25Index::Impl::init(const BM25Params& params, std::size_t expected_docs)
    -> std::expected<void, core::error> {

    if (params.k1 <= 0.0f) {
        return std::vesper_unexpected(core::error{core::error_code::invalid_argument, "k1 must be positive", "bm25"});
    }
    if (params.b < 0.0f || params.b > 1.0f) {
        return std::vesper_unexpected(core::error{core::error_code::invalid_argument, "b must be between 0 and 1", "bm25"});
    }

    params_ = params;
    initialized_ = true;

    if (expected_docs > 0) {
        doc_stats_.reserve(expected_docs);
    }

    return {};
}

auto BM25Index::Impl::get_or_create_term_id(const std::string& term) -> std::uint32_t {
    auto it = term_to_id_.find(term);
    if (it != term_to_id_.end()) {
        return it->second;
    }

    std::uint32_t id = static_cast<std::uint32_t>(term_to_id_.size());
    term_to_id_[term] = id;
    id_to_term_.push_back(term);
    inverted_index_.emplace_back();
    doc_freqs_.push_back(0);

    return id;
}

auto BM25Index::Impl::tokenize_and_count(std::string_view text) const
    -> std::unordered_map<std::uint32_t, std::uint32_t> {

    Tokenizer::Options opts;
    opts.lowercase = params_.lowercase;
    opts.remove_stopwords = params_.remove_stopwords;
    opts.min_length = params_.min_term_length;
    opts.max_length = params_.max_term_length;

    auto tokens = Tokenizer::tokenize(text, opts);

    std::unordered_map<std::uint32_t, std::uint32_t> term_counts;
    for (const auto& token : tokens) {
        auto it = term_to_id_.find(token);
        if (it != term_to_id_.end()) {
            term_counts[it->second]++;
        }
    }

    return term_counts;
}

auto BM25Index::Impl::add_document(std::uint64_t doc_id, std::string_view text,
                                   const std::unordered_map<std::string, std::string>& metadata)
    -> std::expected<void, core::error> {

    if (!initialized_) {
        return std::vesper_unexpected(core::error{core::error_code::not_initialized, "Index not initialized", "bm25"});
    }

    std::unique_lock lock(mutex_);

    // If duplicate doc_id, update: remove prior contributions then re-index
    if (auto it_old = doc_stats_.find(doc_id); it_old != doc_stats_.end()) {
        const auto& old = it_old->second;
        for (std::uint32_t tid : old.term_freqs.indices) {
            if (tid < inverted_index_.size()) {
                inverted_index_[tid].remove(static_cast<uint32_t>(doc_id));
                if (doc_freqs_[tid] > 0) doc_freqs_[tid]--;
            }
        }
        if (total_tokens_ >= old.length) total_tokens_ -= old.length;
        doc_stats_.erase(it_old);
    }

    // Tokenize and count terms
    Tokenizer::Options opts;
    opts.lowercase = params_.lowercase;
    opts.remove_stopwords = params_.remove_stopwords;
    opts.min_length = params_.min_term_length;
    opts.max_length = params_.max_term_length;

    auto tokens = Tokenizer::tokenize(text, opts);

    // Count term frequencies
    std::unordered_map<std::string, std::uint32_t> term_counts;
    for (const auto& token : tokens) {
        term_counts[token]++;
    }

    // Build sparse vector and update inverted index
    DocumentStats stats;
    stats.doc_id = doc_id;
    stats.length = static_cast<std::uint32_t>(tokens.size());

    for (const auto& [term, count] : term_counts) {
        std::uint32_t term_id = get_or_create_term_id(term);

        // Update inverted index
        inverted_index_[term_id].add(doc_id);
        doc_freqs_[term_id]++;

        // Add to sparse vector
        stats.term_freqs.indices.push_back(term_id);
        stats.term_freqs.values.push_back(static_cast<float>(count));
    }

    // Sort indices for efficient dot product
    std::vector<std::size_t> perm(stats.term_freqs.indices.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](std::size_t i, std::size_t j) {
        return stats.term_freqs.indices[i] < stats.term_freqs.indices[j];
    });

    SparseVector sorted;
    sorted.indices.reserve(stats.term_freqs.indices.size());
    sorted.values.reserve(stats.term_freqs.values.size());
    for (std::size_t i : perm) {
        sorted.indices.push_back(stats.term_freqs.indices[i]);
        sorted.values.push_back(stats.term_freqs.values[i]);
    }
    stats.term_freqs = std::move(sorted);

    // Update global statistics
    total_tokens_ += stats.length;
    doc_stats_[doc_id] = std::move(stats);

    // Update average document length
    avg_doc_length_ = static_cast<float>(total_tokens_) / doc_stats_.size();

    return {};
}

auto BM25Index::Impl::compute_idf(std::uint32_t term_id) const -> float {
    if (term_id >= doc_freqs_.size()) {
        return 0.0f;
    }

    std::size_t N = doc_stats_.size();
    std::size_t df = doc_freqs_[term_id];

    if (df == 0) {
        return 0.0f;
    }

    // IDF = log(1 + (N - df + 0.5) / (df + 0.5))
    // Adding 1 ensures non-negative IDF and aligns with common BM25 variants used in practice.
    return std::log(1.0f + (N - df + 0.5f) / (df + 0.5f));
}

auto BM25Index::Impl::score_document(const SparseVector& query_terms, std::uint64_t doc_id) const
    -> float {

    auto it = doc_stats_.find(doc_id);
    if (it == doc_stats_.end()) {
        return 0.0f;
    }

    const auto& doc_stats = it->second;
    float score = 0.0f;

    // BM25 scoring
    float doc_len_norm = 1.0f - params_.b + params_.b * (doc_stats.length / avg_doc_length_);

    std::size_t qi = 0, di = 0;
    while (qi < query_terms.indices.size() && di < doc_stats.term_freqs.indices.size()) {
        if (query_terms.indices[qi] < doc_stats.term_freqs.indices[di]) {
            ++qi;
        } else if (query_terms.indices[qi] > doc_stats.term_freqs.indices[di]) {
            ++di;
        } else {
            // Matching term
            std::uint32_t term_id = query_terms.indices[qi];
            float query_tf = query_terms.values[qi];
            float doc_tf = doc_stats.term_freqs.values[di];
            float idf = compute_idf(term_id);

            // BM25 formula
            float numerator = doc_tf * (params_.k1 + 1.0f);
            float denominator = doc_tf + params_.k1 * doc_len_norm;
            score += idf * query_tf * (numerator / denominator);

            ++qi;
            ++di;
        }
    }

    return score;
}

auto BM25Index::Impl::search(std::string_view query, std::uint32_t k,
                             const roaring::Roaring* filter) const
    -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error> {

    if (!initialized_) {
        return std::vesper_unexpected(core::error{core::error_code::not_initialized, "Index not initialized", "bm25"});
    }

    std::shared_lock lock(mutex_);

    // Tokenize query
    auto query_terms = tokenize_and_count(query);
    if (query_terms.empty()) {
        return std::vector<std::pair<std::uint64_t, float>>{};
    }

    // Build query sparse vector
    SparseVector query_vec;
    for (const auto& [term_id, count] : query_terms) {
        query_vec.indices.push_back(term_id);
        query_vec.values.push_back(static_cast<float>(count));
    }

    // Sort indices
    std::vector<std::size_t> perm(query_vec.indices.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](std::size_t i, std::size_t j) {
        return query_vec.indices[i] < query_vec.indices[j];
    });

    SparseVector sorted_query;
    for (std::size_t i : perm) {
        sorted_query.indices.push_back(query_vec.indices[i]);
        sorted_query.values.push_back(query_vec.values[i]);
    }

    // Find candidate documents (union of posting lists)
    roaring::Roaring candidates;
    for (std::uint32_t term_id : sorted_query.indices) {
        if (term_id < inverted_index_.size()) {
            candidates |= inverted_index_[term_id];
        }
    }

    // Apply filter if provided
    if (filter) {
        candidates &= *filter;
    }

    // Score all candidates
    using ScoredDoc = std::pair<float, std::uint64_t>;
    std::priority_queue<ScoredDoc, std::vector<ScoredDoc>, std::greater<>> top_k;

    for (std::uint64_t doc_id : candidates) {
        float score = score_document(sorted_query, doc_id);

        if (top_k.size() < k) {
            top_k.emplace(score, doc_id);
        } else if (score > top_k.top().first) {
            top_k.pop();
            top_k.emplace(score, doc_id);
        }
    }

    // Extract results in descending score order
    std::vector<std::pair<std::uint64_t, float>> results;
    results.reserve(top_k.size());

    while (!top_k.empty()) {
        auto [score, doc_id] = top_k.top();
        top_k.pop();
        results.emplace_back(doc_id, score);
    }

    std::reverse(results.begin(), results.end());

    return results;
}

auto BM25Index::Impl::encode_text(std::string_view text) const
    -> std::expected<SparseVector, core::error> {

    if (!initialized_) {
        return std::vesper_unexpected(core::error{core::error_code::not_initialized, "Index not initialized", "bm25"});
    }

    std::shared_lock lock(mutex_);

    auto term_counts = tokenize_and_count(text);

    SparseVector vec;
    for (const auto& [term_id, count] : term_counts) {
        float idf = compute_idf(term_id);
        vec.indices.push_back(term_id);
        vec.values.push_back(count * idf);  // TF-IDF weight
    }

    // Sort indices
    std::vector<std::size_t> perm(vec.indices.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](std::size_t i, std::size_t j) {
        return vec.indices[i] < vec.indices[j];
    });

    SparseVector sorted;
    for (std::size_t i : perm) {
        sorted.indices.push_back(vec.indices[i]);
        sorted.values.push_back(vec.values[i]);
    }

    return sorted;
}

auto BM25Index::Impl::get_document_vector(std::uint64_t doc_id) const
    -> std::expected<SparseVector, core::error> {

    std::shared_lock lock(mutex_);

    auto it = doc_stats_.find(doc_id);
    if (it == doc_stats_.end()) {
        return std::vesper_unexpected(core::error{core::error_code::not_found, "Document not found", "bm25"});
    }

    // Convert term frequencies to TF-IDF weights
    SparseVector vec = it->second.term_freqs;
    for (std::size_t i = 0; i < vec.indices.size(); ++i) {
        float idf = compute_idf(vec.indices[i]);
        vec.values[i] *= idf;
    }

    return vec;
}

auto BM25Index::Impl::get_stats() const noexcept -> BM25Stats {
    std::shared_lock lock(mutex_);

    BM25Stats stats;
    stats.num_documents = doc_stats_.size();
    stats.vocabulary_size = term_to_id_.size();
    stats.total_tokens = total_tokens_;
    stats.avg_doc_length = avg_doc_length_;

    // Estimate memory usage
    stats.memory_bytes = sizeof(*this);
    stats.memory_bytes += term_to_id_.size() * (32 + sizeof(std::uint32_t));  // Approx
    stats.memory_bytes += id_to_term_.size() * 32;  // Approx string size

    for (const auto& bitmap : inverted_index_) {
        stats.memory_bytes += bitmap.getSizeInBytes();
    }

    stats.memory_bytes += doc_stats_.size() * sizeof(DocumentStats);

    return stats;
}

auto BM25Index::Impl::clear() -> void {
    std::unique_lock lock(mutex_);

    term_to_id_.clear();
    id_to_term_.clear();
    inverted_index_.clear();
    doc_stats_.clear();
    doc_freqs_.clear();
    avg_doc_length_ = 0.0f;
    total_tokens_ = 0;
}


// --- Serialization (v1.0) ---

namespace {
    constexpr char BM25_MAGIC[8] = {'B','M','2','5','v','1','0','0'}; // 8 bytes
    constexpr char TRAILER_MARK[4] = {'C','H','K','S'};
    constexpr std::uint64_t FNV_OFFSET = 1469598103934665603ull;
    constexpr std::uint64_t FNV_PRIME  = 1099511628211ull;
}

auto BM25Index::Impl::save_to_file(const std::string& path) const -> std::expected<void, core::error> {
    using core::error; using core::error_code;

    std::vector<std::string> vocab_snapshot;
    std::vector<std::pair<std::uint64_t, DocumentStats>> docs_snapshot;
    BM25Params params_snapshot;
    float avg_len_snapshot = 0.0f;
    std::size_t total_tokens_snapshot = 0;

    // Take a consistent snapshot under shared lock
    {
        std::shared_lock lk(mutex_);
        vocab_snapshot = id_to_term_;
        params_snapshot = params_;
        avg_len_snapshot = avg_doc_length_;
        total_tokens_snapshot = total_tokens_;
        docs_snapshot.reserve(doc_stats_.size());
        for (const auto& kv : doc_stats_) {
            docs_snapshot.emplace_back(kv.first, kv.second);
        }
    }
    // Deterministic order for docs
    std::sort(docs_snapshot.begin(), docs_snapshot.end(), [](const auto& a, const auto& b){ return a.first < b.first; });

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return std::vesper_unexpected(error{error_code::io_error, "Failed to open file for writing", "bm25"});
    }

    auto write_raw = [&](const void* p, std::size_t n){ out.write(reinterpret_cast<const char*>(p), static_cast<std::streamsize>(n)); return static_cast<bool>(out); };
    auto fnv_update = [&](std::uint64_t& h, const void* p, std::size_t n){ const auto* b = static_cast<const std::uint8_t*>(p); for (std::size_t i=0;i<n;++i){ h ^= b[i]; h *= FNV_PRIME; } };
    auto write_hash = [&](std::uint64_t& h, const void* p, std::size_t n){ if (!write_raw(p,n)) return false; fnv_update(h,p,n); return true; };

    std::uint64_t checksum = FNV_OFFSET;

    // Header: magic + version(major,minor)
    if (!write_hash(checksum, BM25_MAGIC, sizeof(BM25_MAGIC))) {
        return std::vesper_unexpected(error{error_code::io_error, "Write failed (magic)", "bm25"});
    }
    std::uint16_t ver_major = 1, ver_minor = 0;
    if (!write_hash(checksum, &ver_major, sizeof(ver_major))) return std::vesper_unexpected(error{error_code::io_error, "Write failed (ver_major)", "bm25"});
    if (!write_hash(checksum, &ver_minor, sizeof(ver_minor))) return std::vesper_unexpected(error{error_code::io_error, "Write failed (ver_minor)", "bm25"});

    // Params
    if (!write_hash(checksum, &params_snapshot.k1, sizeof(params_snapshot.k1))) return std::vesper_unexpected(error{error_code::io_error, "Write failed (k1)", "bm25"});
    if (!write_hash(checksum, &params_snapshot.b, sizeof(params_snapshot.b))) return std::vesper_unexpected(error{error_code::io_error, "Write failed (b)", "bm25"});
    std::uint8_t lowercase = params_snapshot.lowercase ? 1u : 0u;
    std::uint8_t remove_sw = params_snapshot.remove_stopwords ? 1u : 0u;
    if (!write_hash(checksum, &lowercase, sizeof(lowercase))) return std::vesper_unexpected(error{error_code::io_error, "Write failed (lowercase)", "bm25"});
    if (!write_hash(checksum, &remove_sw, sizeof(remove_sw))) return std::vesper_unexpected(error{error_code::io_error, "Write failed (remove_stopwords)", "bm25"});
    if (!write_hash(checksum, &params_snapshot.min_term_length, sizeof(params_snapshot.min_term_length))) return std::vesper_unexpected(error{error_code::io_error, "Write failed (min_term_length)", "bm25"});
    if (!write_hash(checksum, &params_snapshot.max_term_length, sizeof(params_snapshot.max_term_length))) return std::vesper_unexpected(error{error_code::io_error, "Write failed (max_term_length)", "bm25"});

    // Global stats and counts
    std::uint32_t vocab_sz = static_cast<std::uint32_t>(vocab_snapshot.size());
    std::uint64_t num_docs = static_cast<std::uint64_t>(docs_snapshot.size());
    if (!write_hash(checksum, &vocab_sz, sizeof(vocab_sz))) return std::vesper_unexpected(error{error_code::io_error, "Write failed (vocab_sz)", "bm25"});
    if (!write_hash(checksum, &num_docs, sizeof(num_docs))) return std::vesper_unexpected(error{error_code::io_error, "Write failed (num_docs)", "bm25"});
    if (!write_hash(checksum, &total_tokens_snapshot, sizeof(total_tokens_snapshot))) return std::vesper_unexpected(error{error_code::io_error, "Write failed (total_tokens)", "bm25"});
    if (!write_hash(checksum, &avg_len_snapshot, sizeof(avg_len_snapshot))) return std::vesper_unexpected(error{error_code::io_error, "Write failed (avg_doc_length)", "bm25"});

    // Vocabulary
    for (const auto& term : vocab_snapshot) {
        std::uint32_t len = static_cast<std::uint32_t>(term.size());
        if (!write_hash(checksum, &len, sizeof(len))) return std::vesper_unexpected(error{error_code::io_error, "Write failed (term_len)", "bm25"});
        if (len) { if (!write_hash(checksum, term.data(), len)) return std::vesper_unexpected(error{error_code::io_error, "Write failed (term_data)", "bm25"}); }
    }

    // Documents: [doc_id][length][nnz][ nnz * (term_id, value) ]
    for (const auto& [doc_id, st] : docs_snapshot) {
        std::uint64_t did = doc_id;
        std::uint32_t dlen = st.length;
        std::uint32_t nnz = static_cast<std::uint32_t>(st.term_freqs.indices.size());
        if (!write_hash(checksum, &did, sizeof(did))) return std::vesper_unexpected(error{error_code::io_error, "Write failed (doc_id)", "bm25"});
        if (!write_hash(checksum, &dlen, sizeof(dlen))) return std::vesper_unexpected(error{error_code::io_error, "Write failed (doc_length)", "bm25"});
        if (!write_hash(checksum, &nnz, sizeof(nnz))) return std::vesper_unexpected(error{error_code::io_error, "Write failed (nnz)", "bm25"});
        for (std::size_t i=0;i<nnz;++i) {
            std::uint32_t tid = st.term_freqs.indices[i];
            float val = st.term_freqs.values[i];
            if (!write_hash(checksum, &tid, sizeof(tid))) return std::vesper_unexpected(error{error_code::io_error, "Write failed (term_id)", "bm25"});
            if (!write_hash(checksum, &val, sizeof(val))) return std::vesper_unexpected(error{error_code::io_error, "Write failed (term_val)", "bm25"});
        }
    }

    // Trailer: CHKS + checksum
    if (!write_raw(TRAILER_MARK, sizeof(TRAILER_MARK))) return std::vesper_unexpected(error{error_code::io_error, "Write failed (trailer mark)", "bm25"});
    if (!write_raw(&checksum, sizeof(checksum))) return std::vesper_unexpected(error{error_code::io_error, "Write failed (checksum)", "bm25"});

    return {};
}

auto BM25Index::Impl::load_from_file(const std::string& path) -> std::expected<std::unique_ptr<Impl>, core::error> {
    using core::error; using core::error_code;

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::vesper_unexpected(error{error_code::io_error, "Failed to open file for reading", "bm25"});
    }

    auto read_raw = [&](void* p, std::size_t n){ in.read(reinterpret_cast<char*>(p), static_cast<std::streamsize>(n)); return static_cast<bool>(in); };
    auto fnv_update = [&](std::uint64_t& h, const void* p, std::size_t n){ const auto* b = static_cast<const std::uint8_t*>(p); for (std::size_t i=0;i<n;++i){ h ^= b[i]; h *= FNV_PRIME; } };
    auto read_hash = [&](std::uint64_t& h, void* p, std::size_t n){ if (!read_raw(p,n)) return false; fnv_update(h,p,n); return true; };

    std::uint64_t checksum = FNV_OFFSET;

    // Header
    char magic[8]{};
    if (!read_hash(checksum, magic, sizeof(magic))) return std::vesper_unexpected(error{error_code::io_error, "Read failed (magic)", "bm25"});
    if (std::memcmp(magic, BM25_MAGIC, sizeof(magic)) != 0) {
        return std::vesper_unexpected(error{error_code::config_invalid, "Invalid BM25 magic", "bm25"});
    }
    std::uint16_t ver_major=0, ver_minor=0;
    if (!read_hash(checksum, &ver_major, sizeof(ver_major))) return std::vesper_unexpected(error{error_code::io_error, "Read failed (ver_major)", "bm25"});
    if (!read_hash(checksum, &ver_minor, sizeof(ver_minor))) return std::vesper_unexpected(error{error_code::io_error, "Read failed (ver_minor)", "bm25"});
    if (!(ver_major == 1 && ver_minor == 0)) {
        return std::vesper_unexpected(error{error_code::config_invalid, "Unsupported BM25 version", "bm25"});
    }

    BM25Params params{};
    if (!read_hash(checksum, &params.k1, sizeof(params.k1))) return std::vesper_unexpected(error{error_code::io_error, "Read failed (k1)", "bm25"});
    if (!read_hash(checksum, &params.b, sizeof(params.b))) return std::vesper_unexpected(error{error_code::io_error, "Read failed (b)", "bm25"});
    std::uint8_t lowercase=1, remove_sw=1;
    if (!read_hash(checksum, &lowercase, sizeof(lowercase))) return std::vesper_unexpected(error{error_code::io_error, "Read failed (lowercase)", "bm25"});
    if (!read_hash(checksum, &remove_sw, sizeof(remove_sw))) return std::vesper_unexpected(error{error_code::io_error, "Read failed (remove_stopwords)", "bm25"});
    params.lowercase = (lowercase != 0);
    params.remove_stopwords = (remove_sw != 0);
    if (!read_hash(checksum, &params.min_term_length, sizeof(params.min_term_length))) return std::vesper_unexpected(error{error_code::io_error, "Read failed (min_term_length)", "bm25"});
    if (!read_hash(checksum, &params.max_term_length, sizeof(params.max_term_length))) return std::vesper_unexpected(error{error_code::io_error, "Read failed (max_term_length)", "bm25"});

    std::uint32_t vocab_sz=0; std::uint64_t num_docs=0; std::size_t total_tokens=0; float avg_len=0.0f;
    if (!read_hash(checksum, &vocab_sz, sizeof(vocab_sz))) return std::vesper_unexpected(error{error_code::io_error, "Read failed (vocab_sz)", "bm25"});
    if (!read_hash(checksum, &num_docs, sizeof(num_docs))) return std::vesper_unexpected(error{error_code::io_error, "Read failed (num_docs)", "bm25"});
    if (!read_hash(checksum, &total_tokens, sizeof(total_tokens))) return std::vesper_unexpected(error{error_code::io_error, "Read failed (total_tokens)", "bm25"});
    if (!read_hash(checksum, &avg_len, sizeof(avg_len))) return std::vesper_unexpected(error{error_code::io_error, "Read failed (avg_doc_length)", "bm25"});

    auto impl = std::make_unique<Impl>();
    impl->params_ = params;
    impl->initialized_ = true;
    impl->id_to_term_.resize(vocab_sz);
    impl->doc_freqs_.assign(vocab_sz, 0);
    impl->inverted_index_.resize(vocab_sz);

    // Vocabulary
    for (std::uint32_t i=0; i<vocab_sz; ++i) {
        std::uint32_t len=0; if (!read_hash(checksum, &len, sizeof(len))) return std::vesper_unexpected(error{error_code::io_error, "Read failed (term_len)", "bm25"});
        std::string term(len, '\0');
        if (len > 0) { if (!read_hash(checksum, term.data(), len)) return std::vesper_unexpected(error{error_code::io_error, "Read failed (term_data)", "bm25"}); }
        impl->id_to_term_[i] = term;
        impl->term_to_id_.emplace(std::move(term), i);
    }

    // Documents
    impl->total_tokens_ = 0;
    for (std::uint64_t di=0; di<num_docs; ++di) {
        std::uint64_t doc_id=0; std::uint32_t dlen=0; std::uint32_t nnz=0;
        if (!read_hash(checksum, &doc_id, sizeof(doc_id))) return std::vesper_unexpected(error{error_code::io_error, "Read failed (doc_id)", "bm25"});
        if (!read_hash(checksum, &dlen, sizeof(dlen))) return std::vesper_unexpected(error{error_code::io_error, "Read failed (doc_length)", "bm25"});
        if (!read_hash(checksum, &nnz, sizeof(nnz))) return std::vesper_unexpected(error{error_code::io_error, "Read failed (nnz)", "bm25"});
        DocumentStats st; st.doc_id = doc_id; st.length = dlen;
        st.term_freqs.indices.resize(nnz); st.term_freqs.values.resize(nnz);
        for (std::uint32_t i=0;i<nnz;++i) {
            std::uint32_t tid=0; float val=0.0f;
            if (!read_hash(checksum, &tid, sizeof(tid))) return std::vesper_unexpected(error{error_code::io_error, "Read failed (term_id)", "bm25"});
            if (!read_hash(checksum, &val, sizeof(val))) return std::vesper_unexpected(error{error_code::io_error, "Read failed (term_val)", "bm25"});
            st.term_freqs.indices[i] = tid; st.term_freqs.values[i] = val;
        }
        // Indices are expected sorted already; defensively ensure
        std::vector<std::size_t> perm(st.term_freqs.indices.size());
        std::iota(perm.begin(), perm.end(), 0);
        std::sort(perm.begin(), perm.end(), [&](std::size_t a, std::size_t b){ return st.term_freqs.indices[a] < st.term_freqs.indices[b]; });
        SparseVector sorted;
        sorted.indices.reserve(st.term_freqs.indices.size());
        sorted.values.reserve(st.term_freqs.values.size());
        for (auto pi : perm) { sorted.indices.push_back(st.term_freqs.indices[pi]); sorted.values.push_back(st.term_freqs.values[pi]); }
        st.term_freqs = std::move(sorted);

        impl->doc_stats_.emplace(doc_id, st);
        impl->total_tokens_ += dlen;
        // Rebuild inverted index and doc_freqs
        for (std::uint32_t tid : st.term_freqs.indices) {
            if (tid < impl->inverted_index_.size()) {
                impl->inverted_index_[tid].add(static_cast<uint32_t>(doc_id));
                impl->doc_freqs_[tid]++;
            }
        }
    }

    // Trailer and checksum validation
    char trailer[4]{};
    if (!read_raw(trailer, sizeof(trailer))) return std::vesper_unexpected(error{error_code::io_error, "Read failed (trailer mark)", "bm25"});
    if (std::memcmp(trailer, TRAILER_MARK, sizeof(trailer)) != 0) {
        return std::vesper_unexpected(error{error_code::data_integrity, "Missing checksum trailer", "bm25"});
    }
    std::uint64_t stored_checksum=0;
    if (!read_raw(&stored_checksum, sizeof(stored_checksum))) return std::vesper_unexpected(error{error_code::io_error, "Read failed (checksum)", "bm25"});
    if (stored_checksum != checksum) {
        return std::vesper_unexpected(error{error_code::data_integrity, "Checksum mismatch", "bm25"});
    }

    // Finalize stats
    impl->avg_doc_length_ = (impl->doc_stats_.empty() ? 0.0f : static_cast<float>(impl->total_tokens_) / impl->doc_stats_.size());

    return impl;
}

// BM25Index implementation (forwarding to Impl)

BM25Index::BM25Index() : impl_(std::make_unique<Impl>()) {
    // Initialize with default parameters for convenience in tests and typical usage
    (void)this->init(BM25Params{}, 0);
}
BM25Index::~BM25Index() = default;
BM25Index::BM25Index(BM25Index&&) noexcept = default;
BM25Index& BM25Index::operator=(BM25Index&&) noexcept = default;

auto BM25Index::init(const BM25Params& params, std::size_t expected_docs)
    -> std::expected<void, core::error> {
    return impl_->init(params, expected_docs);
}

auto BM25Index::add_document(std::uint64_t doc_id, std::string_view text,
                             const std::unordered_map<std::string, std::string>& metadata)
    -> std::expected<void, core::error> {
    return impl_->add_document(doc_id, text, metadata);
}

auto BM25Index::add_batch(const std::uint64_t* doc_ids,
                         const std::string* texts,
                         std::size_t n)
    -> std::expected<void, core::error> {

    // Simple sequential implementation for now
    // TODO: Parallelize tokenization
    for (std::size_t i = 0; i < n; ++i) {
        auto result = add_document(doc_ids[i], texts[i]);
        if (!result) {
            return result;
        }
    }

    return {};
}

auto BM25Index::search(std::string_view query, std::uint32_t k,
                      const roaring::Roaring* filter) const
    -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error> {
    return impl_->search(query, k, filter);
}

auto BM25Index::encode_text(std::string_view text) const
    -> std::expected<SparseVector, core::error> {
    return impl_->encode_text(text);
}

auto BM25Index::get_document_vector(std::uint64_t doc_id) const
    -> std::expected<SparseVector, core::error> {
    return impl_->get_document_vector(doc_id);
}

auto BM25Index::score_document(const SparseVector& query_terms, std::uint64_t doc_id) const
    -> float {
    return impl_->score_document(query_terms, doc_id);
}

auto BM25Index::get_stats() const noexcept -> BM25Stats {
    return impl_->get_stats();
}

auto BM25Index::is_initialized() const noexcept -> bool {
    return impl_->is_initialized();
}

auto BM25Index::size() const noexcept -> std::size_t {
    return impl_->size();
}

auto BM25Index::clear() -> void {
    impl_->clear();
}

auto BM25Index::save(const std::string& path) const
    -> std::expected<void, core::error> {
    return impl_->save_to_file(path);
}

auto BM25Index::load(const std::string& path)
    -> std::expected<BM25Index, core::error> {
    auto r = Impl::load_from_file(path);
    if (!r) return std::vesper_unexpected(r.error());
    BM25Index idx;
    idx.impl_ = std::move(r.value());
    return idx;
}

auto BM25Index::vocabulary_size() const noexcept -> std::size_t {
    return impl_->vocabulary_size();
}

auto BM25Index::avg_doc_length() const noexcept -> float {
    return impl_->avg_doc_length();
}

auto BM25Index::update_params(const BM25Params& params) -> void {
    impl_->update_params(params);
}

} // namespace vesper::index
