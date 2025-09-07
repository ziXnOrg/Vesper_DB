#pragma once

/** \file metadata_store.hpp
 *  \brief Metadata storage and filtering with Roaring bitmaps
 */

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <expected>
#include <shared_mutex>
#include <variant>

#include "vesper/error.hpp"
#include "vesper/filter_expr.hpp"
#include "vesper/filter_eval.hpp"
#include "roaring.hh"

namespace vesper::metadata {

/** \brief Metadata value type */
using MetadataValue = std::variant<std::string, double, std::int64_t, bool>;

/** \brief Document metadata */
struct DocumentMetadata {
    std::uint64_t id;
    std::unordered_map<std::string, MetadataValue> attributes;
};

/** \brief Metadata index configuration */
struct MetadataIndexConfig {
    bool enable_bitmap_index{true};     /**< Use Roaring bitmaps for categorical */
    bool enable_range_index{true};      /**< Build range indexes for numerics */
    std::size_t max_cardinality{10000}; /**< Max unique values for bitmap index */
    std::size_t cache_size_mb{64};      /**< Cache size for bitmap operations */
};

/** \brief Metadata store with bitmap filtering
 *
 * Example usage:
 * ```cpp
 * MetadataStore store;
 * 
 * // Add document metadata
 * DocumentMetadata doc{
 *     .id = 123,
 *     .attributes = {
 *         {"category", "electronics"},
 *         {"price", 99.99},
 *         {"in_stock", true}
 *     }
 * };
 * store.add(doc);
 * 
 * // Filter documents
 * filter_expr expr = ...;
 * auto bitmap = store.evaluate_filter(expr);
 * ```
 */
class MetadataStore {
public:
    MetadataStore();
    explicit MetadataStore(MetadataIndexConfig config);
    ~MetadataStore();
    
    MetadataStore(MetadataStore&&) noexcept;
    MetadataStore& operator=(MetadataStore&&) noexcept;
    MetadataStore(const MetadataStore&) = delete;
    MetadataStore& operator=(const MetadataStore&) = delete;
    
    /** \brief Add document metadata
     *
     * \param doc Document metadata
     * \return Success or error
     *
     * Thread-safety: Thread-safe with concurrent reads
     */
    auto add(const DocumentMetadata& doc) -> std::expected<void, core::error>;
    
    /** \brief Add batch of documents
     *
     * \param docs Documents to add
     * \return Success or error
     */
    auto add_batch(const std::vector<DocumentMetadata>& docs) 
        -> std::expected<void, core::error>;
    
    /** \brief Update document metadata
     *
     * \param doc Updated metadata
     * \return Success or error
     */
    auto update(const DocumentMetadata& doc) -> std::expected<void, core::error>;
    
    /** \brief Remove document metadata
     *
     * \param id Document ID
     * \return Success or error
     */
    auto remove(std::uint64_t id) -> std::expected<void, core::error>;
    
    /** \brief Evaluate filter expression to bitmap
     *
     * \param expr Filter expression
     * \return Bitmap of matching document IDs
     *
     * Thread-safety: Thread-safe for concurrent calls
     */
    auto evaluate_filter(const filter_expr& expr) const 
        -> std::expected<roaring::Roaring, core::error>;
    
    /** \brief Get documents matching filter
     *
     * \param expr Filter expression
     * \param limit Maximum results (0 = no limit)
     * \return Matching document IDs
     */
    auto search(const filter_expr& expr, std::size_t limit = 0) const
        -> std::expected<std::vector<std::uint64_t>, core::error>;
    
    /** \brief Get document metadata
     *
     * \param id Document ID
     * \return Document metadata or error
     */
    auto get(std::uint64_t id) const 
        -> std::expected<DocumentMetadata, core::error>;
    
    /** \brief Check if document exists
     *
     * \param id Document ID
     * \return True if document exists
     */
    auto exists(std::uint64_t id) const -> bool;
    
    /** \brief Get all document IDs
     *
     * \return Bitmap of all document IDs
     */
    auto get_all_ids() const -> roaring::Roaring;
    
    /** \brief Build/rebuild indexes
     *
     * \return Success or error
     */
    auto build_indexes() -> std::expected<void, core::error>;
    
    /** \brief Get statistics */
    struct Stats {
        std::size_t document_count{0};
        std::size_t index_count{0};
        std::size_t memory_usage_bytes{0};
        std::size_t cache_hits{0};
        std::size_t cache_misses{0};
    };
    
    auto get_stats() const -> Stats;
    
    /** \brief Clear all metadata */
    auto clear() -> void;
    
    /** \brief Save to disk
     *
     * \param path File path
     * \return Success or error
     */
    auto save(const std::string& path) const -> std::expected<void, core::error>;
    
    /** \brief Load from disk
     *
     * \param path File path
     * \return Success or error
     */
    auto load(const std::string& path) -> std::expected<void, core::error>;
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/** \brief Metadata filter utilities */
namespace utils {

/** \brief Convert simple key-value pairs to filter expression
 *
 * \param kvs Key-value pairs
 * \return Filter expression (AND of all terms)
 */
auto kv_to_filter(const std::unordered_map<std::string, std::string>& kvs) 
    -> filter_expr;

/** \brief Parse filter expression from JSON string
 *
 * \param json JSON string
 * \return Filter expression or error
 */
auto parse_filter_json(const std::string& json) 
    -> std::expected<filter_expr, core::error>;

/** \brief Serialize filter expression to JSON
 *
 * \param expr Filter expression
 * \return JSON string
 */
auto filter_to_json(const filter_expr& expr) -> std::string;

} // namespace utils

} // namespace vesper::metadata
