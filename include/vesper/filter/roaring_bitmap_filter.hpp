/** \file roaring_bitmap_filter.hpp
 *  \brief High-performance bitmap filtering using CRoaring library
 */

#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

// CRoaring C++ API
#ifdef __cplusplus
extern "C" {
#endif
#include <roaring/roaring.h>
#ifdef __cplusplus
}
#endif

#include "vesper/error.hpp"
#include "vesper/filter_expr.hpp"

namespace vesper::filter {

/**
 * \brief RAII wrapper for CRoaring bitmap
 */
class RoaringBitmapWrapper {
public:
    RoaringBitmapWrapper() : bitmap_(roaring_bitmap_create()) {}
    
    explicit RoaringBitmapWrapper(roaring_bitmap_t* bitmap) : bitmap_(bitmap) {}
    
    ~RoaringBitmapWrapper() {
        if (bitmap_) {
            roaring_bitmap_free(bitmap_);
        }
    }
    
    // Move constructor
    RoaringBitmapWrapper(RoaringBitmapWrapper&& other) noexcept
        : bitmap_(other.bitmap_) {
        other.bitmap_ = nullptr;
    }
    
    // Move assignment
    RoaringBitmapWrapper& operator=(RoaringBitmapWrapper&& other) noexcept {
        if (this != &other) {
            if (bitmap_) {
                roaring_bitmap_free(bitmap_);
            }
            bitmap_ = other.bitmap_;
            other.bitmap_ = nullptr;
        }
        return *this;
    }
    
    // Delete copy operations
    RoaringBitmapWrapper(const RoaringBitmapWrapper&) = delete;
    RoaringBitmapWrapper& operator=(const RoaringBitmapWrapper&) = delete;
    
    // Clone operation
    [[nodiscard]] RoaringBitmapWrapper clone() const {
        return RoaringBitmapWrapper(roaring_bitmap_copy(bitmap_));
    }
    
    // Basic operations
    void add(uint32_t value) {
        roaring_bitmap_add(bitmap_, value);
    }
    
    void add_range(uint32_t start, uint32_t end) {
        roaring_bitmap_add_range_closed(bitmap_, start, end - 1);
    }
    
    void add_many(size_t n, const uint32_t* values) {
        roaring_bitmap_add_many(bitmap_, n, values);
    }
    
    [[nodiscard]] bool contains(uint32_t value) const {
        return roaring_bitmap_contains(bitmap_, value);
    }
    
    [[nodiscard]] uint64_t cardinality() const {
        return roaring_bitmap_get_cardinality(bitmap_);
    }
    
    [[nodiscard]] bool is_empty() const {
        return roaring_bitmap_is_empty(bitmap_);
    }
    
    // Set operations (return new bitmap)
    [[nodiscard]] RoaringBitmapWrapper and_op(const RoaringBitmapWrapper& other) const {
        return RoaringBitmapWrapper(roaring_bitmap_and(bitmap_, other.bitmap_));
    }
    
    [[nodiscard]] RoaringBitmapWrapper or_op(const RoaringBitmapWrapper& other) const {
        return RoaringBitmapWrapper(roaring_bitmap_or(bitmap_, other.bitmap_));
    }
    
    [[nodiscard]] RoaringBitmapWrapper xor_op(const RoaringBitmapWrapper& other) const {
        return RoaringBitmapWrapper(roaring_bitmap_xor(bitmap_, other.bitmap_));
    }
    
    [[nodiscard]] RoaringBitmapWrapper andnot_op(const RoaringBitmapWrapper& other) const {
        return RoaringBitmapWrapper(roaring_bitmap_andnot(bitmap_, other.bitmap_));
    }
    
    [[nodiscard]] RoaringBitmapWrapper flip(uint32_t start, uint32_t end) const {
        auto result = clone();
        roaring_bitmap_flip_inplace(result.bitmap_, start, end);
        return result;
    }
    
    // In-place operations
    void or_inplace(const RoaringBitmapWrapper& other) {
        roaring_bitmap_or_inplace(bitmap_, other.bitmap_);
    }
    
    void and_inplace(const RoaringBitmapWrapper& other) {
        roaring_bitmap_and_inplace(bitmap_, other.bitmap_);
    }
    
    void xor_inplace(const RoaringBitmapWrapper& other) {
        roaring_bitmap_xor_inplace(bitmap_, other.bitmap_);
    }
    
    void andnot_inplace(const RoaringBitmapWrapper& other) {
        roaring_bitmap_andnot_inplace(bitmap_, other.bitmap_);
    }
    
    // Convert to array
    [[nodiscard]] std::vector<uint32_t> to_array() const {
        uint64_t card = cardinality();
        std::vector<uint32_t> result(card);
        roaring_bitmap_to_uint32_array(bitmap_, result.data());
        return result;
    }
    
    // Statistics
    [[nodiscard]] uint64_t size_in_bytes() const {
        return roaring_bitmap_size_in_bytes(bitmap_);
    }
    
    [[nodiscard]] uint64_t portable_size_in_bytes() const {
        return roaring_bitmap_portable_size_in_bytes(bitmap_);
    }
    
    // Serialization
    [[nodiscard]] std::vector<uint8_t> serialize() const {
        size_t size = roaring_bitmap_portable_size_in_bytes(bitmap_);
        std::vector<uint8_t> buffer(size);
        roaring_bitmap_portable_serialize(bitmap_, reinterpret_cast<char*>(buffer.data()));
        return buffer;
    }
    
    static RoaringBitmapWrapper deserialize(const uint8_t* data, size_t size) {
        return RoaringBitmapWrapper(
            roaring_bitmap_portable_deserialize_safe(
                reinterpret_cast<const char*>(data), size)
        );
    }
    
    // Run optimizations
    void run_optimize() {
        roaring_bitmap_run_optimize(bitmap_);
    }
    
    void shrink_to_fit() {
        roaring_bitmap_shrink_to_fit(bitmap_);
    }
    
    // Access raw bitmap for advanced operations
    [[nodiscard]] roaring_bitmap_t* get() const { return bitmap_; }
    
private:
    roaring_bitmap_t* bitmap_;
};

/**
 * \brief Statistics for bitmap filter operations
 */
struct RoaringFilterStats {
    std::uint64_t evaluations{0};
    std::uint64_t bitmap_operations{0};
    std::uint64_t cache_hits{0};
    std::uint64_t cache_misses{0};
    double avg_selectivity{0.0};
    std::size_t memory_bytes{0};
    std::size_t compressed_bytes{0};
};

/**
 * \brief High-performance bitmap filter using CRoaring
 */
class RoaringBitmapFilter {
public:
    explicit RoaringBitmapFilter(std::size_t max_vectors);
    ~RoaringBitmapFilter();
    
    /**
     * \brief Add metadata for a vector
     */
    auto add_metadata(std::uint32_t vector_id, 
                      const std::unordered_map<std::string, std::string>& tags,
                      const std::unordered_map<std::string, double>& nums)
        -> std::expected<void, core::error>;
    
    /**
     * \brief Remove metadata for a vector
     */
    auto remove_metadata(std::uint32_t vector_id)
        -> std::expected<void, core::error>;
    
    /**
     * \brief Compile filter expression to bitmap
     */
    auto compile_filter(const filter_expr& expr)
        -> std::expected<RoaringBitmapWrapper, core::error>;
    
    /**
     * \brief Apply filter and return matching vector IDs
     */
    auto apply_filter(const filter_expr& expr)
        -> std::expected<std::vector<std::uint32_t>, core::error>;
    
    /**
     * \brief Apply filter to a candidate set
     */
    auto filter_candidates(const std::vector<std::uint32_t>& candidates,
                          const filter_expr& expr)
        -> std::expected<std::vector<std::uint32_t>, core::error>;
    
    /**
     * \brief Get filter selectivity estimate
     */
    [[nodiscard]] auto estimate_selectivity(const filter_expr& expr) const
        -> double;
    
    /**
     * \brief Get statistics
     */
    [[nodiscard]] auto get_stats() const -> RoaringFilterStats;
    
    /**
     * \brief Clear all metadata
     */
    auto clear() -> void;
    
    /**
     * \brief Build inverted index for a field
     */
    auto build_index(const std::string& field)
        -> std::expected<void, core::error>;
    
    /**
     * \brief Optimize all bitmaps for better compression
     */
    auto optimize() -> void;
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * \brief RoaringBitmapFilter implementation
 */
class RoaringBitmapFilter::Impl {
public:
    explicit Impl(std::size_t max_vectors) 
        : max_vectors_(max_vectors) {
        // Reserve space for metadata
        vector_tags_.reserve(max_vectors);
        vector_nums_.reserve(max_vectors);
    }
    
    auto add_metadata(std::uint32_t vector_id,
                     const std::unordered_map<std::string, std::string>& tags,
                     const std::unordered_map<std::string, double>& nums)
        -> std::expected<void, core::error> {
        
        if (vector_id >= max_vectors_) {
            return std::unexpected(core::error{
                core::error_code::precondition_failed,
                "Vector ID exceeds maximum",
                "roaring_filter.add_metadata"
            });
        }
        
        // Store metadata
        vector_tags_[vector_id] = tags;
        vector_nums_[vector_id] = nums;
        
        // Update inverted indexes
        for (const auto& [field, value] : tags) {
            tag_index_[field][value].add(vector_id);
        }
        
        for (const auto& [field, value] : nums) {
            num_values_[field][vector_id] = value;
        }
        
        return {};
    }
    
    auto compile_filter(const filter_expr& expr)
        -> std::expected<RoaringBitmapWrapper, core::error> {
        
        return std::visit([this](const auto& node) -> std::expected<RoaringBitmapWrapper, core::error> {
            using T = std::decay_t<decltype(node)>;
            
            if constexpr (std::is_same_v<T, term>) {
                // Term equality
                auto it = tag_index_.find(node.field);
                if (it == tag_index_.end()) {
                    return RoaringBitmapWrapper();  // Empty bitmap
                }
                
                auto value_it = it->second.find(node.value);
                if (value_it == it->second.end()) {
                    return RoaringBitmapWrapper();  // Empty bitmap
                }
                
                stats_.bitmap_operations++;
                return value_it->second.clone();
                
            } else if constexpr (std::is_same_v<T, range>) {
                // Range query
                RoaringBitmapWrapper result;
                
                auto it = num_values_.find(node.field);
                if (it != num_values_.end()) {
                    for (const auto& [id, value] : it->second) {
                        if (value >= node.min_value && value <= node.max_value) {
                            result.add(id);
                        }
                    }
                }
                
                stats_.bitmap_operations++;
                return result;
                
            } else if constexpr (std::is_same_v<T, filter_expr::and_t>) {
                // AND operation
                if (node.children.empty()) {
                    return RoaringBitmapWrapper();
                }
                
                auto result = compile_filter(node.children[0]);
                if (!result) return result;
                
                RoaringBitmapWrapper bitmap = std::move(*result);
                
                for (std::size_t i = 1; i < node.children.size(); ++i) {
                    auto child_result = compile_filter(node.children[i]);
                    if (!child_result) return child_result;
                    
                    bitmap.and_inplace(*child_result);
                    stats_.bitmap_operations++;
                }
                
                return bitmap;
                
            } else if constexpr (std::is_same_v<T, filter_expr::or_t>) {
                // OR operation
                if (node.children.empty()) {
                    return RoaringBitmapWrapper();
                }
                
                auto result = compile_filter(node.children[0]);
                if (!result) return result;
                
                RoaringBitmapWrapper bitmap = std::move(*result);
                
                for (std::size_t i = 1; i < node.children.size(); ++i) {
                    auto child_result = compile_filter(node.children[i]);
                    if (!child_result) return child_result;
                    
                    bitmap.or_inplace(*child_result);
                    stats_.bitmap_operations++;
                }
                
                return bitmap;
                
            } else if constexpr (std::is_same_v<T, filter_expr::not_t>) {
                // NOT operation - flip all bits up to max_vectors
                if (node.children.empty()) {
                    return RoaringBitmapWrapper();
                }
                
                auto child_result = compile_filter(node.children[0]);
                if (!child_result) return child_result;
                
                // Create full bitmap and subtract child
                RoaringBitmapWrapper full;
                full.add_range(0, static_cast<uint32_t>(max_vectors_));
                full.andnot_inplace(*child_result);
                
                stats_.bitmap_operations++;
                return full;
            }
            
            return RoaringBitmapWrapper();
        }, expr.node);
    }
    
    auto apply_filter(const filter_expr& expr)
        -> std::expected<std::vector<std::uint32_t>, core::error> {
        
        auto bitmap_result = compile_filter(expr);
        if (!bitmap_result) {
            return std::unexpected(bitmap_result.error());
        }
        
        stats_.evaluations++;
        return bitmap_result->to_array();
    }
    
    auto filter_candidates(const std::vector<std::uint32_t>& candidates,
                          const filter_expr& expr)
        -> std::expected<std::vector<std::uint32_t>, core::error> {
        
        auto bitmap_result = compile_filter(expr);
        if (!bitmap_result) {
            return std::unexpected(bitmap_result.error());
        }
        
        // Create bitmap from candidates
        RoaringBitmapWrapper candidate_bitmap;
        candidate_bitmap.add_many(candidates.size(), candidates.data());
        
        // Intersect with filter
        candidate_bitmap.and_inplace(*bitmap_result);
        
        stats_.evaluations++;
        return candidate_bitmap.to_array();
    }
    
    auto estimate_selectivity(const filter_expr& expr) const -> double {
        // Use actual cardinalities from Roaring bitmaps
        return std::visit([this](const auto& node) -> double {
            using T = std::decay_t<decltype(node)>;
            
            if constexpr (std::is_same_v<T, term>) {
                auto it = tag_index_.find(node.field);
                if (it == tag_index_.end()) return 0.0;
                
                auto value_it = it->second.find(node.value);
                if (value_it == it->second.end()) return 0.0;
                
                return static_cast<double>(value_it->second.cardinality()) / max_vectors_;
                
            } else if constexpr (std::is_same_v<T, range>) {
                // Estimate based on range width
                auto it = num_values_.find(node.field);
                if (it == num_values_.end()) return 0.0;
                
                // Simple heuristic: 10% for narrow ranges, 50% for wide
                double range_width = node.max_value - node.min_value;
                return std::min(0.5, range_width / 100.0);
                
            } else if constexpr (std::is_same_v<T, filter_expr::and_t>) {
                // Multiply selectivities (independence assumption)
                double sel = 1.0;
                for (const auto& child : node.children) {
                    sel *= estimate_selectivity(child);
                }
                return sel;
                
            } else if constexpr (std::is_same_v<T, filter_expr::or_t>) {
                // Union probability
                double sel = 0.0;
                for (const auto& child : node.children) {
                    double child_sel = estimate_selectivity(child);
                    sel = sel + child_sel - (sel * child_sel);
                }
                return sel;
                
            } else if constexpr (std::is_same_v<T, filter_expr::not_t>) {
                if (node.children.empty()) return 1.0;
                return 1.0 - estimate_selectivity(node.children[0]);
            }
            
            return 0.0;
        }, expr.node);
    }
    
    auto optimize() -> void {
        // Run optimize on all bitmaps
        for (auto& [field, values] : tag_index_) {
            for (auto& [value, bitmap] : values) {
                bitmap.run_optimize();
                bitmap.shrink_to_fit();
            }
        }
    }
    
    auto get_stats() const -> RoaringFilterStats {
        RoaringFilterStats result = stats_;
        
        // Calculate memory usage
        result.memory_bytes = 0;
        result.compressed_bytes = 0;
        
        for (const auto& [field, values] : tag_index_) {
            for (const auto& [value, bitmap] : values) {
                result.memory_bytes += bitmap.size_in_bytes();
                result.compressed_bytes += bitmap.portable_size_in_bytes();
            }
        }
        
        return result;
    }
    
    auto clear() -> void {
        vector_tags_.clear();
        vector_nums_.clear();
        tag_index_.clear();
        num_values_.clear();
        stats_ = {};
    }
    
private:
    std::size_t max_vectors_;
    
    // Metadata storage
    std::unordered_map<std::uint32_t, std::unordered_map<std::string, std::string>> vector_tags_;
    std::unordered_map<std::uint32_t, std::unordered_map<std::string, double>> vector_nums_;
    
    // Inverted indexes using Roaring bitmaps
    std::unordered_map<std::string, std::unordered_map<std::string, RoaringBitmapWrapper>> tag_index_;
    std::unordered_map<std::string, std::unordered_map<std::uint32_t, double>> num_values_;
    
    // Statistics
    mutable RoaringFilterStats stats_;
};

// RoaringBitmapFilter public methods implementation
inline RoaringBitmapFilter::RoaringBitmapFilter(std::size_t max_vectors)
    : impl_(std::make_unique<Impl>(max_vectors)) {}

inline RoaringBitmapFilter::~RoaringBitmapFilter() = default;

inline auto RoaringBitmapFilter::add_metadata(std::uint32_t vector_id,
                                             const std::unordered_map<std::string, std::string>& tags,
                                             const std::unordered_map<std::string, double>& nums)
    -> std::expected<void, core::error> {
    return impl_->add_metadata(vector_id, tags, nums);
}

inline auto RoaringBitmapFilter::compile_filter(const filter_expr& expr)
    -> std::expected<RoaringBitmapWrapper, core::error> {
    return impl_->compile_filter(expr);
}

inline auto RoaringBitmapFilter::apply_filter(const filter_expr& expr)
    -> std::expected<std::vector<std::uint32_t>, core::error> {
    return impl_->apply_filter(expr);
}

inline auto RoaringBitmapFilter::filter_candidates(const std::vector<std::uint32_t>& candidates,
                                                  const filter_expr& expr)
    -> std::expected<std::vector<std::uint32_t>, core::error> {
    return impl_->filter_candidates(candidates, expr);
}

inline auto RoaringBitmapFilter::estimate_selectivity(const filter_expr& expr) const
    -> double {
    return impl_->estimate_selectivity(expr);
}

inline auto RoaringBitmapFilter::get_stats() const -> RoaringFilterStats {
    return impl_->get_stats();
}

inline auto RoaringBitmapFilter::clear() -> void {
    impl_->clear();
}

inline auto RoaringBitmapFilter::optimize() -> void {
    impl_->optimize();
}

} // namespace vesper::filter