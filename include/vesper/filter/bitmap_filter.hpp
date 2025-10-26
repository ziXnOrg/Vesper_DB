/** \file bitmap_filter.hpp
 *  \brief High-performance bitmap filtering using Roaring Bitmaps
 */

#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>
#include <roaring/roaring.h>

#include "vesper/error.hpp"
#include "vesper/filter_expr.hpp"

namespace vesper::filter {

/**
 * \brief Roaring bitmap wrapper for efficient set operations
 * 
 * Uses CRoaring library for memory-efficient compressed bitmaps
 * with fast set operations. Typically achieves 10-100x compression
 * compared to std::vector<bool>.
 */
class RoaringBitmap {
public:
    RoaringBitmap() : bitmap_(roaring_bitmap_create()) {}
    
    explicit RoaringBitmap(std::size_t /*size*/) : bitmap_(roaring_bitmap_create()) {
        // Size hint not needed for roaring bitmaps - they grow dynamically
    }
    
    ~RoaringBitmap() {
        if (bitmap_) {
            roaring_bitmap_free(bitmap_);
        }
    }
    
    // Move constructor
    RoaringBitmap(RoaringBitmap&& other) noexcept : bitmap_(other.bitmap_) {
        other.bitmap_ = nullptr;
    }
    
    // Move assignment
    RoaringBitmap& operator=(RoaringBitmap&& other) noexcept {
        if (this != &other) {
            if (bitmap_) {
                roaring_bitmap_free(bitmap_);
            }
            bitmap_ = other.bitmap_;
            other.bitmap_ = nullptr;
        }
        return *this;
    }
    
    // Copy constructor
    RoaringBitmap(const RoaringBitmap& other) 
        : bitmap_(other.bitmap_ ? roaring_bitmap_copy(other.bitmap_) : nullptr) {}
    
    // Copy assignment
    RoaringBitmap& operator=(const RoaringBitmap& other) {
        if (this != &other) {
            if (bitmap_) {
                roaring_bitmap_free(bitmap_);
            }
            bitmap_ = other.bitmap_ ? roaring_bitmap_copy(other.bitmap_) : nullptr;
        }
        return *this;
    }
    
    void add(std::uint32_t value) {
        if (bitmap_) {
            roaring_bitmap_add(bitmap_, value);
        }
    }
    
    void add_range(std::uint32_t start, std::uint32_t end) {
        if (bitmap_) {
            roaring_bitmap_add_range_closed(bitmap_, start, end - 1);
        }
    }
    
    [[nodiscard]] bool contains(std::uint32_t value) const {
        return bitmap_ && roaring_bitmap_contains(bitmap_, value);
    }
    
    [[nodiscard]] std::size_t cardinality() const {
        return bitmap_ ? roaring_bitmap_get_cardinality(bitmap_) : 0;
    }
    
    [[nodiscard]] bool is_empty() const {
        return !bitmap_ || roaring_bitmap_is_empty(bitmap_);
    }
    
    // Set operations
    RoaringBitmap operator&(const RoaringBitmap& other) const {
        RoaringBitmap result;
        if (bitmap_ && other.bitmap_) {
            roaring_bitmap_free(result.bitmap_);
            result.bitmap_ = roaring_bitmap_and(bitmap_, other.bitmap_);
        }
        return result;
    }
    
    RoaringBitmap operator|(const RoaringBitmap& other) const {
        RoaringBitmap result;
        if (bitmap_ && other.bitmap_) {
            roaring_bitmap_free(result.bitmap_);
            result.bitmap_ = roaring_bitmap_or(bitmap_, other.bitmap_);
        } else if (bitmap_) {
            roaring_bitmap_free(result.bitmap_);
            result.bitmap_ = roaring_bitmap_copy(bitmap_);
        } else if (other.bitmap_) {
            roaring_bitmap_free(result.bitmap_);
            result.bitmap_ = roaring_bitmap_copy(other.bitmap_);
        }
        return result;
    }
    
    RoaringBitmap operator~() const {
        RoaringBitmap result;
        if (bitmap_) {
            roaring_bitmap_free(result.bitmap_);
            result.bitmap_ = roaring_bitmap_flip(bitmap_, 0, roaring_bitmap_maximum(bitmap_) + 1);
        }
        return result;
    }
    
    // Iteration support
    std::vector<std::uint32_t> to_array() const {
        std::vector<std::uint32_t> result;
        if (bitmap_) {
            std::size_t card = roaring_bitmap_get_cardinality(bitmap_);
            result.resize(card);
            roaring_bitmap_to_uint32_array(bitmap_, result.data());
        }
        return result;
    }
    
    // Additional useful methods
    void run_optimize() {
        if (bitmap_) {
            roaring_bitmap_run_optimize(bitmap_);
        }
    }
    
    [[nodiscard]] std::size_t size_in_bytes() const {
        return bitmap_ ? roaring_bitmap_size_in_bytes(bitmap_) : 0;
    }
    
    [[nodiscard]] std::size_t portable_size_in_bytes() const {
        return bitmap_ ? roaring_bitmap_portable_size_in_bytes(bitmap_) : 0;
    }
    
    // Serialization
    auto serialize() const -> std::vector<uint8_t> {
        if (!bitmap_) return {};
        
        std::size_t size = roaring_bitmap_portable_size_in_bytes(bitmap_);
        std::vector<uint8_t> buffer(size);
        roaring_bitmap_portable_serialize(bitmap_, reinterpret_cast<char*>(buffer.data()));
        return buffer;
    }
    
    static auto deserialize(const std::vector<uint8_t>& buffer) -> RoaringBitmap {
        RoaringBitmap result;
        if (!buffer.empty()) {
            roaring_bitmap_free(result.bitmap_);
            result.bitmap_ = roaring_bitmap_portable_deserialize_safe(
                reinterpret_cast<const char*>(buffer.data()), buffer.size()
            );
        }
        return result;
    }
    
private:
    roaring_bitmap_t* bitmap_{nullptr};
};

/**
 * \brief Statistics for bitmap filter operations
 */
struct BitmapFilterStats {
    std::uint64_t evaluations{0};
    std::uint64_t bitmap_operations{0};
    std::uint64_t cache_hits{0};
    std::uint64_t cache_misses{0};
    double avg_selectivity{0.0};
    std::size_t memory_bytes{0};
};

/**
 * \brief Bitmap-based filter for efficient metadata filtering
 */
class BitmapFilter {
public:
    explicit BitmapFilter(std::size_t max_vectors);
    ~BitmapFilter();
    
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
        -> std::expected<RoaringBitmap, core::error>;
    
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
    [[nodiscard]] auto get_stats() const -> BitmapFilterStats;
    
    /**
     * \brief Clear all metadata
     */
    auto clear() -> void;
    
    /**
     * \brief Build inverted index for a field
     */
    auto build_index(const std::string& field)
        -> std::expected<void, core::error>;
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * \brief Bitmap filter implementation
 */
class BitmapFilter::Impl {
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
                "bitmap_filter.add_metadata"
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
            // For numeric fields, we'd typically use interval trees
            // For simplicity, we just track all values
            num_values_[field][vector_id] = value;
        }
        
        return {};
    }
    
    auto compile_filter(const filter_expr& expr)
        -> std::expected<RoaringBitmap, core::error> {
        
        return std::visit([this](const auto& node) -> std::expected<RoaringBitmap, core::error> {
            using T = std::decay_t<decltype(node)>;
            
            if constexpr (std::is_same_v<T, term>) {
                // Term equality
                auto it = tag_index_.find(node.field);
                if (it == tag_index_.end()) {
                    return RoaringBitmap(max_vectors_);  // Empty bitmap
                }
                
                auto value_it = it->second.find(node.value);
                if (value_it == it->second.end()) {
                    return RoaringBitmap(max_vectors_);  // Empty bitmap
                }
                
                stats_.bitmap_operations++;
                return value_it->second;
                
            } else if constexpr (std::is_same_v<T, range>) {
                // Range query
                RoaringBitmap result(max_vectors_);
                
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
                    return RoaringBitmap(max_vectors_);
                }
                
                auto result = compile_filter(node.children[0]);
                if (!result) return result;
                
                RoaringBitmap bitmap = std::move(*result);
                
                for (std::size_t i = 1; i < node.children.size(); ++i) {
                    auto child_result = compile_filter(node.children[i]);
                    if (!child_result) return child_result;
                    
                    bitmap = bitmap & *child_result;
                    stats_.bitmap_operations++;
                }
                
                return bitmap;
                
            } else if constexpr (std::is_same_v<T, filter_expr::or_t>) {
                // OR operation
                if (node.children.empty()) {
                    return RoaringBitmap(max_vectors_);
                }
                
                auto result = compile_filter(node.children[0]);
                if (!result) return result;
                
                RoaringBitmap bitmap = std::move(*result);
                
                for (std::size_t i = 1; i < node.children.size(); ++i) {
                    auto child_result = compile_filter(node.children[i]);
                    if (!child_result) return child_result;
                    
                    bitmap = bitmap | *child_result;
                    stats_.bitmap_operations++;
                }
                
                return bitmap;
                
            } else if constexpr (std::is_same_v<T, filter_expr::not_t>) {
                // NOT operation
                if (node.children.empty()) {
                    return RoaringBitmap(max_vectors_);
                }
                
                auto child_result = compile_filter(node.children[0]);
                if (!child_result) return child_result;
                
                stats_.bitmap_operations++;
                return ~(*child_result);
            }
            
            return RoaringBitmap(max_vectors_);
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
    
    auto estimate_selectivity(const filter_expr& expr) const -> double {
        // Simplified selectivity estimation
        // In production, use histogram-based estimation
        return std::visit([this](const auto& node) -> double {
            using T = std::decay_t<decltype(node)>;
            
            if constexpr (std::is_same_v<T, term>) {
                auto it = tag_index_.find(node.field);
                if (it == tag_index_.end()) return 0.0;
                
                auto value_it = it->second.find(node.value);
                if (value_it == it->second.end()) return 0.0;
                
                return static_cast<double>(value_it->second.cardinality()) / max_vectors_;
                
            } else if constexpr (std::is_same_v<T, range>) {
                // Rough estimate: assume 10% selectivity for ranges
                return 0.1;
                
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
    
private:
    std::size_t max_vectors_;
    
    // Metadata storage
    std::unordered_map<std::uint32_t, std::unordered_map<std::string, std::string>> vector_tags_;
    std::unordered_map<std::uint32_t, std::unordered_map<std::string, double>> vector_nums_;
    
    // Inverted indexes
    std::unordered_map<std::string, std::unordered_map<std::string, RoaringBitmap>> tag_index_;
    std::unordered_map<std::string, std::unordered_map<std::uint32_t, double>> num_values_;
    
    // Statistics
    mutable BitmapFilterStats stats_;
};

// BitmapFilter public methods implementation
inline BitmapFilter::BitmapFilter(std::size_t max_vectors)
    : impl_(std::make_unique<Impl>(max_vectors)) {}

inline BitmapFilter::~BitmapFilter() = default;

inline auto BitmapFilter::add_metadata(std::uint32_t vector_id,
                                      const std::unordered_map<std::string, std::string>& tags,
                                      const std::unordered_map<std::string, double>& nums)
    -> std::expected<void, core::error> {
    return impl_->add_metadata(vector_id, tags, nums);
}

inline auto BitmapFilter::compile_filter(const filter_expr& expr)
    -> std::expected<RoaringBitmap, core::error> {
    return impl_->compile_filter(expr);
}

inline auto BitmapFilter::apply_filter(const filter_expr& expr)
    -> std::expected<std::vector<std::uint32_t>, core::error> {
    return impl_->apply_filter(expr);
}

inline auto BitmapFilter::estimate_selectivity(const filter_expr& expr) const
    -> double {
    return impl_->estimate_selectivity(expr);
}

} // namespace vesper::filter