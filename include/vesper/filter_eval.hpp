#pragma once

/** \file filter_eval.hpp
 *  \brief In-memory evaluation of filter_expr against simple tag/numeric maps.
 */

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "vesper/filter_expr.hpp"

namespace vesper::filter_eval {

using tags_t = std::unordered_map<std::string, std::string>;
using nums_t = std::unordered_map<std::string, double>;

// Evaluate whether a document with given tags/numerics matches the expression.
auto matches(const filter_expr& expr, const tags_t& tags, const nums_t& nums) -> bool;

// Apply filter to an in-memory store: returns ids satisfying expr; if expr==nullptr include all
struct id_view { std::uint64_t id; const tags_t* tags; const nums_t* nums; };
auto apply_filter(const filter_expr* expr, const std::vector<id_view>& store) -> std::vector<std::uint64_t>;

} // namespace vesper::filter_eval

