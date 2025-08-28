#pragma once

/** \file filter_expr.hpp
 *  \brief Filter expression AST for metadata predicates.
 *
 * Use cases: compile to Roaring bitmaps and intersect during candidate generation.
 * Ownership: this AST is value-semantic and self-contained.
 */

#include <string>
#include <variant>
#include <vector>

namespace vesper {

/** \brief A simple term equality predicate field == value. */
struct term {
  std::string field; /**< attribute name */
  std::string value; /**< string-encoded value (typed variants later) */
};

/** \brief A numeric range predicate min_value ≤ field ≤ max_value. */
struct range {
  std::string field; /**< attribute name */
  double min_value{}; /**< inclusive */
  double max_value{}; /**< inclusive */
};

/** \brief Recursive filter expression. */
struct filter_expr {
  struct and_t { std::vector<filter_expr> children; };
  struct or_t  { std::vector<filter_expr> children; };
  struct not_t { std::vector<filter_expr> children; };

  std::variant<term, range, and_t, or_t, not_t> node; /**< root node */
};

} // namespace vesper

