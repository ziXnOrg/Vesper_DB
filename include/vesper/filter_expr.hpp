#pragma once

#include <string>
#include <variant>
#include <vector>

namespace vesper {

struct term {
  std::string field;
  std::string value; // or variant for typed fields in a later phase
};

struct range {
  std::string field;
  double min_value{};
  double max_value{};
};

struct filter_expr {
  struct and_t { std::vector<filter_expr> children; };
  struct or_t  { std::vector<filter_expr> children; };
  struct not_t { std::vector<filter_expr> children; };

  std::variant<term, range, and_t, or_t, not_t> node;
};

} // namespace vesper

