#include "vesper/filter_eval.hpp"

namespace vesper::filter_eval {

static auto matches_node(const filter_expr& e, const tags_t& tags, const nums_t& nums) -> bool {
  if (std::holds_alternative<term>(e.node)) {
    const auto& t = std::get<term>(e.node);
    auto it = tags.find(t.field);
    return it != tags.end() && it->second == t.value;
  } else if (std::holds_alternative<range>(e.node)) {
    const auto& r = std::get<range>(e.node);
    auto it = nums.find(r.field);
    if (it == nums.end()) return false;
    return (it->second >= r.min_value) && (it->second <= r.max_value);
  } else if (std::holds_alternative<filter_expr::and_t>(e.node)) {
    const auto& a = std::get<filter_expr::and_t>(e.node);
    for (const auto& c : a.children) if (!matches_node(c, tags, nums)) return false;
    return true; // and([]) == true
  } else if (std::holds_alternative<filter_expr::or_t>(e.node)) {
    const auto& o = std::get<filter_expr::or_t>(e.node);
    for (const auto& c : o.children) if (matches_node(c, tags, nums)) return true;
    return false; // or([]) == false
  } else if (std::holds_alternative<filter_expr::not_t>(e.node)) {
    const auto& n = std::get<filter_expr::not_t>(e.node);
    bool v = true; // not([]) == true
    for (const auto& c : n.children) v = v && (!matches_node(c, tags, nums));
    return v;
  }
  return false;
}

auto matches(const filter_expr& expr, const tags_t& tags, const nums_t& nums) -> bool {
  return matches_node(expr, tags, nums);
}

auto apply_filter(const filter_expr* expr, const std::vector<id_view>& store) -> std::vector<std::uint64_t> {
  std::vector<std::uint64_t> ids;
  ids.reserve(store.size());
  if (!expr) {
    for (auto& v : store) ids.push_back(v.id);
    return ids;
  }
  for (auto& v : store) if (matches(*expr, *v.tags, *v.nums)) ids.push_back(v.id);
  return ids;
}

} // namespace vesper::filter_eval

