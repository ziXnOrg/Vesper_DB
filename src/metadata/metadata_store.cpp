/** \file metadata_store.cpp
 *  \brief Implementation of metadata store with Roaring bitmaps
 */

#include "vesper/metadata/metadata_store.hpp"

#include <algorithm>
#include <fstream>
#include <utility>

namespace vesper::metadata {

namespace {
    // Helpers for serialization of strings
    inline void write_string(std::ofstream& os, const std::string& s) {
        std::uint64_t n = static_cast<std::uint64_t>(s.size());
        os.write(reinterpret_cast<const char*>(&n), sizeof(n));
        os.write(s.data(), static_cast<std::streamsize>(s.size()));
    }

    inline bool read_string(std::ifstream& is, std::string& out) {
        std::uint64_t n{};
        if (!is.read(reinterpret_cast<char*>(&n), sizeof(n))) return false;
        out.resize(static_cast<std::size_t>(n));
        return static_cast<bool>(is.read(out.data(), static_cast<std::streamsize>(n)));
    }

    inline std::string bool_to_str(bool b) { return b ? "true" : "false"; }
}

class MetadataStore::Impl {
public:
    explicit Impl(MetadataIndexConfig cfg)
        : config_(std::move(cfg)) {}

    auto add(const DocumentMetadata& doc) -> std::expected<void, core::error> {
        if (doc.id > std::numeric_limits<std::uint32_t>::max()) {
            return std::unexpected(core::error{
                core::error_code::precondition_failed,
                "Document id exceeds 32-bit limit required by Roaring",
                "metadata.add"
            });
        }
        std::unique_lock lock(mutex_);
        if (documents_.find(doc.id) != documents_.end()) {
            // Treat as update to keep semantics predictable
            remove_locked(doc.id);
        }
        documents_.emplace(doc.id, doc);
        all_ids_.add(static_cast<std::uint32_t>(doc.id));
        index_document_locked(doc);
        return {};
    }

    auto add_batch(const std::vector<DocumentMetadata>& docs) -> std::expected<void, core::error> {
        std::unique_lock lock(mutex_);
        for (const auto& d : docs) {
            if (d.id > std::numeric_limits<std::uint32_t>::max()) {
                return std::unexpected(core::error{
                    core::error_code::precondition_failed,
                    "Document id exceeds 32-bit limit required by Roaring",
                    "metadata.add_batch"
                });
            }
        }
        for (const auto& d : docs) {
            if (documents_.find(d.id) != documents_.end()) {
                remove_locked(d.id);
            }
            documents_.emplace(d.id, d);
            all_ids_.add(static_cast<std::uint32_t>(d.id));
            index_document_locked(d);
        }
        return {};
    }

    auto update(const DocumentMetadata& doc) -> std::expected<void, core::error> {
        std::unique_lock lock(mutex_);
        if (documents_.find(doc.id) == documents_.end()) {
            return std::unexpected(core::error{
                core::error_code::not_found,
                "Document not found",
                "metadata.update"
            });
        }
        remove_locked(doc.id);
        documents_.emplace(doc.id, doc);
        all_ids_.add(static_cast<std::uint32_t>(doc.id));
        index_document_locked(doc);
        return {};
    }

    auto remove(std::uint64_t id) -> std::expected<void, core::error> {
        std::unique_lock lock(mutex_);
        if (documents_.find(id) == documents_.end()) {
            return std::unexpected(core::error{
                core::error_code::not_found,
                "Document not found",
                "metadata.remove"
            });
        }
        remove_locked(id);
        return {};
    }

    auto evaluate_filter(const filter_expr& expr) const -> std::expected<roaring::Roaring, core::error> {
        std::shared_lock lock(mutex_);
        return compile_locked(expr);
    }

    auto search(const filter_expr& expr, std::size_t limit) const -> std::expected<std::vector<std::uint64_t>, core::error> {
        auto bitmap_e = evaluate_filter(expr);
        if (!bitmap_e) return std::unexpected(bitmap_e.error());
        const auto& bm = *bitmap_e;
        std::vector<std::uint64_t> ids;
        ids.reserve(static_cast<std::size_t>(bm.cardinality()));
        for (auto it = bm.begin(); it != bm.end(); ++it) {
            ids.push_back(*it);
            if (limit && ids.size() >= limit) break;
        }
        return ids;
    }

    auto get(std::uint64_t id) const -> std::expected<DocumentMetadata, core::error> {
        std::shared_lock lock(mutex_);
        auto it = documents_.find(id);
        if (it == documents_.end()) {
            return std::unexpected(core::error{core::error_code::not_found, "Document not found", "metadata.get"});
        }
        return it->second;
    }

    auto exists(std::uint64_t id) const -> bool {
        std::shared_lock lock(mutex_);
        return documents_.find(id) != documents_.end();
    }

    auto get_all_ids() const -> roaring::Roaring {
        std::shared_lock lock(mutex_);
        return all_ids_; // Roaring has cheap copy-on-write semantics
    }

    auto build_indexes() -> std::expected<void, core::error> {
        std::unique_lock lock(mutex_);
        tag_index_.clear();
        num_values_.clear();
        for (const auto& [id, doc] : documents_) {
            index_document_locked(doc);
        }
        return {};
    }

    auto get_stats() const -> MetadataStore::Stats {
        std::shared_lock lock(mutex_);
        Stats s{};
        s.document_count = documents_.size();
        // Count bitmaps
        for (const auto& [field, values] : tag_index_) {
            s.index_count += values.size();
            for (const auto& [val, bm] : values) {
                s.memory_usage_bytes += bm.getSizeInBytes();
            }
        }
        // Rough contribution from numerics maps
        for (const auto& [field, m] : num_values_) {
            (void)field;
            s.index_count += 1;
            s.memory_usage_bytes += m.size() * (sizeof(std::uint64_t) + sizeof(double));
        }
        // all_ids_ size
        s.memory_usage_bytes += all_ids_.getSizeInBytes();
        return s;
    }

    void clear() {
        std::unique_lock lock(mutex_);
        documents_.clear();
        tag_index_.clear();
        num_values_.clear();
        all_ids_ = roaring::Roaring();
    }

    auto save(const std::string& path) const -> std::expected<void, core::error> {
        std::shared_lock lock(mutex_);
        std::ofstream os(path, std::ios::binary);
        if (!os) {
            return std::unexpected(core::error{core::error_code::io_failed, "Failed to open file for writing", "metadata.save"});
        }
        const std::uint32_t version = 1;
        os.write(reinterpret_cast<const char*>(&version), sizeof(version));
        // Persist only documents; indexes are rebuilt on load
        std::uint64_t n_docs = static_cast<std::uint64_t>(documents_.size());
        os.write(reinterpret_cast<const char*>(&n_docs), sizeof(n_docs));
        for (const auto& [id, doc] : documents_) {
            (void)doc;
            os.write(reinterpret_cast<const char*>(&id), sizeof(id));
            std::uint64_t n_attrs = static_cast<std::uint64_t>(doc.attributes.size());
            os.write(reinterpret_cast<const char*>(&n_attrs), sizeof(n_attrs));
            for (const auto& [k, v] : doc.attributes) {
                write_string(os, k);
                // type tag
                std::uint8_t tag{};
                if (std::holds_alternative<std::string>(v)) tag = 0;
                else if (std::holds_alternative<double>(v)) tag = 1;
                else if (std::holds_alternative<std::int64_t>(v)) tag = 2;
                else if (std::holds_alternative<bool>(v)) tag = 3;
                os.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
                switch (tag) {
                    case 0: {
                        const auto& s = std::get<std::string>(v);
                        write_string(os, s);
                        break;
                    }
                    case 1: {
                        double d = std::get<double>(v);
                        os.write(reinterpret_cast<const char*>(&d), sizeof(d));
                        break;
                    }
                    case 2: {
                        std::int64_t i = std::get<std::int64_t>(v);
                        os.write(reinterpret_cast<const char*>(&i), sizeof(i));
                        break;
                    }
                    case 3: {
                        bool b = std::get<bool>(v);
                        os.write(reinterpret_cast<const char*>(&b), sizeof(b));
                        break;
                    }
                    default: break;
                }
            }
        }
        return {};
    }

    auto load(const std::string& path) -> std::expected<void, core::error> {
        std::unique_lock lock(mutex_);
        std::ifstream is(path, std::ios::binary);
        if (!is) {
            return std::unexpected(core::error{core::error_code::io_failed, "Failed to open file for reading", "metadata.load"});
        }
        std::uint32_t version{};
        if (!is.read(reinterpret_cast<char*>(&version), sizeof(version))) {
            return std::unexpected(core::error{core::error_code::data_integrity, "Corrupt header", "metadata.load"});
        }
        if (version != 1) {
            return std::unexpected(core::error{core::error_code::data_integrity, "Unsupported version", "metadata.load"});
        }
        std::uint64_t n_docs{};
        if (!is.read(reinterpret_cast<char*>(&n_docs), sizeof(n_docs))) {
            return std::unexpected(core::error{core::error_code::data_integrity, "Corrupt count", "metadata.load"});
        }
        documents_.clear();
        tag_index_.clear();
        num_values_.clear();
        all_ids_ = roaring::Roaring();
        for (std::uint64_t idx = 0; idx < n_docs; ++idx) {
            DocumentMetadata d{};
            if (!is.read(reinterpret_cast<char*>(&d.id), sizeof(d.id))) {
                return std::unexpected(core::error{core::error_code::data_integrity, "Corrupt id", "metadata.load"});
            }
            std::uint64_t n_attrs{};
            if (!is.read(reinterpret_cast<char*>(&n_attrs), sizeof(n_attrs))) {
                return std::unexpected(core::error{core::error_code::data_integrity, "Corrupt attr count", "metadata.load"});
            }
            for (std::uint64_t j = 0; j < n_attrs; ++j) {
                std::string key;
                if (!read_string(is, key)) {
                    return std::unexpected(core::error{core::error_code::data_integrity, "Corrupt key", "metadata.load"});
                }
                std::uint8_t tag{};
                if (!is.read(reinterpret_cast<char*>(&tag), sizeof(tag))) {
                    return std::unexpected(core::error{core::error_code::data_integrity, "Corrupt type tag", "metadata.load"});
                }
                switch (tag) {
                    case 0: {
                        std::string s;
                        if (!read_string(is, s)) {
                            return std::unexpected(core::error{core::error_code::data_integrity, "Corrupt string", "metadata.load"});
                        }
                        d.attributes.emplace(std::move(key), std::move(s));
                        break;
                    }
                    case 1: {
                        double val{};
                        if (!is.read(reinterpret_cast<char*>(&val), sizeof(val))) {
                            return std::unexpected(core::error{core::error_code::data_integrity, "Corrupt double", "metadata.load"});
                        }
                        d.attributes.emplace(std::move(key), val);
                        break;
                    }
                    case 2: {
                        std::int64_t val{};
                        if (!is.read(reinterpret_cast<char*>(&val), sizeof(val))) {
                            return std::unexpected(core::error{core::error_code::data_integrity, "Corrupt int64", "metadata.load"});
                        }
                        d.attributes.emplace(std::move(key), val);
                        break;
                    }
                    case 3: {
                        bool val{};
                        if (!is.read(reinterpret_cast<char*>(&val), sizeof(val))) {
                            return std::unexpected(core::error{core::error_code::data_integrity, "Corrupt bool", "metadata.load"});
                        }
                        d.attributes.emplace(std::move(key), val);
                        break;
                    }
                    default:
                        return std::unexpected(core::error{core::error_code::data_integrity, "Unknown type tag", "metadata.load"});
                }
            }
            // Insert and index
            documents_.emplace(d.id, d);
            if (d.id <= std::numeric_limits<std::uint32_t>::max()) {
                all_ids_.add(static_cast<std::uint32_t>(d.id));
            }
            index_document_locked(d);
        }
        return {};
    }

private:
    // Storage
    MetadataIndexConfig config_{};
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::uint64_t, DocumentMetadata> documents_;

    // Indexes
    std::unordered_map<std::string, std::unordered_map<std::string, roaring::Roaring>> tag_index_;
    std::unordered_map<std::string, std::unordered_map<std::uint64_t, double>> num_values_;

    roaring::Roaring all_ids_{};

    void index_document_locked(const DocumentMetadata& doc) {
        const auto did = static_cast<std::uint32_t>(doc.id);
        for (const auto& [k, v] : doc.attributes) {
            if (std::holds_alternative<std::string>(v)) {
                const auto& s = std::get<std::string>(v);
                tag_index_[k][s].add(did);
            } else if (std::holds_alternative<bool>(v)) {
                tag_index_[k][bool_to_str(std::get<bool>(v))].add(did);
            } else if (std::holds_alternative<std::int64_t>(v)) {
                // Index as numeric and also as categorical string (for equality filters)
                const auto ival = std::get<std::int64_t>(v);
                num_values_[k][doc.id] = static_cast<double>(ival);
                tag_index_[k][std::to_string(ival)].add(did);
            } else if (std::holds_alternative<double>(v)) {
                const auto dval = std::get<double>(v);
                num_values_[k][doc.id] = dval;
            }
        }
    }

    void remove_locked(std::uint64_t id) {
        auto it = documents_.find(id);
        if (it == documents_.end()) return;
        const auto did = static_cast<std::uint32_t>(id);
        const auto& doc = it->second;
        // Remove from tag bitmaps
        for (const auto& [k, v] : doc.attributes) {
            if (std::holds_alternative<std::string>(v)) {
                const auto& s = std::get<std::string>(v);
                auto fit = tag_index_.find(k);
                if (fit != tag_index_.end()) {
                    auto vit = fit->second.find(s);
                    if (vit != fit->second.end()) vit->second.remove(did);
                }
            } else if (std::holds_alternative<bool>(v)) {
                auto fit = tag_index_.find(k);
                if (fit != tag_index_.end()) {
                    auto vit = fit->second.find(bool_to_str(std::get<bool>(v)));
                    if (vit != fit->second.end()) vit->second.remove(did);
                }
            } else if (std::holds_alternative<std::int64_t>(v)) {
                auto fit = tag_index_.find(k);
                if (fit != tag_index_.end()) {
                    auto vit = fit->second.find(std::to_string(std::get<std::int64_t>(v)));
                    if (vit != fit->second.end()) vit->second.remove(did);
                }
                auto nit = num_values_.find(k);
                if (nit != num_values_.end()) {
                    nit->second.erase(id);
                }
            } else if (std::holds_alternative<double>(v)) {
                auto nit = num_values_.find(k);
                if (nit != num_values_.end()) {
                    nit->second.erase(id);
                }
            }
        }
        documents_.erase(it);
        all_ids_.remove(did);
    }

    auto compile_locked(const filter_expr& expr) const -> std::expected<roaring::Roaring, core::error> {
        return std::visit([this](const auto& node) -> std::expected<roaring::Roaring, core::error> {
            using T = std::decay_t<decltype(node)>;
            if constexpr (std::is_same_v<T, term>) {
                auto fit = tag_index_.find(node.field);
                if (fit == tag_index_.end()) return roaring::Roaring();
                auto vit = fit->second.find(node.value);
                if (vit == fit->second.end()) return roaring::Roaring();
                return vit->second; // copy-on-write
            } else if constexpr (std::is_same_v<T, range>) {
                roaring::Roaring bm;
                auto nit = num_values_.find(node.field);
                if (nit != num_values_.end()) {
                    for (const auto& [id, value] : nit->second) {
                        if (value >= node.min_value && value <= node.max_value) {
                            if (id <= std::numeric_limits<std::uint32_t>::max())
                                bm.add(static_cast<std::uint32_t>(id));
                        }
                    }
                }
                return bm;
            } else if constexpr (std::is_same_v<T, filter_expr::and_t>) {
                if (node.children.empty()) {
                    // and([]) == true => full set
                    return all_ids_;
                }
                auto acc_e = compile_locked(node.children[0]);
                if (!acc_e) return acc_e;
                auto acc = std::move(*acc_e);
                for (std::size_t i = 1; i < node.children.size(); ++i) {
                    auto rhs_e = compile_locked(node.children[i]);
                    if (!rhs_e) return rhs_e;
                    acc &= *rhs_e; // inplace and
                }
                return acc;
            } else if constexpr (std::is_same_v<T, filter_expr::or_t>) {
                if (node.children.empty()) {
                    // or([]) == false => empty set
                    return roaring::Roaring();
                }
                auto acc_e = compile_locked(node.children[0]);
                if (!acc_e) return acc_e;
                auto acc = std::move(*acc_e);
                for (std::size_t i = 1; i < node.children.size(); ++i) {
                    auto rhs_e = compile_locked(node.children[i]);
                    if (!rhs_e) return rhs_e;
                    acc |= *rhs_e; // inplace or
                }
                return acc;
            } else if constexpr (std::is_same_v<T, filter_expr::not_t>) {
                if (node.children.empty()) {
                    // not([]) == true => full set
                    return all_ids_;
                }
                auto child_e = compile_locked(node.children[0]);
                if (!child_e) return child_e;
                auto res = all_ids_;
                res -= *child_e; // andnot
                return res;
            }
            return roaring::Roaring();
        }, expr.node);
    }
};

// Public API forwarding
MetadataStore::MetadataStore() : impl_(std::make_unique<Impl>(MetadataIndexConfig{})) {}
MetadataStore::MetadataStore(MetadataIndexConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}
MetadataStore::~MetadataStore() = default;
MetadataStore::MetadataStore(MetadataStore&&) noexcept = default;
MetadataStore& MetadataStore::operator=(MetadataStore&&) noexcept = default;

auto MetadataStore::add(const DocumentMetadata& doc) -> std::expected<void, core::error> { return impl_->add(doc); }
auto MetadataStore::add_batch(const std::vector<DocumentMetadata>& docs) -> std::expected<void, core::error> { return impl_->add_batch(docs); }
auto MetadataStore::update(const DocumentMetadata& doc) -> std::expected<void, core::error> { return impl_->update(doc); }
auto MetadataStore::remove(std::uint64_t id) -> std::expected<void, core::error> { return impl_->remove(id); }
auto MetadataStore::evaluate_filter(const filter_expr& expr) const -> std::expected<roaring::Roaring, core::error> { return impl_->evaluate_filter(expr); }
auto MetadataStore::search(const filter_expr& expr, std::size_t limit) const -> std::expected<std::vector<std::uint64_t>, core::error> { return impl_->search(expr, limit); }
auto MetadataStore::get(std::uint64_t id) const -> std::expected<DocumentMetadata, core::error> { return impl_->get(id); }
auto MetadataStore::exists(std::uint64_t id) const -> bool { return impl_->exists(id); }
auto MetadataStore::get_all_ids() const -> roaring::Roaring { return impl_->get_all_ids(); }
auto MetadataStore::build_indexes() -> std::expected<void, core::error> { return impl_->build_indexes(); }
auto MetadataStore::get_stats() const -> Stats { return impl_->get_stats(); }
void MetadataStore::clear() { impl_->clear(); }
auto MetadataStore::save(const std::string& path) const -> std::expected<void, core::error> { return impl_->save(path); }
auto MetadataStore::load(const std::string& path) -> std::expected<void, core::error> { return impl_->load(path); }

// Utilities
namespace utils {

auto kv_to_filter(const std::unordered_map<std::string, std::string>& kvs) -> filter_expr {
    filter_expr::and_t a{};
    a.children.reserve(kvs.size());
    for (const auto& [k, v] : kvs) {
        a.children.push_back(filter_expr{term{k, v}});
    }
    return filter_expr{std::move(a)};
}

auto parse_filter_json(const std::string& json) -> std::expected<filter_expr, core::error> {
    // JSON parsing is out-of-scope without a dependency; return unavailable
    return std::unexpected(core::error{core::error_code::unavailable, "JSON parsing not enabled", "metadata.parse_filter_json"});
}

static void to_json_impl(const filter_expr& e, std::string& out) {
    if (std::holds_alternative<term>(e.node)) {
        const auto& t = std::get<term>(e.node);
        out += "{\"term\":{\"field\":\"" + t.field + "\",\"value\":\"" + t.value + "\"}}";
    } else if (std::holds_alternative<range>(e.node)) {
        const auto& r = std::get<range>(e.node);
        out += "{\"range\":{\"field\":\"" + r.field + "\",\"min\":";
        out += std::to_string(r.min_value);
        out += ",\"max\":";
        out += std::to_string(r.max_value);
        out += "}}";
    } else if (std::holds_alternative<filter_expr::and_t>(e.node)) {
        const auto& a = std::get<filter_expr::and_t>(e.node);
        out += "{\"and\":[";
        for (std::size_t i = 0; i < a.children.size(); ++i) {
            if (i) out += ",";
            to_json_impl(a.children[i], out);
        }
        out += "]}";
    } else if (std::holds_alternative<filter_expr::or_t>(e.node)) {
        const auto& o = std::get<filter_expr::or_t>(e.node);
        out += "{\"or\":[";
        for (std::size_t i = 0; i < o.children.size(); ++i) {
            if (i) out += ",";
            to_json_impl(o.children[i], out);
        }
        out += "]}";
    } else if (std::holds_alternative<filter_expr::not_t>(e.node)) {
        const auto& n = std::get<filter_expr::not_t>(e.node);
        out += "{\"not\":[";
        for (std::size_t i = 0; i < n.children.size(); ++i) {
            if (i) out += ",";
            to_json_impl(n.children[i], out);
        }
        out += "]}";
    }
}

auto filter_to_json(const filter_expr& expr) -> std::string {
    std::string out;
    to_json_impl(expr, out);
    return out;
}

} // namespace utils

} // namespace vesper::metadata
