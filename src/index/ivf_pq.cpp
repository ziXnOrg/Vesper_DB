#include "vesper/index/ivf_pq.hpp"
#include "vesper/index/kmeans_elkan.hpp"
#include "vesper/index/rabitq_quantizer.hpp"
#include "vesper/index/hnsw.hpp"
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
#include "vesper/index/pq_fastscan.hpp"
namespace vesper::index {
    using PqImpl = FastScanPq;
}
#else
#include "vesper/index/pq_simple.hpp"
namespace vesper::index {
    using PqImpl = SimplePq;
}
#endif
#include "vesper/kernels/dispatch.hpp"
#include "vesper/kernels/batch_distances.hpp"
#include "vesper/core/memory_pool.hpp"
#include "vesper/core/cpu_features.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
#include <immintrin.h>
#endif

#include <algorithm>
#include <numeric>
#include <queue>
#include <thread>
#include <fstream>
#include <cstring>
#include <array>
#include <random>

#include <filesystem>
#include <chrono>

#include <iostream>
#include <limits>
#include <cstdlib>
#include <functional>
#include "expected"

#include <atomic>

#ifdef VESPER_HAS_ZSTD
#include <zstd.h>
#endif
#ifdef VESPER_HAS_CBLAS
#include <cblas.h>
#endif



// Max size for metadata JSON blob (64 KiB)
static constexpr std::size_t kMaxMetadataSize = 64u * 1024u;

// Platform-specific includes for memory-mapped I/O
#if defined(_WIN32)
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #include <windows.h>
#else
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
#endif

// --- Metadata JSON validation helpers (UTF-8 + structural caps) ---
namespace {
    using vesper::core::error;
    using vesper::core::error_code;

    static bool is_valid_utf8(const char* s, std::size_t len) {
        const unsigned char* p = reinterpret_cast<const unsigned char*>(s);
        const unsigned char* e = p + len;
        while (p < e) {
            unsigned char c = *p++;
            if (c < 0x80) continue; // ASCII
            if ((c >> 5) == 0x6) { // 2-byte
                if (p >= e) return false; unsigned char c1 = *p++;
                if ((c1 & 0xC0) != 0x80) return false;
                if (c < 0xC2) return false; // overlong
            } else if ((c >> 4) == 0xE) { // 3-byte
                if (e - p < 2) return false; unsigned char c1 = *p++; unsigned char c2 = *p++;
                if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80) return false;
                if (c == 0xE0 && c1 < 0xA0) return false; // overlong
                if (c == 0xED && c1 >= 0xA0) return false; // surrogates
            } else if ((c >> 3) == 0x1E) { // 4-byte
                if (e - p < 3) return false; unsigned char c1 = *p++; unsigned char c2 = *p++; unsigned char c3 = *p++;
                if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80) return false;
                if (c == 0xF0 && c1 < 0x90) return false; // overlong
                if (c == 0xF4 && c1 > 0x8F) return false; // > U+10FFFF
                if (c > 0xF4) return false;
            } else {
                return false;
            }
        }
        return true;
    }

    struct JsonCaps { std::size_t max_depth; std::size_t max_keys; };

    static std::expected<void, error> validate_json_structure(std::string_view json, JsonCaps caps) {
        if (!is_valid_utf8(json.data(), json.size())) {
            return std::vesper_unexpected(error{error_code::config_invalid, "Metadata JSON is not valid UTF-8", "ivf_pq"});
        }
        // Lightweight JSON validator with depth/key caps.
        const char* s = json.data(); const char* const end = s + json.size();
        auto skip_ws = [&](){ while (s < end && (*s==' '||*s=='\n'||*s=='\r'||*s=='\t')) ++s; };
        auto parse_hex4 = [&](){ for (int i=0;i<4;++i){ if (s>=end) return false; char ch=*s++; if(!((ch>='0'&&ch<='9')||(ch>='A'&&ch<='F')||(ch>='a'&&ch<='f'))) return false;} return true; };
        auto parse_string = [&](){ if (s>=end || *s!='"') return false; ++s; while (s<end){ char ch=*s++; if (ch=='"') return true; if (ch=='\\'){ if (s>=end) return false; char e=*s++; switch(e){ case '"': case '\\': case '/': case 'b': case 'f': case 'n': case 'r': case 't': break; case 'u': if (!parse_hex4()) return false; break; default: return false; } } else { /* UTF-8 already validated */ } } return false; };
        auto parse_number = [&](){ const char* b=s; if (s<end && *s=='-') ++s; if (s>=end) return false; if (*s=='0'){ ++s; } else { if (!(*s>='1'&&*s<='9')) return false; while (s<end && *s>='0'&&*s<='9') ++s; } if (s<end && *s=='.'){ ++s; if (s>=end||!(*s>='0'&&*s<='9')) return false; while (s<end && *s>='0'&&*s<='9') ++s; } if (s<end && (*s=='e'||*s=='E')){ ++s; if (s<end && (*s=='+'||*s=='-')) ++s; if (s>=end||!(*s>='0'&&*s<='9')) return false; while (s<end && *s>='0'&&*s<='9') ++s; } return s>b; };

        std::size_t depth = 0; std::size_t key_count = 0;
        std::vector<char> stack; stack.reserve(16);

        std::function<bool()> parse_value; // forward decl
        auto parse_array = [&]() -> bool {
            if (*s != '[') return false; ++s; ++depth; if (depth > caps.max_depth) return false; skip_ws();
            if (s<end && *s==']'){ ++s; --depth; return true; }
            while (s<end){ if (!parse_value()) return false; skip_ws(); if (s>=end) return false; if (*s==','){ ++s; skip_ws(); continue; } if (*s==']'){ ++s; --depth; return true; } return false; }
            return false;
        };
        auto parse_object = [&]() -> bool {
            if (*s != '{') return false; ++s; ++depth; if (depth > caps.max_depth) return false; skip_ws();
            if (s<end && *s=='}'){ ++s; --depth; return true; }
            while (s<end){ if (!parse_string()) return false; ++key_count; if (key_count > caps.max_keys) return false; skip_ws(); if (s>=end || *s!=':') return false; ++s; skip_ws(); if (!parse_value()) return false; skip_ws(); if (s>=end) return false; if (*s==','){ ++s; skip_ws(); continue; } if (*s=='}'){ ++s; --depth; return true; } return false; }
            return false;
        };
        parse_value = [&]() -> bool {
            skip_ws(); if (s>=end) return false; char c=*s;
            if (c=='"') return parse_string();
            if (c=='{') return parse_object();
            if (c=='[') return parse_array();
            if (c=='t'){ if (end-s<4) return false; return (s[0]=='t'&&s[1]=='r'&&s[2]=='u'&&s[3]=='e') ? (s+=4, true) : false; }
            if (c=='f'){ if (end-s<5) return false; return (s[0]=='f'&&s[1]=='a'&&s[2]=='l'&&s[3]=='s'&&s[4]=='e') ? (s+=5, true) : false; }
            if (c=='n'){ if (end-s<4) return false; return (s[0]=='n'&&s[1]=='u'&&s[2]=='l'&&s[3]=='l') ? (s+=4, true) : false; }
            return parse_number();
        };

        skip_ws(); if (!parse_value()) {
            return std::vesper_unexpected(error{error_code::config_invalid, "Metadata JSON is structurally invalid", "ivf_pq"});
        }
        skip_ws(); if (s != end) {
            return std::vesper_unexpected(error{error_code::config_invalid, "Trailing characters after JSON", "ivf_pq"});
        }
        return {};
    }
} // namespace


// Small fixed-size top-K buffer for k<=64 to avoid heap overhead
struct TopKBuffer {
    explicit TopKBuffer(std::size_t K) : K_(K) {
        dists.assign(K_, std::numeric_limits<float>::infinity());
        ids.assign(K_, 0ull);
    }
    inline void consider(float dist, std::uint64_t id) {
        if (count_ < K_) {
            dists[count_] = dist; ids[count_] = id;
            if (count_ == 0 || dist > worst_val_) { worst_val_ = dist; worst_idx_ = count_; }
            ++count_;
            return;
        }
        if (dist >= worst_val_) return;
        dists[worst_idx_] = dist; ids[worst_idx_] = id;
        // Recompute worst slot (K is small)
        worst_idx_ = 0; worst_val_ = dists[0];
        for (std::size_t i = 1; i < K_; ++i) {
            if (dists[i] > worst_val_) { worst_val_ = dists[i]; worst_idx_ = i; }
        }
    }
    auto to_sorted_pairs() const -> std::vector<std::pair<std::uint64_t, float>> {
        std::vector<std::pair<std::uint64_t, float>> out;
        out.reserve(count_);
        for (std::size_t i = 0; i < count_; ++i) out.emplace_back(ids[i], dists[i]);
        std::sort(out.begin(), out.end(), [](const auto& a, const auto& b){ return a.second < b.second; });
        return out;
    }
    std::size_t size() const noexcept { return count_; }
private:
    std::size_t K_;
    std::size_t count_{0};
    std::vector<float> dists;
    std::vector<std::uint64_t> ids;
    std::size_t worst_idx_{0};
    float worst_val_{std::numeric_limits<float>::infinity()};
};

namespace vesper::index {

/** \brief Internal implementation of IVF-PQ index. */
class IvfPqIndex::Impl {
public:
    Impl() = default;
    ~Impl() = default;

    /** \brief Training state and parameters. */
    struct TrainingState {
        bool trained{false};
        std::size_t dim{0};
        std::size_t dsub{0};
        IvfPqTrainParams params;
        AlignedCentroidBuffer coarse_centroids{1, 1};
        std::unique_ptr<PqImpl> pq;
        std::unique_ptr<RaBitQuantizer> rabitq;  // RaBitQ quantizer
        std::vector<float> rotation_matrix;  // For OPQ
        std::string metadata_json;            // Optional user metadata (JSON)
        bool using_rabitq{false};  // Track which quantizer is active
    } state_;

    /** \brief Inverted list entry. */
    struct InvertedListEntry {
        std::uint64_t id;
        PqCode code;  // Standard PQ code
        QuantizedVector rabitq_code;  // RaBitQ code (if using RaBitQ)
    };

    /** \brief Inverted lists for each coarse centroid. */
    std::vector<std::vector<InvertedListEntry>> inverted_lists_;

    /** \brief Optional HNSW over coarse centroids for fast assignment. */
    std::unique_ptr<HnswIndex> centroid_hnsw_;


    // KD-tree over coarse centroids (built at train, used for exact fast assignment)
    struct KDNode {
        int left{-1}, right{-1};
        std::uint32_t split_dim{0};
        float split_val{0};
        std::uint32_t begin{0}, end{0}; // range into kd_order_
        bool leaf{true};
        std::vector<float> center; // mean of centroids in this subtree (size = dim)
        float radius2{0.0f};       // max squared distance from center to any centroid in subtree
        std::array<float, 8> center_proj{}; // projected center (p=8)
        float radius_proj2{0.0f};           // projected radius^2 (p=8)
        std::vector<float> bb_min;          // axis-aligned bbox min (size = dim)
        std::vector<float> bb_max;          // axis-aligned bbox max (size = dim)
    };
    std::vector<KDNode> kd_nodes_;
    std::vector<std::uint32_t> kd_order_;
    AlignedCentroidBuffer kd_leaf_centroids_{1, 1};
    int kd_root_{-1};
    std::uint32_t kd_leaf_size_{256};
    std::vector<float> kd_proj_P_; // PCA rows (p=8) row-major size = 8*dim, orthonormal rows

    // Projection-based coarse assignment state (when coarse_assigner==Projection)
    std::uint32_t proj_dim_{16};                 // effective projection dim
    std::vector<float> proj_rows_;               // row-major [proj_dim_ x dim], orthonormal rows
    std::vector<float> proj_centroids_rm_;       // row-major [nlist x proj_dim_]
    std::vector<float> proj_centroid_norms_;     // [nlist] = ||proj_centroid||^2
    std::vector<float> proj_centroids_pack8_;    // packed panels [ceil(nlist/8) x proj_dim_ x 8] for AVX microkernel

    void kd_build_();
    std::uint32_t kd_nearest_(const float* vec) const;
    void kd_assign_batch_(const float* data, std::size_t n, std::uint32_t* out_assignments) const;

    // Build projection rows from coarse centroids and precompute projected centroids
    void build_projection_();

    /** \brief Statistics. */
    std::size_t n_vectors_{0};
    std::mutex lists_mutex_;

    // ANN assignment counters
    std::atomic<std::uint64_t> ann_assignments_{0};
    std::atomic<std::uint64_t> ann_validated_{0};
    std::atomic<std::uint64_t> ann_mismatches_{0};

    // KD-tree instrumentation (mutable to allow updates in const methods)
    mutable std::atomic<std::uint64_t> kd_nodes_pushed_{0};
    mutable std::atomic<std::uint64_t> kd_nodes_popped_{0};
    mutable std::atomic<std::uint64_t> kd_leaves_scanned_{0};
    // KD timing accumulators (nanoseconds, aggregated across queries)
    mutable std::atomic<std::uint64_t> kd_trav_ns_{0};
    mutable std::atomic<std::uint64_t> kd_leaf_ns_{0};

        // Timing accumulators (nanoseconds)
        std::atomic<std::uint64_t> t_assign_ns_{0};
        std::atomic<std::uint64_t> t_encode_ns_{0};
        std::atomic<std::uint64_t> t_lists_ns_{0};



    // Optional metadata schema validator hook
    std::function<std::expected<void, core::error>(std::string_view)> metadata_validator_{};

    /** \brief Train the index. */
    auto train(const float* data, std::size_t dim, std::size_t n,
               const IvfPqTrainParams& params)
        -> std::expected<IvfPqTrainStats, core::error>;

    /** \brief Add vectors to index. */
    auto add(const std::uint64_t* ids, const float* data, std::size_t n)
        -> std::expected<void, core::error>;

    /** \brief Search for nearest neighbors. */
    auto search(const float* query, const IvfPqSearchParams& params) const
        -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error>;

    /** \brief Batch search. */
    auto search_batch(const float* queries, std::size_t n_queries,
                      const IvfPqSearchParams& params) const
        -> std::expected<std::vector<std::vector<std::pair<std::uint64_t, float>>>, core::error>;

    // Expose rotation helpers for outer class reconstruction paths
    auto apply_rotation(const float* input, float* output, std::size_t n) const -> void;
    auto apply_rotation_T(const float* input, float* output, std::size_t n) const -> void;


private:
    /** \brief Train coarse quantizer. */
    auto train_coarse_quantizer(const float* data, std::size_t n, std::size_t dim)
        -> std::expected<void, core::error>;

    /** \brief Train product quantizer. */
    auto train_product_quantizer(const float* data, std::size_t n)
        -> std::expected<void, core::error>;

    /** \brief Learn OPQ rotation. */
    auto learn_opq_rotation(const float* data, std::size_t n)
        -> std::expected<void, core::error>;



    /** \brief Find nearest coarse centroids. */
    auto find_nearest_centroids(const float* query, std::uint32_t nprobe) const
        -> std::vector<std::pair<std::uint32_t, float>>;

    /** \brief Compute ADC distance. */
    auto compute_adc_distance(const float* query, const PqCode& code,
                             const AlignedCentroidBuffer& luts) const -> float;
};

auto IvfPqIndex::Impl::train(const float* data, std::size_t dim, std::size_t n,
                             const IvfPqTrainParams& params)
    -> std::expected<IvfPqTrainStats, core::error> {
    using core::error;
    using core::error_code;

    if (n < params.nlist) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Not enough training data for nlist centroids",
            "ivf_pq"
        });
    }

    if (dim % params.m != 0) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Dimension must be divisible by number of subquantizers",
            "ivf_pq"
        });
    }

    const auto start_time = std::chrono::steady_clock::now();

    state_.params = params;
    state_.dim = dim;
    state_.dsub = dim / params.m;

    // Step 1: Train coarse quantizer
    if (auto result = train_coarse_quantizer(data, n, dim); !result.has_value()) {
        return std::vesper_unexpected(result.error());
    }

    // Step 2: Learn OPQ rotation if enabled and PCA init selected (on residuals)
    if (params.use_opq && params.opq_init == OpqInit::PCA) {
        // Compute residuals of training data with respect to nearest coarse centroid
        std::vector<float> residuals(n * state_.dim);
        const auto& ops = kernels::select_backend_auto();
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(n); ++i) {
            const float* vec = data + static_cast<std::size_t>(i) * state_.dim;
            float min_dist = std::numeric_limits<float>::max();
            std::uint32_t best_idx = 0;
            for (std::uint32_t c = 0; c < state_.params.nlist; ++c) {
                const float dist = ops.l2_sq(std::span(vec, state_.dim), state_.coarse_centroids.get_centroid(c));
                if (dist < min_dist) { min_dist = dist; best_idx = c; }
            }
            const float* centroid = state_.coarse_centroids[best_idx];
            float* out = residuals.data() + static_cast<std::size_t>(i) * state_.dim;
            for (std::size_t d = 0; d < state_.dim; ++d) {
                out[d] = vec[d] - centroid[d];
            }
        }
        // Clamp PCA initialization to a sample to avoid O(n*dim^2) cost on huge n
        const std::size_t cap = (state_.params.opq_sample_n > 0 ? state_.params.opq_sample_n : static_cast<std::size_t>(50000));
        const std::size_t n_pca = std::min<std::size_t>(n, cap);
        if (state_.params.verbose) {
            std::cout << "[OPQ] PCA init sample=" << n_pca << " of n=" << n << std::endl;
        }
        if (n_pca == n) {
            if (auto result = learn_opq_rotation(residuals.data(), n_pca); !result.has_value()) {
                return std::vesper_unexpected(result.error());
            }
        } else {
            std::vector<float> residuals_pca(n_pca * state_.dim);
            const double stride = static_cast<double>(n) / static_cast<double>(n_pca);
            for (std::size_t i = 0; i < n_pca; ++i) {
                std::size_t idx = static_cast<std::size_t>(i * stride);
                if (idx >= n) idx = n - 1;
                std::copy_n(residuals.data() + idx * state_.dim, state_.dim,
                            residuals_pca.data() + i * state_.dim);
            }
            if (auto result = learn_opq_rotation(residuals_pca.data(), n_pca); !result.has_value()) {
                return std::vesper_unexpected(result.error());
            }
        }
    }

    // Step 3: Train product quantizer
    if (auto result = train_product_quantizer(data, n); !result.has_value()) {
        return std::vesper_unexpected(result.error());
    }

    // Initialize inverted lists
    inverted_lists_.clear();
    inverted_lists_.resize(params.nlist);

    state_.trained = true;

    const auto end_time = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration<float>(end_time - start_time);

    IvfPqTrainStats stats;
    stats.train_time_sec = duration.count();
    stats.iterations = 25;  // From k-means config

    return stats;
}

auto IvfPqIndex::Impl::train_coarse_quantizer(const float* data, std::size_t n, std::size_t dim)
    -> std::expected<void, core::error> {

    // Use Elkan's k-means for efficient training
    KmeansElkan elkan;
    KmeansElkan::Config config{
        .k = state_.params.nlist,
        .max_iter = state_.params.max_iter,
        .epsilon = state_.params.epsilon,
        .seed = state_.params.seed,
        .verbose = state_.params.verbose
    };

    auto result = elkan.cluster(data, n, dim, config);
    if (!result.has_value()) {
        return std::vesper_unexpected(result.error());
    }

    // Store centroids in aligned buffer
    state_.coarse_centroids = AlignedCentroidBuffer(state_.params.nlist, dim);
    state_.coarse_centroids.from_vectors(result->centroids);

    // Optionally build a small HNSW index over coarse centroids to accelerate assignments
    if (state_.params.use_centroid_ann && state_.params.coarse_assigner == CoarseAssigner::HNSW) {
        centroid_hnsw_ = std::make_unique<HnswIndex>();
        HnswBuildParams hparams;
        hparams.M = 16;
        hparams.efConstruction = state_.params.centroid_ann_ef_construction;
        hparams.extend_candidates = true;
        auto init_res = centroid_hnsw_->init(dim, hparams, state_.params.nlist);
        if (!init_res.has_value()) { return std::vesper_unexpected(init_res.error()); }
        for (std::uint32_t c = 0; c < state_.params.nlist; ++c) {
            auto add_res = centroid_hnsw_->add(c, state_.coarse_centroids[c]);
            if (!add_res.has_value()) { return std::vesper_unexpected(add_res.error()); }
        }
    } else {
        centroid_hnsw_.reset();
    }

    // Build projection structures if selected
    if (state_.params.coarse_assigner == CoarseAssigner::Projection) {
        build_projection_();
    } else {
        proj_rows_.clear(); proj_centroids_rm_.clear(); proj_centroid_norms_.clear(); proj_centroids_pack8_.clear();
    }

    // Build KD-tree over coarse centroids if selected
    if (state_.params.coarse_assigner == CoarseAssigner::KDTree) {
        kd_build_();
    } else {
        kd_nodes_.clear(); kd_order_.clear(); kd_root_ = -1;
    }

    return {};

}

void IvfPqIndex::Impl::kd_build_() {
    // Optional env override for leaf size and allow quick sweeps externally
    if (const char* v = std::getenv("VESPER_KD_LEAF_SIZE")) {
        const unsigned s = static_cast<unsigned>(std::strtoul(v, nullptr, 10));
        if (s >= 16 && s <= 1024) kd_leaf_size_ = static_cast<std::uint32_t>(s);
    }

    kd_order_.resize(state_.params.nlist);
    for (std::uint32_t i = 0; i < state_.params.nlist; ++i) kd_order_[i] = i;
    kd_nodes_.clear();
    kd_nodes_.reserve(state_.params.nlist * 2u);

    const std::size_t dim = state_.dim;
    auto& centroids = state_.coarse_centroids;

    // Compute PCA projection (p=8) over coarse centroids using power iteration (orthonormal rows)
    static constexpr std::size_t P = 8;
    kd_proj_P_.assign(P * dim, 0.0f);
    // Global mean (double for accumulation)
    std::vector<double> mu(dim, 0.0);
    for (std::uint32_t c = 0; c < state_.params.nlist; ++c) {
        const float* x = centroids[c];
        for (std::size_t d = 0; d < dim; ++d) mu[d] += static_cast<double>(x[d]);
    }
    const double Ninvd = (state_.params.nlist > 0) ? 1.0 / static_cast<double>(state_.params.nlist) : 0.0;
    for (std::size_t d = 0; d < dim; ++d) mu[d] *= Ninvd;

    auto dotf = [&](const float* a, const float* b) -> double {
        double s = 0.0; for (std::size_t d = 0; d < dim; ++d) s += static_cast<double>(a[d]) * static_cast<double>(b[d]); return s; };
    auto dotfd = [&](const float* a, const std::vector<double>& b) -> double {
        double s = 0.0; for (std::size_t d = 0; d < dim; ++d) s += static_cast<double>(a[d]) * b[d]; return s; };
    auto dotdd = [&](const std::vector<double>& a, const std::vector<double>& b) -> double {
        double s = 0.0; for (std::size_t d = 0; d < dim; ++d) s += a[d] * b[d]; return s; };

    auto cov_mv = [&](const std::vector<float>& v, std::vector<float>& outf) {
        std::vector<double> out(dim, 0.0);
        // mu_v = mu dot v
        double mu_v = 0.0; for (std::size_t d = 0; d < dim; ++d) mu_v += mu[d] * static_cast<double>(v[d]);
        for (std::uint32_t c = 0; c < state_.params.nlist; ++c) {
            const float* x = centroids[c];
            double y = 0.0; for (std::size_t d = 0; d < dim; ++d) y += static_cast<double>(x[d]) * static_cast<double>(v[d]);
            const double alpha = y - mu_v; // (x - mu) dot v
            for (std::size_t d = 0; d < dim; ++d) out[d] += (static_cast<double>(x[d]) - mu[d]) * alpha;
        }
        const double denom = (state_.params.nlist > 1) ? 1.0 / static_cast<double>(state_.params.nlist - 1) : 0.0;
        for (std::size_t d = 0; d < dim; ++d) outf[d] = static_cast<float>(out[d] * denom);
    };

    std::mt19937 rng(42);
    std::normal_distribution<float> ndist(0.0f, 1.0f);
    std::vector<float> v(dim), w(dim);
    auto gs_ortho = [&](std::vector<float>& vec){
        for (std::size_t k = 0; k < P; ++k) {
            const float* pk = kd_proj_P_.data() + k * dim;
            double norm_pk = 0.0; for (std::size_t d = 0; d < dim; ++d) norm_pk += static_cast<double>(pk[d]) * pk[d];
            if (norm_pk < 1e-12) continue;
            double proj = 0.0; for (std::size_t d = 0; d < dim; ++d) proj += static_cast<double>(vec[d]) * pk[d];
            const double scale = proj / norm_pk;
            for (std::size_t d = 0; d < dim; ++d) vec[d] = static_cast<float>(static_cast<double>(vec[d]) - scale * pk[d]);
        }
    };

    for (std::size_t k = 0; k < P; ++k) {
        // init v
        for (std::size_t d = 0; d < dim; ++d) v[d] = ndist(rng);
        gs_ortho(v);
        // normalize
        double nv = 0.0; for (float x : v) nv += static_cast<double>(x) * x; nv = std::sqrt(nv) + 1e-12;
        for (float& x : v) x = static_cast<float>(x / nv);
        // power iterations
        for (int it = 0; it < 12; ++it) {
            cov_mv(v, w);
            gs_ortho(w);
            double nw = 0.0; for (float x : w) nw += static_cast<double>(x) * x; nw = std::sqrt(nw) + 1e-12;
            for (std::size_t d = 0; d < dim; ++d) v[d] = static_cast<float>(w[d] / nw);
        }
        // store row k
        float* pk = kd_proj_P_.data() + k * dim;
        // final orthonormalize & copy
        gs_ortho(v);
        double nv2 = 0.0; for (float x : v) nv2 += static_cast<double>(x) * x; nv2 = std::sqrt(nv2) + 1e-12;
        for (std::size_t d = 0; d < dim; ++d) pk[d] = static_cast<float>(v[d] / nv2);
    }


    std::function<int(std::uint32_t,std::uint32_t)> build = [&](std::uint32_t begin, std::uint32_t end) -> int {
        const std::uint32_t count = end - begin;
        const int node_id = static_cast<int>(kd_nodes_.size());
        kd_nodes_.push_back(KDNode{});
        KDNode& node = kd_nodes_.back();
        node.begin = begin; node.end = end;

        // Compute mean and variance for this node's range
        std::vector<double> mean(dim, 0.0), m2(dim, 0.0);
        std::size_t processed = 0;
        for (std::uint32_t it = begin; it < end; ++it) {
            const std::uint32_t c = kd_order_[it];
            const float* ctr = centroids[c];
            ++processed;
            for (std::size_t d = 0; d < dim; ++d) {
                const double x = static_cast<double>(ctr[d]);
                const double delta = x - mean[d];
                mean[d] += delta / static_cast<double>(processed);
                const double delta2 = x - mean[d];
                m2[d] += delta * delta2;
            }
        }

        // Set node center (mean)
        node.center.resize(dim);
        for (std::size_t d = 0; d < dim; ++d) node.center[d] = static_cast<float>(mean[d]);
        // Projected center (p=8) via PCA rows
        for (std::size_t k = 0; k < 8; ++k) {
            const float* pk = kd_proj_P_.data() + k * dim;
            double s = 0.0; for (std::size_t d = 0; d < dim; ++d) s += static_cast<double>(pk[d]) * static_cast<double>(node.center[d]);
            node.center_proj[k] = static_cast<float>(s);
        }

        // Compute bounding sphere radius^2 (full), projected radius^2, and axis-aligned bbox
        float r2 = 0.0f;
        float rp2 = 0.0f;
        node.bb_min.assign(dim, std::numeric_limits<float>::infinity());
        node.bb_max.assign(dim, -std::numeric_limits<float>::infinity());
        for (std::uint32_t it = begin; it < end; ++it) {
            const std::uint32_t c = kd_order_[it];
            const float* ctr = centroids[c];
            float acc = 0.0f;
            float accp = 0.0f;
            for (std::size_t d = 0; d < dim; ++d) {
                const float v = ctr[d];
                const float diff = v - node.center[d];
                acc += diff * diff;
                if (v < node.bb_min[d]) node.bb_min[d] = v;
                if (v > node.bb_max[d]) node.bb_max[d] = v;
            }
            // projected via PCA rows
            for (std::size_t k = 0; k < 8; ++k) {
                const float* pk = kd_proj_P_.data() + k * dim;
                double s = 0.0; for (std::size_t d = 0; d < dim; ++d) s += static_cast<double>(pk[d]) * static_cast<double>(ctr[d]);
                const float diffp = static_cast<float>(s) - node.center_proj[k];
                accp += diffp * diffp;
            }
            if (acc > r2) r2 = acc;
            if (accp > rp2) rp2 = accp;
        }
        node.radius2 = r2;
        node.radius_proj2 = rp2;

        if (count <= kd_leaf_size_) {
            node.leaf = true; node.left = node.right = -1; node.split_dim = 0; node.split_val = 0.0f;
            return node_id;
        }

        // Choose split dimension by maximum variance
        std::uint32_t best_dim = 0; double best_var = -1.0;
        const std::size_t cnt = static_cast<std::size_t>(count);
        if (cnt > 1) {
            for (std::size_t d = 0; d < dim; ++d) {
                const double var = m2[d] / static_cast<double>(cnt - 1);
                if (var > best_var) { best_var = var; best_dim = static_cast<std::uint32_t>(d); }
            }
        }
        node.leaf = false;
        node.split_dim = best_dim;
        const std::uint32_t mid = begin + count / 2u;
        std::nth_element(kd_order_.begin() + begin, kd_order_.begin() + mid, kd_order_.begin() + end,
                         [&](std::uint32_t a, std::uint32_t b){ return centroids[a][node.split_dim] < centroids[b][node.split_dim]; });
        node.split_val = centroids[kd_order_[mid]][node.split_dim];
        node.left  = build(begin, mid);
        node.right = build(mid, end);
        return node_id;
    };

    kd_root_ = build(0u, state_.params.nlist);

    // Pack centroids in kd_order_ into a contiguous buffer for cache-friendly leaf scans
    kd_leaf_centroids_ = AlignedCentroidBuffer(state_.params.nlist, dim);
    for (std::uint32_t i = 0; i < state_.params.nlist; ++i) {
        std::memcpy(kd_leaf_centroids_[i], state_.coarse_centroids[kd_order_[i]], sizeof(float) * dim);
    }
}


void IvfPqIndex::Impl::build_projection_() {
    const std::size_t dim = state_.dim;
    proj_dim_ = std::max<std::uint32_t>(1u, std::min<std::uint32_t>(state_.params.projection_dim, static_cast<std::uint32_t>(dim)));
    // Allocate rows [proj_dim_ x dim]
    proj_rows_.assign(static_cast<std::size_t>(proj_dim_) * dim, 0.0f);

    // Compute mean of coarse centroids
    std::vector<double> mu(dim, 0.0);
    for (std::uint32_t c = 0; c < state_.params.nlist; ++c) {
        const float* x = state_.coarse_centroids[c];
        for (std::size_t d = 0; d < dim; ++d) mu[d] += static_cast<double>(x[d]);
    }
    const double Ninvd = (state_.params.nlist > 0) ? 1.0 / static_cast<double>(state_.params.nlist) : 0.0;
    for (std::size_t d = 0; d < dim; ++d) mu[d] *= Ninvd;

    // Covariance * v (without explicitly forming covariance)
    auto cov_mv = [&](const std::vector<float>& v, std::vector<float>& outf) {
        std::vector<double> out(dim, 0.0);
        double mu_v = 0.0; for (std::size_t d = 0; d < dim; ++d) mu_v += mu[d] * static_cast<double>(v[d]);
        for (std::uint32_t c = 0; c < state_.params.nlist; ++c) {
            const float* x = state_.coarse_centroids[c];
            double y = 0.0; for (std::size_t d = 0; d < dim; ++d) y += static_cast<double>(x[d]) * static_cast<double>(v[d]);
            const double alpha = y - mu_v; // (x - mu) dot v
            for (std::size_t d = 0; d < dim; ++d) out[d] += (static_cast<double>(x[d]) - mu[d]) * alpha;
        }
        const double denom = (state_.params.nlist > 1) ? 1.0 / static_cast<double>(state_.params.nlist - 1) : 0.0;
        for (std::size_t d = 0; d < dim; ++d) outf[d] = static_cast<float>(out[d] * denom);
    };

    // Power iteration to obtain top proj_dim_ orthonormal rows
    std::mt19937 rng(12345);
    std::normal_distribution<float> ndist(0.0f, 1.0f);
    std::vector<float> v(dim), w(dim);

    auto gs_ortho = [&](std::vector<float>& vec){
        for (std::uint32_t k = 0; k < proj_dim_; ++k) {
            const float* pk = proj_rows_.data() + static_cast<std::size_t>(k) * dim;
            double norm_pk = 0.0; for (std::size_t d = 0; d < dim; ++d) norm_pk += static_cast<double>(pk[d]) * pk[d];
            if (norm_pk < 1e-20) continue;
            double proj = 0.0; for (std::size_t d = 0; d < dim; ++d) proj += static_cast<double>(vec[d]) * pk[d];
            const double scale = proj / norm_pk;
            for (std::size_t d = 0; d < dim; ++d) vec[d] = static_cast<float>(static_cast<double>(vec[d]) - scale * pk[d]);
        }
    };

    for (std::uint32_t k = 0; k < proj_dim_; ++k) {
        for (std::size_t d = 0; d < dim; ++d) v[d] = ndist(rng);
        gs_ortho(v);
        double nv = 0.0; for (float x : v) nv += static_cast<double>(x) * x; nv = std::sqrt(nv) + 1e-12;
        for (float& x : v) x = static_cast<float>(x / nv);
        for (int it = 0; it < 12; ++it) {
            cov_mv(v, w);
            gs_ortho(w);
            double nw = 0.0; for (float x : w) nw += static_cast<double>(x) * x; nw = std::sqrt(nw) + 1e-12;
            for (std::size_t d = 0; d < dim; ++d) v[d] = static_cast<float>(w[d] / nw);
        }
        // store normalized row k
        gs_ortho(v);
        double nv2 = 0.0; for (float x : v) nv2 += static_cast<double>(x) * x; nv2 = std::sqrt(nv2) + 1e-12;
        float* pk = proj_rows_.data() + static_cast<std::size_t>(k) * dim;
        for (std::size_t d = 0; d < dim; ++d) pk[d] = static_cast<float>(v[d] / nv2);
    }

    // Project all centroids to proj_dim_
    proj_centroids_rm_.assign(static_cast<std::size_t>(state_.params.nlist) * proj_dim_, 0.0f);
    proj_centroid_norms_.assign(state_.params.nlist, 0.0f);
    for (std::uint32_t c = 0; c < state_.params.nlist; ++c) {
        const float* x = state_.coarse_centroids[c];
        float* yc = proj_centroids_rm_.data() + static_cast<std::size_t>(c) * proj_dim_;
        double norm = 0.0;
        for (std::uint32_t k = 0; k < proj_dim_; ++k) {
            const float* pk = proj_rows_.data() + static_cast<std::size_t>(k) * dim;
            double s = 0.0; for (std::size_t d = 0; d < dim; ++d) s += static_cast<double>(pk[d]) * static_cast<double>(x[d]);
            yc[k] = static_cast<float>(s);
            norm += s * s;
        }
        proj_centroid_norms_[c] = static_cast<float>(norm);
    }
    // Pack centroids into 8-wide panels for AVX microkernel
    proj_centroids_pack8_.assign(((state_.params.nlist + 7u)/8u) * static_cast<std::size_t>(proj_dim_) * 8u, 0.0f);
    {
        const std::size_t C = state_.params.nlist;
        const std::size_t blocks = (C + 7u) / 8u;
        for (std::size_t b = 0; b < blocks; ++b) {
            for (std::size_t k = 0; k < proj_dim_; ++k) {
                for (std::size_t lane = 0; lane < 8; ++lane) {
                    const std::size_t cj = b * 8 + lane;
                    float v = 0.0f;
                    if (cj < C) v = proj_centroids_rm_[cj * static_cast<std::size_t>(proj_dim_) + k];
                    proj_centroids_pack8_[b * static_cast<std::size_t>(proj_dim_) * 8 + k * 8 + lane] = v;
                }
            }
        }
    }

}

// Fast L2 microkernels for dim=128 to reduce per-centroid overhead in KD leaf scans
namespace {
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

inline float l2_128_scalar(const float* a, const float* b) noexcept {
    float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f;
    for (int i=0;i<128;i+=4){
        const float d0=a[i]-b[i];
        const float d1=a[i+1]-b[i+1];
        const float d2=a[i+2]-b[i+2];
        const float d3=a[i+3]-b[i+3];
        s0+=d0*d0; s1+=d1*d1; s2+=d2*d2; s3+=d3*d3;
    }
    return (s0+s1)+(s2+s3);
}

#ifdef __AVX512F__
inline float l2_128_avx512(const float* a, const float* b) noexcept {
    __m512 acc = _mm512_setzero_ps();
    for (int i=0;i<128;i+=16){
        __m512 va = _mm512_loadu_ps(a+i);
        __m512 vb = _mm512_loadu_ps(b+i);
        __m512 vd = _mm512_sub_ps(va, vb);
        acc = _mm512_fmadd_ps(vd, vd, acc);
    }
    alignas(64) float tmp[16];
    _mm512_store_ps(tmp, acc);
    float s=0.0f; for (int i=0;i<16;++i) s+=tmp[i];
    return s;
}
#endif

#ifdef __AVX2__
inline float l2_128_avx2(const float* a, const float* b) noexcept {
    __m256 acc0=_mm256_setzero_ps(), acc1=_mm256_setzero_ps(), acc2=_mm256_setzero_ps(), acc3=_mm256_setzero_ps();
    const float* pa=a; const float* pb=b;
    for (int i=0;i<128;i+=32){
        __m256 a0=_mm256_loadu_ps(pa+i+0),  b0=_mm256_loadu_ps(pb+i+0);
        __m256 d0=_mm256_sub_ps(a0,b0); acc0=_mm256_fmadd_ps(d0,d0,acc0);
        __m256 a1=_mm256_loadu_ps(pa+i+8),  b1=_mm256_loadu_ps(pb+i+8);
        __m256 d1=_mm256_sub_ps(a1,b1); acc1=_mm256_fmadd_ps(d1,d1,acc1);
        __m256 a2=_mm256_loadu_ps(pa+i+16), b2=_mm256_loadu_ps(pb+i+16);
        __m256 d2=_mm256_sub_ps(a2,b2); acc2=_mm256_fmadd_ps(d2,d2,acc2);
        __m256 a3=_mm256_loadu_ps(pa+i+24), b3=_mm256_loadu_ps(pb+i+24);
        __m256 d3=_mm256_sub_ps(a3,b3); acc3=_mm256_fmadd_ps(d3,d3,acc3);
    }
    acc0=_mm256_add_ps(acc0,acc1); acc2=_mm256_add_ps(acc2,acc3); acc0=_mm256_add_ps(acc0,acc2);
    alignas(32) float tmp[8]; _mm256_store_ps(tmp, acc0);
    float s=0.0f; for (int i=0;i<8;++i) s+=tmp[i];
    return s;
}
#endif

inline float l2_128_fast(const float* a, const float* b) noexcept {
#if defined(__AVX512F__)
    return l2_128_avx512(a,b);
#elif defined(__AVX2__)
    return l2_128_avx2(a,b);
#else
    return l2_128_scalar(a,b);
#endif
}

// Prunable variants using running threshold to early-abort
inline float l2_128_scalar_prune(const float* a, const float* b, float thresh) noexcept {
    float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f;
    for (int i=0;i<128;i+=4){
        const float d0=a[i]-b[i]; s0+=d0*d0; if ((s0+s1+s2+s3)>thresh) return thresh+1.0f;
        const float d1=a[i+1]-b[i+1]; s1+=d1*d1; if ((s0+s1+s2+s3)>thresh) return thresh+1.0f;
        const float d2=a[i+2]-b[i+2]; s2+=d2*d2; if ((s0+s1+s2+s3)>thresh) return thresh+1.0f;
        const float d3=a[i+3]-b[i+3]; s3+=d3*d3; if ((s0+s1+s2+s3)>thresh) return thresh+1.0f;
    }
    return (s0+s1)+(s2+s3);
}

#ifdef __AVX512F__
inline float l2_128_avx512_prune(const float* a, const float* b, float thresh) noexcept {
    __m512 acc = _mm512_setzero_ps();
    alignas(64) float tmp[16];
    float partial = 0.0f;
    for (int i=0;i<128;i+=32){
        __m512 va0 = _mm512_loadu_ps(a+i);
        __m512 vb0 = _mm512_loadu_ps(b+i);
        __m512 d0 = _mm512_sub_ps(va0, vb0);
        acc = _mm512_fmadd_ps(d0, d0, acc);
        __m512 va1 = _mm512_loadu_ps(a+i+16);
        __m512 vb1 = _mm512_loadu_ps(b+i+16);
        __m512 d1 = _mm512_sub_ps(va1, vb1);
        acc = _mm512_fmadd_ps(d1, d1, acc);
        _mm512_store_ps(tmp, acc);
        partial = 0.0f; for (int t=0;t<16;++t) partial += tmp[t];
        if (partial > thresh) return thresh + 1.0f;
    }
    return partial;
}
#endif

#ifdef __AVX2__
inline float l2_128_avx2_prune(const float* a, const float* b, float thresh) noexcept {
    __m256 acc0=_mm256_setzero_ps(), acc1=_mm256_setzero_ps(), acc2=_mm256_setzero_ps(), acc3=_mm256_setzero_ps();
    alignas(32) float tmp[8];
    float partial=0.0f;
    for (int i=0;i<128;i+=32){
        __m256 a0=_mm256_loadu_ps(a+i+0),  b0=_mm256_loadu_ps(b+i+0);
        __m256 d0=_mm256_sub_ps(a0,b0); acc0=_mm256_fmadd_ps(d0,d0,acc0);
        __m256 a1=_mm256_loadu_ps(a+i+8),  b1=_mm256_loadu_ps(b+i+8);
        __m256 d1=_mm256_sub_ps(a1,b1); acc1=_mm256_fmadd_ps(d1,d1,acc1);
        __m256 a2=_mm256_loadu_ps(a+i+16), b2=_mm256_loadu_ps(b+i+16);
        __m256 d2=_mm256_sub_ps(a2,b2); acc2=_mm256_fmadd_ps(d2,d2,acc2);
        __m256 a3=_mm256_loadu_ps(a+i+24), b3=_mm256_loadu_ps(b+i+24);
        __m256 d3=_mm256_sub_ps(a3,b3); acc3=_mm256_fmadd_ps(d3,d3,acc3);
        __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0,acc1), _mm256_add_ps(acc2,acc3));
        _mm256_store_ps(tmp, acc);
        partial = 0.0f; for (int t=0;t<8;++t) partial += tmp[t];
        if (partial > thresh) return thresh + 1.0f;
    }
    return partial;
}
#endif

inline float l2_128_fast_prune(const float* a, const float* b, float thresh) noexcept {
#if defined(__AVX512F__)
    return l2_128_avx512_prune(a,b,thresh);
#elif defined(__AVX2__)
    return l2_128_avx2_prune(a,b,thresh);
#else
    return l2_128_scalar_prune(a,b,thresh);
#endif
}

// Axis-aligned bounding box lower bound for a node
inline float kd_box_lb_ptr(const float* q, const float* bb_min, const float* bb_max, std::size_t dim) noexcept {
    float lb = 0.0f;
    for (std::size_t d=0; d<dim; ++d) {
        const float v = q[d];
        float diff = 0.0f;
        const float mn = bb_min[d];
        const float mx = bb_max[d];
        if (v < mn) diff = mn - v;
        else if (v > mx) diff = v - mx;
        lb += diff * diff;
    }
    return lb;
}

} // anonymous namespace


std::uint32_t IvfPqIndex::Impl::kd_nearest_(const float* vec) const {
    const auto& ops = kernels::select_backend_auto();
    const std::size_t dim = state_.dim;

    std::uint32_t best_idx = 0;
    float best = std::numeric_limits<float>::infinity();

    const bool use_proj = ([](){ const char* v = std::getenv("VESPER_KD_USE_PROJ"); return v && v[0] == '1'; })();

    // Precompute 8-D projection of query once (if enabled)
    float qproj[8] = {0};
    if (use_proj) {
        for (std::size_t k = 0; k < 8; ++k) {
            const float* pk = kd_proj_P_.data() + k * dim;
            double s = 0.0; for (std::size_t d = 0; d < dim; ++d) s += static_cast<double>(pk[d]) * static_cast<double>(vec[d]);
            qproj[k] = static_cast<float>(s);
        }
    }

    // Small fixed-capacity candidate set (best-first by lower bound)
    constexpr int CAP = 512;
    int cand_id[CAP];
    float cand_lb[CAP];
    int sz = 0;

    auto push = [&](int nid, float lb) {
        if (nid < 0) return;
        if (lb < 0.0f) lb = 0.0f;
        if (sz < CAP) { cand_id[sz] = nid; cand_lb[sz] = lb; ++sz; }
        else {
            int worst = 0;
            for (int i = 1; i < CAP; ++i) if (cand_lb[i] > cand_lb[worst]) worst = i;
            if (lb < cand_lb[worst]) { cand_id[worst] = nid; cand_lb[worst] = lb; }
        }
        kd_nodes_pushed_.fetch_add(1, std::memory_order_relaxed);
    };

    if (kd_root_ == -1) return 0u;
    {
        const KDNode& root = kd_nodes_[static_cast<std::size_t>(kd_root_)];
        float lb = 0.0f;
        if (use_proj) {
            float d2p = 0.0f;
            for (std::size_t k = 0; k < 8; ++k) { const float diff = qproj[k] - root.center_proj[k]; d2p += diff * diff; }
            lb = d2p - root.radius_proj2;
        } else {
            float d2f = kernels::select_backend_auto().l2_sq(std::span(vec, dim), std::span<const float>(root.center.data(), dim));
            lb = d2f - root.radius2;
        }
        // tighten with AABB lower bound
        const float lb_box = kd_box_lb_ptr(vec, root.bb_min.data(), root.bb_max.data(), dim);
        if (lb_box > lb) lb = lb_box;
        push(kd_root_, lb);
    }

    while (sz > 0) {
        int best_i = 0;
        for (int i = 1; i < sz; ++i) if (cand_lb[i] < cand_lb[best_i]) best_i = i;
        const float lb = cand_lb[best_i];
        const int nid = cand_id[best_i];
        // remove best_i
        cand_id[best_i] = cand_id[--sz];
        cand_lb[best_i] = cand_lb[sz];

        kd_nodes_popped_.fetch_add(1, std::memory_order_relaxed);
        if (lb >= best) continue; // prune

        const KDNode& node = kd_nodes_[static_cast<std::size_t>(nid)];
        // Re-evaluate bound at pop using full-dim (and proj if enabled)
        float lb_curr = 0.0f;
        if (use_proj) {
            float d2p_curr = 0.0f; for (std::size_t k = 0; k < 8; ++k) { const float diff = qproj[k] - node.center_proj[k]; d2p_curr += diff * diff; }
            lb_curr = d2p_curr - node.radius_proj2;
        }
        {
            float d2f_curr = kernels::select_backend_auto().l2_sq(std::span(vec, dim), std::span<const float>(node.center.data(), dim));
            float lbf_curr = d2f_curr - node.radius2;
            if (lbf_curr > lb_curr) lb_curr = lbf_curr;
        }
        // tighten with AABB bound
        {
            const float lb_box = kd_box_lb_ptr(vec, node.bb_min.data(), node.bb_max.data(), dim);
            if (lb_box > lb_curr) lb_curr = lb_box;
        }
        if (lb_curr >= best) continue;
        if (node.leaf) {
            if (dim == 128) {
                for (std::uint32_t it = node.begin; it < node.end; ++it) {
                    const float d = l2_128_fast_prune(vec, kd_leaf_centroids_[it], best);
                    if (d < best) { best = d; best_idx = kd_order_[it]; }
                }
            } else {
                const auto qspan = std::span(vec, dim);
                for (std::uint32_t it = node.begin; it < node.end; ++it) {
                    const float d = kernels::select_backend_auto().l2_sq(qspan, kd_leaf_centroids_.get_centroid(it));
                    if (d < best) { best = d; best_idx = kd_order_[it]; }
                }
            }
            kd_leaves_scanned_.fetch_add(static_cast<std::uint64_t>(node.end - node.begin), std::memory_order_relaxed);
            continue;
        }

        if (node.left != -1) {
            const KDNode& L = kd_nodes_[static_cast<std::size_t>(node.left)];
            float lbL = 0.0f;
            if (use_proj) {
                float d2pL = 0.0f; for (std::size_t k = 0; k < 8; ++k) { const float diff = qproj[k] - L.center_proj[k]; d2pL += diff * diff; }
                lbL = d2pL - L.radius_proj2;
            } else {
                float d2fL = kernels::select_backend_auto().l2_sq(std::span(vec, dim), std::span<const float>(L.center.data(), dim));
                lbL = d2fL - L.radius2;
            }
            const float lbL_box = kd_box_lb_ptr(vec, L.bb_min.data(), L.bb_max.data(), dim);
            if (lbL_box > lbL) lbL = lbL_box;
            if (lbL < best) push(node.left, lbL);
        }
        if (node.right != -1) {
            const KDNode& R = kd_nodes_[static_cast<std::size_t>(node.right)];
            float lbR = 0.0f;
            if (use_proj) {
                float d2pR = 0.0f; for (std::size_t k = 0; k < 8; ++k) { const float diff = qproj[k] - R.center_proj[k]; d2pR += diff * diff; }
                lbR = d2pR - R.radius_proj2;
            } else {
                float d2fR = kernels::select_backend_auto().l2_sq(std::span(vec, dim), std::span<const float>(R.center.data(), dim));
                lbR = d2fR - R.radius2;
            }
            const float lbR_box = kd_box_lb_ptr(vec, R.bb_min.data(), R.bb_max.data(), dim);
            if (lbR_box > lbR) lbR = lbR_box;
            if (lbR < best) push(node.right, lbR);
        }
    }

    return best_idx;
}


void IvfPqIndex::Impl::kd_assign_batch_(const float* data, std::size_t n, std::uint32_t* out_assignments) const {
    const std::size_t dim = state_.dim;
    if (kd_root_ == -1 || n == 0) return;

    const auto& ops = kernels::select_backend_auto();
    const std::size_t num_nodes = kd_nodes_.size();

    // Per-query best distance and index
    std::vector<float> best(n, std::numeric_limits<float>::infinity());
    std::vector<std::uint32_t> best_idx(n, 0u);

    // Precompute 8-D projections for all queries (PCA rows)
    std::vector<float> qproj(n * 8);
    for (std::size_t i = 0; i < n; ++i) {
        const float* qv = data + i * dim;
        float* qp = qproj.data() + i * 8;
        for (std::size_t k = 0; k < 8; ++k) {
            const float* pk = kd_proj_P_.data() + k * dim;
            double s = 0.0; for (std::size_t d = 0; d < dim; ++d) s += static_cast<double>(pk[d]) * static_cast<double>(qv[d]);
            qp[k] = static_cast<float>(s);
        }
    }

    // Per-node query lists and node lower-bound (min over queued queries)
    std::vector<std::vector<std::uint32_t>> node_q(num_nodes);
    std::vector<float> node_min_lb(num_nodes, std::numeric_limits<float>::infinity());

    // Min-heap of nodes by node_min_lb (implemented as simple arrays; sizes are small)
    std::vector<int> heap_ids; heap_ids.reserve(num_nodes);
    std::vector<float> heap_keys; heap_keys.reserve(num_nodes);
    auto heap_push = [&](int nid){
        heap_ids.push_back(nid); heap_keys.push_back(node_min_lb[static_cast<std::size_t>(nid)]);
    };
    auto heap_pop_min = [&](){
        if (heap_ids.empty()) return -1;
        int best_i = 0; float best_k = heap_keys[0];
        for (int i = 1; i < static_cast<int>(heap_ids.size()); ++i) {
            if (heap_keys[i] < best_k) { best_k = heap_keys[i]; best_i = i; }
        }
        const int nid = heap_ids[best_i];
        heap_ids[best_i] = heap_ids.back(); heap_ids.pop_back();
        heap_keys[best_i] = heap_keys.back(); heap_keys.pop_back();
        return nid;
    };
    auto consider_push_node = [&](int nid){
        if (nid < 0) return;
        const std::size_t u = static_cast<std::size_t>(nid);
        if (node_q[u].empty()) return;
        // If already in heap, we allow duplicates; processing will see empty list next time
        heap_push(nid);
    };

    // Initialize: assign all queries to root with their individual projected LB (p=8)
    const KDNode& root = kd_nodes_[static_cast<std::size_t>(kd_root_)];
    for (std::uint32_t q = 0; q < n; ++q) {
        const float* qp = qproj.data() + static_cast<std::size_t>(q) * 8;
        float d2p = 0.0f;
        for (std::size_t k = 0; k < 8; ++k) {
            const float diff = qp[k] - root.center_proj[k];
            d2p += diff * diff;
        }
        float lb = d2p - root.radius_proj2; if (lb < 0.0f) lb = 0.0f;
        node_q[static_cast<std::size_t>(kd_root_)].push_back(q);
        if (lb < node_min_lb[static_cast<std::size_t>(kd_root_)]) node_min_lb[static_cast<std::size_t>(kd_root_)] = lb;
    }
    consider_push_node(kd_root_);

    // Process nodes best-first
    while (true) {
        const int nid = heap_pop_min();
        if (nid < 0) break;
        const std::size_t u = static_cast<std::size_t>(nid);
        auto& qlist = node_q[u];
        if (qlist.empty()) continue;
        const KDNode& node = kd_nodes_[u];

        if (node.leaf) {
            // Filter queries by per-query projected bound and scan per-query with early cutoff
            std::vector<std::uint32_t> qs; qs.reserve(qlist.size());
            for (std::uint32_t qi : qlist) {
                const float* qp = qproj.data() + static_cast<std::size_t>(qi) * 8;
                float d2p = 0.0f;
                for (std::size_t k = 0; k < 8; ++k) {
                    const float diff = qp[k] - node.center_proj[k];
                    d2p += diff * diff;
                }
                float lb = d2p - node.radius_proj2; if (lb < 0.0f) lb = 0.0f;
                if (lb < best[qi]) qs.push_back(qi);
            }
            qlist.clear();
            node_min_lb[u] = std::numeric_limits<float>::infinity();
            if (qs.empty()) continue;

            const std::uint32_t L = node.end - node.begin;
            // Per-query SIMD distance using backend kernels; avoid packing
            for (std::size_t ii = 0; ii < qs.size(); ++ii) {
                const std::uint32_t qi = qs[ii];
                const float* qv = data + static_cast<std::size_t>(qi) * dim;
                float b = best[qi];
                std::uint32_t bidx = best_idx[qi];
                if (dim == 128) {
                    for (std::uint32_t j = 0; j < L; ++j) {
                        const std::uint32_t idx = node.begin + j;
                        const float v = l2_128_fast_prune(qv, kd_leaf_centroids_[idx], b);
                        if (v < b) { b = v; bidx = kd_order_[idx]; }
                    }
                } else {
                    const auto qspan = std::span(qv, dim);
                    for (std::uint32_t j = 0; j < L; ++j) {
                        const std::uint32_t idx = node.begin + j;
                        const float v = kernels::select_backend_auto().l2_sq(qspan, kd_leaf_centroids_.get_centroid(idx));
                        if (v < b) { b = v; bidx = kd_order_[idx]; }
                    }
                }
                best[qi] = b; best_idx[qi] = bidx;
            }
            continue;
        }

        // Internal node: distribute queries to children using projected bounds
        const int Lid = node.left;
        const int Rid = node.right;
        float minL = std::numeric_limits<float>::infinity();
        float minR = std::numeric_limits<float>::infinity();
        for (std::uint32_t qi : qlist) {
            const float* qp = qproj.data() + static_cast<std::size_t>(qi) * 8;
            const float* qv = data + static_cast<std::size_t>(qi) * dim;
            if (Lid != -1) {
                const KDNode& L = kd_nodes_[static_cast<std::size_t>(Lid)];
                float d2pL = 0.0f;
                for (std::size_t k = 0; k < 8; ++k) { const float diff = qp[k] - L.center_proj[k]; d2pL += diff * diff; }
                float lbL = d2pL - L.radius_proj2; if (lbL < 0.0f) lbL = 0.0f;
                if (lbL < best[qi]) {
                    const float d2fL = kernels::select_backend_auto().l2_sq(std::span(qv, dim), std::span<const float>(L.center.data(), dim));
                    const float lbfL = d2fL - L.radius2;
                    lbL = (lbfL > lbL ? lbfL : lbL);
                    node_q[static_cast<std::size_t>(Lid)].push_back(qi); if (lbL < minL) minL = lbL;
                }
            }
            if (Rid != -1) {
                const KDNode& R = kd_nodes_[static_cast<std::size_t>(Rid)];
                float d2pR = 0.0f;
                for (std::size_t k = 0; k < 8; ++k) { const float diff = qp[k] - R.center_proj[k]; d2pR += diff * diff; }
                float lbR = d2pR - R.radius_proj2; if (lbR < 0.0f) lbR = 0.0f;
                if (lbR < best[qi]) {
                    const float d2fR = kernels::select_backend_auto().l2_sq(std::span(qv, dim), std::span<const float>(R.center.data(), dim));
                    const float lbfR = d2fR - R.radius2;
                    lbR = (lbfR > lbR ? lbfR : lbR);
                    node_q[static_cast<std::size_t>(Rid)].push_back(qi); if (lbR < minR) minR = lbR;
                }
            }
        }
        // Clear current node queue and reset bound
        qlist.clear(); node_min_lb[u] = std::numeric_limits<float>::infinity();
        if (Lid != -1 && !node_q[static_cast<std::size_t>(Lid)].empty()) { node_min_lb[static_cast<std::size_t>(Lid)] = std::min(node_min_lb[static_cast<std::size_t>(Lid)], minL); consider_push_node(Lid); }
        if (Rid != -1 && !node_q[static_cast<std::size_t>(Rid)].empty()) { node_min_lb[static_cast<std::size_t>(Rid)] = std::min(node_min_lb[static_cast<std::size_t>(Rid)], minR); consider_push_node(Rid); }
    }

    // Write out assignments
    for (std::size_t i = 0; i < n; ++i) out_assignments[i] = best_idx[i];
}


auto IvfPqIndex::Impl::train_product_quantizer(const float* data, std::size_t n)
    -> std::expected<void, core::error> {

    // Compute residuals of training data with respect to nearest coarse centroid
    std::vector<float> residuals(n * state_.dim);

    const auto& ops = kernels::select_backend_auto();

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(n); ++i) {
        const float* vec = data + static_cast<std::size_t>(i) * state_.dim;
        // Find nearest centroid (original space)
        float min_dist = std::numeric_limits<float>::max();
        std::uint32_t best_idx = 0;
        for (std::uint32_t c = 0; c < state_.params.nlist; ++c) {
            const float dist = ops.l2_sq(
                std::span(vec, state_.dim),
                state_.coarse_centroids.get_centroid(c)
            );
            if (dist < min_dist) { min_dist = dist; best_idx = c; }
        }
        const float* centroid = state_.coarse_centroids[best_idx];
        float* out = residuals.data() + static_cast<std::size_t>(i) * state_.dim;
        for (std::size_t d = 0; d < state_.dim; ++d) {
            out[d] = vec[d] - centroid[d];
        }
    }

    // Configure PQ
    FastScanPqConfig pq_config{
        .m = state_.params.m,
        .nbits = state_.params.nbits,
        .block_size = 32,
        .use_avx512 = core::decide_use_avx512_from_env_and_cpu()
    };

    // Helper: compute average reconstruction error in original residual space
    auto compute_error = [&](PqImpl& pq, const float* train_ptr, bool trained_on_rotated) -> double {
        std::vector<std::uint8_t> codes(n * state_.params.m);
        pq.encode(train_ptr, n, codes.data());
        std::vector<float> decoded(n * state_.dim);
        pq.decode(codes.data(), n, decoded.data());

        // Map decoded back to original residual space if needed
        if (trained_on_rotated && state_.params.use_opq && !state_.rotation_matrix.empty()) {
            std::vector<float> tmp(n * state_.dim);
            apply_rotation_T(decoded.data(), tmp.data(), n);
            decoded.swap(tmp);
        }

        const auto& ops2 = kernels::select_backend_auto();
        double total = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            total += ops2.l2_sq(
                std::span(decoded.data() + i * state_.dim, state_.dim),
                std::span(residuals.data() + i * state_.dim, state_.dim)
            );
        }
        return total / static_cast<double>(n);
    };

    // Always train unrotated baseline
    auto pq_unrot = std::make_unique<PqImpl>(pq_config);
    if (auto r = pq_unrot->train(residuals.data(), n, state_.dim); !r.has_value()) {
        return std::vesper_unexpected(r.error());
    }
    const double err_unrot = compute_error(*pq_unrot, residuals.data(), false);
    if (state_.params.verbose) {
        std::cout << "[OPQ] baseline unrot error=" << err_unrot << std::endl;
    }


    // If OPQ requested, perform alternating optimization to learn R and PQ
    std::unique_ptr<PqImpl> pq_opq_best;
    double err_opq_best = std::numeric_limits<double>::infinity();
    std::vector<float> R_best;

    if (state_.params.use_opq) {
        // Initialize rotation (identity if none exists)
        if (state_.rotation_matrix.empty()) {
            state_.rotation_matrix.resize(state_.dim * state_.dim, 0.0f);
            for (std::size_t d = 0; d < state_.dim; ++d) state_.rotation_matrix[d * state_.dim + d] = 1.0f;
        }

        // Alternating optimization for a few iterations
        const std::uint32_t iters = std::max<std::uint32_t>(1, std::min<std::uint32_t>(state_.params.opq_iter, 3));
        const std::size_t cap = state_.params.opq_sample_n > 0 ? state_.params.opq_sample_n : static_cast<std::size_t>(20000);
        const std::size_t n_opq = std::min<std::size_t>(n, cap);
        std::vector<float> rotated_samp(n_opq * state_.dim);
        std::vector<std::uint8_t> codes; codes.reserve(n_opq * state_.params.m);
        std::vector<float> Y(n_opq * state_.dim);

        const auto& ops2 = kernels::select_backend_auto();

        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();
        const std::uint64_t max_opq_ms = 120000; // time budget guard for OPQ alternating (ms)
        double prev_err = std::numeric_limits<double>::infinity();
        std::uint32_t stall = 0;
        const double rel_eps = 1e-4;      // minimum relative improvement per iteration
        const std::uint32_t patience = 2;  // stop after this many consecutive stalls

        for (std::uint32_t it = 0; it < iters; ++it) {
            // Step 1: Rotate residual sample
            apply_rotation(residuals.data(), rotated_samp.data(), n_opq);

            // Step 2: Train PQ on rotated sample
            auto pq_temp = std::make_unique<PqImpl>(pq_config);
            auto tr = pq_temp->train(rotated_samp.data(), n_opq, state_.dim);
            if (!tr.has_value()) { break; }

            // Step 3: Compute sample reconstruction error in original space
            codes.assign(n_opq * state_.params.m, 0);
            pq_temp->encode(rotated_samp.data(), n_opq, codes.data());
            pq_temp->decode(codes.data(), n_opq, Y.data());

            // Map Y back via R^T to original residual space
            std::vector<float> Y_back(n_opq * state_.dim);
            apply_rotation_T(Y.data(), Y_back.data(), n_opq);

            double total = 0.0;
            for (std::size_t i = 0; i < n_opq; ++i) {
                total += ops2.l2_sq(
                    std::span(Y_back.data() + i * state_.dim, state_.dim),
                    std::span(residuals.data() + i * state_.dim, state_.dim)
                );
            }
            const double err_it = total / static_cast<double>(n_opq);
            if (state_.params.verbose) {
                std::cout << "[OPQ] iter " << it << ": sample_err=" << err_it
                          << ", baseline=" << err_unrot << std::endl;
            }
            if (err_it < err_opq_best) {
                err_opq_best = err_it;
                R_best = state_.rotation_matrix; // snapshot best R
            }
            // Convergence and timeout guards
            if (std::isfinite(prev_err)) {
                const double denom = std::max(prev_err, 1e-12);
                const double rel_impr = (prev_err - err_it) / denom;
                if (rel_impr < rel_eps) { ++stall; } else { stall = 0; }
            }
            prev_err = err_it;
            if (stall >= patience) {
                if (state_.params.verbose) {
                    std::cout << "[OPQ] early-stop: stalled for " << stall
                              << " iters (rel_eps=" << rel_eps << ")" << std::endl;
                }
                break;
            }
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
            if (elapsed_ms > max_opq_ms) {
                if (state_.params.verbose) {
                    std::cout << "[OPQ] early-stop: time budget exceeded (" << elapsed_ms
                              << " ms > " << max_opq_ms << " ms)" << std::endl;
                }
                break;
            }


            // Step 4: Update rotation via Procrustes on sample: argmin_R ||X R - Y||
            std::vector<double> C(state_.dim * state_.dim, 0.0);
            for (std::size_t i = 0; i < n_opq; ++i) {
                const float* x = residuals.data() + i * state_.dim;
                const float* y = Y.data() + i * state_.dim; // rotated space
                for (std::size_t j = 0; j < state_.dim; ++j) {
                    const double xv = static_cast<double>(x[j]);
                    for (std::size_t k = 0; k < state_.dim; ++k) {
                        C[j * state_.dim + k] += xv * static_cast<double>(y[k]);
                    }
                }
            }
            const double invn = 1.0 / static_cast<double>(n_opq);
            for (double& v : C) v *= invn;

            // Jacobi SVD of C to get R = U * V^T
            const std::size_t d = state_.dim;
            std::vector<double> U(d * d, 0.0), V(d * d, 0.0), A = C;
            for (std::size_t i = 0; i < d; ++i) { U[i * d + i] = 1.0; V[i * d + i] = 1.0; }
            const int max_sweeps = 20; const double tol = 1e-10;
            for (int sweep = 0; sweep < max_sweeps; ++sweep) {
                double off = 0.0;
                for (std::size_t p = 0; p + 1 < d; ++p) {
                    for (std::size_t q = p + 1; q < d; ++q) {
                        double app = 0.0, aqq = 0.0, apq = 0.0;
                        for (std::size_t i = 0; i < d; ++i) {
                            app += A[i * d + p] * A[i * d + p];
                            aqq += A[i * d + q] * A[i * d + q];
                            apq += A[i * d + p] * A[i * d + q];
                        }
                        off += apq * apq;
                        if (std::abs(apq) < tol) continue;
                        double tau = (aqq - app) / (2.0 * apq);
                        double t = (tau >= 0) ? 1.0 / (tau + std::sqrt(1.0 + tau * tau)) :
                                                -1.0 / (-tau + std::sqrt(1.0 + tau * tau));
                        double c = 1.0 / std::sqrt(1.0 + t * t);
                        double s = t * c;
                        for (std::size_t i = 0; i < d; ++i) {
                            double aip = A[i * d + p];
                            double aiq = A[i * d + q];
                            A[i * d + p] = c * aip - s * aiq;
                            A[i * d + q] = s * aip + c * aiq;
                        }
                        for (std::size_t i = 0; i < d; ++i) {
                            double vip = V[i * d + p];
                            double viq = V[i * d + q];
                            V[i * d + p] = c * vip - s * viq;
                            V[i * d + q] = s * vip + c * viq;
                        }
                    }
                }
                if (off < tol * d * d) break;
            }
            // Compute U from A's columns normalization
            std::vector<double> S(d, 0.0);
            for (std::size_t i = 0; i < d; ++i) {
                for (std::size_t j = 0; j < d; ++j) S[i] += A[j * d + i] * A[j * d + i];
                S[i] = std::sqrt(S[i]);
                if (S[i] > tol) {
                    for (std::size_t j = 0; j < d; ++j) U[j * d + i] = A[j * d + i] / S[i];
                }
            }
            // R = U * V^T
            std::vector<float> R_new(d * d, 0.0f);
            for (std::size_t i = 0; i < d; ++i) {
                for (std::size_t j = 0; j < d; ++j) {
                    double sum = 0.0;
                    for (std::size_t k = 0; k < d; ++k) sum += U[i * d + k] * V[j * d + k];
                    R_new[i * d + j] = static_cast<float>(sum);
                }
            }
            // Replace current rotation and continue
            state_.rotation_matrix.swap(R_new);
        }

        // Restore best found rotation
        if (!R_best.empty()) state_.rotation_matrix = std::move(R_best);

        // If OPQ improved error, train final PQ on full rotated residuals
        if (err_opq_best < std::numeric_limits<double>::infinity()) {
            std::vector<float> rotated_full(n * state_.dim);
            apply_rotation(residuals.data(), rotated_full.data(), n);
            auto pq_final = std::make_unique<PqImpl>(pq_config);
            auto trf = pq_final->train(rotated_full.data(), n, state_.dim);
            if (trf.has_value()) {
                pq_opq_best = std::move(pq_final);
            } else {
                err_opq_best = std::numeric_limits<double>::infinity();
            }
        }
    }

    if (state_.params.verbose) {
        const bool kept = (err_opq_best < err_unrot);
        std::cout << "[OPQ] decision: " << (kept ? "kept" : "disabled")
                  << ", err_opq=" << (std::isfinite(err_opq_best) ? err_opq_best : -1.0)
                  << ", err_unrot=" << err_unrot << std::endl;
    }

    // Choose best between OPQ (if any) and unrotated
    if (err_opq_best < err_unrot) {
        state_.pq = std::move(pq_opq_best);
    } else {
        state_.pq = std::move(pq_unrot);
        state_.rotation_matrix.clear();
    }

    return {};
}

auto IvfPqIndex::Impl::learn_opq_rotation(const float* data, std::size_t n)
    -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;

    if (n < state_.dim * 2) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Not enough data for OPQ rotation learning",
            "ivf_pq"
        });
    }

    const std::size_t dim = state_.dim;
    const std::size_t matrix_size = dim * dim;

    // Step 1: Compute covariance matrix of residuals
    std::vector<double> covariance(matrix_size, 0.0);
    std::vector<double> mean(dim, 0.0);

    // Compute mean
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t d = 0; d < dim; ++d) {
            mean[d] += data[i * dim + d];
        }
    }
    for (std::size_t d = 0; d < dim; ++d) {
        mean[d] /= n;
    }

    // Compute covariance
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> centered(dim);
        for (std::size_t d = 0; d < dim; ++d) {
            centered[d] = data[i * dim + d] - mean[d];
        }

        for (std::size_t j = 0; j < dim; ++j) {
            for (std::size_t k = 0; k < dim; ++k) {
                covariance[j * dim + k] += centered[j] * centered[k];
            }
        }
    }

    // Normalize covariance
    for (std::size_t i = 0; i < matrix_size; ++i) {
        covariance[i] /= (n - 1);
    }

    // Step 2: Eigendecomposition for PCA rotation using power iteration
    std::vector<double> eigenvectors(matrix_size, 0.0);
    std::vector<double> eigenvalues(dim);

    // Initialize with identity
    for (std::size_t i = 0; i < dim; ++i) {
        eigenvectors[i * dim + i] = 1.0;
    }

    // Power iteration for each eigenvector
    const int max_iters = 100;
    const double tol = 1e-6;

    for (std::size_t evec_idx = 0; evec_idx < dim; ++evec_idx) {
        std::vector<double> v(dim);
        // Random initialization
        for (std::size_t i = 0; i < dim; ++i) {
            v[i] = (i == evec_idx) ? 1.0 : 0.1 * (i % 3 - 1);
        }

        // Orthogonalize against previous eigenvectors
        for (std::size_t prev = 0; prev < evec_idx; ++prev) {
            double dot = 0.0;
            for (std::size_t i = 0; i < dim; ++i) {
                dot += v[i] * eigenvectors[prev * dim + i];
            }
            for (std::size_t i = 0; i < dim; ++i) {
                v[i] -= dot * eigenvectors[prev * dim + i];
            }
        }

        // Normalize
        double norm = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            norm += v[i] * v[i];
        }
        norm = std::sqrt(norm);
        for (std::size_t i = 0; i < dim; ++i) {
            v[i] /= norm;
        }

        // Power iteration
        double prev_eigenvalue = 0.0;
        for (int iter = 0; iter < max_iters; ++iter) {
            std::vector<double> Av(dim, 0.0);

            // Av = covariance * v
            for (std::size_t i = 0; i < dim; ++i) {
                for (std::size_t j = 0; j < dim; ++j) {
                    Av[i] += covariance[i * dim + j] * v[j];
                }
            }

            // Orthogonalize against previous eigenvectors
            for (std::size_t prev = 0; prev < evec_idx; ++prev) {
                double dot = 0.0;
                for (std::size_t i = 0; i < dim; ++i) {
                    dot += Av[i] * eigenvectors[prev * dim + i];
                }
                for (std::size_t i = 0; i < dim; ++i) {
                    Av[i] -= dot * eigenvectors[prev * dim + i];
                }
            }

            // Compute eigenvalue (Rayleigh quotient)
            double eigenvalue = 0.0;
            double v_norm_sq = 0.0;
            for (std::size_t i = 0; i < dim; ++i) {
                eigenvalue += v[i] * Av[i];
                v_norm_sq += v[i] * v[i];
            }
            eigenvalue /= v_norm_sq;

            // Check convergence
            if (std::abs(eigenvalue - prev_eigenvalue) < tol) {
                break;
            }
            prev_eigenvalue = eigenvalue;

            // Normalize Av and update v
            norm = 0.0;
            for (std::size_t i = 0; i < dim; ++i) {
                norm += Av[i] * Av[i];
            }
            norm = std::sqrt(norm);
            for (std::size_t i = 0; i < dim; ++i) {
                v[i] = Av[i] / norm;
            }
        }

        // Store eigenvector and eigenvalue
        for (std::size_t i = 0; i < dim; ++i) {
            eigenvectors[evec_idx * dim + i] = v[i];
        }
        eigenvalues[evec_idx] = prev_eigenvalue;
    }

    // Step 3: Build rotation matrix (eigenvectors form orthonormal basis)
    state_.rotation_matrix.resize(matrix_size);
    for (std::size_t i = 0; i < matrix_size; ++i) {
        state_.rotation_matrix[i] = static_cast<float>(eigenvectors[i]);
    }

    // Verify orthogonality (optional check)
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = i + 1; j < dim; ++j) {



            double dot = 0.0;
            for (std::size_t k = 0; k < dim; ++k) {
                dot += state_.rotation_matrix[i * dim + k] *
                       state_.rotation_matrix[j * dim + k];
            }
            if (std::abs(dot) > 0.01) {
                // Fallback to identity if orthogonalization failed
                state_.rotation_matrix.clear();
                state_.rotation_matrix.resize(matrix_size, 0.0f);
                for (std::size_t k = 0; k < dim; ++k) {
                    state_.rotation_matrix[k * dim + k] = 1.0f;
                }
                break;
            }
        }
    }

    return {};
}

auto IvfPqIndex::Impl::apply_rotation(const float* input, float* output, std::size_t n) const
    -> void {
    if (state_.rotation_matrix.empty()) {
        std::memcpy(output, input, n * state_.dim * sizeof(float));
        return;
    }

    // Matrix multiplication: output = input * rotation
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(n); ++i) {
        const float* vec_in = input + i * state_.dim;
        float* vec_out = output + i * state_.dim;

        for (std::size_t j = 0; j < state_.dim; ++j) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < state_.dim; ++k) {
                sum += vec_in[k] * state_.rotation_matrix[k * state_.dim + j];
            }
            vec_out[j] = sum;
        }
    }
}

auto IvfPqIndex::Impl::add(const std::uint64_t* ids, const float* data, std::size_t n)
    -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;

    if (!state_.trained) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Index must be trained before adding vectors",
            "ivf_pq"
        });
    }

    // 1) Assign each vector to nearest coarse centroid (use HNSW if available)
    std::vector<std::uint32_t> assignments(n);
    const auto& ops = kernels::select_backend_auto();

    // Optional timing
    const bool kTimingAdd = (state_.params.timings_enabled || state_.params.verbose || ([](){ const char* v = std::getenv("VESPER_TIMING"); return v && v[0] != '0' && v[0] != '\0'; })());
    auto t_assign_start = kTimingAdd ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    // Reset KD counters for this add() run
    kd_nodes_pushed_.store(0, std::memory_order_relaxed);
    kd_nodes_popped_.store(0, std::memory_order_relaxed);
    kd_leaves_scanned_.store(0, std::memory_order_relaxed);
    kd_trav_ns_.store(0, std::memory_order_relaxed);
    kd_leaf_ns_.store(0, std::memory_order_relaxed);

    if (state_.params.use_centroid_ann && state_.params.coarse_assigner == CoarseAssigner::KDTree && kd_root_ != -1) {
        const bool use_batch = ([](){ const char* v = std::getenv("VESPER_KD_BATCH"); return v && v[0] == '1'; })();
        if (use_batch) {
            // KD-tree exact nearest centroid assignment (batched, serial within assignment)
            kd_assign_batch_(data, n, assignments.data());

            // Optional correctness validation against brute-force on a sample
            if (state_.params.validate_ann_assignment) {
                float rate = std::clamp(state_.params.validate_ann_sample_rate, 0.0f, 1.0f);
                if (rate > 0.0f) {
                    for (std::size_t i = 0; i < n; ++i) {
                        const float* vec = data + i * state_.dim;
                        const std::uint32_t aidx = assignments[i];
                        const std::uint32_t h = static_cast<std::uint32_t>(i) * 2654435761u;
                        const float u = (h & 0x00FFFFFFu) * (1.0f / 16777216.0f);
                        if (u < rate) {
                            float min_dist_bf = std::numeric_limits<float>::max();
                            std::uint32_t bf_idx = 0;
                            for (std::uint32_t c = 0; c < state_.params.nlist; ++c) {
                                const float dist = ops.l2_sq(std::span(vec, state_.dim), state_.coarse_centroids.get_centroid(c));
                                if (dist < min_dist_bf) { min_dist_bf = dist; bf_idx = c; }
                            }
                            ann_validated_.fetch_add(1, std::memory_order_relaxed);
                            if (bf_idx != aidx) {
                                ann_mismatches_.fetch_add(1, std::memory_order_relaxed);
                            }
                        }
                    }
                }
            }
        } else {
            // KD-tree exact nearest centroid assignment (parallel per-query)
            #pragma omp parallel for
            for (int i = 0; i < static_cast<int>(n); ++i) {
                const std::size_t idx = static_cast<std::size_t>(i);
                const float* vec = data + idx * state_.dim;
                const std::uint32_t best_idx_val = kd_nearest_(vec);
                assignments[idx] = best_idx_val;

                if (state_.params.validate_ann_assignment) {
                    float rate = std::clamp(state_.params.validate_ann_sample_rate, 0.0f, 1.0f);
                    if (rate > 0.0f) {
                        const std::uint32_t h = static_cast<std::uint32_t>(idx) * 2654435761u;
                        const float u = (h & 0x00FFFFFFu) * (1.0f / 16777216.0f);
                        if (u < rate) {
                            float min_dist_bf = std::numeric_limits<float>::max();
                            std::uint32_t bf_idx = 0;
                            for (std::uint32_t c = 0; c < state_.params.nlist; ++c) {
                                const float dist = ops.l2_sq(std::span(vec, state_.dim), state_.coarse_centroids.get_centroid(c));
                                if (dist < min_dist_bf) { min_dist_bf = dist; bf_idx = c; }
                            }
                            ann_validated_.fetch_add(1, std::memory_order_relaxed);
                            if (bf_idx != best_idx_val) {
                                ann_mismatches_.fetch_add(1, std::memory_order_relaxed);
                            }
                        }
                    }
                }
            }
        }
        ann_assignments_.fetch_add(static_cast<std::uint64_t>(n), std::memory_order_relaxed);
    } else if (centroid_hnsw_ && state_.params.use_centroid_ann && state_.params.coarse_assigner == CoarseAssigner::HNSW) {
        // Batch HNSW ANN search over the entire add() batch with top-L refine
        HnswSearchParams sp; sp.efSearch = state_.params.centroid_ann_ef_search; sp.k = std::max<std::uint32_t>(1u, std::min(state_.params.centroid_ann_refine_k, state_.params.nlist));
        auto batch_res = centroid_hnsw_->search_batch(data, n, sp);
        if (batch_res.has_value()) {
            const auto& all = *batch_res;
            for (std::size_t i = 0; i < n; ++i) {
                std::uint32_t best_idx = 0;
                const float* vec = data + i * state_.dim;
                if (!all[i].empty()) {
                    const std::size_t L = std::min<std::size_t>(all[i].size(), state_.params.centroid_ann_refine_k);
                    float min_dist = std::numeric_limits<float>::max();
                    for (std::size_t j = 0; j < L; ++j) {
                        const std::uint32_t cand = static_cast<std::uint32_t>(all[i][j].first);
                        const float d = ops.l2_sq(std::span(vec, state_.dim), state_.coarse_centroids.get_centroid(cand));
                        if (d < min_dist) { min_dist = d; best_idx = cand; }
                    }
                } else {
                    // Fallback to brute-force on failure
                    float min_dist = std::numeric_limits<float>::max();
                    for (std::uint32_t c = 0; c < state_.params.nlist; ++c) {
                        const float dist = ops.l2_sq(std::span(vec, state_.dim), state_.coarse_centroids.get_centroid(c));
                        if (dist < min_dist) { min_dist = dist; best_idx = c; }
                    }
                }

                // Optional correctness validation against brute-force on a sample
                if (state_.params.validate_ann_assignment) {
                    float rate = std::clamp(state_.params.validate_ann_sample_rate, 0.0f, 1.0f);
                    if (rate > 0.0f) {
                        const std::uint32_t h = static_cast<std::uint32_t>(i) * 2654435761u;
                        const float u = (h & 0x00FFFFFFu) * (1.0f / 16777216.0f);
                        if (u < rate) {
                            float min_dist_bf = std::numeric_limits<float>::max();
                            std::uint32_t bf_idx = 0;
                            for (std::uint32_t c = 0; c < state_.params.nlist; ++c) {
                                const float dist = ops.l2_sq(std::span(vec, state_.dim), state_.coarse_centroids.get_centroid(c));
                                if (dist < min_dist_bf) { min_dist_bf = dist; bf_idx = c; }
                            }
                            ann_validated_.fetch_add(1, std::memory_order_relaxed);
                            if (bf_idx != best_idx) {
                                ann_mismatches_.fetch_add(1, std::memory_order_relaxed);
                                if (state_.params.verbose) {
                                    std::cerr << "[IVFPQ] ANN centroid assignment mismatch: ann=" << best_idx
                                              << " bf=" << bf_idx << " (sample)\n";
                                }
                            }
                        }
                    }
                }
                assignments[i] = best_idx;
            }
            ann_assignments_.fetch_add(static_cast<std::uint64_t>(n), std::memory_order_relaxed);
        } else {
            // If batch search fails entirely, fall back to brute-force for all
            #pragma omp parallel for
            for (int i = 0; i < static_cast<int>(n); ++i) {
                const float* vec = data + static_cast<std::size_t>(i) * state_.dim;
                float min_dist = std::numeric_limits<float>::max();
                std::uint32_t best_idx = 0;
                for (std::uint32_t c = 0; c < state_.params.nlist; ++c) {
                    const float dist = ops.l2_sq(std::span(vec, state_.dim), state_.coarse_centroids.get_centroid(c));
                    if (dist < min_dist) { min_dist = dist; best_idx = c; }
                }
                assignments[static_cast<std::size_t>(i)] = best_idx;
            }
        }
    } else if (state_.params.use_centroid_ann && state_.params.coarse_assigner == CoarseAssigner::Projection && !proj_rows_.empty()) {
        // Projection-based screening with exact refinement
        const std::uint32_t L = std::max<std::uint32_t>(1u, std::min(state_.params.centroid_ann_refine_k, state_.params.nlist));
        const std::size_t p = static_cast<std::size_t>(proj_dim_);
        // Project queries and compute norms
        std::vector<float> qproj(n * p);
        std::vector<float> qnorm(n, 0.0f);
        for (std::size_t i = 0; i < n; ++i) {
            const float* qv = data + i * state_.dim;
            float* qp = qproj.data() + i * p;
            double accn = 0.0;
            for (std::size_t k = 0; k < p; ++k) {
                const float* pk = proj_rows_.data() + k * state_.dim;
                double s = 0.0; for (std::size_t d = 0; d < state_.dim; ++d) s += static_cast<double>(pk[d]) * static_cast<double>(qv[d]);
                qp[k] = static_cast<float>(s);
                accn += s * s;
            }
            qnorm[i] = static_cast<float>(accn);
        }
        // Candidate heaps per query
        struct Cand { float dist; std::uint32_t idx; };
        auto cmp = [](const Cand& a, const Cand& b){ return a.dist < b.dist; };
        std::vector<std::size_t> heap_size(n, 0);
        std::vector<Cand> heap_storage(n * static_cast<std::size_t>(L), Cand{std::numeric_limits<float>::infinity(), 0});
        std::vector<std::size_t> off(n+1, 0); for (std::size_t i = 0; i <= n; ++i) off[i] = i * static_cast<std::size_t>(L);
#ifdef VESPER_HAS_CBLAS
{
    // SGEMM-based projection screening: compute [qb x jb] dot blocks and update heaps
    const std::size_t QB = 256;
    std::vector<float> dots; dots.resize(QB * jb);
    for (std::size_t i0 = 0; i0 < n; i0 += QB) {
        const std::size_t qb = std::min<std::size_t>(QB, n - i0);
        // C = A * B^T where A = qproj[i0:i0+qb, :], B = proj_centroids_rm_[j0:j0+jb, :]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    static_cast<int>(qb), static_cast<int>(jb), static_cast<int>(p),
                    1.0f,
                    qproj.data() + i0 * p, static_cast<int>(p),
                    proj_centroids_rm_.data() + j0 * p, static_cast<int>(p),
                    0.0f,
                    dots.data(), static_cast<int>(jb));
        for (std::size_t r = 0; r < qb; ++r) {
            const std::size_t i = i0 + r;
            const float qi = qnorm[i];
            const std::size_t base = off[i];
            std::size_t& hs = heap_size[i];
            for (std::size_t jj = 0; jj < jb; ++jj) {
                const std::size_t cj = j0 + jj;
                const float dot = dots[r * jb + jj];
                const float distp = qi + proj_centroid_norms_[cj] - 2.0f * dot;
                if (hs < L) {
                    heap_storage[base + hs++] = Cand{distp, static_cast<std::uint32_t>(cj)};
                    if (hs == L) std::make_heap(heap_storage.begin() + base, heap_storage.begin() + base + hs, cmp);
                } else if (distp < heap_storage[base].dist) {
                    std::pop_heap(heap_storage.begin() + base, heap_storage.begin() + base + hs, cmp);
                    heap_storage[base + hs - 1] = Cand{distp, static_cast<std::uint32_t>(cj)};
                    std::push_heap(heap_storage.begin() + base, heap_storage.begin() + base + hs, cmp);
                }
            }
        }
    }
}
#else

        // Blocked fused dot-and-select with OpenMP over queries
        const std::size_t C = state_.params.nlist; const std::size_t CB = 256;
        for (std::size_t j0 = 0; j0 < C; j0 += CB) {
            const std::size_t jb = std::min<std::size_t>(CB, C - j0);
            if (p == 16 && !proj_centroids_pack8_.empty()) {
                const std::size_t C = state_.params.nlist;
                const std::size_t ntiles = (n / 16) * 16;
                // Process full 16x8 tiles
                #pragma omp parallel for schedule(static)
                for (int tile = 0; tile < static_cast<int>(ntiles/16); ++tile) {
                    const int i0 = tile * 16;
                    // Pack A-panel: Apack[k*16 + r] = qproj[(i0+r)*p + k]
                    alignas(32) float Apack[16*16];
                    for (int k = 0; k < 16; ++k) {
                        for (int r = 0; r < 16; ++r) {
                            Apack[k*16 + r] = qproj[(static_cast<std::size_t>(i0 + r) * p) + k];
                        }
                    }
                    const std::size_t cb_start = (j0 / 8) * 8;
                    const std::size_t cb_end = j0 + jb;
                    for (std::size_t cjblk = cb_start; cjblk < cb_end; cjblk += 8) {
                        const std::size_t block_id = cjblk / 8;
                        const float* Bpack = proj_centroids_pack8_.data() + block_id * (16 * 8);
                        // Accumulators for top and bottom 8 rows
                        __m256 Ctop[8]; __m256 Cbot[8];
                        for (int j = 0; j < 8; ++j) { Ctop[j] = _mm256_setzero_ps(); Cbot[j] = _mm256_setzero_ps(); }
                        // Rank-1 updates over k=0..15
                        for (int k = 0; k < 16; ++k) {
                            __m256 a_top = _mm256_loadu_ps(Apack + k*16 + 0);
                            __m256 a_bot = _mm256_loadu_ps(Apack + k*16 + 8);
                            float yk_arr[8];
                            #if defined(__AVX__)
                            __m256 yk = _mm256_loadu_ps(Bpack + k*8);
                            _mm256_storeu_ps(yk_arr, yk);
                            #else
                            for (int lane=0; lane<8; ++lane) yk_arr[lane] = Bpack[k*8 + lane];
                            #endif
                            for (int j = 0; j < 8; ++j) {
                                __m256 b = _mm256_set1_ps(yk_arr[j]);
                                Ctop[j] = _mm256_fmadd_ps(a_top, b, Ctop[j]);
                                Cbot[j] = _mm256_fmadd_ps(a_bot, b, Cbot[j]);
                            }
                        }
                        // Epilogue: convert to distances and update heaps
                        float dots_top[8][8]; float dots_bot[8][8];
                        for (int j = 0; j < 8; ++j) { _mm256_storeu_ps(dots_top[j], Ctop[j]); _mm256_storeu_ps(dots_bot[j], Cbot[j]); }
                        for (int r = 0; r < 16; ++r) {
                            const std::size_t i = static_cast<std::size_t>(i0 + r);
                            const float qi = qnorm[i];
                            const std::size_t base = off[i];
                            std::size_t& hs = heap_size[i];
                            for (int lane = 0; lane < 8; ++lane) {
                                const std::size_t cj = cjblk + static_cast<std::size_t>(lane);
                                if (cj < j0 || cj >= (j0 + jb) || cj >= C) continue;
                                const float dot = (r < 8 ? dots_top[lane][r] : dots_bot[lane][r - 8]);
                                const float distp = qi + proj_centroid_norms_[cj] - 2.0f * dot;
                                if (hs < L) {
                                    heap_storage[base + hs++] = Cand{distp, static_cast<std::uint32_t>(cj)};
                                    if (hs == L) std::make_heap(heap_storage.begin() + base, heap_storage.begin() + base + hs, cmp);
                                } else if (distp < heap_storage[base].dist) {
                                    std::pop_heap(heap_storage.begin() + base, heap_storage.begin() + base + hs, cmp);
                                    heap_storage[base + hs - 1] = Cand{distp, static_cast<std::uint32_t>(cj)};
                                    std::push_heap(heap_storage.begin() + base, heap_storage.begin() + base + hs, cmp);
                                }
                            }
                        }
                    }
                    // Handle centroid tail (<8) for these rows
                    const std::size_t tail_start = ((j0 + jb) / 8) * 8;
                    for (std::size_t cj = std::max<std::size_t>(j0, tail_start); cj < j0 + jb; ++cj) {
                        const float* yc = proj_centroids_rm_.data() + cj * p;
                        for (int r = 0; r < 16; ++r) {
                            const std::size_t i = static_cast<std::size_t>(i0 + r);
                            const float* qp = qproj.data() + i * p;
                            float dot = 0.0f; for (int k = 0; k < 16; ++k) dot += qp[k] * yc[k];
                            const float distp = qnorm[i] + proj_centroid_norms_[cj] - 2.0f * dot;
                            const std::size_t base = off[i];
                            std::size_t& hs = heap_size[i];
                            if (hs < L) { heap_storage[base + hs++] = Cand{distp, static_cast<std::uint32_t>(cj)}; if (hs == L) std::make_heap(heap_storage.begin()+base, heap_storage.begin()+base+hs, cmp); }
                            else if (distp < heap_storage[base].dist) { std::pop_heap(heap_storage.begin()+base, heap_storage.begin()+base+hs, cmp); heap_storage[base+hs-1] = Cand{distp, static_cast<std::uint32_t>(cj)}; std::push_heap(heap_storage.begin()+base, heap_storage.begin()+base+hs, cmp); }
                        }
                    }
                }
                // Process remaining rows (<16) with fallback
                for (std::size_t i = ntiles; i < n; ++i) {
                    const float* qp = qproj.data() + i * p;
                    const float qi = qnorm[i];
                    const std::size_t base = off[i];
                    std::size_t& hs = heap_size[i];
                    for (std::size_t jj = 0; jj < jb; ++jj) {
                        const std::size_t cj = j0 + jj;
                        const float* yc = proj_centroids_rm_.data() + cj * p;
                        float dot = 0.0f; for (int k = 0; k < 16; ++k) dot += qp[k] * yc[k];
                        const float distp = qi + proj_centroid_norms_[cj] - 2.0f * dot;
                        if (hs < L) { heap_storage[base + hs++] = Cand{distp, static_cast<std::uint32_t>(cj)}; if (hs == L) std::make_heap(heap_storage.begin()+base, heap_storage.begin()+base+hs, cmp); }
                        else if (distp < heap_storage[base].dist) { std::pop_heap(heap_storage.begin()+base, heap_storage.begin()+base+hs, cmp); heap_storage[base+hs-1] = Cand{distp, static_cast<std::uint32_t>(cj)}; std::push_heap(heap_storage.begin()+base, heap_storage.begin()+base+hs, cmp); }
                    }
                }
            } else {
                // Original per-query fallback path (any p)
                #pragma omp parallel for schedule(static)
                for (int ii = 0; ii < static_cast<int>(n); ++ii) {
                    const std::size_t i = static_cast<std::size_t>(ii);
                    const float* qp = qproj.data() + i * p;
                    const float qi_norm = qnorm[i];
                    const std::size_t base = off[i];
                    std::size_t& hs = heap_size[i];
                    for (std::size_t jj = 0; jj < jb; ++jj) {
                        const std::size_t cj = j0 + jj;
                        const float* yc = proj_centroids_rm_.data() + cj * p;
                        float dot = 0.0f; for (std::size_t k = 0; k < p; ++k) dot += qp[k] * yc[k];
                        const float distp = qi_norm + proj_centroid_norms_[cj] - 2.0f * dot;
                        if (hs < L) { heap_storage[base + hs++] = Cand{distp, static_cast<std::uint32_t>(cj)}; if (hs == L) std::make_heap(heap_storage.begin()+base, heap_storage.begin()+base+hs, cmp); }
                        else if (distp < heap_storage[base].dist) { std::pop_heap(heap_storage.begin()+base, heap_storage.begin()+base+hs, cmp); heap_storage[base+hs-1] = Cand{distp, static_cast<std::uint32_t>(cj)}; std::push_heap(heap_storage.begin()+base, heap_storage.begin()+base+hs, cmp); }
                    }
                }
            }
        }
#endif // VESPER_HAS_CBLAS

        // Exact refinement over candidates
        for (std::size_t i = 0; i < n; ++i) {
            const float* qv = data + i * state_.dim;
            const std::size_t base = off[i];
            const std::size_t hs = heap_size[i];
            float bestd = std::numeric_limits<float>::infinity(); std::uint32_t bestc = 0u;
            for (std::size_t t = 0; t < hs; ++t) {
                const std::uint32_t cand = heap_storage[base + t].idx;
                const float d = ops.l2_sq(std::span(qv, state_.dim), state_.coarse_centroids.get_centroid(cand));
                if (d < bestd) { bestd = d; bestc = cand; }
            }
            assignments[i] = bestc;

            // Optional correctness validation against brute-force on a sample
            if (state_.params.validate_ann_assignment) {
                float rate = std::clamp(state_.params.validate_ann_sample_rate, 0.0f, 1.0f);
                if (rate > 0.0f) {
                    const std::uint32_t h = static_cast<std::uint32_t>(i) * 2654435761u;
                    const float u = (h & 0x00FFFFFFu) * (1.0f / 16777216.0f);
                    if (u < rate) {
                        float min_dist_bf = std::numeric_limits<float>::max();
                        std::uint32_t bf_idx = 0;
                        for (std::uint32_t c = 0; c < state_.params.nlist; ++c) {
                            const float dist = ops.l2_sq(std::span(qv, state_.dim), state_.coarse_centroids.get_centroid(c));
                            if (dist < min_dist_bf) { min_dist_bf = dist; bf_idx = c; }
                        }
                        ann_validated_.fetch_add(1, std::memory_order_relaxed);
                        if (bf_idx != bestc) { ann_mismatches_.fetch_add(1, std::memory_order_relaxed); }
                    }
                }
            }
        }
        ann_assignments_.fetch_add(static_cast<std::uint64_t>(n), std::memory_order_relaxed);
    } else {
        // Brute-force assignment
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(n); ++i) {
            const float* vec = data + static_cast<std::size_t>(i) * state_.dim;
            float min_dist = std::numeric_limits<float>::max();
            std::uint32_t best_idx = 0;
            for (std::uint32_t c = 0; c < state_.params.nlist; ++c) {
                const float dist = ops.l2_sq(std::span(vec, state_.dim), state_.coarse_centroids.get_centroid(c));
                if (dist < min_dist) { min_dist = dist; best_idx = c; }
            }
            assignments[static_cast<std::size_t>(i)] = best_idx;
        }
    }

    if (kTimingAdd) {
        auto t_assign_end = std::chrono::steady_clock::now();
        auto ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t_assign_end - t_assign_start).count());
        t_assign_ns_.fetch_add(ns, std::memory_order_relaxed);
    }

    // 2) Compute residuals (original space), then rotate if OPQ before encoding
    std::vector<float> residuals(n * state_.dim);
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(n); ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        const float* vec = data + idx * state_.dim;
        const float* centroid = state_.coarse_centroids[assignments[idx]];
        float* out = residuals.data() + idx * state_.dim;
        for (std::size_t d = 0; d < state_.dim; ++d) {
            out[d] = vec[d] - centroid[d];
        }
    }

    const float* encode_data = residuals.data();
    std::vector<float> rotated_residuals;
    if (state_.params.use_opq && !state_.rotation_matrix.empty()) {
        rotated_residuals.resize(n * state_.dim);
        apply_rotation(residuals.data(), rotated_residuals.data(), n);
        encode_data = rotated_residuals.data();
    }

    // 3) Encode residuals with PQ
    std::vector<std::uint8_t> all_codes(n * state_.params.m);
    auto t_encode_start = kTimingAdd ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    state_.pq->encode(encode_data, n, all_codes.data());
    if (kTimingAdd) {
        auto t_encode_end = std::chrono::steady_clock::now();
        auto ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t_encode_end - t_encode_start).count());
        t_encode_ns_.fetch_add(ns, std::memory_order_relaxed);
    }

    // 4) Add to inverted lists (serialize access)
    auto t_lists_start = kTimingAdd ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    {
        std::lock_guard<std::mutex> lock(lists_mutex_);

        for (std::size_t i = 0; i < n; ++i) {
            const std::uint32_t list_idx = assignments[i];

            InvertedListEntry entry;
            entry.id = ids[i];
            entry.code.codes.assign(
                all_codes.data() + i * state_.params.m,
                all_codes.data() + (i + 1) * state_.params.m
            );

            inverted_lists_[list_idx].push_back(std::move(entry));
        }

        n_vectors_ += n;
    }
    if (kTimingAdd) {
        auto t_lists_end = std::chrono::steady_clock::now();
        auto ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t_lists_end - t_lists_start).count());
        t_lists_ns_.fetch_add(ns, std::memory_order_relaxed);
    }

    return {};
}

auto IvfPqIndex::Impl::find_nearest_centroids(const float* query, std::uint32_t nprobe) const
    -> std::vector<std::pair<std::uint32_t, float>> {

    const std::uint32_t n_centroids = state_.params.nlist;
    nprobe = std::min(nprobe, n_centroids);

    // Compute distances to all coarse centroids
    std::vector<std::pair<float, std::uint32_t>> distances;
    distances.reserve(n_centroids);

    const auto& ops = kernels::select_backend_auto();

    for (std::uint32_t i = 0; i < n_centroids; ++i) {
        const float dist = ops.l2_sq(
            std::span(query, state_.dim),
            state_.coarse_centroids.get_centroid(i)
        );
        distances.emplace_back(dist, i);
    }

    // Partial sort to find nprobe nearest
    std::partial_sort(distances.begin(), distances.begin() + nprobe, distances.end());

    // Convert to result format
    std::vector<std::pair<std::uint32_t, float>> result;
    result.reserve(nprobe);

    for (std::uint32_t i = 0; i < nprobe; ++i) {
        result.emplace_back(distances[i].second, distances[i].first);
    }

    return result;
}

auto IvfPqIndex::Impl::compute_adc_distance(const float* /* query */,
                                           const PqCode& code,
                                           const AlignedCentroidBuffer& luts) const -> float {
    float distance = 0.0f;
    const float* lut_data = luts.data();

    for (std::uint32_t m = 0; m < state_.params.m; ++m) {
        // luts is flat array: [m * ksub + code]
        std::uint32_t ksub = 1U << state_.params.nbits;
        distance += lut_data[m * ksub + code.codes[m]];
    }


    return distance;
}

auto IvfPqIndex::Impl::search(const float* query, const IvfPqSearchParams& params) const
    -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error> {
    using core::error;
    using core::error_code;

    if (!state_.trained) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Index must be trained before searching",
            "ivf_pq"
        });
    }

    // We compute list-specific LUTs using residual query = (query - centroid)
    // Coarse assignment (nprobe) is done in original space.
    auto probe_lists = find_nearest_centroids(query, params.nprobe);

    // Collect candidates from inverted lists
    using Candidate = std::pair<float, std::uint64_t>;
    std::priority_queue<Candidate> heap;

    // Decide candidate pool size: default to k; allow larger shortlist via cand_k
    const std::size_t pool_k = static_cast<std::size_t>(
        std::max<std::uint32_t>(params.k, (params.cand_k > 0 ? params.cand_k : params.k))
    );


    const bool use_heapless = (pool_k <= 64);
    TopKBuffer topk(pool_k);
    static const bool kTiming = [](){ const char* v = std::getenv("VESPER_TIMING"); return v && v[0] != '0' && v[0] != '\0'; }();
    auto t_adc_start = std::chrono::steady_clock::now();

    std::vector<float> residual_query(state_.dim);
    std::vector<float> residual_query_rot(state_.dim);

    for (const auto& pair : probe_lists) {
        const std::uint32_t list_idx = pair.first;
        const auto& list = inverted_lists_[list_idx];
        const float* centroid = state_.coarse_centroids[list_idx];

        // residual in original space
        for (std::size_t d = 0; d < state_.dim; ++d) {
            residual_query[d] = query[d] - centroid[d];
        }


        const float* lut_query = residual_query.data();

        if (state_.params.use_opq && !state_.rotation_matrix.empty()) {
            apply_rotation(residual_query.data(), residual_query_rot.data(), 1);
            lut_query = residual_query_rot.data();
        }

        // Fast path: batch ADC via FastScanPq blocks if available; fallback to scalar
        bool fast_path_used = false;
        #ifdef __x86_64__
        if (auto* fs = dynamic_cast<FastScanPq*>(state_.pq.get())) {
            const std::uint32_t m = state_.params.m;
            const std::uint32_t block_size = fs->config().block_size;
            const std::size_t L = list.size();
            const std::size_t TILE = 2048;

            // Linear id array mapped to computed distances order
            std::vector<std::uint64_t> ids_linear;
            ids_linear.reserve(L);

            for (std::size_t base = 0; base < L; base += TILE) {
                const std::size_t tile_n = std::min<std::size_t>(TILE, L - base);

                std::vector<PqCodeBlock> blocks;
                blocks.reserve((tile_n + block_size - 1) / block_size);
                PqCodeBlock block(m, block_size);

                for (std::size_t i = 0; i < tile_n; ++i) {
                    const auto& e = list[base + i];
                    if (!block.add_code(e.code.codes.data())) {
                        blocks.push_back(block);
                        block.clear();
                        (void)block.add_code(e.code.codes.data());
                    }
                    ids_linear.push_back(e.id);
                }
                if (block.size() > 0) blocks.push_back(block);

                const std::size_t total_codes = blocks.size() * block_size;
                std::vector<float> dists(total_codes, 0.0f);

                #ifdef __AVX512F__
                if (fs->config().use_avx512) {
                    fs->compute_distances_avx512(lut_query, blocks, dists.data());
                } else
                #endif
                #ifdef __AVX2__
                {
                    fs->compute_distances_avx2(lut_query, blocks, dists.data());
                }
                #else
                {
                    fs->compute_distances(lut_query, blocks, dists.data());
                }
                #endif

                for (std::size_t i = 0; i < tile_n; ++i) {
                    const float dist = dists[i];
                    const std::uint64_t id = ids_linear[base + i];
                    if (use_heapless) {
                        topk.consider(dist, id);
                    } else {
                        if (heap.size() < pool_k) {
                            heap.emplace(dist, id);
                        } else if (dist < heap.top().first) {
                            heap.pop();
                            heap.emplace(dist, id);
                        }
                    }
                }
            }

            fast_path_used = true;
        }
        #endif

        if (!fast_path_used) {
            // Compute lookup tables for this list and use scalar accumulation
            auto luts = state_.pq->compute_lookup_tables(lut_query);
            for (const auto& entry : list) {
                const float dist = compute_adc_distance(lut_query, entry.code, luts);
                if (use_heapless) {
                    topk.consider(dist, entry.id);
                } else {
                    if (heap.size() < pool_k) {
                        heap.emplace(dist, entry.id);
                    } else if (dist < heap.top().first) {
                        heap.pop();
                        heap.emplace(dist, entry.id);
                    }
                }
            }
        }
    }

    if (kTiming) {
        auto t_adc_end = std::chrono::steady_clock::now();
        auto adc_us = std::chrono::duration_cast<std::chrono::microseconds>(t_adc_end - t_adc_start).count();
        std::cerr << "ivf_pq.timing adc_us=" << adc_us << " nprobe=" << probe_lists.size() << " pool_k=" << pool_k << "\n";
    }

    // Extract results in sorted order
    std::vector<std::pair<std::uint64_t, float>> results;
    if (use_heapless) {
        results = topk.to_sorted_pairs();
    } else {
        results.reserve(heap.size());
        while (!heap.empty()) {
            const auto& [dist, id] = heap.top();
            results.emplace_back(id, dist);
            heap.pop();
        }
        std::reverse(results.begin(), results.end());
    }

    return results;
}

auto IvfPqIndex::Impl::search_batch(const float* queries, std::size_t n_queries,
                                   const IvfPqSearchParams& params) const
    -> std::expected<std::vector<std::vector<std::pair<std::uint64_t, float>>>, core::error> {

    std::vector<std::vector<std::pair<std::uint64_t, float>>> results(n_queries);

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(n_queries); ++i) {
        const float* query = queries + i * state_.dim;
        auto result = search(query, params);

        if (result.has_value()) {
            results[i] = std::move(result.value());
        }
    }

    return results;
}

// IvfPqIndex public interface implementation

IvfPqIndex::IvfPqIndex() : impl_(std::make_unique<Impl>()) {}

void IvfPqIndex::set_metadata_json(std::string_view json) {
    impl_->state_.metadata_json.assign(json.begin(), json.end());
}

auto IvfPqIndex::set_metadata_json_checked(std::string_view json)
    -> std::expected<void, core::error> {
    using core::error; using core::error_code;
    if (json.size() > kMaxMetadataSize) {
        return std::vesper_unexpected(error{ error_code::config_invalid, "Metadata JSON too large (limit 64 KiB)", "ivf_pq"});
    }
    if (auto vr = validate_json_structure(json, {64, 4096}); !vr.has_value()) {
        return std::vesper_unexpected(vr.error());
    }
    if (impl_->metadata_validator_) {
        auto r = impl_->metadata_validator_(json);
        if (!r.has_value()) return std::vesper_unexpected(r.error());
    }
    impl_->state_.metadata_json.assign(json.begin(), json.end());
    return {};
}

void IvfPqIndex::set_metadata_validator(MetadataValidator validator) {
    impl_->metadata_validator_ = std::move(validator);
}


auto IvfPqIndex::get_metadata_json() const -> std::string {
    return impl_->state_.metadata_json;
}

IvfPqIndex::~IvfPqIndex() = default;
IvfPqIndex::IvfPqIndex(IvfPqIndex&&) noexcept = default;
IvfPqIndex& IvfPqIndex::operator=(IvfPqIndex&&) noexcept = default;

auto IvfPqIndex::train(const float* data, std::size_t dim, std::size_t n,
                      const IvfPqTrainParams& params)
    -> std::expected<IvfPqTrainStats, core::error> {
    return impl_->train(data, dim, n, params);
}

auto IvfPqIndex::add(const std::uint64_t* ids, const float* data, std::size_t n)
    -> std::expected<void, core::error> {
    return impl_->add(ids, data, n);
}

auto IvfPqIndex::search(const float* query, const IvfPqSearchParams& params) const
    -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error> {
    return impl_->search(query, params);
}

auto IvfPqIndex::search_batch(const float* queries, std::size_t n_queries,
                             const IvfPqSearchParams& params) const
    -> std::expected<std::vector<std::vector<std::pair<std::uint64_t, float>>>, core::error> {
    return impl_->search_batch(queries, n_queries, params);
}

auto IvfPqIndex::get_stats() const noexcept -> Stats {
    Stats stats;
    stats.n_vectors = impl_->n_vectors_;
    stats.n_lists = impl_->state_.params.nlist;
    stats.m = impl_->state_.params.m;
    stats.code_size = impl_->state_.params.m;

    // Calculate memory usage
    stats.memory_bytes = sizeof(Impl);
    stats.memory_bytes += impl_->state_.coarse_centroids.size() *
                          impl_->state_.dim * sizeof(float);

    for (const auto& list : impl_->inverted_lists_) {
        stats.memory_bytes += list.size() *
                              (sizeof(Impl::InvertedListEntry) + impl_->state_.params.m);
    }

    stats.avg_list_size = impl_->n_vectors_ > 0 ?
        static_cast<float>(impl_->n_vectors_) / impl_->state_.params.nlist : 0.0f;

    // ANN/coarse-assigner telemetry: true when alternative assigner is enabled
    stats.ann_enabled = impl_->state_.params.use_centroid_ann;
    stats.ann_assignments = impl_->ann_assignments_.load(std::memory_order_relaxed);
    stats.ann_validated = impl_->ann_validated_.load(std::memory_order_relaxed);
    stats.ann_mismatches = impl_->ann_mismatches_.load(std::memory_order_relaxed);
    // KD-tree instrumentation
    stats.kd_nodes_pushed = impl_->kd_nodes_pushed_.load(std::memory_order_relaxed);
    stats.kd_nodes_popped = impl_->kd_nodes_popped_.load(std::memory_order_relaxed);
    stats.kd_leaves_scanned = impl_->kd_leaves_scanned_.load(std::memory_order_relaxed);

    // Timing telemetry (enabled if verbose or timings flag or env VESPER_TIMING)
    const bool env_timing = [](){ const char* v = std::getenv("VESPER_TIMING"); return v && v[0] != '0' && v[0] != '\0'; }();
    stats.timings_enabled = impl_->state_.params.verbose || impl_->state_.params.timings_enabled || env_timing;
    if (stats.timings_enabled) {
        const double inv_million = 1.0 / 1'000'000.0;
        stats.t_assign_ms = static_cast<double>(impl_->t_assign_ns_.load(std::memory_order_relaxed)) * inv_million;
        stats.t_encode_ms = static_cast<double>(impl_->t_encode_ns_.load(std::memory_order_relaxed)) * inv_million;
        stats.t_lists_ms  = static_cast<double>(impl_->t_lists_ns_.load(std::memory_order_relaxed)) * inv_million;
        // KD timing breakdown (aggregated across queries)
        stats.kd_traversal_ms = static_cast<double>(impl_->kd_trav_ns_.load(std::memory_order_relaxed)) * inv_million;
        stats.kd_leaf_ms      = static_cast<double>(impl_->kd_leaf_ns_.load(std::memory_order_relaxed)) * inv_million;
    }

    return stats;
}

auto IvfPqIndex::is_trained() const noexcept -> bool {
    return impl_->state_.trained;
}

auto IvfPqIndex::dimension() const noexcept -> std::size_t {
    return impl_->state_.dim;
}

auto IvfPqIndex::clear() -> void {
    impl_->inverted_lists_.clear();
    impl_->inverted_lists_.resize(impl_->state_.params.nlist);
    impl_->n_vectors_ = 0;
}

auto IvfPqIndex::reset() -> void {
    impl_ = std::make_unique<Impl>();
}

auto IvfPqIndex::save(const std::string& path) const -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;
    namespace fs = std::filesystem;

    // Ensure directory exists
    std::error_code fec;
    fs::create_directories(path, fec);
    if (fec) {
        return std::vesper_unexpected(error{
            error_code::io_failed, "Failed to create directory: " + path, "ivf_pq.save"});
    }

    const std::string file_path = path + "/ivfpq.bin";
    std::ofstream file(file_path, std::ios::binary);
    if (!file) {
        return std::vesper_unexpected(error{ error_code::io_failed,
            "Failed to open file for writing: " + file_path, "ivf_pq.save"});
    }

    auto write_bytes = [&](const void* ptr, std::size_t nbytes){ file.write(reinterpret_cast<const char*>(ptr), static_cast<std::streamsize>(nbytes)); };

    // FNV-1a 64-bit checksum updated as we write
    auto fnv_update = [&](std::uint64_t& h, const void* ptr, std::size_t nbytes){
        const auto* p = static_cast<const std::uint8_t*>(ptr);
        constexpr std::uint64_t FNV_PRIME = 1099511628211ull;
        for (std::size_t i = 0; i < nbytes; ++i) { h ^= p[i]; h *= FNV_PRIME; }
    };
    std::uint64_t checksum = 1469598103934665603ull;

    auto write_and_hash = [&](const void* ptr, std::size_t nbytes){ write_bytes(ptr, nbytes); fnv_update(checksum, ptr, nbytes); };

    // Optional v1.1 sectioned format with optional zstd compression
    auto env_get = [](const char* k)->const char*{ return std::getenv(k); };
    bool use_v11 = false;
    if (const char* e = env_get("VESPER_IVFPQ_SAVE_V11")) { use_v11 = (*e == '1'); }
    if (use_v11) {
        // write v1.1 header
        const char magic11[8] = {'I','V','F','P','Q','v','1','1'};
        write_and_hash(magic11, sizeof(magic11));
        std::uint16_t ver_major = 1, ver_minor = 1; write_and_hash(&ver_major, sizeof(ver_major)); write_and_hash(&ver_minor, sizeof(ver_minor));

        // Flags
        std::uint32_t flags = 0;
        if (impl_->state_.params.use_opq && !impl_->state_.rotation_matrix.empty()) flags |= 0x1u;
        if (impl_->state_.using_rabitq) flags |= 0x2u; // reserved
        write_and_hash(&flags, sizeof(flags));

        // Core dims/params
        std::uint32_t dim = static_cast<std::uint32_t>(impl_->state_.dim);
        std::uint32_t nlist = impl_->state_.params.nlist;
        std::uint32_t m = impl_->state_.params.m;
        std::uint32_t nbits = impl_->state_.params.nbits;
        std::uint32_t dsub = static_cast<std::uint32_t>(impl_->state_.dsub);
        std::uint64_t nvec = static_cast<std::uint64_t>(impl_->n_vectors_);
        std::uint32_t code_size = m;
        std::uint64_t build_ts = static_cast<std::uint64_t>(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));

        write_and_hash(&dim, sizeof(dim));
        write_and_hash(&nlist, sizeof(nlist));
        write_and_hash(&m, sizeof(m));
        write_and_hash(&nbits, sizeof(nbits));
        write_and_hash(&dsub, sizeof(dsub));
        write_and_hash(&nvec, sizeof(nvec));
        write_and_hash(&code_size, sizeof(code_size));
        write_and_hash(&build_ts, sizeof(build_ts));

        // Metadata (reserved)
        std::uint32_t meta_len = 0; write_and_hash(&meta_len, sizeof(meta_len));

        // Section header structure: type, uncompressed_size, compressed_size, section_checksum
        struct SectionHdr { std::uint32_t type; std::uint64_t unc; std::uint64_t comp; std::uint64_t shash; };
        auto fnv64 = [&](const void* ptr, std::size_t nbytes){ std::uint64_t h=1469598103934665603ull; const auto* p=(const std::uint8_t*)ptr; constexpr std::uint64_t P=1099511628211ull; for (size_t i=0;i<nbytes;++i){ h^=p[i]; h*=P; } return h; };

        // Compression control

        int zstd_level = 0; // 0 => no compression
    #ifdef VESPER_HAS_ZSTD
        if (const char* zl = env_get("VESPER_IVFPQ_ZSTD_LEVEL")) { int v = std::atoi(zl); if (v >= 1 && v <= 3) zstd_level = v; }
    #endif

        auto write_section = [&](std::uint32_t type, const void* data, std::size_t bytes){
            SectionHdr hdr{}; hdr.type = type; hdr.unc = static_cast<std::uint64_t>(bytes);
            std::vector<char> out;
            const void* payload = data; std::size_t payload_size = bytes;
        #ifdef VESPER_HAS_ZSTD
            if (zstd_level > 0 && bytes > 0) {
                size_t bound = ZSTD_compressBound(bytes);
                out.resize(bound);
                size_t got = ZSTD_compress(out.data(), bound, data, bytes, zstd_level);
                if (!ZSTD_isError(got) && got < bytes) { payload = out.data(); payload_size = got; }
                else { out.clear(); }
            }
        #endif
            hdr.comp = static_cast<std::uint64_t>(payload_size);
            hdr.shash = fnv64(data, bytes); // checksum of uncompressed contents
            write_and_hash(&hdr, sizeof(hdr));
            if (payload_size) write_and_hash(payload, payload_size);
        };

        // SECTION TYPES
        constexpr std::uint32_t SEC_CENTROIDS = 1;
        constexpr std::uint32_t SEC_CODEBOOKS = 2;
        constexpr std::uint32_t SEC_INVERTED  = 3;
        constexpr std::uint32_t SEC_OPQ       = 4;
        constexpr std::uint32_t SEC_METADATA  = 5; // New: JSON metadata

        // Centroids: flatten into contiguous [nlist*dim]
        std::vector<float> centroids;
        centroids.resize(static_cast<std::size_t>(nlist) * dim);
        for (std::uint32_t c = 0; c < nlist; ++c) {
            std::memcpy(centroids.data() + static_cast<std::size_t>(c) * dim,
                        impl_->state_.coarse_centroids[c], sizeof(float) * dim);
        }
        write_section(SEC_CENTROIDS, centroids.data(), centroids.size() * sizeof(float));

        // PQ codebooks
        if (!impl_->state_.pq) {
            return std::vesper_unexpected(error{ error_code::internal, "PQ not trained", "ivf_pq.save"});
        }
        std::vector<float> codebooks;
        impl_->state_.pq->export_codebooks(codebooks);
        write_section(SEC_CODEBOOKS, codebooks.data(), codebooks.size() * sizeof(float));

        // OPQ rotation matrix if present
        if (flags & 0x1u) {
            write_section(SEC_OPQ, impl_->state_.rotation_matrix.data(), impl_->state_.rotation_matrix.size() * sizeof(float));
        }

        // Optional JSON metadata
        if (!impl_->state_.metadata_json.empty()) {
            const auto& mj = impl_->state_.metadata_json;
            if (mj.size() > kMaxMetadataSize) {
                return std::vesper_unexpected(error{ error_code::config_invalid, "Metadata JSON too large (limit 64 KiB)", "ivf_pq.save"});
            }
            if (auto vr = validate_json_structure(mj, {64, 4096}); !vr.has_value()) {
                return std::vesper_unexpected(error{ error_code::config_invalid, std::string("Metadata JSON invalid: ")+vr.error().message, "ivf_pq.save"});
            }
            if (impl_->metadata_validator_) {
                auto r = impl_->metadata_validator_(mj);
                if (!r.has_value()) {
                    return std::vesper_unexpected(error{ error_code::config_invalid, std::string("Metadata schema validation failed: ")+r.error().message, "ivf_pq.save"});
                }
            }
            write_section(SEC_METADATA, mj.data(), mj.size());
        }

        // Inverted lists: serialize as [lists_count][ each list: sz][ entries: id + m code bytes]
        std::vector<char> inv;
        inv.reserve(16u * 1024u); // small start; will grow
        auto append = [&](const void* p, std::size_t nb){ const char* pc=(const char*)p; inv.insert(inv.end(), pc, pc+nb); };
        std::uint32_t lists_count = nlist; append(&lists_count, sizeof(lists_count));
        for (std::uint32_t li = 0; li < nlist; ++li) {
            const auto& list = impl_->inverted_lists_[li];
            std::uint64_t sz = static_cast<std::uint64_t>(list.size()); append(&sz, sizeof(sz));
            for (const auto& e : list) {
                append(&e.id, sizeof(e.id));
                append(e.code.codes.data(), e.code.codes.size());
            }
        }
        write_section(SEC_INVERTED, inv.data(), inv.size());

        // Trailer checksum
        const char tail[4] = {'C','H','K','S'}; write_bytes(tail, sizeof(tail)); write_bytes(&checksum, sizeof(checksum));
        return {};
    }


    // Header
    const char magic[8] = {'I','V','F','P','Q','v','1','0'};
    write_and_hash(magic, sizeof(magic));
    std::uint16_t ver_major = 1, ver_minor = 0; write_and_hash(&ver_major, sizeof(ver_major)); write_and_hash(&ver_minor, sizeof(ver_minor));

    // Flags
    std::uint32_t flags = 0;
    if (impl_->state_.params.use_opq && !impl_->state_.rotation_matrix.empty()) flags |= 0x1u;
    if (impl_->state_.using_rabitq) flags |= 0x2u; // reserved
    write_and_hash(&flags, sizeof(flags));

    // Core dims/params
    std::uint32_t dim = static_cast<std::uint32_t>(impl_->state_.dim);
    std::uint32_t nlist = impl_->state_.params.nlist;
    std::uint32_t m = impl_->state_.params.m;
    std::uint32_t nbits = impl_->state_.params.nbits;
    std::uint32_t dsub = static_cast<std::uint32_t>(impl_->state_.dsub);
    std::uint64_t nvec = static_cast<std::uint64_t>(impl_->n_vectors_);
    std::uint32_t code_size = m;
    std::uint64_t build_ts = static_cast<std::uint64_t>(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));

    write_and_hash(&dim, sizeof(dim));
    write_and_hash(&nlist, sizeof(nlist));
    write_and_hash(&m, sizeof(m));
    write_and_hash(&nbits, sizeof(nbits));
    write_and_hash(&dsub, sizeof(dsub));
    write_and_hash(&nvec, sizeof(nvec));
    write_and_hash(&code_size, sizeof(code_size));
    write_and_hash(&build_ts, sizeof(build_ts));

    // Metadata (reserved)
    std::uint32_t meta_len = 0; write_and_hash(&meta_len, sizeof(meta_len));

    // Coarse centroids [nlist x dim]
    for (std::uint32_t c = 0; c < nlist; ++c) {
        auto span = impl_->state_.coarse_centroids.get_centroid(c);
        write_and_hash(span.data(), sizeof(float) * dim);
    }

    // PQ codebooks [m*ksub x dsub]
    if (!impl_->state_.pq) {
        return std::vesper_unexpected(error{ error_code::internal, "PQ not trained", "ivf_pq.save"});
    }
    std::vector<float> codebooks;
    impl_->state_.pq->export_codebooks(codebooks);
    if (codebooks.size() != static_cast<std::size_t>(m) * (1u << nbits) * dsub) {
        return std::vesper_unexpected(error{ error_code::internal, "PQ codebooks size mismatch", "ivf_pq.save"});
    }
    write_and_hash(codebooks.data(), codebooks.size() * sizeof(float));

    // OPQ rotation matrix if present [dim x dim]
    if (flags & 0x1u) {
        if (impl_->state_.rotation_matrix.size() != static_cast<std::size_t>(dim) * dim) {
            return std::vesper_unexpected(error{ error_code::internal, "OPQ rotation size mismatch", "ivf_pq.save"});
        }
        write_and_hash(impl_->state_.rotation_matrix.data(), impl_->state_.rotation_matrix.size() * sizeof(float));
    }

    // Inverted lists
    std::uint32_t lists_count = nlist; write_and_hash(&lists_count, sizeof(lists_count));
    for (std::uint32_t li = 0; li < nlist; ++li) {
        const auto& list = impl_->inverted_lists_[li];
        std::uint64_t sz = static_cast<std::uint64_t>(list.size());
        write_and_hash(&sz, sizeof(sz));
        for (const auto& e : list) {
            write_and_hash(&e.id, sizeof(e.id));
            if (e.code.codes.size() != m) {
                return std::vesper_unexpected(error{ error_code::internal, "PQ code size mismatch in list", "ivf_pq.save"});
            }
            write_and_hash(e.code.codes.data(), m);
        }
    }

    // Trailer checksum
    const char tail[4] = {'C','H','K','S'};
    write_bytes(tail, sizeof(tail));
    write_bytes(&checksum, sizeof(checksum));

    return {};
}

auto IvfPqIndex::load(const std::string& path)
    -> std::expected<IvfPqIndex, core::error> {
    using core::error; using core::error_code;
    const std::string file_path = path + "/ivfpq.bin";
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        return std::vesper_unexpected(error{ error_code::not_found, "Index file not found: " + file_path, "ivf_pq.load"});
    }

    auto read_exact = [&](void* ptr, std::size_t nbytes) -> bool { file.read(reinterpret_cast<char*>(ptr), static_cast<std::streamsize>(nbytes)); return static_cast<std::size_t>(file.gcount()) == nbytes; };
    auto fnv_update = [&](std::uint64_t& h, const void* ptr, std::size_t nbytes){
        const auto* p = static_cast<const std::uint8_t*>(ptr); constexpr std::uint64_t FNV_PRIME = 1099511628211ull; for (std::size_t i=0;i<nbytes;++i){ h ^= p[i]; h *= FNV_PRIME; }
    };

    std::uint64_t checksum_calc = 1469598103934665603ull;

    // Read whole file size
    file.seekg(0, std::ios::end);
    const std::streamoff fsize = file.tellg();
    if (fsize < 64) {
        return std::vesper_unexpected(error{ error_code::config_invalid, "File too small", "ivf_pq.load"});
    }
    file.seekg(0, std::ios::beg);

    auto read_and_hash = [&](void* ptr, std::size_t nbytes){ if (!read_exact(ptr, nbytes)) return false; fnv_update(checksum_calc, ptr, nbytes); return true; };

    char magic[8]; if (!read_and_hash(magic, sizeof(magic))) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading magic", "ivf_pq.load"});
    bool is_v11 = false;
    if (std::string_view(magic, 8) == std::string_view("IVFPQv11",8)) {
        is_v11 = true;
    } else if (std::string_view(magic, 8) != std::string_view("IVFPQv10",8)) {
        return std::vesper_unexpected(error{ error_code::config_invalid, "Magic mismatch", "ivf_pq.load"});
    }

    // If mmap is requested and v1.1 file, switch to mmap-based parser
    bool want_mmap = false; if (const char* e = std::getenv("VESPER_IVFPQ_LOAD_MMAP")) want_mmap = (*e=='1');
    if (want_mmap && is_v11) {
        file.close();

        auto parse_v11_mmap = [&](const std::string& fpath) -> std::expected<IvfPqIndex, error> {
            // Map whole file read-only
            const std::uint8_t* base = nullptr; std::size_t sz = 0;
        #if defined(_WIN32)
            HANDLE hFile = CreateFileA(fpath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
            if (hFile == INVALID_HANDLE_VALUE) return std::vesper_unexpected(error{ error_code::io_failed, "CreateFile failed", "ivf_pq.load"});
            LARGE_INTEGER fsz{}; if (!GetFileSizeEx(hFile, &fsz)) { CloseHandle(hFile); return std::vesper_unexpected(error{ error_code::io_failed, "GetFileSizeEx failed", "ivf_pq.load"}); }
            HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
            if (!hMap) { CloseHandle(hFile); return std::vesper_unexpected(error{ error_code::io_failed, "CreateFileMapping failed", "ivf_pq.load"}); }
            void* view = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
            if (!view) { CloseHandle(hMap); CloseHandle(hFile); return std::vesper_unexpected(error{ error_code::io_failed, "MapViewOfFile failed", "ivf_pq.load"}); }
            base = static_cast<const std::uint8_t*>(view); sz = static_cast<std::size_t>(fsz.QuadPart);
            auto unmap = [&](){ UnmapViewOfFile(view); CloseHandle(hMap); CloseHandle(hFile); };
        #else
            int fd = ::open(fpath.c_str(), O_RDONLY);
            if (fd < 0) return std::vesper_unexpected(error{ error_code::io_failed, "open failed", "ivf_pq.load"});
            struct stat st{}; if (fstat(fd, &st) != 0) { ::close(fd); return std::vesper_unexpected(error{ error_code::io_failed, "fstat failed", "ivf_pq.load"}); }
            sz = static_cast<std::size_t>(st.st_size);
            void* addr = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
            if (addr == MAP_FAILED) { ::close(fd); return std::vesper_unexpected(error{ error_code::io_failed, "mmap failed", "ivf_pq.load"}); }
            base = static_cast<const std::uint8_t*>(addr);
            auto unmap = [&](){ munmap(const_cast<std::uint8_t*>(base), sz); ::close(fd); };
        #endif

            // RAII guard
            struct MMapGuard { std::function<void()> f; ~MMapGuard(){ if(f) f(); } };
            MMapGuard guard; guard.f = [&](){ unmap(); };

            if (sz < 64) return std::vesper_unexpected(error{ error_code::config_invalid, "File too small", "ivf_pq.load"});
            const std::uint8_t* p = base; const std::uint8_t* tail = base + sz - 12; // exclude trailer

            auto rd = [&](auto& v) -> bool { using T=std::decay_t<decltype(v)>; if (p + sizeof(T) > tail) return false; std::memcpy(&v, p, sizeof(T)); p += sizeof(T); return true; };
            auto rd_bytes = [&](const void*& out, std::size_t n) -> bool { if (p + n > tail) return false; out = p; p += n; return true; };
            auto fnv64 = [&](const void* ptr, std::size_t n){ const auto* q = static_cast<const std::uint8_t*>(ptr); std::uint64_t h=1469598103934665603ull; constexpr std::uint64_t F=1099511628211ull; for (std::size_t i=0;i<n;++i){ h^=q[i]; h*=F;} return h; };

            // Check magic
            if (std::memcmp(base, "IVFPQv11", 8) != 0) return std::vesper_unexpected(error{ error_code::config_invalid, "Magic mismatch (mmap)", "ivf_pq.load"});

            // Compute file checksum over [base, tail)
            std::uint64_t checksum_calc = fnv64(base, static_cast<std::size_t>(tail - base));
            // Verify trailer
            if (std::memcmp(base + sz - 12, "CHKS", 4) != 0) return std::vesper_unexpected(error{ error_code::config_invalid, "Missing checksum trailer (mmap)", "ivf_pq.load"});
            std::uint64_t checksum_file = 0; std::memcpy(&checksum_file, base + sz - 8, 8);
            if (checksum_calc != checksum_file) return std::vesper_unexpected(error{ error_code::config_invalid, "Checksum mismatch (mmap)", "ivf_pq.load"});

            // Parse header
            p = base + 8; // after magic
            std::uint16_t ver_major=0, ver_minor=0; if (!rd(ver_major) || !rd(ver_minor)) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading version (mmap)", "ivf_pq.load"});
            if (ver_major != 1) return std::vesper_unexpected(error{ error_code::config_invalid, "Unsupported version (mmap)", "ivf_pq.load"});
            std::uint32_t flags=0; if (!rd(flags)) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading flags (mmap)", "ivf_pq.load"});
            std::uint32_t dim=0, nlist=0, m=0, nbits=0, dsub=0; std::uint64_t nvec=0; std::uint32_t code_size=0; std::uint64_t build_ts=0;
            if (!rd(dim) || !rd(nlist) || !rd(m) || !rd(nbits) || !rd(dsub) || !rd(nvec) || !rd(code_size) || !rd(build_ts))
                return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading header (mmap)", "ivf_pq.load"});
            std::string meta_hdr_json;
            std::uint32_t meta_len=0; if (!rd(meta_len)) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading meta_len (mmap)", "ivf_pq.load"});
            if (meta_len) {
                if (meta_len > kMaxMetadataSize) return std::vesper_unexpected(error{ error_code::config_invalid, "Metadata too large (mmap)", "ivf_pq.load"});
                const void* meta=nullptr; if (!rd_bytes(meta, meta_len)) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading metadata (mmap)", "ivf_pq.load"});
                meta_hdr_json.assign(static_cast<const char*>(meta), static_cast<std::size_t>(meta_len));
                if (auto vr = validate_json_structure(meta_hdr_json, {64, 4096}); !vr.has_value()) {
                    return std::vesper_unexpected(error{ error_code::config_invalid, std::string("Metadata JSON invalid (mmap header): ")+vr.error().message, "ivf_pq.load"});
                }
            }

            IvfPqIndex index; index.impl_ = std::make_unique<Impl>(); auto& impl = *index.impl_;
            impl.state_.dim = dim; impl.state_.params.nlist = nlist; impl.state_.params.m = m; impl.state_.params.nbits = nbits; impl.state_.dsub = dsub;
            impl.state_.params.use_opq = (flags & 0x1u) != 0;

            struct SectionHdr { std::uint32_t type; std::uint64_t unc; std::uint64_t comp; std::uint64_t shash; };
            constexpr std::uint32_t SEC_CENTROIDS = 1, SEC_CODEBOOKS=2, SEC_INVERTED=3, SEC_OPQ=4, SEC_METADATA=5;

            while (p < tail) {
                // Read section header fields individually to avoid padding issues
                std::uint32_t sec_type = 0; std::uint64_t sec_unc = 0, sec_comp = 0, sec_shash = 0;
                if (!rd(sec_type) || !rd(sec_unc) || !rd(sec_comp) || !rd(sec_shash)) break; // stop if header incomplete
                const void* payload = nullptr; if (!rd_bytes(payload, static_cast<std::size_t>(sec_comp))) return std::vesper_unexpected(error{ error_code::config_invalid, "Section truncated (mmap)", "ivf_pq.load"});

                auto get_span = [&](std::vector<std::uint8_t>& scratch)->std::span<const std::byte> {
                    if (sec_comp == sec_unc || sec_unc == 0) {
                        return std::span<const std::byte>(reinterpret_cast<const std::byte*>(payload), static_cast<std::size_t>(sec_comp));
                    }
                #ifdef VESPER_HAS_ZSTD
                    scratch.resize(static_cast<std::size_t>(sec_unc));
                    size_t got = ZSTD_decompress(scratch.data(), scratch.size(), payload, static_cast<std::size_t>(sec_comp));
                    if (ZSTD_isError(got) || got != scratch.size()) return {};
                    return std::span<const std::byte>(reinterpret_cast<const std::byte*>(scratch.data()), scratch.size());
                #else
                    return {};
                #endif
                };

                std::vector<std::uint8_t> scratch;
                auto sp = get_span(scratch); if (sp.empty() && sec_unc!=0) return std::vesper_unexpected(error{ error_code::config_invalid, "Decompression failed (mmap)", "ivf_pq.load"});
                // Verify per-section checksum
                if (sec_unc) {
                    std::uint64_t sh = fnv64(sp.data(), sp.size()); if (sh != sec_shash) return std::vesper_unexpected(error{ error_code::config_invalid, "Section checksum mismatch (mmap)", "ivf_pq.load"});
                }

                switch (sec_type) {
                    case SEC_CENTROIDS: {
                        if (sp.size_bytes() != static_cast<std::size_t>(nlist) * dim * sizeof(float)) return std::vesper_unexpected(error{ error_code::config_invalid, "Centroids size mismatch (mmap)", "ivf_pq.load"});
                        impl.state_.coarse_centroids = AlignedCentroidBuffer(nlist, dim);
                        auto* fptr = reinterpret_cast<const float*>(sp.data());
                        for (std::uint32_t c=0;c<nlist;++c) std::memcpy(impl.state_.coarse_centroids[c], fptr + static_cast<std::size_t>(c)*dim, sizeof(float)*dim);
                        break;
                    }
                    case SEC_CODEBOOKS: {
                        const std::size_t ksub = 1u << nbits; const std::size_t cb_elems = static_cast<std::size_t>(m) * ksub * dsub;
                        if (sp.size_bytes() != cb_elems * sizeof(float)) return std::vesper_unexpected(error{ error_code::config_invalid, "Codebooks size mismatch (mmap)", "ivf_pq.load"});
                        std::vector<float> codebooks(cb_elems); std::memcpy(codebooks.data(), sp.data(), sp.size());
                        FastScanPqConfig pqcfg{ .m = m, .nbits = nbits, .block_size = 32, .use_avx512 = core::decide_use_avx512_from_env_and_cpu() };
                        impl.state_.pq = std::make_unique<PqImpl>(pqcfg);
                        impl.state_.pq->import_pretrained(dsub, std::span<const float>(codebooks.data(), codebooks.size()));
                        break;
                    }
                    case SEC_OPQ: {
                        if (sp.size_bytes() != static_cast<std::size_t>(dim) * dim * sizeof(float)) return std::vesper_unexpected(error{ error_code::config_invalid, "OPQ size mismatch (mmap)", "ivf_pq.load"});
                        impl.state_.params.use_opq = true; impl.state_.rotation_matrix.resize(static_cast<std::size_t>(dim) * dim);
                        std::memcpy(impl.state_.rotation_matrix.data(), sp.data(), sp.size());
                        break;
                    }
                    case SEC_METADATA: {
                        if (sp.size_bytes() > kMaxMetadataSize) return std::vesper_unexpected(error{ error_code::config_invalid, "Metadata too large (mmap)", "ivf_pq.load"});
                        std::string tmp(reinterpret_cast<const char*>(sp.data()), sp.size_bytes());
                        if (auto vr = validate_json_structure(tmp, {64, 4096}); !vr.has_value()) {
                            return std::vesper_unexpected(error{ error_code::config_invalid, std::string("Metadata JSON invalid (mmap section): ")+vr.error().message, "ivf_pq.load"});
                        }
                        impl.state_.metadata_json = std::move(tmp);
                        break;
                    }
                    case SEC_INVERTED: {
                        const auto* q = reinterpret_cast<const std::uint8_t*>(sp.data()); const auto* qe = q + sp.size_bytes();
                        if (qe - q < static_cast<ptrdiff_t>(sizeof(std::uint32_t))) return std::vesper_unexpected(error{ error_code::config_invalid, "Inverted too small (mmap)", "ivf_pq.load"});
                        std::uint32_t lists_count=0; std::memcpy(&lists_count, q, sizeof(lists_count)); q += sizeof(lists_count);
                        if (lists_count != nlist) return std::vesper_unexpected(error{ error_code::config_invalid, "lists_count mismatch (mmap)", "ivf_pq.load"});
                        impl.inverted_lists_.clear(); impl.inverted_lists_.resize(nlist);
                        for (std::uint32_t li=0; li<nlist; ++li) {
                            if (qe - q < static_cast<ptrdiff_t>(sizeof(std::uint64_t))) return std::vesper_unexpected(error{ error_code::config_invalid, "List header truncated (mmap)", "ivf_pq.load"});
                            std::uint64_t szl=0; std::memcpy(&szl, q, sizeof(szl)); q += sizeof(szl);
                            auto& list = impl.inverted_lists_[li]; list.reserve(static_cast<std::size_t>(szl));
                            for (std::uint64_t i=0;i<szl;++i) {
                                if (qe - q < static_cast<ptrdiff_t>(sizeof(std::uint64_t) + m)) return std::vesper_unexpected(error{ error_code::config_invalid, "Entry truncated (mmap)", "ivf_pq.load"});
                                Impl::InvertedListEntry e; std::memcpy(&e.id, q, sizeof(e.id)); q += sizeof(e.id);
                                e.code.codes.resize(m); std::memcpy(e.code.codes.data(), q, m); q += m; list.push_back(std::move(e));
                            }
                        }
                        if (q != qe) return std::vesper_unexpected(error{ error_code::config_invalid, "Inverted extra bytes (mmap)", "ivf_pq.load"});
                        break;
                    }
                    default: { /* skip unknown */ }
                }
            }

            impl.n_vectors_ = static_cast<std::size_t>(nvec); impl.state_.trained = true;
            if (impl.state_.metadata_json.empty() && !meta_hdr_json.empty()) {
                impl.state_.metadata_json = std::move(meta_hdr_json);
            }
            return std::expected<IvfPqIndex, core::error>(std::move(index));
        };

        if (auto r = parse_v11_mmap(file_path); r.has_value()) return r; // success via mmap
        // If mmap path fails for any reason, fall back to streaming path
        // Re-open stream and reset checksum
        file = std::ifstream(file_path, std::ios::binary); if (!file) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reopening file after mmap fallback", "ivf_pq.load"});
        checksum_calc = 1469598103934665603ull; file.seekg(0, std::ios::beg);
        if (!read_and_hash(magic, sizeof(magic))) return std::vesper_unexpected(error{ error_code::io_failed, "Failed re-reading magic", "ivf_pq.load"});
        // proceed as normal below
    }

    std::uint16_t ver_major=0, ver_minor=0; if (!read_and_hash(&ver_major, sizeof(ver_major))) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading version", "ivf_pq.load"}); if (!read_and_hash(&ver_minor, sizeof(ver_minor))) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading version", "ivf_pq.load"});
    if (ver_major != 1) return std::vesper_unexpected(error{ error_code::config_invalid, "Unsupported version", "ivf_pq.load"});

    std::uint32_t flags=0; if (!read_and_hash(&flags, sizeof(flags))) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading flags", "ivf_pq.load"});

    std::uint32_t dim=0, nlist=0, m=0, nbits=0, dsub=0; std::uint64_t nvec=0; std::uint32_t code_size=0; std::uint64_t build_ts=0;
    if (!read_and_hash(&dim, sizeof(dim))) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading dim", "ivf_pq.load"});
    if (!read_and_hash(&nlist, sizeof(nlist))) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading nlist", "ivf_pq.load"});
    if (!read_and_hash(&m, sizeof(m))) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading m", "ivf_pq.load"});
    if (!read_and_hash(&nbits, sizeof(nbits))) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading nbits", "ivf_pq.load"});
    if (!read_and_hash(&dsub, sizeof(dsub))) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading dsub", "ivf_pq.load"});
    if (!read_and_hash(&nvec, sizeof(nvec))) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading n_vectors", "ivf_pq.load"});
    if (!read_and_hash(&code_size, sizeof(code_size))) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading code_size", "ivf_pq.load"});
    if (!read_and_hash(&build_ts, sizeof(build_ts))) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading timestamp", "ivf_pq.load"});

    std::string meta_hdr_json;
    std::uint32_t meta_len=0; if (!read_and_hash(&meta_len, sizeof(meta_len))) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading meta_len", "ivf_pq.load"});
    if (meta_len > 0) {
        if (meta_len > kMaxMetadataSize) return std::vesper_unexpected(error{ error_code::config_invalid, "Metadata too large", "ivf_pq.load"});
        std::vector<char> meta(meta_len);
        if (!read_and_hash(meta.data(), meta_len)) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading metadata", "ivf_pq.load"});
        meta_hdr_json.assign(meta.data(), meta.size());
        if (auto vr = validate_json_structure(meta_hdr_json, {64, 4096}); !vr.has_value()) {
            return std::vesper_unexpected(error{ error_code::config_invalid, std::string("Metadata JSON invalid (header): ")+vr.error().message, "ivf_pq.load"});
        }
    }

    if (is_v11) {
        // Early trailer checksum verification to fail-fast on corrupted files
        // This mirrors the mmap path behavior and prevents parsing of bad sections
        {
            std::ifstream fin(file_path, std::ios::binary);
            if (!fin) return std::vesper_unexpected(error{ error_code::io_failed, "Failed to reopen for checksum", "ivf_pq.load"});
            fin.seekg(0, std::ios::end); std::streamoff sz = fin.tellg(); fin.seekg(0, std::ios::beg);
            if (sz < 12) return std::vesper_unexpected(error{ error_code::config_invalid, "File too small (v1.1)", "ivf_pq.load"});
            std::vector<char> all(static_cast<std::size_t>(sz)); fin.read(all.data(), sz);
            if (fin.gcount() != sz) return std::vesper_unexpected(error{ error_code::io_failed, "Failed to read file for checksum", "ivf_pq.load"});
            if (std::memcmp(all.data() + sz - 12, "CHKS", 4) != 0) return std::vesper_unexpected(error{ error_code::config_invalid, "Missing checksum trailer", "ivf_pq.load"});
            std::uint64_t file_ch = 0; std::memcpy(&file_ch, all.data() + sz - 8, 8);
            std::uint64_t calc_ch = 1469598103934665603ull; fnv_update(calc_ch, all.data(), static_cast<std::size_t>(sz - 12));
            if (calc_ch != file_ch) return std::vesper_unexpected(error{ error_code::config_invalid, "Checksum mismatch (early)", "ivf_pq.load"});
            // Keep current stream position (we already consumed the fixed header and optional metadata)
            file.clear(); // do not reposition here; continue parsing sections from current position
        }

        // Parse sectioned layout
        IvfPqIndex index;
        index.impl_ = std::make_unique<Impl>();
        auto& impl = *index.impl_;
        impl.state_.dim = dim;
        impl.state_.params.nlist = nlist;
        impl.state_.params.m = m;
        impl.state_.params.nbits = nbits;
        impl.state_.dsub = dsub;

        constexpr std::uint32_t SEC_CENTROIDS = 1;
        constexpr std::uint32_t SEC_CODEBOOKS = 2;
        constexpr std::uint32_t SEC_INVERTED  = 3;
        constexpr std::uint32_t SEC_OPQ       = 4;
        constexpr std::uint32_t SEC_METADATA  = 5;

        auto fnv64 = [&](const void* ptr, std::size_t n){ const auto* q = static_cast<const std::uint8_t*>(ptr); std::uint64_t h=1469598103934665603ull; constexpr std::uint64_t F=1099511628211ull; for (std::size_t i=0;i<n;++i){ h^=q[i]; h*=F;} return h; };

        bool got_centroids=false, got_codebooks=false, got_inverted=false;
        struct SectionHdr { std::uint32_t type; std::uint64_t unc; std::uint64_t comp; std::uint64_t shash; };
        while (file.tellg() < fsize - static_cast<std::streamoff>(12)) {
            SectionHdr hdr{}; if (!read_and_hash(&hdr, sizeof(hdr))) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading section header", "ivf_pq.load"});
            if (hdr.unc == 0 && hdr.comp == 0) continue;
            std::vector<char> payload(static_cast<std::size_t>(hdr.comp));
            if (hdr.comp) { if (!read_and_hash(payload.data(), static_cast<std::size_t>(hdr.comp))) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading section payload", "ivf_pq.load"}); }

            auto get_data = [&](std::vector<char>& buf_unc)->std::span<const std::byte>{
                if (hdr.comp == hdr.unc || hdr.unc == 0) {
                    return std::span<const std::byte>(reinterpret_cast<const std::byte*>(payload.data()), payload.size());
                }
            #ifdef VESPER_HAS_ZSTD
                buf_unc.resize(static_cast<std::size_t>(hdr.unc));
                size_t got = ZSTD_decompress(buf_unc.data(), buf_unc.size(), payload.data(), payload.size());
                if (ZSTD_isError(got) || got != buf_unc.size()) {
                    return std::span<const std::byte>();
                }
                return std::span<const std::byte>(reinterpret_cast<const std::byte*>(buf_unc.data()), buf_unc.size());
            #else
                // No zstd compiled in; cannot decompress
                return std::span<const std::byte>();
            #endif
            };

            switch (hdr.type) {
                case SEC_CENTROIDS: {
                    std::vector<char> unc; auto sp = get_data(unc); if (sp.empty()) return std::vesper_unexpected(error{ error_code::config_invalid, "Centroids decompress/parse failed", "ivf_pq.load"});

                    if (sp.size_bytes() != static_cast<std::size_t>(nlist) * dim * sizeof(float)) return std::vesper_unexpected(error{ error_code::config_invalid, "Centroids size mismatch", "ivf_pq.load"});
                    impl.state_.coarse_centroids = AlignedCentroidBuffer(nlist, dim);
                    auto* fptr = reinterpret_cast<const float*>(sp.data());
                    for (std::uint32_t c = 0; c < nlist; ++c) { std::memcpy(impl.state_.coarse_centroids[c], fptr + static_cast<std::size_t>(c) * dim, sizeof(float)*dim); }
                    got_centroids = true; break;
                }
                case SEC_CODEBOOKS: {
                    std::vector<char> unc; auto sp = get_data(unc); if (sp.empty()) return std::vesper_unexpected(error{ error_code::config_invalid, "Codebooks decompress/parse failed", "ivf_pq.load"});

                    const std::size_t ksub = 1u << nbits; const std::size_t cb_elems = static_cast<std::size_t>(m) * ksub * dsub;
                    if (sp.size_bytes() != cb_elems * sizeof(float)) return std::vesper_unexpected(error{ error_code::config_invalid, "Codebooks size mismatch", "ivf_pq.load"});
                    std::vector<float> codebooks(cb_elems);
                    std::memcpy(codebooks.data(), sp.data(), sp.size_bytes());
                    FastScanPqConfig pqcfg{ .m = m, .nbits = nbits, .block_size = 32, .use_avx512 = core::decide_use_avx512_from_env_and_cpu() };
                    impl.state_.pq = std::make_unique<PqImpl>(pqcfg);
                    impl.state_.pq->import_pretrained(dsub, std::span<const float>(codebooks.data(), codebooks.size()));
                    got_codebooks = true; break;
                }
                case SEC_OPQ: {
                    std::vector<char> unc; auto sp = get_data(unc); if (sp.empty()) return std::vesper_unexpected(error{ error_code::config_invalid, "OPQ decompress/parse failed", "ivf_pq.load"});
                    if (sp.size_bytes() != static_cast<std::size_t>(dim) * dim * sizeof(float)) return std::vesper_unexpected(error{ error_code::config_invalid, "OPQ size mismatch", "ivf_pq.load"});
                    impl.state_.params.use_opq = true; impl.state_.rotation_matrix.resize(static_cast<std::size_t>(dim) * dim);
                    std::memcpy(impl.state_.rotation_matrix.data(), sp.data(), sp.size_bytes());
                    break;
                }
                case SEC_METADATA: {
                    std::vector<char> unc; auto sp = get_data(unc); if (sp.empty()) return std::vesper_unexpected(error{ error_code::config_invalid, "Metadata decompress/parse failed", "ivf_pq.load"});
                    if (sp.size_bytes() > kMaxMetadataSize) return std::vesper_unexpected(error{ error_code::config_invalid, "Metadata too large", "ivf_pq.load"});
                    std::string tmp(reinterpret_cast<const char*>(sp.data()), sp.size_bytes());
                    if (auto vr = validate_json_structure(tmp, {64, 4096}); !vr.has_value()) {
                        return std::vesper_unexpected(error{ error_code::config_invalid, std::string("Metadata JSON invalid (section): ")+vr.error().message, "ivf_pq.load"});
                    }
                    impl.state_.metadata_json = std::move(tmp);
                    break;
                }
                case SEC_INVERTED: {
                    std::vector<char> unc; auto sp = get_data(unc); if (sp.empty()) return std::vesper_unexpected(error{ error_code::config_invalid, "Inverted decompress/parse failed", "ivf_pq.load"});

                    const auto* p = reinterpret_cast<const std::byte*>(sp.data()); const auto* pe = p + sp.size_bytes();
                    if (pe - p < static_cast<ptrdiff_t>(sizeof(std::uint32_t))) return std::vesper_unexpected(error{ error_code::config_invalid, "Inverted too small", "ivf_pq.load"});
                    std::uint32_t lists_count = 0; std::memcpy(&lists_count, p, sizeof(lists_count)); p += sizeof(lists_count);
                    if (lists_count != nlist) return std::vesper_unexpected(error{ error_code::config_invalid, "lists_count mismatch", "ivf_pq.load"});
                    impl.inverted_lists_.clear(); impl.inverted_lists_.resize(nlist);
                    for (std::uint32_t li=0; li<nlist; ++li) {
                        if (pe - p < static_cast<ptrdiff_t>(sizeof(std::uint64_t))) return std::vesper_unexpected(error{ error_code::config_invalid, "List header truncated", "ivf_pq.load"});
                        std::uint64_t sz=0; std::memcpy(&sz, p, sizeof(sz)); p += sizeof(sz);
                        auto& list = impl.inverted_lists_[li]; list.reserve(static_cast<std::size_t>(sz));
                        for (std::uint64_t i=0;i<sz;++i) {
                            if (pe - p < static_cast<ptrdiff_t>(sizeof(std::uint64_t) + m)) return std::vesper_unexpected(error{ error_code::config_invalid, "Entry truncated", "ivf_pq.load"});
                            Impl::InvertedListEntry e; std::memcpy(&e.id, p, sizeof(e.id)); p += sizeof(e.id);
                            e.code.codes.resize(m); std::memcpy(e.code.codes.data(), p, m); p += m;
                            list.push_back(std::move(e));
                        }
                    }
                    if (p != pe) return std::vesper_unexpected(error{ error_code::config_invalid, "Inverted section extra bytes", "ivf_pq.load"});
                    got_inverted = true; break;
                }
                default: {
                    // Unknown section: skip
                    break;
                }
            }
        }

        // Trailer
        char tail[4]; file.read(tail, 4);
        if (file.gcount() != 4 || std::string_view(tail,4) != std::string_view("CHKS",4)) {
            return std::vesper_unexpected(error{ error_code::config_invalid, "Missing checksum trailer", "ivf_pq.load"});
        }
        std::uint64_t checksum_file=0; file.read(reinterpret_cast<char*>(&checksum_file), sizeof(checksum_file));
        if (!file || checksum_file != checksum_calc) {
            return std::vesper_unexpected(error{ error_code::config_invalid, "Checksum mismatch", "ivf_pq.load"});
        }

        impl.n_vectors_ = static_cast<std::size_t>(nvec);
        impl.state_.trained = true;
        if (impl.state_.metadata_json.empty() && !meta_hdr_json.empty()) {
            impl.state_.metadata_json = std::move(meta_hdr_json);
        }
        return std::expected<IvfPqIndex, core::error>(std::move(index));
    }

    IvfPqIndex index;
    index.impl_ = std::make_unique<Impl>();
    auto& impl = *index.impl_;

    impl.state_.dim = dim;
    impl.state_.params.nlist = nlist;
    impl.state_.params.m = m;
    impl.state_.params.nbits = nbits;
    impl.state_.dsub = dsub;

    // Centroids: allow toggling between baseline (temp buffer + copy) and direct streaming into aligned buffer
    // Env: VESPER_IVFPQ_LOAD_STREAM_CENTROIDS -> 1 (default) streams directly; 0 uses baseline copy path
    bool stream_centroids = true;
    if (const char* e = std::getenv("VESPER_IVFPQ_LOAD_STREAM_CENTROIDS")) {
        if (*e == '0') stream_centroids = false;
        else if (*e == '1') stream_centroids = true;
    }

    impl.state_.coarse_centroids = AlignedCentroidBuffer(nlist, dim);
    if (stream_centroids) {
        for (std::uint32_t c = 0; c < nlist; ++c) {
            float* dest = impl.state_.coarse_centroids[c];
            if (!read_and_hash(dest, sizeof(float) * dim)) {
                return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading centroid row", "ivf_pq.load"});
            }
        }
    } else {
        std::vector<float> row(dim);
        for (std::uint32_t c = 0; c < nlist; ++c) {
            if (!read_and_hash(row.data(), sizeof(float) * dim)) {
                return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading centroid row", "ivf_pq.load"});
            }
            std::memcpy(impl.state_.coarse_centroids[c], row.data(), sizeof(float) * dim);
        }
    }

    // PQ codebooks
    const std::size_t ksub = 1u << nbits;
    const std::size_t cb_elems = static_cast<std::size_t>(m) * ksub * dsub;
    std::vector<float> codebooks(cb_elems);
    if (!read_and_hash(codebooks.data(), codebooks.size()*sizeof(float))) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading PQ codebooks", "ivf_pq.load"});
    FastScanPqConfig pqcfg{ .m = m, .nbits = nbits, .block_size = 32, .use_avx512 = core::decide_use_avx512_from_env_and_cpu() };
    impl.state_.pq = std::make_unique<PqImpl>(pqcfg);
    impl.state_.pq->import_pretrained(dsub, std::span<const float>(codebooks.data(), codebooks.size()));

    // OPQ rotation (optional)
    impl.state_.params.use_opq = (flags & 0x1u) != 0;
    if (impl.state_.params.use_opq) {
        impl.state_.rotation_matrix.resize(static_cast<std::size_t>(dim) * dim);
        if (!read_and_hash(impl.state_.rotation_matrix.data(), impl.state_.rotation_matrix.size() * sizeof(float)))
            return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading OPQ rotation", "ivf_pq.load"});
    }

    // Inverted lists
    std::uint32_t lists_count=0; if (!read_and_hash(&lists_count, sizeof(lists_count))) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading lists_count", "ivf_pq.load"});
    if (lists_count != nlist) return std::vesper_unexpected(error{ error_code::config_invalid, "lists_count mismatch", "ivf_pq.load"});
    impl.inverted_lists_.clear(); impl.inverted_lists_.resize(nlist);
    for (std::uint32_t li=0; li<nlist; ++li) {
        std::uint64_t sz=0; if (!read_and_hash(&sz, sizeof(sz))) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading list size", "ivf_pq.load"});
        auto& list = impl.inverted_lists_[li]; list.reserve(static_cast<std::size_t>(sz));
        for (std::uint64_t i=0;i<sz;++i) {
            Impl::InvertedListEntry e; if (!read_and_hash(&e.id, sizeof(e.id))) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading id", "ivf_pq.load"});
            e.code.codes.resize(m); if (!read_and_hash(e.code.codes.data(), m)) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading code", "ivf_pq.load"});
            list.push_back(std::move(e));
        }
    }

    // Read and verify trailer checksum (not hashed)
    char tail[4]; file.read(tail, 4);
    if (file.gcount() != 4 || std::string_view(tail,4) != std::string_view("CHKS",4)) {
        return std::vesper_unexpected(error{ error_code::config_invalid, "Missing checksum trailer", "ivf_pq.load"});
    }
    std::uint64_t checksum_file=0; file.read(reinterpret_cast<char*>(&checksum_file), sizeof(checksum_file));
    if (!file || checksum_file != checksum_calc) {
        return std::vesper_unexpected(error{ error_code::config_invalid, "Checksum mismatch", "ivf_pq.load"});
    }

    impl.n_vectors_ = static_cast<std::size_t>(nvec);
    impl.state_.trained = true;

    return std::expected<IvfPqIndex, core::error>(std::move(index));
}

auto compute_recall(const IvfPqIndex& index,
                   const float* queries, std::size_t n_queries,
                   const std::uint64_t* ground_truth, std::size_t k,
                   const IvfPqSearchParams& params) -> float {

    auto results = index.search_batch(queries, n_queries, params);
    if (!results.has_value()) {
        return 0.0f;
    }

    std::size_t total_found = 0;

    for (std::size_t q = 0; q < n_queries; ++q) {
        const auto& search_results = results.value()[q];
        const std::uint64_t* gt = ground_truth + q * k;

        for (const auto& [id, dist] : search_results) {
            for (std::size_t i = 0; i < k; ++i) {
                if (id == gt[i]) {
                    total_found++;
                    break;
                }
            }
        }
    }

    return static_cast<float>(total_found) / (n_queries * k);
}

auto IvfPqIndex::reconstruct_cluster(std::uint32_t cluster_id,
                                    std::vector<std::uint64_t>& ids,
                                    std::vector<float>& vectors) const
    -> std::expected<void, core::error> {

    using core::error;
    using core::error_code;

    if (!impl_->state_.trained) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Index not trained",
            "ivf_pq"
        });
    }

    if (cluster_id >= impl_->state_.params.nlist) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Invalid cluster ID",
            "ivf_pq"
        });
    }

    // Clear output vectors
    ids.clear();
    vectors.clear();

    // Get inverted list for this cluster
    const auto& inverted_list = impl_->inverted_lists_[cluster_id];
    if (inverted_list.empty()) {
        return {};  // Empty cluster
    }

    // Reserve space
    ids.reserve(inverted_list.size());
    vectors.reserve(inverted_list.size() * impl_->state_.dim);

    // Get coarse centroid for this cluster
    const float* centroid = impl_->state_.coarse_centroids[cluster_id];

    // Reconstruct each vector in the cluster
    for (const auto& entry : inverted_list) {
        ids.push_back(entry.id);

        // Decode PQ code to residual (may be rotated if OPQ was used)
        std::vector<float> residual_code(impl_->state_.dim);
        impl_->state_.pq->decode(entry.code.data(), 1, residual_code.data());

        // If OPQ used, apply inverse rotation to get residual in original space
        std::vector<float> residual_orig;
        const float* residual_ptr = residual_code.data();
        if (impl_->state_.params.use_opq && !impl_->state_.rotation_matrix.empty()) {
            residual_orig.resize(impl_->state_.dim);
            impl_->apply_rotation_T(residual_code.data(), residual_orig.data(), 1);
            residual_ptr = residual_orig.data();
        }

        // Add centroid to get full vector
        for (std::size_t d = 0; d < impl_->state_.dim; ++d) {
            vectors.push_back(centroid[d] + residual_ptr[d]);
        }
    }

    return {};
}

auto IvfPqIndex::reconstruct(std::uint64_t id) const
    -> std::expected<std::vector<float>, core::error> {

    using core::error;
    using core::error_code;

    if (!impl_->state_.trained) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Index not trained",
            "ivf_pq"
        });
    }

    // Search all inverted lists for the ID
    for (std::uint32_t cluster_id = 0; cluster_id < impl_->state_.params.nlist; ++cluster_id) {
        const auto& inverted_list = impl_->inverted_lists_[cluster_id];

        for (const auto& entry : inverted_list) {
            if (entry.id == id) {
                // Found the vector, reconstruct it
                std::vector<float> reconstructed(impl_->state_.dim);

                // Get coarse centroid
                const float* centroid = impl_->state_.coarse_centroids[cluster_id];

                // Decode PQ code to residual (may be rotated if OPQ was used)
                std::vector<float> residual_code(impl_->state_.dim);
                impl_->state_.pq->decode(entry.code.data(), 1, residual_code.data());

                // If OPQ used, apply inverse rotation to get residual in original space
                const float* residual_ptr = residual_code.data();
                std::vector<float> residual_orig;
                if (impl_->state_.params.use_opq && !impl_->state_.rotation_matrix.empty()) {
                    residual_orig.resize(impl_->state_.dim);
                    impl_->apply_rotation_T(residual_code.data(), residual_orig.data(), 1);
                    residual_ptr = residual_orig.data();
                }

                // Add centroid to get full vector
                for (std::size_t d = 0; d < impl_->state_.dim; ++d) {
                    reconstructed[d] = centroid[d] + residual_ptr[d];
                }

                return reconstructed;
            }
        }
    }

    return std::vesper_unexpected(error{
        error_code::not_found,
        "Vector ID not found",
        "ivf_pq"
    });
}

auto IvfPqIndex::get_num_clusters() const -> std::expected<std::uint32_t, core::error> {
    if (!impl_->state_.trained) {
        return std::vesper_unexpected(core::error{
            core::error_code::not_initialized,
            "Index not trained",
            "ivf_pq.get_num_clusters"
        });
    }
    return static_cast<std::uint32_t>(impl_->state_.params.nlist);
}

auto IvfPqIndex::get_dimension() const -> std::expected<std::size_t, core::error> {
    if (!impl_->state_.trained) {
        return std::vesper_unexpected(core::error{
            core::error_code::not_initialized,
            "Index not trained",
            "ivf_pq.get_dimension"
        });
    }
    return impl_->state_.dim;
}

auto IvfPqIndex::get_cluster_assignment(std::uint64_t id) const
    -> std::expected<std::uint32_t, core::error> {
    if (!impl_->state_.trained) {
        return std::vesper_unexpected(core::error{
            core::error_code::not_initialized,
            "Index not trained",
            "ivf_pq.get_cluster_assignment"
        });
    }

    // Search through all inverted lists
    for (std::uint32_t cluster_id = 0; cluster_id < impl_->inverted_lists_.size(); ++cluster_id) {
        const auto& inverted_list = impl_->inverted_lists_[cluster_id];
        for (const auto& entry : inverted_list) {
            if (entry.id == id) {
                return cluster_id;
            }
        }
    }

    return std::vesper_unexpected(core::error{
        core::error_code::not_found,
        "Vector ID not found",
        "ivf_pq.get_cluster_assignment"
    });
}

auto IvfPqIndex::get_cluster_centroid(std::uint32_t cluster_id) const
    -> std::expected<std::vector<float>, core::error> {
    if (!impl_->state_.trained) {
        return std::vesper_unexpected(core::error{
            core::error_code::not_initialized,
            "Index not trained",
            "ivf_pq.get_cluster_centroid"
        });
    }

    if (cluster_id >= impl_->state_.params.nlist) {
        return std::vesper_unexpected(core::error{
            core::error_code::out_of_range,
            "Cluster ID out of range",
            "ivf_pq.get_cluster_centroid"
        });
    }

    const float* centroid = impl_->state_.coarse_centroids[cluster_id];
    std::vector<float> result(impl_->state_.dim);
    std::memcpy(result.data(), centroid, impl_->state_.dim * sizeof(float));
    return result;
}

auto IvfPqIndex::update_cluster_centroid(std::uint32_t cluster_id, const float* centroid)
    -> std::expected<void, core::error> {
    if (!impl_->state_.trained) {
        return std::vesper_unexpected(core::error{
            core::error_code::not_initialized,
            "Index not trained",
            "ivf_pq.update_cluster_centroid"
        });
    }

    if (cluster_id >= impl_->state_.params.nlist) {
        return std::vesper_unexpected(core::error{
            core::error_code::out_of_range,
            "Cluster ID out of range",
            "ivf_pq.update_cluster_centroid"
        });
    }

    if (!centroid) {
        return std::vesper_unexpected(core::error{
            core::error_code::invalid_argument,
            "Null centroid pointer",
            "ivf_pq.update_cluster_centroid"
        });
    }

    // Update the centroid
    float* target = impl_->state_.coarse_centroids[cluster_id];
    std::memcpy(target, centroid, impl_->state_.dim * sizeof(float));

    return {};
}

auto IvfPqIndex::reassign_vector(std::uint64_t id, std::uint32_t new_cluster)
    -> std::expected<void, core::error> {
    if (!impl_->state_.trained) {
        return std::vesper_unexpected(core::error{
            core::error_code::not_initialized,
            "Index not trained",
            "ivf_pq.reassign_vector"
        });
    }

    if (new_cluster >= impl_->state_.params.nlist) {
        return std::vesper_unexpected(core::error{
            core::error_code::out_of_range,
            "New cluster ID out of range",
            "ivf_pq.reassign_vector"
        });
    }

    std::lock_guard<std::mutex> lock(impl_->lists_mutex_);

    // Find the vector in current cluster
    Impl::InvertedListEntry found_entry;
    bool found = false;
    std::uint32_t old_cluster = 0;

    for (std::uint32_t cluster_id = 0; cluster_id < impl_->inverted_lists_.size(); ++cluster_id) {
        auto& inverted_list = impl_->inverted_lists_[cluster_id];
        auto it = std::find_if(inverted_list.begin(), inverted_list.end(),
                               [id](const auto& entry) { return entry.id == id; });

        if (it != inverted_list.end()) {
            found_entry = *it;
            inverted_list.erase(it);
            old_cluster = cluster_id;
            found = true;
            break;
        }
    }

    if (!found) {
        return std::vesper_unexpected(core::error{
            core::error_code::not_found,
            "Vector ID not found",
            "ivf_pq.reassign_vector"
        });
    }

    // Recompute the residual for the new cluster
    // First reconstruct the original vector
    const float* old_centroid = impl_->state_.coarse_centroids[old_cluster];
    std::vector<float> original_vector(impl_->state_.dim);
    std::vector<float> residual(impl_->state_.dim);

    // Decode old residual (may be rotated if OPQ was used)
    impl_->state_.pq->decode(found_entry.code.data(), 1, residual.data());

    // If OPQ used, invert rotation to get residual in original space
    std::vector<float> residual_orig;
    const float* residual_ptr = residual.data();
    if (impl_->state_.params.use_opq && !impl_->state_.rotation_matrix.empty()) {
        residual_orig.resize(impl_->state_.dim);
        impl_->apply_rotation_T(residual.data(), residual_orig.data(), 1);
        residual_ptr = residual_orig.data();
    }

    // Reconstruct original vector
    for (std::size_t d = 0; d < impl_->state_.dim; ++d) {
        original_vector[d] = old_centroid[d] + residual_ptr[d];
    }

    // Compute new residual with respect to new centroid (original space)
    const float* new_centroid = impl_->state_.coarse_centroids[new_cluster];
    for (std::size_t d = 0; d < impl_->state_.dim; ++d) {
        residual[d] = original_vector[d] - new_centroid[d];
    }

    // If OPQ used, rotate new residual before encoding
    const float* enc_ptr = residual.data();
    std::vector<float> residual_rot;
    if (impl_->state_.params.use_opq && !impl_->state_.rotation_matrix.empty()) {
        residual_rot.resize(impl_->state_.dim);
        impl_->apply_rotation(residual.data(), residual_rot.data(), 1);
        enc_ptr = residual_rot.data();
    }

    // Encode new residual
    Impl::InvertedListEntry new_entry;
    new_entry.id = id;
    new_entry.code.codes.resize(impl_->state_.params.m);
    impl_->state_.pq->encode(enc_ptr, 1, new_entry.code.codes.data());

    // Add to new cluster
    impl_->inverted_lists_[new_cluster].push_back(new_entry);

    return {};
}

#ifndef VESPER_NO_ROARING
auto IvfPqIndex::compact_inverted_list(std::uint32_t cluster_id, const roaring_bitmap_t* deleted_ids)
    -> std::expected<void, core::error> {
    if (!impl_->state_.trained) {
        return std::vesper_unexpected(core::error{
            core::error_code::not_initialized,
            "Index not trained",
            "ivf_pq.compact_inverted_list"
        });
    }

    if (cluster_id >= impl_->state_.params.nlist) {
        return std::vesper_unexpected(core::error{
            core::error_code::out_of_range,
            "Cluster ID out of range",
            "ivf_pq.compact_inverted_list"
        });
    }

    std::lock_guard<std::mutex> lock(impl_->lists_mutex_);

    auto& inverted_list = impl_->inverted_lists_[cluster_id];

    // Remove deleted entries
    auto new_end = std::remove_if(inverted_list.begin(), inverted_list.end(),
        [&deleted_ids](const auto& entry) {
            return deleted_ids && roaring_bitmap_contains(deleted_ids, static_cast<uint32_t>(entry.id));
        });

    std::size_t removed = std::distance(new_end, inverted_list.end());
    inverted_list.erase(new_end, inverted_list.end());

    // Shrink to fit if we removed many entries
    if (removed > inverted_list.size() / 4) {
        inverted_list.shrink_to_fit();
    }

    // Update global vector count
    impl_->n_vectors_ -= removed;

    return {};
}
#endif // VESPER_NO_ROARING


// Inverse rotation helper (definition placed after other methods)
auto IvfPqIndex::Impl::apply_rotation_T(const float* input, float* output, std::size_t n) const
    -> void {
    if (state_.rotation_matrix.empty()) {
        std::memcpy(output, input, n * state_.dim * sizeof(float));
        return;
    }
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(n); ++i) {
        const float* vec_in = input + i * state_.dim;
        float* vec_out = output + i * state_.dim;
        for (std::size_t j = 0; j < state_.dim; ++j) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < state_.dim; ++k) {
                sum += vec_in[k] * state_.rotation_matrix[j * state_.dim + k];
            }
            vec_out[j] = sum;
        }
    }
}

} // namespace vesper::index