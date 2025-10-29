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
#include "vesper/core/platform_utils.hpp"

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
#include <unordered_map>

#include <filesystem>
#include <chrono>

#include <iostream>
#include <limits>
#include <cstdlib>
#include <functional>
#include "vesper/expected_polyfill.hpp"

#include <atomic>


#include <new>

#ifdef VESPER_HAS_ZSTD
#include <zstd.h>
#endif
#ifdef VESPER_HAS_CBLAS
#include <cblas.h>
#endif


#include "vesper/index/projection_assigner.hpp"
#include "vesper/index/fast_hadamard.hpp"


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
    std::uint32_t kd_leaf_size_{64};  // Optimized from 256 based on benchmarking

    // Van Emde Boas layout for cache-oblivious traversal
    std::vector<KDNode> kd_nodes_veb_;
    std::vector<int> kd_veb_map_;  // Maps original node index to VEB position
    bool use_veb_layout_{false};
    std::vector<float> kd_proj_P_; // PCA rows (p=8) row-major size = 8*dim, orthonormal rows

    // Hierarchical KD-tree structures for improved performance
    struct HierarchicalKDTree {
        // Top-level tree over cluster representatives
        std::vector<KDNode> top_nodes;
        std::vector<std::uint32_t> top_order;  // indices into representatives
        int top_root{-1};
        std::vector<std::vector<float>> representatives;  // cluster centers for top level
        std::vector<std::uint32_t> rep_to_clusters;  // maps representative to cluster range

        // Sub-trees for each top-level leaf
        struct SubTree {
            std::vector<KDNode> nodes;
            std::vector<std::uint32_t> order;  // indices into actual centroids
            int root{-1};
            std::uint32_t centroid_begin{0};  // range in original centroids
            std::uint32_t centroid_end{0};
        };
        std::vector<SubTree> subtrees;

        // Configuration
        std::uint32_t num_representatives{128};  // number of top-level clusters
        std::uint32_t subtree_leaf_size{32};     // leaf size for sub-trees
        bool enabled{false};
    } hierarchical_kd_;

    // Hierarchical KD-tree methods
    void hierarchical_kd_build_();
    std::uint32_t hierarchical_kd_nearest_(const float* vec) const;
    void hierarchical_kd_assign_batch_(const float* data, std::size_t n, std::uint32_t* out_assignments) const;


    // Packed leaf centroid panels for 128-D fast path (SoA panels per leaf)
    std::vector<float> kd_leaf_panels_;
    std::vector<std::uint64_t> kd_leaf_panel_offset_; // offset (in floats) into kd_leaf_panels_
    std::vector<std::uint32_t> kd_leaf_panel_blocks_; // number of 16-centroid blocks per leaf

    // Projection-based coarse assignment state (when coarse_assigner==Projection)
    std::uint32_t proj_dim_{16};                 // effective projection dim
    std::vector<float> proj_rows_;               // row-major [proj_dim_ x dim], orthonormal rows
    std::vector<float> proj_centroids_rm_;       // row-major [nlist x proj_dim_]
    std::vector<float> proj_centroid_norms_;     // [nlist] = ||proj_centroid||^2
    std::vector<float> proj_centroids_pack8_;    // packed panels [ceil(nlist/8) x proj_dim_ x 8] for AVX microkernel

    // Fast rotational quantization for centroids (2.3× speedup)
    std::unique_ptr<FastRotationalQuantizer> centroid_quantizer_;
    std::vector<std::uint8_t> quantized_centroids_;   // [nlist × dim] quantized centroids
    std::vector<float> centroid_scales_;              // [nlist] scales for reconstruction
    std::vector<float> centroid_offsets_;             // [nlist] offsets for reconstruction
    bool use_rotational_quantization_{false};

    void kd_build_();
    void kd_build_veb_layout_();  // Build Van Emde Boas cache-oblivious layout
    std::uint32_t kd_nearest_(const float* vec) const;
    std::uint32_t kd_nearest_veb_(const float* vec) const;  // VEB-optimized traversal
    std::uint32_t kd_nearest_approx_(const float* vec, float early_termination_factor = 1.2f) const;
    void kd_assign_batch_(const float* data, std::size_t n, std::uint32_t* out_assignments) const;
    bool should_use_hierarchical_kd_() const;

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

    // Debug-only explain hooks
    auto debug_explain_adc_rank(const float* query, std::uint64_t target_id) const
        -> std::expected<std::pair<std::size_t, float>, core::error>;

    // Debug-only: centroid rank of GT's assigned centroid among all centroids
    auto debug_explain_centroid_rank(const float* query, std::uint64_t target_id) const
        -> std::expected<std::pair<std::size_t, std::uint32_t>, core::error>;

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


    const bool dbg = ([](){ auto v = vesper::core::safe_getenv("VESPER_IVFPQ_DEBUG"); return v && !v->empty() && ((*v)[0] != '0'); })();
    if (dbg) { state_.params.verbose = true; }

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

    // Disable ANN coarse assignment for small nlist values where bruteforce is more efficient
    // Empirical testing shows bruteforce outperforms ANN assigners for nlist < 1024
    if (params.nlist < 1024 && params.use_centroid_ann) {
        state_.params.use_centroid_ann = false;
    }

    // Step 1: Train coarse quantizer

    if (dbg) {
        std::cerr << "[IVFPQ][train] starting coarse k-means" << std::endl;
    }

    if (auto result = train_coarse_quantizer(data, n, dim); !result.has_value()) {
        return std::vesper_unexpected(result.error());
    }

    if (dbg) {
        std::cerr << "[IVFPQ][train] coarse k-means completed" << std::endl;
    }


    // Step 2: Learn OPQ rotation if enabled and PCA init selected (on residuals)
    if (params.use_opq && params.opq_init == OpqInit::PCA) {

        if (dbg) {
            std::cerr << "[IVFPQ][train] OPQ(PCA) residuals computation + rotation learning" << std::endl;
        }
        // Debug: after learning rotation, report how non-identity it is (Frobenius norm of R-I)
        auto report_rotation = [&](const std::vector<float>& R, std::size_t d){
            double fro_sq = 0.0; std::size_t idx = 0;
            for (std::size_t i = 0; i < d; ++i) {
                for (std::size_t j = 0; j < d; ++j, ++idx) {
                    const double ref = (i==j) ? 1.0 : 0.0;
                    const double diff = static_cast<double>(R[idx]) - ref;
                    fro_sq += diff * diff;
                }
            }
            if (dbg) {
                std::cerr << "[IVFPQ][train][diag] OPQ ||R-I||_F=" << std::sqrt(fro_sq) << " dim=" << d << "\n";
            }
        };
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


    if (dbg) {
        std::cerr << "[IVFPQ][train] starting PQ training" << std::endl;
    }

    // Step 3: Train product quantizer
    if (auto result = train_product_quantizer(data, n); !result.has_value()) {
        return std::vesper_unexpected(result.error());
    }

    if (dbg) {
        std::cerr << "[IVFPQ][train] PQ trained" << std::endl;
    }


    // Initialize inverted lists
    if (dbg) {
        std::cerr << "[IVFPQ][train] initialized inverted lists: nlist=" << params.nlist << std::endl;
    }

    inverted_lists_.clear();
    inverted_lists_.resize(params.nlist);

    state_.trained = true;

    const auto end_time = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration<float>(end_time - start_time);

    IvfPqTrainStats stats;

    if (dbg) {
        std::cerr << "[IVFPQ][train] complete in " << duration.count() << " sec" << std::endl;
    }

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
        .use_parallel = true,
        .n_threads = 0,
        .verbose = state_.params.verbose,
        .init_method = static_cast<KmeansElkan::Config::InitMethod>(state_.params.kmeans_init_method),
        .kmeans_parallel_rounds = state_.params.kmeans_parallel_rounds,
        .kmeans_parallel_oversampling = state_.params.kmeans_parallel_oversampling
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

        // Build Van Emde Boas layout for better cache performance
        if (kd_nodes_.size() > 64) {  // Only use VEB for larger trees
            kd_build_veb_layout_();
            use_veb_layout_ = true;
        }

        // Build hierarchical KD-tree for larger datasets
        if (should_use_hierarchical_kd_()) {
            hierarchical_kd_build_();
        }

        // Initialize fast rotational quantizer for centroids (optional speedup)
        // Accuracy-first default: disabled unless explicitly enabled via env.
        auto use_rq_env = vesper::core::safe_getenv("VESPER_USE_ROTATIONAL_QUANTIZATION");
        const bool enable_rq = (use_rq_env && !use_rq_env->empty() && (*use_rq_env)[0] == '1');
        if (enable_rq) {
            centroid_quantizer_ = std::make_unique<FastRotationalQuantizer>();
            FastRotationalQuantizer::TrainParams rq_params;
            rq_params.seed = 42;
            rq_params.num_rotations = 3;  // Multiple rotation blocks for better quantization
            rq_params.normalize = true;

            // Train on coarse centroids
            centroid_quantizer_->train(
                state_.coarse_centroids.data(),
                state_.params.nlist,
                state_.dim,
                rq_params
            );

            // Quantize all centroids
            quantized_centroids_.resize(state_.params.nlist * state_.dim);
            centroid_scales_.resize(state_.params.nlist);
            centroid_offsets_.resize(state_.params.nlist);

            centroid_quantizer_->quantize_batch(
                state_.coarse_centroids.data(),
                state_.params.nlist,
                quantized_centroids_.data(),
                centroid_scales_.data(),
                centroid_offsets_.data()
            );

            use_rotational_quantization_ = true;
        }
    } else {
        kd_nodes_.clear(); kd_order_.clear(); kd_root_ = -1;
        kd_nodes_veb_.clear(); kd_veb_map_.clear();
        use_veb_layout_ = false;
        hierarchical_kd_.enabled = false;
        use_rotational_quantization_ = false;
    }

    return {};

}

void IvfPqIndex::Impl::kd_build_veb_layout_() {
    if (kd_nodes_.empty() || kd_root_ == -1) return;

    const size_t num_nodes = kd_nodes_.size();
    kd_nodes_veb_.clear();
    kd_nodes_veb_.reserve(num_nodes);
    kd_veb_map_.resize(num_nodes, -1);

    // Helper to estimate subtree size
    std::function<size_t(int)> estimate_subtree_size;
    estimate_subtree_size = [&](int node_idx) -> size_t {
        if (node_idx < 0) return 0;
        const KDNode& node = kd_nodes_[static_cast<size_t>(node_idx)];
        if (node.leaf) return 1;

        // Simple heuristic: count nodes up to depth 3
        size_t count = 1;
        std::queue<std::pair<int, int>> q;
        q.push({node_idx, 0});

        while (!q.empty() && count < 64) {
            auto [idx, depth] = q.front();
            q.pop();
            if (depth >= 3) break;

            const KDNode& n = kd_nodes_[static_cast<size_t>(idx)];
            if (n.left >= 0) {
                count++;
                q.push({n.left, depth + 1});
            }
            if (n.right >= 0) {
                count++;
                q.push({n.right, depth + 1});
            }
        }
        return count;
    };

    // Recursive VEB layout construction
    std::function<void(int, size_t&)> build_veb;
    build_veb = [&](int node_idx, size_t& veb_pos) {
        if (node_idx < 0) return;

        const KDNode& src = kd_nodes_[static_cast<size_t>(node_idx)];

        // Allocate position in VEB layout
        size_t my_pos = veb_pos++;
        kd_veb_map_[static_cast<size_t>(node_idx)] = static_cast<int>(my_pos);
        kd_nodes_veb_.push_back(src);

        // For cache efficiency, layout children close to parent
        // Small subtrees are laid out contiguously
        if (src.left >= 0 && src.right >= 0) {
            // Estimate subtree sizes for balanced layout
            size_t left_size = estimate_subtree_size(src.left);
            size_t right_size = estimate_subtree_size(src.right);

            // Van Emde Boas: split at sqrt(n) boundary
            if (left_size + right_size > 16) {
                // Large subtree: recursive VEB layout
                build_veb(src.left, veb_pos);
                build_veb(src.right, veb_pos);
            } else {
                // Small subtree: contiguous layout for cache line packing
                build_veb(src.left, veb_pos);
                build_veb(src.right, veb_pos);
            }
        } else if (src.left >= 0) {
            build_veb(src.left, veb_pos);
        } else if (src.right >= 0) {
            build_veb(src.right, veb_pos);
        }
    };

    size_t veb_pos = 0;
    build_veb(kd_root_, veb_pos);

    // Update child pointers to use VEB indices
    for (auto& node : kd_nodes_veb_) {
        if (node.left >= 0) {
            node.left = kd_veb_map_[static_cast<size_t>(node.left)];
        }
        if (node.right >= 0) {
            node.right = kd_veb_map_[static_cast<size_t>(node.right)];
        }
    }

    // VEB root is always at position 0
    if (!kd_nodes_veb_.empty()) {
        kd_root_ = 0;
    }
}

void IvfPqIndex::Impl::kd_build_() {
    // Optional env override for leaf size and allow quick sweeps externally
    if (auto v = vesper::core::safe_getenv("VESPER_KD_LEAF_SIZE")) {
        const unsigned s = static_cast<unsigned>(std::strtoul(v->c_str(), nullptr, 10));
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

    // Prepack 128D leaf centroids into 16-wide SoA panels for AVX2 kernels
    kd_leaf_panels_.clear();
    kd_leaf_panel_offset_.assign(kd_nodes_.size(), 0);
    kd_leaf_panel_blocks_.assign(kd_nodes_.size(), 0);
    if (dim == 128) {
        std::size_t total_floats = 0;
        for (std::size_t u = 0; u < kd_nodes_.size(); ++u) {
            const KDNode& nd = kd_nodes_[u];
            if (!nd.leaf) continue;
            const std::uint32_t L = nd.end - nd.begin;
            const std::uint32_t blocks = L / 16u;
            if (blocks == 0) continue;
            total_floats += static_cast<std::size_t>(blocks) * 128u * 16u; // 2048 floats per block
        }
        kd_leaf_panels_.resize(total_floats);
        std::size_t off = 0;
        for (std::size_t u = 0; u < kd_nodes_.size(); ++u) {
            const KDNode& nd = kd_nodes_[u];
            if (!nd.leaf) continue;
            const std::uint32_t L = nd.end - nd.begin;
            const std::uint32_t blocks = L / 16u;
            if (blocks == 0) continue;
            kd_leaf_panel_offset_[u] = static_cast<std::uint64_t>(off);
            kd_leaf_panel_blocks_[u] = blocks;
            for (std::uint32_t b = 0; b < blocks; ++b) {
                float* panel = kd_leaf_panels_.data() + off;
                for (int b8 = 0; b8 < 16; ++b8) {
                    for (int lane = 0; lane < 16; ++lane) {
                        const std::uint32_t idx = nd.begin + b * 16u + static_cast<std::uint32_t>(lane);
                        const float* src = kd_leaf_centroids_[idx] + b8 * 8;
                        std::memcpy(panel + (static_cast<std::size_t>(b8) * 16 + static_cast<std::size_t>(lane)) * 8,
                                    src, sizeof(float) * 8);
                    }
                }
                off += 128u * 16u;
            }
        }
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

#ifdef __AVX2__
inline void l2_128x8_avx2_panel(const float* q, const float* panel_ptr, int lane_offset, float out[8]) noexcept {
    __m256 acc0=_mm256_setzero_ps(), acc1=_mm256_setzero_ps(), acc2=_mm256_setzero_ps(), acc3=_mm256_setzero_ps();
    __m256 acc4=_mm256_setzero_ps(), acc5=_mm256_setzero_ps(), acc6=_mm256_setzero_ps(), acc7=_mm256_setzero_ps();
    for (int b8=0; b8<16; ++b8) {
        const float* qblk = q + b8*8;
        __m256 qv = _mm256_loadu_ps(qblk);
        const std::size_t base = static_cast<std::size_t>(b8)*16 + static_cast<std::size_t>(lane_offset);
        __m256 d0=_mm256_sub_ps(qv, _mm256_loadu_ps(panel_ptr + (base + 0)*8)); acc0=_mm256_fmadd_ps(d0,d0,acc0);
        __m256 d1=_mm256_sub_ps(qv, _mm256_loadu_ps(panel_ptr + (base + 1)*8)); acc1=_mm256_fmadd_ps(d1,d1,acc1);
        __m256 d2=_mm256_sub_ps(qv, _mm256_loadu_ps(panel_ptr + (base + 2)*8)); acc2=_mm256_fmadd_ps(d2,d2,acc2);
        __m256 d3=_mm256_sub_ps(qv, _mm256_loadu_ps(panel_ptr + (base + 3)*8)); acc3=_mm256_fmadd_ps(d3,d3,acc3);
        __m256 d4=_mm256_sub_ps(qv, _mm256_loadu_ps(panel_ptr + (base + 4)*8)); acc4=_mm256_fmadd_ps(d4,d4,acc4);
        __m256 d5=_mm256_sub_ps(qv, _mm256_loadu_ps(panel_ptr + (base + 5)*8)); acc5=_mm256_fmadd_ps(d5,d5,acc5);
        __m256 d6=_mm256_sub_ps(qv, _mm256_loadu_ps(panel_ptr + (base + 6)*8)); acc6=_mm256_fmadd_ps(d6,d6,acc6);
        __m256 d7=_mm256_sub_ps(qv, _mm256_loadu_ps(panel_ptr + (base + 7)*8)); acc7=_mm256_fmadd_ps(d7,d7,acc7);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc0); out[0]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    _mm256_store_ps(tmp, acc1); out[1]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    _mm256_store_ps(tmp, acc2); out[2]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    _mm256_store_ps(tmp, acc3); out[3]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    _mm256_store_ps(tmp, acc4); out[4]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    _mm256_store_ps(tmp, acc5); out[5]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    _mm256_store_ps(tmp, acc6); out[6]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    _mm256_store_ps(tmp, acc7); out[7]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
}
#endif

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


#ifdef __AVX2__
inline void l2_128x8_avx2_panel(const float* q, const float* panel_ptr, int lane_offset, float out[8]) noexcept {
    __m256 acc0=_mm256_setzero_ps(), acc1=_mm256_setzero_ps(), acc2=_mm256_setzero_ps(), acc3=_mm256_setzero_ps();
    __m256 acc4=_mm256_setzero_ps(), acc5=_mm256_setzero_ps(), acc6=_mm256_setzero_ps(), acc7=_mm256_setzero_ps();
    for (int b8=0; b8<16; ++b8) {
        const float* qblk = q + b8*8;
        __m256 qv = _mm256_loadu_ps(qblk);
        const std::size_t base = static_cast<std::size_t>(b8)*16 + static_cast<std::size_t>(lane_offset);
        __m256 d0=_mm256_sub_ps(qv, _mm256_loadu_ps(panel_ptr + (base + 0)*8)); acc0=_mm256_fmadd_ps(d0,d0,acc0);
        __m256 d1=_mm256_sub_ps(qv, _mm256_loadu_ps(panel_ptr + (base + 1)*8)); acc1=_mm256_fmadd_ps(d1,d1,acc1);
        __m256 d2=_mm256_sub_ps(qv, _mm256_loadu_ps(panel_ptr + (base + 2)*8)); acc2=_mm256_fmadd_ps(d2,d2,acc2);
        __m256 d3=_mm256_sub_ps(qv, _mm256_loadu_ps(panel_ptr + (base + 3)*8)); acc3=_mm256_fmadd_ps(d3,d3,acc3);
        __m256 d4=_mm256_sub_ps(qv, _mm256_loadu_ps(panel_ptr + (base + 4)*8)); acc4=_mm256_fmadd_ps(d4,d4,acc4);
        __m256 d5=_mm256_sub_ps(qv, _mm256_loadu_ps(panel_ptr + (base + 5)*8)); acc5=_mm256_fmadd_ps(d5,d5,acc5);
        __m256 d6=_mm256_sub_ps(qv, _mm256_loadu_ps(panel_ptr + (base + 6)*8)); acc6=_mm256_fmadd_ps(d6,d6,acc6);
        __m256 d7=_mm256_sub_ps(qv, _mm256_loadu_ps(panel_ptr + (base + 7)*8)); acc7=_mm256_fmadd_ps(d7,d7,acc7);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc0); out[0]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    _mm256_store_ps(tmp, acc1); out[1]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    _mm256_store_ps(tmp, acc2); out[2]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    _mm256_store_ps(tmp, acc3); out[3]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    _mm256_store_ps(tmp, acc4); out[4]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    _mm256_store_ps(tmp, acc5); out[5]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    _mm256_store_ps(tmp, acc6); out[6]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    _mm256_store_ps(tmp, acc7); out[7]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
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


std::uint32_t IvfPqIndex::Impl::kd_nearest_veb_(const float* vec) const {
    // VEB-optimized traversal with better cache locality
    if (kd_nodes_veb_.empty() || kd_root_ != 0) return kd_nearest_(vec);

    const auto& ops = kernels::select_backend_auto();
    const std::size_t dim = state_.dim;

    std::uint32_t best_idx = 0;
    float best = std::numeric_limits<float>::infinity();

    // Use the same priority queue structure
    struct KDNodeCandidate {
        int node_id;
        float lower_bound;
        bool operator>(const KDNodeCandidate& other) const {
            return lower_bound > other.lower_bound;
        }
    };

    std::priority_queue<KDNodeCandidate,
                       std::vector<KDNodeCandidate>,
                       std::greater<KDNodeCandidate>> queue;

    // Start with root (always at index 0 in VEB layout)
    queue.push({0, 0.0f});
    kd_nodes_pushed_.fetch_add(1, std::memory_order_relaxed);

    while (!queue.empty()) {
        const int nid = queue.top().node_id;
        queue.pop();

        if (nid < 0 || static_cast<size_t>(nid) >= kd_nodes_veb_.size()) continue;

        kd_nodes_popped_.fetch_add(1, std::memory_order_relaxed);

        const KDNode& node = kd_nodes_veb_[static_cast<size_t>(nid)];

        // Compute lower bound for this node
        float lb_curr = 0.0f;
        float d2f_curr = ops.l2_sq(std::span(vec, dim), std::span<const float>(node.center.data(), dim));
        lb_curr = d2f_curr - node.radius2;

        // Prune if this node can't contain better solution
        if (lb_curr >= best) continue;

        if (node.leaf) {
            // Process leaf node - scan centroids
            for (std::uint32_t it = node.begin; it < node.end; ++it) {
                const float d = ops.l2_sq(std::span(vec, dim), kd_leaf_centroids_.get_centroid(it));
                if (d < best) {
                    best = d;
                    best_idx = kd_order_[it];
                }
            }
            kd_leaves_scanned_.fetch_add(static_cast<std::uint64_t>(node.end - node.begin), std::memory_order_relaxed);
        } else {
            // Internal node - add children to queue
            // Children are laid out nearby in VEB layout for cache efficiency
            if (node.left >= 0) {
                const KDNode& left_child = kd_nodes_veb_[static_cast<size_t>(node.left)];
                float lb_left = ops.l2_sq(std::span(vec, dim), std::span<const float>(left_child.center.data(), dim)) - left_child.radius2;
                if (lb_left < best) {
                    queue.push({node.left, lb_left});
                    kd_nodes_pushed_.fetch_add(1, std::memory_order_relaxed);
                }
            }

            if (node.right >= 0) {
                const KDNode& right_child = kd_nodes_veb_[static_cast<size_t>(node.right)];
                float lb_right = ops.l2_sq(std::span(vec, dim), std::span<const float>(right_child.center.data(), dim)) - right_child.radius2;
                if (lb_right < best) {
                    queue.push({node.right, lb_right});
                    kd_nodes_pushed_.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }

    return best_idx;
}

std::uint32_t IvfPqIndex::Impl::kd_nearest_(const float* vec) const {
    const auto& ops = kernels::select_backend_auto();
    const std::size_t dim = state_.dim;

    std::uint32_t best_idx = 0;
    float best = std::numeric_limits<float>::infinity();

    const bool use_proj = ([](){ auto v = vesper::core::safe_getenv("VESPER_KD_USE_PROJ"); return v && !v->empty() && ((*v)[0] == '1'); })();

    // Precompute 8-D projection of query once (if enabled)
    float qproj[8] = {0};
    if (use_proj) {
        for (std::size_t k = 0; k < 8; ++k) {
            const float* pk = kd_proj_P_.data() + k * dim;
            double s = 0.0; for (std::size_t d = 0; d < dim; ++d) s += static_cast<double>(pk[d]) * static_cast<double>(vec[d]);
            qproj[k] = static_cast<float>(s);
        }
    }

    // Priority queue for candidate nodes (min-heap by lower bound)
    struct KDNodeCandidate {
        int node_id;
        float lower_bound;

        // For min-heap: greater comparison gives smallest element at top
        bool operator>(const KDNodeCandidate& other) const {
            return lower_bound > other.lower_bound;
        }
    };

    std::priority_queue<KDNodeCandidate,
                       std::vector<KDNodeCandidate>,
                       std::greater<KDNodeCandidate>> pq;

    // Reserve capacity to avoid reallocation
    std::vector<KDNodeCandidate> container;
    container.reserve(kd_nodes_.size());
    std::priority_queue<KDNodeCandidate,
                       std::vector<KDNodeCandidate>,
                       std::greater<KDNodeCandidate>> queue(std::greater<KDNodeCandidate>(), std::move(container));

    auto heap_push = [&](int nid, float lb){
        if (nid < 0) return;
        if (lb < 0.0f) lb = 0.0f;

        // Prefetch node data for future access (3-iteration lookahead)
        if (static_cast<size_t>(nid) < kd_nodes_.size()) {
            const KDNode* node_ptr = &kd_nodes_[static_cast<size_t>(nid)];
            #if defined(__builtin_prefetch)
                __builtin_prefetch(node_ptr, 0, 3);  // Read, high temporal locality
                __builtin_prefetch(&node_ptr->center[0], 0, 2);
                __builtin_prefetch(&node_ptr->bb_min[0], 0, 1);
                __builtin_prefetch(&node_ptr->bb_max[0], 0, 1);

                // Speculatively prefetch children if they exist
                if (node_ptr->left >= 0 && static_cast<size_t>(node_ptr->left) < kd_nodes_.size()) {
                    __builtin_prefetch(&kd_nodes_[static_cast<size_t>(node_ptr->left)], 0, 1);
                }
                if (node_ptr->right >= 0 && static_cast<size_t>(node_ptr->right) < kd_nodes_.size()) {
                    __builtin_prefetch(&kd_nodes_[static_cast<size_t>(node_ptr->right)], 0, 1);
                }
            #elif defined(_MSC_VER) && (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
                _mm_prefetch(reinterpret_cast<const char*>(node_ptr), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(&node_ptr->center[0]), _MM_HINT_T1);
                _mm_prefetch(reinterpret_cast<const char*>(&node_ptr->bb_min[0]), _MM_HINT_T2);
                _mm_prefetch(reinterpret_cast<const char*>(&node_ptr->bb_max[0]), _MM_HINT_T2);
            #endif
        }

        queue.push({nid, lb});
        kd_nodes_pushed_.fetch_add(1, std::memory_order_relaxed);
    };

    auto heap_pop_min = [&]() -> int {
        if (queue.empty()) return -1;
        int nid = queue.top().node_id;
        queue.pop();
        return nid;
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
        heap_push(kd_root_, lb);
    }

    while (true) {
        const int nid = heap_pop_min();
        if (nid < 0) break;

        kd_nodes_popped_.fetch_add(1, std::memory_order_relaxed);

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
        if (lb_curr >= best) continue; // prune

        if (node.leaf) {
            // Prefetch leaf centroids before processing
            #if defined(__builtin_prefetch)
                for (std::uint32_t i = node.begin; i < std::min(node.begin + 8u, node.end); ++i) {
                    __builtin_prefetch(kd_leaf_centroids_[i], 0, 1);
                }
            #elif defined(_MSC_VER) && (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
                for (std::uint32_t i = node.begin; i < std::min(node.begin + 8u, node.end); ++i) {
                    _mm_prefetch(reinterpret_cast<const char*>(kd_leaf_centroids_[i]), _MM_HINT_T2);
                }
            #endif

            if (dim == 128) {
                const std::size_t uidx = static_cast<std::size_t>(nid);
                const std::uint32_t L = node.end - node.begin;
                const std::uint32_t blocks = (uidx < kd_leaf_panel_blocks_.size()) ? kd_leaf_panel_blocks_[uidx] : 0u;
            #ifdef __AVX2__
                if (blocks) {
                    const float* panel = kd_leaf_panels_.data() + kd_leaf_panel_offset_[uidx];
                    for (std::uint32_t b = 0; b < blocks; ++b) {
                        float d0[8], d1[8];
                        const float* pb = panel + static_cast<std::size_t>(b) * 128u * 16u;
                        l2_128x8_avx2_panel(vec, pb, 0, d0);
                        l2_128x8_avx2_panel(vec, pb, 8, d1);
                        for (int t = 0; t < 8; ++t) {
                            const std::uint32_t it0 = node.begin + b * 16u + static_cast<std::uint32_t>(t);
                            if (d0[t] < best) { best = d0[t]; best_idx = kd_order_[it0]; }
                            const std::uint32_t it1 = it0 + 8u;
                            if (d1[t] < best) { best = d1[t]; best_idx = kd_order_[it1]; }
                        }
                    }
                }
            #endif
                for (std::uint32_t it = node.begin + blocks * 16u; it < node.end; ++it) {
                    const float d = l2_128_fast_prune(vec, kd_leaf_centroids_[it], best);
                    if (d < best) { best = d; best_idx = kd_order_[it]; }
                }
            } else {
                // Enhanced SIMD leaf scanning for arbitrary dimensions
                #if defined(__AVX2__) && (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
                if (dim >= 8 && dim % 8 == 0) {
                    // AVX2 optimized path for dimensions divisible by 8
                    const size_t vec_size = dim / 8;
                    __m256 best_vec = _mm256_set1_ps(best);

                    for (std::uint32_t it = node.begin; it < node.end; ++it) {
                        const float* centroid = kd_leaf_centroids_[it];
                        __m256 sum = _mm256_setzero_ps();

                        // Process 8 floats at a time
                        for (size_t i = 0; i < dim; i += 8) {
                            __m256 v_query = _mm256_loadu_ps(vec + i);
                            __m256 v_centroid = _mm256_loadu_ps(centroid + i);
                            __m256 diff = _mm256_sub_ps(v_query, v_centroid);
                            sum = _mm256_fmadd_ps(diff, diff, sum);
                        }

                        // Horizontal sum
                        __m128 hi = _mm256_extractf128_ps(sum, 1);
                        __m128 lo = _mm256_castps256_ps128(sum);
                        __m128 sum128 = _mm_add_ps(hi, lo);
                        sum128 = _mm_hadd_ps(sum128, sum128);
                        sum128 = _mm_hadd_ps(sum128, sum128);
                        float d = _mm_cvtss_f32(sum128);

                        if (d < best) {
                            best = d;
                            best_idx = kd_order_[it];
                        }
                    }
                } else
                #endif
                {
                    // Fallback to kernel dispatch for non-AVX2 or non-aligned dimensions
                    const auto qspan = std::span(vec, dim);
                    for (std::uint32_t it = node.begin; it < node.end; ++it) {
                        const float d = kernels::select_backend_auto().l2_sq(qspan, kd_leaf_centroids_.get_centroid(it));
                        if (d < best) { best = d; best_idx = kd_order_[it]; }
                    }
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
            if (lbL < best) heap_push(node.left, lbL);
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
            if (lbR < best) heap_push(node.right, lbR);
        }
    }

    return best_idx;
}

std::uint32_t IvfPqIndex::Impl::kd_nearest_approx_(const float* vec, float early_termination_factor) const {
    // Approximate search with early termination based on distance bounds
    const auto& ops = kernels::select_backend_auto();
    const std::size_t dim = state_.dim;

    std::uint32_t best_idx = 0;
    float best = std::numeric_limits<float>::infinity();

    // Early termination threshold - stop exploring nodes if lower bound exceeds this
    float termination_threshold = std::numeric_limits<float>::infinity();

    // Priority queue for candidate nodes
    struct KDNodeCandidate {
        int node_id;
        float lower_bound;
        bool operator>(const KDNodeCandidate& other) const {
            return lower_bound > other.lower_bound;
        }
    };

    std::priority_queue<KDNodeCandidate,
                       std::vector<KDNodeCandidate>,
                       std::greater<KDNodeCandidate>> pq;

    // Start with root
    pq.push({kd_root_, 0.0f});

    int nodes_explored = 0;
    const int max_nodes = static_cast<int>(kd_nodes_.size() * 0.1f); // Explore at most 10% of nodes

    while (!pq.empty() && nodes_explored < max_nodes) {
        auto [nid, lb] = pq.top();
        pq.pop();

        // Early termination: skip if lower bound exceeds threshold
        if (lb > termination_threshold) break;

        nodes_explored++;

        const auto& node = kd_nodes_[static_cast<size_t>(nid)];

        if (node.leaf) {
            // Scan leaf centroids with branchless SIMD + rotational quantization
            const std::uint32_t* order = kd_order_.data() + node.begin;
            const std::size_t ncentroids = node.end - node.begin;

            // Use rotational quantization for fast approximate distances
            if (use_rotational_quantization_ && centroid_quantizer_ && ncentroids >= 16) {
                // Quantize query vector once
                std::vector<std::uint8_t> q_codes(dim);
                auto [q_scale, q_offset] = centroid_quantizer_->quantize(
                    std::span<const float>(vec, dim),
                    q_codes.data()
                );

                // Batch estimate distances using quantized representations
                std::vector<float> approx_distances(ncentroids);
                centroid_quantizer_->estimate_distances_batch(
                    q_codes.data(),
                    quantized_centroids_.data() + node.begin * dim,
                    ncentroids,
                    q_scale, q_offset,
                    centroid_scales_.data() + node.begin,
                    centroid_offsets_.data() + node.begin,
                    approx_distances.data()
                );

                // Find top-k candidates using branchless min reduction
                const size_t k = std::min<size_t>(4, ncentroids);
                std::vector<size_t> top_k(k);
                std::vector<float> top_k_dists(k, std::numeric_limits<float>::max());

                // Branchless scan for top-k
                for (size_t i = 0; i < ncentroids; ++i) {
                    float dist = approx_distances[i];
                    for (size_t j = 0; j < k; ++j) {
                        bool is_better = dist < top_k_dists[j];
                        float temp_dist = is_better ? dist : top_k_dists[j];
                        size_t temp_idx = is_better ? i : top_k[j];
                        dist = is_better ? top_k_dists[j] : dist;
                        top_k_dists[j] = temp_dist;
                        top_k[j] = temp_idx;
                    }
                }

                // Refine top-k with exact distances
                for (size_t j = 0; j < k; ++j) {
                    size_t idx = top_k[j];
                    const float* centroid = kd_leaf_centroids_.data() + (node.begin + idx) * dim;
                    float exact_dist = ops.l2_sq(std::span<const float>(vec, dim),
                                                std::span<const float>(centroid, dim));
                    if (exact_dist < best) {
                        best = exact_dist;
                        best_idx = order[idx];
                        termination_threshold = best * early_termination_factor;
                    }
                }
            } else if (ops.batch_l2_sq && ncentroids >= 8) {
                // Original batch computation path
                std::vector<float> distances(ncentroids);
                ops.batch_l2_sq(std::span<const float>(vec, dim),
                               kd_leaf_centroids_.data() + node.begin * dim,
                               ncentroids, dim, distances.data());

                for (size_t i = 0; i < ncentroids; ++i) {
                    if (distances[i] < best) {
                        best = distances[i];
                        best_idx = order[i];
                        termination_threshold = best * early_termination_factor;
                    }
                }
            } else {
                // Fallback to scalar
                for (size_t i = 0; i < ncentroids; ++i) {
                    const float* centroid = kd_leaf_centroids_.data() + (node.begin + i) * dim;
                    float dist = ops.l2_sq(std::span<const float>(vec, dim),
                                          std::span<const float>(centroid, dim));
                    if (dist < best) {
                        best = dist;
                        best_idx = order[i];
                        termination_threshold = best * early_termination_factor;
                    }
                }
            }
        } else {
            // Compute distance to split plane
            float plane_dist = vec[node.split_dim] - node.split_val;

            // Order children by distance
            int near_child = (plane_dist <= 0) ? node.left : node.right;
            int far_child = (plane_dist <= 0) ? node.right : node.left;

            // Always explore near child
            if (near_child >= 0) {
                pq.push({near_child, lb});
            }

            // Only explore far child if it could contain better results
            float far_lb = lb + plane_dist * plane_dist;
            if (far_child >= 0 && far_lb < termination_threshold) {
                pq.push({far_child, far_lb});
            }
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

    // Min-heap of nodes by node_min_lb (binary heap)
    std::vector<int> heap_ids; heap_ids.reserve(num_nodes);
    std::vector<float> heap_keys; heap_keys.reserve(num_nodes);
    auto heap_sift_up = [&](int i){
        while (i > 0) {
            int p = (i - 1) >> 1;
            if (heap_keys[p] <= heap_keys[i]) break;
            std::swap(heap_keys[p], heap_keys[i]);
            std::swap(heap_ids[p], heap_ids[i]);
            i = p;
        }
    };
    auto heap_sift_down = [&](int i){
        const int n = static_cast<int>(heap_ids.size());
        while (true) {
            int l = (i << 1) + 1, r = l + 1, m = i;
            if (l < n && heap_keys[l] < heap_keys[m]) m = l;
            if (r < n && heap_keys[r] < heap_keys[m]) m = r;
            if (m == i) break;
            std::swap(heap_keys[i], heap_keys[m]);
            std::swap(heap_ids[i], heap_ids[m]);
            i = m;
        }
    };
    auto heap_push = [&](int nid){
        const float key = node_min_lb[static_cast<std::size_t>(nid)];
        heap_ids.push_back(nid); heap_keys.push_back(key);
        heap_sift_up(static_cast<int>(heap_ids.size()) - 1);
    };
    auto heap_pop_min = [&](){
        if (heap_ids.empty()) return -1;
        const int nid = heap_ids[0];
        heap_ids[0] = heap_ids.back(); heap_ids.pop_back();
        heap_keys[0] = heap_keys.back(); heap_keys.pop_back();
        if (!heap_ids.empty()) heap_sift_down(0);
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
                    const std::uint32_t blocks = kd_leaf_panel_blocks_[u];
                #ifdef __AVX2__
                    if (blocks) {
                        const float* panel = kd_leaf_panels_.data() + kd_leaf_panel_offset_[u];
                        for (std::uint32_t bblk = 0; bblk < blocks; ++bblk) {
                            float d0[8], d1[8];
                            const float* pb = panel + static_cast<std::size_t>(bblk) * 128u * 16u;
                            l2_128x8_avx2_panel(qv, pb, 0, d0);
                            l2_128x8_avx2_panel(qv, pb, 8, d1);
                            for (int t = 0; t < 8; ++t) {
                                const std::uint32_t it0 = node.begin + bblk * 16u + static_cast<std::uint32_t>(t);
                                if (d0[t] < b) { b = d0[t]; bidx = kd_order_[it0]; }
                                const std::uint32_t it1 = it0 + 8u;
                                if (d1[t] < b) { b = d1[t]; bidx = kd_order_[it1]; }
                            }
                        }
                    }
                #endif
                    for (std::uint32_t j = blocks * 16u; j < L; ++j) {
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

bool IvfPqIndex::Impl::should_use_hierarchical_kd_() const {
    // Use hierarchical KD-tree for larger centroid counts
    return (state_.params.nlist >= 512) &&
           (state_.params.coarse_assigner == CoarseAssigner::KDTree) &&
           (vesper::core::safe_getenv("VESPER_HIERARCHICAL_KD") &&
            vesper::core::safe_getenv("VESPER_HIERARCHICAL_KD")->front() == '1');
}

void IvfPqIndex::Impl::hierarchical_kd_build_() {
    const std::size_t dim = state_.dim;
    const std::uint32_t nlist = state_.params.nlist;
    auto& centroids = state_.coarse_centroids;

    // Configuration: adjust based on nlist
    hierarchical_kd_.num_representatives = (nlist >= 2048) ? 256 :
                                           (nlist >= 1024) ? 128 : 64;
    hierarchical_kd_.subtree_leaf_size = 32;

    // K-means clustering to create representatives
    const std::uint32_t num_reps = hierarchical_kd_.num_representatives;
    hierarchical_kd_.representatives.resize(num_reps);
    for (auto& rep : hierarchical_kd_.representatives) {
        rep.resize(dim);
    }

    // Simple k-means++ initialization for representatives
    std::mt19937 rng(42);
    std::vector<std::uint32_t> rep_indices;
    std::vector<float> min_dists(nlist, std::numeric_limits<float>::max());

    // Choose first center randomly
    std::uniform_int_distribution<std::uint32_t> first_dist(0, nlist - 1);
    rep_indices.push_back(first_dist(rng));

    // K-means++ selection for remaining centers
    const auto& ops = kernels::select_backend_auto();
    for (std::uint32_t k = 1; k < num_reps; ++k) {
        // Update min distances
        const float* last_center = centroids[rep_indices.back()];
        for (std::uint32_t i = 0; i < nlist; ++i) {
            float dist = ops.l2_sq(std::span(centroids[i], dim),
                                   std::span(last_center, dim));
            min_dists[i] = std::min(min_dists[i], dist);
        }

        // Sample next center proportional to squared distance
        std::vector<float> probs(min_dists);
        std::discrete_distribution<std::uint32_t> dist(probs.begin(), probs.end());
        rep_indices.push_back(dist(rng));
    }

    // Copy selected centroids as initial representatives
    for (std::uint32_t k = 0; k < num_reps; ++k) {
        std::memcpy(hierarchical_kd_.representatives[k].data(),
                   centroids[rep_indices[k]], dim * sizeof(float));
    }

    // Assign centroids to representatives
    std::vector<std::vector<std::uint32_t>> clusters(num_reps);
    std::vector<std::uint32_t> assignments(nlist);

    // Lloyd's iterations for clustering
    for (int iter = 0; iter < 10; ++iter) {
        // Clear clusters
        for (auto& cluster : clusters) cluster.clear();

        // Assign each centroid to nearest representative
        for (std::uint32_t i = 0; i < nlist; ++i) {
            float best_dist = std::numeric_limits<float>::max();
            std::uint32_t best_rep = 0;

            for (std::uint32_t k = 0; k < num_reps; ++k) {
                float dist = ops.l2_sq(std::span(centroids[i], dim),
                                       std::span(hierarchical_kd_.representatives[k].data(), dim));
                if (dist < best_dist) {
                    best_dist = dist;
                    best_rep = k;
                }
            }
            clusters[best_rep].push_back(i);
            assignments[i] = best_rep;
        }

        // Update representatives as mean of assigned centroids
        for (std::uint32_t k = 0; k < num_reps; ++k) {
            if (clusters[k].empty()) continue;

            std::vector<double> mean(dim, 0.0);
            for (std::uint32_t idx : clusters[k]) {
                for (std::size_t d = 0; d < dim; ++d) {
                    mean[d] += centroids[idx][d];
                }
            }

            const double scale = 1.0 / clusters[k].size();
            for (std::size_t d = 0; d < dim; ++d) {
                hierarchical_kd_.representatives[k][d] = static_cast<float>(mean[d] * scale);
            }
        }
    }

    // Build top-level KD-tree over representatives
    hierarchical_kd_.top_order.resize(num_reps);
    for (std::uint32_t i = 0; i < num_reps; ++i) {
        hierarchical_kd_.top_order[i] = i;
    }
    hierarchical_kd_.top_nodes.clear();
    hierarchical_kd_.top_nodes.reserve(num_reps * 2);

    std::function<int(std::uint32_t, std::uint32_t)> build_top =
        [&](std::uint32_t begin, std::uint32_t end) -> int {
        const std::uint32_t count = end - begin;
        const int node_id = static_cast<int>(hierarchical_kd_.top_nodes.size());
        hierarchical_kd_.top_nodes.push_back(KDNode{});
        KDNode& node = hierarchical_kd_.top_nodes.back();
        node.begin = begin;
        node.end = end;

        // Compute center and radius for this node
        node.center.resize(dim);
        std::fill(node.center.begin(), node.center.end(), 0.0f);

        for (std::uint32_t it = begin; it < end; ++it) {
            const std::uint32_t rep_idx = hierarchical_kd_.top_order[it];
            for (std::size_t d = 0; d < dim; ++d) {
                node.center[d] += hierarchical_kd_.representatives[rep_idx][d];
            }
        }

        const float scale = 1.0f / count;
        for (float& v : node.center) v *= scale;

        // Compute radius
        node.radius2 = 0.0f;
        for (std::uint32_t it = begin; it < end; ++it) {
            const std::uint32_t rep_idx = hierarchical_kd_.top_order[it];
            float dist = ops.l2_sq(std::span(node.center.data(), dim),
                                   std::span(hierarchical_kd_.representatives[rep_idx].data(), dim));
            node.radius2 = std::max(node.radius2, dist);
        }

        // Leaf node for small counts
        if (count <= 4) {
            node.leaf = true;
            return node_id;
        }

        // Find split dimension (max variance)
        std::vector<double> variance(dim, 0.0);
        for (std::uint32_t it = begin; it < end; ++it) {
            const std::uint32_t rep_idx = hierarchical_kd_.top_order[it];
            for (std::size_t d = 0; d < dim; ++d) {
                float diff = hierarchical_kd_.representatives[rep_idx][d] - node.center[d];
                variance[d] += diff * diff;
            }
        }

        node.split_dim = static_cast<std::uint32_t>(
            std::max_element(variance.begin(), variance.end()) - variance.begin());

        // Partition at median
        auto* order_begin = hierarchical_kd_.top_order.data() + begin;
        auto* order_mid = order_begin + count / 2;
        auto* order_end = hierarchical_kd_.top_order.data() + end;

        std::nth_element(order_begin, order_mid, order_end,
            [&](std::uint32_t a, std::uint32_t b) {
                return hierarchical_kd_.representatives[a][node.split_dim] <
                       hierarchical_kd_.representatives[b][node.split_dim];
            });

        const std::uint32_t mid = begin + count / 2;
        node.split_val = hierarchical_kd_.representatives[hierarchical_kd_.top_order[mid]][node.split_dim];

        node.leaf = false;
        node.left = build_top(begin, mid);
        node.right = build_top(mid, end);

        return node_id;
    };

    hierarchical_kd_.top_root = build_top(0, num_reps);

    // Build sub-trees for each cluster
    hierarchical_kd_.subtrees.resize(num_reps);
    hierarchical_kd_.rep_to_clusters.resize(num_reps);

    for (std::uint32_t k = 0; k < num_reps; ++k) {
        auto& subtree = hierarchical_kd_.subtrees[k];
        const auto& cluster = clusters[k];

        if (cluster.empty()) {
            subtree.root = -1;
            continue;
        }

        subtree.centroid_begin = cluster.front();
        subtree.centroid_end = cluster.back() + 1;
        subtree.order = cluster;
        subtree.nodes.clear();
        subtree.nodes.reserve(cluster.size() * 2);

        std::function<int(std::uint32_t, std::uint32_t)> build_sub =
            [&](std::uint32_t begin, std::uint32_t end) -> int {
            const std::uint32_t count = end - begin;
            const int node_id = static_cast<int>(subtree.nodes.size());
            subtree.nodes.push_back(KDNode{});
            KDNode& node = subtree.nodes.back();
            node.begin = begin;
            node.end = end;

            // Compute center for this node
            node.center.resize(dim);
            std::fill(node.center.begin(), node.center.end(), 0.0f);

            for (std::uint32_t it = begin; it < end; ++it) {
                const std::uint32_t centroid_idx = subtree.order[it];
                for (std::size_t d = 0; d < dim; ++d) {
                    node.center[d] += centroids[centroid_idx][d];
                }
            }

            const float scale = 1.0f / count;
            for (float& v : node.center) v *= scale;

            // Compute radius
            node.radius2 = 0.0f;
            for (std::uint32_t it = begin; it < end; ++it) {
                const std::uint32_t centroid_idx = subtree.order[it];
                float dist = ops.l2_sq(std::span(node.center.data(), dim),
                                       std::span(centroids[centroid_idx], dim));
                node.radius2 = std::max(node.radius2, dist);
            }

            // Leaf node
            if (count <= hierarchical_kd_.subtree_leaf_size) {
                node.leaf = true;
                return node_id;
            }

            // Find split dimension
            std::vector<double> variance(dim, 0.0);
            for (std::uint32_t it = begin; it < end; ++it) {
                const std::uint32_t centroid_idx = subtree.order[it];
                for (std::size_t d = 0; d < dim; ++d) {
                    float diff = centroids[centroid_idx][d] - node.center[d];
                    variance[d] += diff * diff;
                }
            }

            node.split_dim = static_cast<std::uint32_t>(
                std::max_element(variance.begin(), variance.end()) - variance.begin());

            // Partition at median
            auto* order_begin = subtree.order.data() + begin;
            auto* order_mid = order_begin + count / 2;
            auto* order_end = subtree.order.data() + end;

            std::nth_element(order_begin, order_mid, order_end,
                [&](std::uint32_t a, std::uint32_t b) {
                    return centroids[a][node.split_dim] < centroids[b][node.split_dim];
                });

            const std::uint32_t mid = begin + count / 2;
            node.split_val = centroids[subtree.order[mid]][node.split_dim];

            node.leaf = false;
            node.left = build_sub(begin, mid);
            node.right = build_sub(mid, end);

            return node_id;
        };

        subtree.root = build_sub(0, static_cast<std::uint32_t>(cluster.size()));
    }

    hierarchical_kd_.enabled = true;
}

std::uint32_t IvfPqIndex::Impl::hierarchical_kd_nearest_(const float* vec) const {
    if (!hierarchical_kd_.enabled || hierarchical_kd_.top_root == -1) {
        return kd_nearest_(vec);  // Fallback to regular KD-tree
    }

    const auto& ops = kernels::select_backend_auto();
    const std::size_t dim = state_.dim;

    // First, find best representatives at top level
    struct Candidate {
        int node_id;
        float lower_bound;
        bool operator>(const Candidate& other) const {
            return lower_bound > other.lower_bound;
        }
    };

    // Search top-level tree for best k representatives
    const std::uint32_t top_k = std::min(4u, hierarchical_kd_.num_representatives);
    std::vector<std::uint32_t> best_reps;
    best_reps.reserve(top_k);

    {
        std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> pq;
        pq.push({hierarchical_kd_.top_root, 0.0f});

        std::priority_queue<std::pair<float, std::uint32_t>> top_heap;

        while (!pq.empty()) {
            auto [nid, lb] = pq.top();
            pq.pop();

            if (nid < 0) continue;
            const KDNode& node = hierarchical_kd_.top_nodes[static_cast<size_t>(nid)];

            // Prune if can't improve
            if (!top_heap.empty() && top_heap.size() >= top_k && lb >= top_heap.top().first) {
                continue;
            }

            if (node.leaf) {
                // Evaluate representatives in this leaf
                for (std::uint32_t it = node.begin; it < node.end; ++it) {
                    const std::uint32_t rep_idx = hierarchical_kd_.top_order[it];
                    float dist = ops.l2_sq(std::span(vec, dim),
                                          std::span(hierarchical_kd_.representatives[rep_idx].data(), dim));

                    if (top_heap.size() < top_k) {
                        top_heap.emplace(dist, rep_idx);
                    } else if (dist < top_heap.top().first) {
                        top_heap.pop();
                        top_heap.emplace(dist, rep_idx);
                    }
                }
            } else {
                // Compute bounds for children
                float plane_dist = vec[node.split_dim] - node.split_val;
                int near_child = (plane_dist <= 0) ? node.left : node.right;
                int far_child = (plane_dist <= 0) ? node.right : node.left;

                if (near_child >= 0) {
                    const KDNode& child = hierarchical_kd_.top_nodes[static_cast<size_t>(near_child)];
                    float child_lb = ops.l2_sq(std::span(vec, dim),
                                              std::span(child.center.data(), dim)) - child.radius2;
                    pq.push({near_child, std::max(0.0f, child_lb)});
                }

                if (far_child >= 0) {
                    const KDNode& child = hierarchical_kd_.top_nodes[static_cast<size_t>(far_child)];
                    float child_lb = ops.l2_sq(std::span(vec, dim),
                                              std::span(child.center.data(), dim)) - child.radius2;
                    pq.push({far_child, std::max(0.0f, child_lb)});
                }
            }
        }

        // Extract best representatives
        while (!top_heap.empty()) {
            best_reps.push_back(top_heap.top().second);
            top_heap.pop();
        }
    }

    // Search within best representative clusters
    std::uint32_t best_idx = 0;
    float best_dist = std::numeric_limits<float>::max();

    for (std::uint32_t rep_idx : best_reps) {
        const auto& subtree = hierarchical_kd_.subtrees[rep_idx];
        if (subtree.root == -1) continue;

        // Search this subtree
        std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> pq;
        pq.push({subtree.root, 0.0f});

        while (!pq.empty()) {
            auto [nid, lb] = pq.top();
            pq.pop();

            if (nid < 0) continue;
            if (lb >= best_dist) continue;  // Prune

            const KDNode& node = subtree.nodes[static_cast<size_t>(nid)];

            if (node.leaf) {
                // Scan centroids in this leaf
                for (std::uint32_t it = node.begin; it < node.end; ++it) {
                    const std::uint32_t centroid_idx = subtree.order[it];
                    float dist = ops.l2_sq(std::span(vec, dim),
                                          state_.coarse_centroids.get_centroid(centroid_idx));
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_idx = centroid_idx;
                    }
                }
            } else {
                // Traverse children
                float plane_dist = vec[node.split_dim] - node.split_val;
                int near_child = (plane_dist <= 0) ? node.left : node.right;
                int far_child = (plane_dist <= 0) ? node.right : node.left;

                if (near_child >= 0) {
                    const KDNode& child = subtree.nodes[static_cast<size_t>(near_child)];
                    float child_lb = ops.l2_sq(std::span(vec, dim),
                                              std::span(child.center.data(), dim)) - child.radius2;
                    pq.push({near_child, std::max(0.0f, child_lb)});
                }

                if (far_child >= 0) {
                    const KDNode& child = subtree.nodes[static_cast<size_t>(far_child)];
                    float child_lb = ops.l2_sq(std::span(vec, dim),
                                              std::span(child.center.data(), dim)) - child.radius2;
                    pq.push({far_child, std::max(0.0f, child_lb)});
                }
            }
        }
    }

    return best_idx;
}

void IvfPqIndex::Impl::hierarchical_kd_assign_batch_(const float* data, std::size_t n,
                                                     std::uint32_t* out_assignments) const {
    // For now, use serial assignment with hierarchical search
    // Future: implement true batch processing with query distribution
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(n); ++i) {
        const float* vec = data + static_cast<size_t>(i) * state_.dim;
        out_assignments[i] = hierarchical_kd_nearest_(vec);
    }
}

auto IvfPqIndex::Impl::train_product_quantizer(const float* data, std::size_t n)
    -> std::expected<void, core::error> {

    // Compute residuals of training data with respect to nearest coarse centroid
    const bool dbg = ([](){ auto v = vesper::core::safe_getenv("VESPER_IVFPQ_DEBUG"); return v && !v->empty() && ((*v)[0] != '0'); })();
    std::vector<float> residuals;
    try {
        residuals.resize(n * state_.dim);
        if (dbg) {
            std::cerr << "[IVFPQ][train] residuals allocated (n=" << n
                      << ", dim=" << state_.dim << ")" << std::endl;
        }


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
        if (dbg) {
            std::cerr << "[IVFPQ][train] residuals computed" << std::endl;
            // Compute residual norm statistics on a sample
            const std::size_t sample_n = std::min<std::size_t>(n, 50000);
            double mean = 0.0, m2 = 0.0;
            for (std::size_t i = 0; i < sample_n; ++i) {
                const float* r = residuals.data() + i * state_.dim;
                double s = 0.0;
                for (std::size_t d = 0; d < state_.dim; ++d) { double v = r[d]; s += static_cast<double>(v) * static_cast<double>(v); }
                double delta = s - mean; mean += delta / static_cast<double>(i + 1); m2 += delta * (s - mean);
            }
            const double var = (sample_n > 1) ? (m2 / static_cast<double>(sample_n - 1)) : 0.0;
            const double stddev = std::sqrt(var);
            std::cerr << "[IVFPQ][train] residual_norm2 mean=" << mean << " std=" << stddev
                      << " sample_n=" << sample_n << "\n";
        }

    } catch (const std::bad_alloc& e) {
        if (dbg) {
            std::cerr << "[IVFPQ][train] OOM while allocating/computing residuals: " << e.what() << std::endl;
        }
        return std::vesper_unexpected(error{ error_code::out_of_memory, "bad_alloc in train_product_quantizer residuals", "ivf_pq" });
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
    if (dbg) {
        std::cerr << "[IVFPQ][train] PQ baseline train begin" << std::endl;
        const std::size_t subdim = state_.dim / state_.params.m;
        const std::uint32_t ksub = (1u << state_.params.nbits);
        std::cerr << "[IVFPQ][train] PQ cfg m=" << state_.params.m
                  << " nbits=" << static_cast<unsigned>(state_.params.nbits)
                  << " subdim=" << subdim
                  << " ksub=" << ksub << std::endl;
    }

    try {
        if (auto r = pq_unrot->train(residuals.data(), n, state_.dim); !r.has_value()) {
            if (dbg) {
                std::cerr << "[IVFPQ][train] PQ baseline train returned error" << std::endl;
            }
            return std::vesper_unexpected(r.error());
        }
    } catch (const std::bad_alloc& e) {
        if (dbg) {
            std::cerr << "[IVFPQ][train] OOM during PQ baseline training: " << e.what() << std::endl;
        }
        return std::vesper_unexpected(error{ error_code::out_of_memory, "bad_alloc in PQ baseline train()", "ivf_pq" });
    } catch (const std::exception& e) {
        if (dbg) {
            std::cerr << "[IVFPQ][train] exception during PQ baseline training: " << e.what() << std::endl;
        }
        throw; // propagate non-OOM exceptions; debug log provides context
    } catch (...) {
        if (dbg) {
            std::cerr << "[IVFPQ][train] non-std exception during PQ baseline training" << std::endl;
        }
        throw; // propagate
    }
    if (dbg) {
        std::cerr << "[IVFPQ][train] PQ baseline train done" << std::endl;
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
            std::expected<void, error> trf;
            try {
                trf = pq_final->train(rotated_full.data(), n, state_.dim);
            } catch (const std::bad_alloc& e) {
                if (dbg) {
                    std::cerr << "[IVFPQ][train] OOM during PQ(OPQ) final training: " << e.what() << std::endl;
                }
                return std::vesper_unexpected(error{ error_code::out_of_memory, "bad_alloc in PQ(OPQ) final train()", "ivf_pq" });
            }
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
    const bool force_opq = ([](){ auto v = vesper::core::safe_getenv("VESPER_IVFPQ_FORCE_OPQ"); return v && !v->empty() && ((*v)[0] == '1'); })();
    if (state_.params.verbose) {
        std::cout << "[OPQ] summary: err_opq_best=" << (std::isfinite(err_opq_best) ? err_opq_best : -1.0)
                  << ", err_unrot=" << err_unrot
                  << ", force_opq=" << (force_opq?1:0) << std::endl;
    }

    bool kept_opq = false;
    if (force_opq && state_.params.use_opq && !state_.rotation_matrix.empty() && pq_opq_best) {
        state_.pq = std::move(pq_opq_best);
        kept_opq = true;
    } else if (err_opq_best < err_unrot) {
        state_.pq = std::move(pq_opq_best);
        kept_opq = true;
    } else {
        state_.pq = std::move(pq_unrot);
        state_.rotation_matrix.clear();
    }

    if (dbg) {
        if (kept_opq && !state_.rotation_matrix.empty()) {
            double fro_sq = 0.0; std::size_t idx = 0;
            for (std::size_t i = 0; i < state_.dim; ++i) {
                for (std::size_t j = 0; j < state_.dim; ++j, ++idx) {
                    const double ref = (i==j) ? 1.0 : 0.0;
                    const double diff = static_cast<double>(state_.rotation_matrix[idx]) - ref;
                    fro_sq += diff * diff;
                }
            }
            std::cerr << "[IVFPQ][train][diag] OPQ kept; ||R-I||_F=" << std::sqrt(fro_sq) << " dim=" << state_.dim << "\n";
        } else {
            std::cerr << "[IVFPQ][train][diag] OPQ disabled (no rotation stored)\n";
        }
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

    const bool dbg = ([](){ auto v = vesper::core::safe_getenv("VESPER_IVFPQ_DEBUG"); return v && !v->empty() && ((*v)[0] != '0'); })();
    if (dbg) {
        std::cerr << "[IVFPQ][add] start n=" << n << " dim=" << state_.dim << " nlist=" << state_.params.nlist << std::endl;
    }

    using core::error;
    using core::error_code;


    if (dbg) {
        std::cerr << "[IVFPQ][add] assigning nearest centroids (coarse)" << std::endl;
    }

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
    const bool kTimingAdd = (state_.params.timings_enabled || state_.params.verbose || ([](){ auto v = vesper::core::safe_getenv("VESPER_TIMING"); return v && !v->empty() && ((*v)[0] != '0'); })());
    auto t_assign_start = kTimingAdd ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    // Reset KD counters for this add() run
    kd_nodes_pushed_.store(0, std::memory_order_relaxed);
    kd_nodes_popped_.store(0, std::memory_order_relaxed);
    kd_leaves_scanned_.store(0, std::memory_order_relaxed);
    kd_trav_ns_.store(0, std::memory_order_relaxed);
    kd_leaf_ns_.store(0, std::memory_order_relaxed);

    if (state_.params.use_centroid_ann && state_.params.coarse_assigner == CoarseAssigner::KDTree && kd_root_ != -1) {
        // Batch mode is default (better performance); disable with VESPER_KD_BATCH=0
        bool use_batch = true;
        if (auto v = vesper::core::safe_getenv("VESPER_KD_BATCH")) {
            if (!v->empty() && (*v)[0] == '0') {
                use_batch = false;
            }
        }
        const bool force_no_veb = ([](){ auto v = vesper::core::safe_getenv("VESPER_NO_VEB"); return v && !v->empty() && ((*v)[0] == '1'); })();

        if (use_batch) {
            // Adaptive algorithm selection based on runtime characteristics
            const bool use_approximate = ([]() {
                // Approximate KD assignment is opt-in only to preserve exactness by default
                auto approx_env = vesper::core::safe_getenv("VESPER_KD_APPROX");
                return approx_env && !approx_env->empty() && ((*approx_env)[0] == '1');
            })();

            const bool use_batch_gemm = ([&]() {
                // Use batch GEMM when we have AVX-512 and suitable batch size
                auto& ops = kernels::select_backend_auto();
                if (!ops.batch_l2_sq) return false;  // No batch support

                // Check for AVX-512 support
                auto backend_env = vesper::core::safe_getenv("VESPER_KERNEL_BACKEND");
                bool has_avx512 = backend_env && backend_env->find("avx512") != std::string::npos;

                // Use batch GEMM for medium to large leaf sizes
                return has_avx512 && kd_leaf_size_ >= 128;
            })();

            // Select and execute the best algorithm
            if (use_approximate) {
                // Use approximate search with early termination
                #pragma omp parallel for
                for (int i = 0; i < static_cast<int>(n); ++i) {
                    const float* vec = data + static_cast<size_t>(i) * state_.dim;
                    assignments[i] = kd_nearest_approx_(vec, 1.1f);  // 10% tolerance
                }
            } else if (hierarchical_kd_.enabled) {
                hierarchical_kd_assign_batch_(data, n, assignments.data());
            } else {
                kd_assign_batch_(data, n, assignments.data());
            }

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
                const std::uint32_t best_idx_val = hierarchical_kd_.enabled
                    ? hierarchical_kd_nearest_(vec)
                    : (use_veb_layout_ && !force_no_veb)
                        ? kd_nearest_veb_(vec)
                        : kd_nearest_(vec);
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
    } else if (state_.params.coarse_assigner == CoarseAssigner::Projection && !proj_rows_.empty()) {
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
        // Projection screening via helper (CBLAS backend when available; scalar fallback otherwise)
        std::vector<std::uint32_t> cand_idx(n * static_cast<std::size_t>(L));
        std::vector<float> cand_dist(n * static_cast<std::size_t>(L));
        {
            using vesper::index::ProjScreenInputs;
            using vesper::index::ProjScreenOutputs;
            ProjScreenInputs psi{
                qproj.data(),
                qnorm.data(),
                n,
                p,
                proj_centroids_rm_.data(),
                proj_centroid_norms_.data(),
                proj_centroids_pack8_.empty() ? nullptr : proj_centroids_pack8_.data(),
                static_cast<std::size_t>(state_.params.nlist),
                L
            };
            ProjScreenOutputs pso{ cand_idx.data(), cand_dist.data() };
            projection_screen_select(psi, pso);
        }

        // Exact refinement over candidates returned by screening helper
        for (std::size_t i = 0; i < n; ++i) {
            const float* qv = data + i * state_.dim;
            float bestd = std::numeric_limits<float>::infinity(); std::uint32_t bestc = 0u;
            const std::size_t base = i * static_cast<std::size_t>(L);
            for (std::size_t t = 0; t < static_cast<std::size_t>(L); ++t) {
                const std::uint32_t cand = cand_idx[base + t];
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

    if (dbg) {
        std::cerr << "[IVFPQ][add] allocating residuals/rotation buffers (n=" << n << ", dim=" << state_.dim << ")" << std::endl;
    }
    std::vector<float> residuals;
    try {
        residuals.resize(n * state_.dim);
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

    // Stage 1 validation: Residual computation
    if (auto v = core::safe_getenv("VESPER_IVFPQ_DEBUG"); v && !v->empty() && (*v)[0] == '1') {
        const std::size_t sample_n = std::min<std::size_t>(n, 10);
        for (std::size_t i = 0; i < sample_n; ++i) {
            const float* vec = data + i * state_.dim;
            const float* centroid = state_.coarse_centroids[assignments[i]];
            const float* residual = residuals.data() + i * state_.dim;

            // Compute residual norm and verify no NaN/Inf
            double norm_sq = 0.0;
            bool has_nan_inf = false;
            for (std::size_t d = 0; d < state_.dim; ++d) {
                const float r = residual[d];
                if (std::isnan(r) || std::isinf(r)) has_nan_inf = true;
                norm_sq += static_cast<double>(r) * static_cast<double>(r);
            }

            std::cerr << "[ADC_VAL][stage1] i=" << i << " cluster=" << assignments[i]
                      << " residual_norm=" << std::sqrt(norm_sq)
                      << " has_nan_inf=" << has_nan_inf << "\n";
        }
    }

    } catch (const std::bad_alloc& e) {
        if (dbg) {
            std::cerr << "[IVFPQ][add] OOM while allocating residuals/rotation: " << e.what() << std::endl;
        }
        return std::vesper_unexpected(error{ error_code::out_of_memory, "bad_alloc in add() residuals/rotation", "ivf_pq"});
    }


    const float* encode_data = residuals.data();
    std::vector<float> rotated_residuals;
    if (state_.params.use_opq && !state_.rotation_matrix.empty()) {
        rotated_residuals.resize(n * state_.dim);
        apply_rotation(residuals.data(), rotated_residuals.data(), n);
        encode_data = rotated_residuals.data();
    }


    if (dbg) {
        std::cerr << "[IVFPQ][add] encoding PQ codes (m=" << state_.params.m << ")" << std::endl;
    }

    // 3) Encode residuals with PQ
    std::vector<std::uint8_t> all_codes(n * state_.params.m);
    auto t_encode_start = kTimingAdd ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    state_.pq->encode(encode_data, n, all_codes.data());

    if (dbg) {
        std::cerr << "[IVFPQ][add] encoded PQ codes" << std::endl;
    }

    // Stage 2 validation: PQ encoding/decoding roundtrip
    if (auto v = core::safe_getenv("VESPER_IVFPQ_DEBUG"); v && !v->empty() && (*v)[0] == '1') {
        const std::size_t sample_n = std::min<std::size_t>(n, 10);
        std::vector<float> decoded(sample_n * state_.dim);
        state_.pq->decode(all_codes.data(), sample_n, decoded.data());

        for (std::size_t i = 0; i < sample_n; ++i) {
            const float* original_residual = encode_data + i * state_.dim;
            const float* decoded_residual = decoded.data() + i * state_.dim;
            const std::uint8_t* code = all_codes.data() + i * state_.params.m;

            // Compute quantization error
            double error_sq = 0.0;
            for (std::size_t d = 0; d < state_.dim; ++d) {
                const double diff = static_cast<double>(original_residual[d]) - static_cast<double>(decoded_residual[d]);
                error_sq += diff * diff;
            }

            // Check code validity
            bool valid_codes = true;
            for (std::uint32_t sub = 0; sub < state_.params.m; ++sub) {
                if (code[sub] >= (1u << state_.params.nbits)) {
                    valid_codes = false;
                    break;
                }
            }

            std::cerr << "[ADC_VAL][stage2] i=" << i
                      << " quant_error=" << std::sqrt(error_sq)
                      << " valid_codes=" << valid_codes << "\n";
        }
    }
    if (dbg) {
    // Debug: validate coarse assignment accuracy on a sample (brute force vs KD exact)
    if (dbg) {
        const std::size_t sample = std::min<std::size_t>(n, 50000);
        const auto& ops = kernels::select_backend_auto();
        std::size_t mismatches = 0;
        for (std::size_t i = 0; i < sample; ++i) {
            const float* x = data + i * state_.dim;
            // brute force nearest centroid
            float best = std::numeric_limits<float>::infinity();
            std::uint32_t best_c = 0;
            for (std::uint32_t c = 0; c < state_.params.nlist; ++c) {
                float d = ops.l2_sq(std::span(x, state_.dim), state_.coarse_centroids.get_centroid(c));
                if (d < best) { best = d; best_c = c; }
            }
            if (assignments[i] != best_c) mismatches++;
        }
        const double rate = sample ? static_cast<double>(mismatches) / static_cast<double>(sample) : 0.0;
        std::cerr << "[IVFPQ][add][diag] coarse_assign_mismatch_rate=" << rate
                  << " sample_n=" << sample << std::endl;
    }

        std::cerr << "[IVFPQ][add] adding to inverted lists" << std::endl;
    }

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

    if (dbg) {
        std::cerr << "[IVFPQ][add] completed inverted list insertion" << std::endl;
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

    // CRITICAL FIX: Use luts[m][code] instead of flat indexing with wrong stride
    // The AlignedCentroidBuffer uses cache-aligned stride, not ksub
    for (std::uint32_t m = 0; m < state_.params.m; ++m) {
        const float* lut_m = luts[m];  // Get pointer to m-th subquantizer's LUT
        distance += lut_m[code.codes[m]];
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

    // Debug toggle (make available to whole function scope)
    const bool dbg = [](){ auto v = vesper::core::safe_getenv("VESPER_IVFPQ_DEBUG"); return v && !v->empty() && ((*v)[0] != '0'); }();


    // Top-k/merging audit gate (debug-only)
    const bool validate_topk = ([](){
        auto v = vesper::core::safe_getenv("VESPER_VALIDATE_TOPK");
        return v && !v->empty() && ((*v)[0] == '1');
    })();
    // Containers for audit
    std::vector<std::pair<float, std::uint64_t>> dbg_candidates; // (dist,id)
    std::vector<std::size_t> dbg_scanned_per_list;
    if (validate_topk) {
        dbg_scanned_per_list.assign(state_.params.nlist, 0);
    }
    // Pool validation diagnostic gate
    const bool validate_pool = ([](){
        auto v = vesper::core::safe_getenv("VESPER_VALIDATE_POOL");
        return v && !v->empty() && ((*v)[0] == '1');
    })();
    std::size_t dbg_scanned_total = 0;    // number of candidates considered before truncation


    // We compute list-specific LUTs using residual query = (query - centroid)
    // Coarse assignment (nprobe) is done in original space.
    std::uint32_t effective_nprobe = params.nprobe;
    const bool dbg_probe_all = ([](){ auto v = vesper::core::safe_getenv("VESPER_IVFPQ_PROBE_ALL"); return v && !v->empty() && ((*v)[0] == '1'); })();
    if (dbg_probe_all) { effective_nprobe = state_.params.nlist; }
    auto probe_lists = find_nearest_centroids(query, effective_nprobe);
    if (dbg) {
        std::cerr << "[IVFPQ][search] effective_nprobe=" << effective_nprobe << " (probe_all=" << (dbg_probe_all?1:0) << ")\n";
    }

    // Collect candidates from inverted lists
    using Candidate = std::pair<float, std::uint64_t>;
    std::priority_queue<Candidate> heap;

    // Decide candidate pool size: default to k; allow larger shortlist via cand_k
    // When reranking is enabled, fetch more candidates
    const bool dbg_force_rerank = ([](){ auto v = vesper::core::safe_getenv("VESPER_IVFPQ_RERANK"); return v && !v->empty() && ((*v)[0] == '1'); })();
    std::size_t pool_k = static_cast<std::size_t>(
        std::max<std::uint32_t>(params.k, (params.cand_k > 0 ? params.cand_k : params.k))
    );

    if (params.use_exact_rerank || dbg_force_rerank) {
        // When reranking, fetch more candidates for better recall
        const std::size_t rerank_k = params.rerank_k > 0 ? params.rerank_k : std::max<std::uint32_t>(params.k * 3, 1000);
        pool_k = std::max(pool_k, rerank_k);
    }

    // New diagnostic knobs: minimum pool size and proportional scaling by k
    if (auto v = vesper::core::safe_getenv("VESPER_IVFPQ_POOL_MIN"); v && !v->empty()) {
        try {
            const std::size_t pool_min = static_cast<std::size_t>(std::stoull(*v));
            if (pool_min > 0) pool_k = std::max(pool_k, pool_min);
        } catch (...) {}
    }
    if (auto v = vesper::core::safe_getenv("VESPER_IVFPQ_POOL_ALPHA"); v && !v->empty()) {
        try {
            const double alpha = std::stod(*v);
            if (alpha > 0.0) {
                const std::size_t alpha_k = static_cast<std::size_t>(params.k * alpha);
                if (alpha_k > 0) pool_k = std::max(pool_k, alpha_k);
            }
        } catch (...) {}
    }

    // Existing diagnostic override: allow env to increase candidate pool size (takes max)
    if (auto v = vesper::core::safe_getenv("VESPER_IVFPQ_POOL_K"); v && !v->empty()) {
        try {
            const std::size_t env_pool = static_cast<std::size_t>(std::stoull(*v));
            if (env_pool > 0) pool_k = std::max(pool_k, env_pool);
        } catch (...) {}
    }
    if (dbg) {
        std::cerr << "[IVFPQ][search] pool_k=" << pool_k << (dbg_force_rerank?" (rerank)":"") << "\n";
    }

    const bool use_heapless = (pool_k <= 64);
    TopKBuffer topk(pool_k);
    static const bool kTiming = [](){ auto v = vesper::core::safe_getenv("VESPER_TIMING"); return v && !v->empty() && ((*v)[0] != '0'); }();
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
        #if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
        // Allow forcing scalar PQ path for diagnostics: set VESPER_PQ_SCALAR=1
        const bool force_scalar_pq = ([](){ auto v = vesper::core::safe_getenv("VESPER_PQ_SCALAR"); return v && !v->empty() && ((*v)[0] == '1'); })();
        if (dbg) {
            bool fs_ok = (dynamic_cast<FastScanPq*>(state_.pq.get()) != nullptr);
            auto simd_env0 = vesper::core::safe_getenv("VESPER_SIMD_MAX");
            std::cerr << "[IVFPQ][search] pre fs_ok=" << (fs_ok?1:0)
                      << " force_scalar=" << (force_scalar_pq?1:0)
                      << " simd_max=" << (simd_env0? *simd_env0 : std::string("-")) << "\n";
        }
        if (!force_scalar_pq) {
            if (auto* fs = dynamic_cast<FastScanPq*>(state_.pq.get())) {
                if (dbg) {
                    auto simd_env = vesper::core::safe_getenv("VESPER_SIMD_MAX");
                    std::cerr << "[IVFPQ][search] pq=ok fs=yes scalar_forced=" << (force_scalar_pq?1:0)
                              << " simd_max=" << (simd_env? *simd_env : std::string("-")) << "\n";
                }
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
                        if (dbg) { std::cerr << "[IVFPQ][search] path=avx512\n"; }
                        fs->compute_distances_avx512(lut_query, blocks, dists.data());
                    } else
                    #endif
                    #ifdef __AVX2__
                    {
                        if (dbg) { std::cerr << "[IVFPQ][search] path=avx2\n"; }
                        fs->compute_distances_avx2(lut_query, blocks, dists.data());
                    }
                    #else
                    {
                        if (dbg) { std::cerr << "[IVFPQ][search] path=scalar_fastscan\n"; }
                        fs->compute_distances(lut_query, blocks, dists.data());
                    }
                    #endif

                    // Step 1: Distance parity validation (optional, debug-only)
                    const bool validate_adc = ([](){
                        auto v = vesper::core::safe_getenv("VESPER_VALIDATE_ADC");
                        return v && !v->empty() && ((*v)[0] == '1');
                    })();

                    std::vector<float> dists_scalar;
                    if (dbg && validate_adc) {
                        // Compute scalar FastScan distances for cross-check
                        dists_scalar.assign(total_codes, 0.0f);
                        fs->compute_distances(lut_query, blocks, dists_scalar.data());
                    }

                    // Push candidates and optionally validate a small sample per tile
                    const std::size_t sample_n = (dbg && validate_adc) ? std::min<std::size_t>(static_cast<std::size_t>(5), tile_n) : 0;


                    for (std::size_t i = 0; i < tile_n; ++i) {
                        const float dist = dists[i];
                        const std::uint64_t id = ids_linear[base + i];

                        if (dbg && validate_adc && i < sample_n) {
                            // Reference ADC via LUT sum from codes
                            const auto& entry = list[base + i];
                            const auto luts_dbg = state_.pq->compute_lookup_tables(lut_query);
                            const float ref_adc = compute_adc_distance(lut_query, entry.code, luts_dbg);

                            // Scalar FastScan distance (if available)
                            float scalar_fs = std::numeric_limits<float>::quiet_NaN();
                            if (!dists_scalar.empty()) scalar_fs = dists_scalar[i];

                            auto rel_err = [](float a, float b){
                                const float denom = std::max(1.0f, std::fabs(b));
                                return std::fabs(a - b) / denom;
                            };
                            const float eps = 1e-5f;
                            const bool fast_vs_ref_ok = (rel_err(dist, ref_adc) <= eps);
                            const bool scalar_vs_ref_ok = (!dists_scalar.empty()) ? (rel_err(scalar_fs, ref_adc) <= eps) : true;

                            if (!(fast_vs_ref_ok && scalar_vs_ref_ok)) {
                                // Dump detailed diagnostics: codes and per-sub contributions
                                const std::uint32_t mdbg = state_.params.m;
                                const std::uint32_t ksub = (1U << state_.params.nbits);
                                const float* lut_data = luts_dbg.data();
                                float accum_dbg = 0.0f;
                                std::cerr << "[ADC][mismatch] list=" << list_idx
                                          << " base=" << base
                                          << " i=" << i
                                          << " id=" << id
                                          << " fast=" << dist
                                          << " scalar_fs=" << scalar_fs
                                          << " ref_adc=" << ref_adc
                                          << " m=" << mdbg
                                          << " ksub=" << ksub
                                          << "\n";
                                std::cerr << "  codes:";
                                for (std::uint32_t sub = 0; sub < mdbg; ++sub) {
                                    const std::uint8_t codeb = entry.code.codes[sub];
                                    const float lutv = lut_data[sub * ksub + codeb];
                                    accum_dbg += lutv;
                                    std::cerr << " " << static_cast<unsigned>(codeb);
                                }


                                std::cerr << "\n  accum_from_lut=" << accum_dbg << "\n";
                            } else if (dbg) {
                                // Optional brief alignment confirmation
                                std::cerr << "[ADC][align] list=" << list_idx
                                          << " base=" << base
                                          << " i=" << i
                                          << " id=" << id
                                          << " fast=" << dist
                                          << " scalar_fs=" << scalar_fs
                                          << " ref_adc=" << ref_adc
                                          << "\n";
                            }
                        }

                        // Stage 3 validation: ADC vs true L2 distance (sample-based)
                        static std::atomic<std::size_t> adc_val_count{0};
                        const std::size_t adc_val_limit = 10;
                        if (auto v = core::safe_getenv("VESPER_IVFPQ_DEBUG"); v && !v->empty() && (*v)[0] == '1') {
                            if (adc_val_count.load(std::memory_order_relaxed) < adc_val_limit && i < 5) {
                                adc_val_count.fetch_add(1, std::memory_order_relaxed);

                                const auto& entry = list[base + i];
                                const std::uint64_t id = entry.id;

                                // Reconstruct the vector from PQ code
                                std::vector<float> reconstructed(state_.dim);
                                std::vector<float> residual_decoded(state_.dim);
                                state_.pq->decode(entry.code.codes.data(), 1, residual_decoded.data());

                                // Apply inverse rotation if OPQ was used
                                const float* residual_ptr = residual_decoded.data();
                                std::vector<float> residual_orig;
                                if (state_.params.use_opq && !state_.rotation_matrix.empty()) {
                                    residual_orig.resize(state_.dim);
                                    apply_rotation_T(residual_decoded.data(), residual_orig.data(), 1);
                                    residual_ptr = residual_orig.data();
                                }

                                // Add centroid to get full vector
                                for (std::size_t d = 0; d < state_.dim; ++d) {
                                    reconstructed[d] = centroid[d] + residual_ptr[d];
                                }

                                // Compute true L2 distance from query to reconstructed vector
                                double true_dist_sq = 0.0;
                                for (std::size_t d = 0; d < state_.dim; ++d) {
                                    const double diff = static_cast<double>(query[d]) - static_cast<double>(reconstructed[d]);
                                    true_dist_sq += diff * diff;
                                }

                                // Compute residual-based distance (query_residual - vector_residual)
                                double residual_dist_sq = 0.0;
                                for (std::size_t d = 0; d < state_.dim; ++d) {
                                    const double diff = static_cast<double>(residual_query[d]) - static_cast<double>(residual_ptr[d]);
                                    residual_dist_sq += diff * diff;
                                }

                                const float adc_dist = dists[i];
                                const float true_dist = static_cast<float>(true_dist_sq);
                                const float residual_dist = static_cast<float>(residual_dist_sq);

                                std::cerr << "[ADC_VAL][stage3] list=" << list_idx
                                          << " i=" << i << " id=" << id
                                          << " adc=" << adc_dist
                                          << " true_l2=" << true_dist
                                          << " residual_l2=" << residual_dist
                                          << " adc_error=" << std::abs(adc_dist - residual_dist)
                                          << "\n";
                            }
                        }


	                        // Debug: record candidate for offline top-k audit (before heap handling)
	                        if (validate_topk) {
	                            dbg_candidates.emplace_back(dist, id);
	                            if (list_idx < dbg_scanned_per_list.size()) dbg_scanned_per_list[list_idx]++;
	                        }

                        if (validate_pool) { ++dbg_scanned_total; }
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
        }
        #endif

        if (!fast_path_used) {
            if (dbg) { std::cerr << "[IVFPQ][search] path=scalar_fallback\n"; }
            // Compute lookup tables for this list and use scalar accumulation
            auto luts = state_.pq->compute_lookup_tables(lut_query);

            // ADC debug: detailed breakdown for specific vectors
            const bool adc_debug = dbg;
            const std::uint64_t debug_gt_id = 932085;
            const std::uint64_t debug_top1_id = 695756;

            for (const auto& entry : list) {
                const float dist = compute_adc_distance(lut_query, entry.code, luts);

                // ADC debug output for specific IDs
                if (adc_debug && (entry.id == debug_gt_id || entry.id == debug_top1_id)) {
                    std::cerr << "[ADC_DEBUG][id=" << entry.id << "] centroid=" << list_idx << " codes=[";
                    for (std::uint32_t m = 0; m < state_.params.m; ++m) {
                        if (m > 0) std::cerr << ",";
                        std::cerr << static_cast<int>(entry.code.codes[m]);
                    }
                    std::cerr << "]\n";

                    std::cerr << "[ADC_DEBUG][id=" << entry.id << "] lut_values=[";
                    for (std::uint32_t m = 0; m < state_.params.m; ++m) {
                        if (m > 0) std::cerr << ",";
                        const float* lut_m = luts[m];
                        const float lut_val = lut_m[entry.code.codes[m]];
                        std::cerr << lut_val;
                    }
                    std::cerr << "] adc_sum=" << dist << "\n";
                }

                // Debug: record candidate for offline top-k audit (scalar path)
                if (validate_topk) {
                    dbg_candidates.emplace_back(dist, entry.id);
                    if (list_idx < dbg_scanned_per_list.size()) dbg_scanned_per_list[list_idx]++;
                }

                if (validate_pool) { ++dbg_scanned_total; }
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

    // Pool validation logging (pre-rerank, after top-k extraction)
    if (validate_pool) {
        const std::size_t after_trunc = std::min<std::size_t>(pool_k, dbg_scanned_total);
        std::cerr << "[POOL][k] pool_k=" << pool_k
                  << " use_heapless=" << (use_heapless?1:0) << "\n";
        std::cerr << "[POOL][counts] scanned=" << dbg_scanned_total
                  << " after_trunc=" << after_trunc
                  << " final_topk=" << results.size()
                  << " early_terminate=0" << "\n";
    }

    } else {
        results.reserve(heap.size());
        while (!heap.empty()) {
            const auto& [dist, id] = heap.top();
            results.emplace_back(id, dist);
            heap.pop();
        }
        std::reverse(results.begin(), results.end());
    }

    // Pool validation logging (pre-rerank, after top-k extraction)
    if (validate_pool) {
        const std::size_t after_trunc = std::min<std::size_t>(pool_k, dbg_scanned_total);
        std::cerr << "[POOL][k] pool_k=" << pool_k
                  << " use_heapless=" << (use_heapless?1:0) << "\n";
        std::cerr << "[POOL][counts] scanned=" << dbg_scanned_total
                  << " after_trunc=" << after_trunc
                  << " final_topk=" << results.size()
                  << " early_terminate=0" << "\n";
    }


    // Top-k/merging audit: compare online heap result vs offline sort of full candidate set
    if (dbg && validate_topk) {
        const std::size_t total_candidates = dbg_candidates.size();
        // Deduplicate by ID taking the best (smallest) distance per ID
        std::unordered_map<std::uint64_t, float> best_by_id;
        best_by_id.reserve(total_candidates);
        for (const auto& cd : dbg_candidates) {
            const float d = cd.first; const std::uint64_t id = cd.second;
            auto it = best_by_id.find(id);
            if (it == best_by_id.end() || d < it->second) {
                best_by_id[id] = d;
            }
        }
        std::vector<std::pair<std::uint64_t, float>> offline;
        offline.reserve(best_by_id.size());
        for (const auto& kv : best_by_id) offline.emplace_back(kv.first, kv.second);
        std::sort(offline.begin(), offline.end(), [](const auto& a, const auto& b){ return a.second < b.second; });

        const std::size_t unique_ids = best_by_id.size();
        const std::size_t duplicates = (total_candidates >= unique_ids) ? (total_candidates - unique_ids) : 0;
        std::cerr << "[TOPK][audit] candidates_total=" << total_candidates
                  << " unique_ids=" << unique_ids
                  << " duplicates=" << duplicates << "\n";
        // Per-list scanned summary (non-zero lists only, capped)
        std::size_t nonzero = 0, printed = 0; for (auto c : dbg_scanned_per_list) if (c) ++nonzero;
        if (nonzero) {
            std::cerr << "[TOPK][lists] nonzero_lists=" << nonzero << ":";
            for (std::size_t li = 0; li < dbg_scanned_per_list.size() && printed < 24; ++li) {
                const auto cnt = dbg_scanned_per_list[li];
                if (cnt) { std::cerr << " (" << li << "," << cnt << ")"; ++printed; }
            }
            std::cerr << "\n";
        }

        // Compare top-k (pre-rerank) between offline and online
        const std::size_t topN = std::min<std::size_t>(params.k, std::min(results.size(), offline.size()));
        std::size_t diverges = 0;
        for (std::size_t i = 0; i < topN; ++i) {
            const auto& on = results[i];
            const auto& off = offline[i];
            const float eps = 1e-5f;
            const float denom = std::max(1.0f, std::fabs(off.second));
            const bool same_id = (on.first == off.first);
            const bool close_dist = (std::fabs(on.second - off.second) / denom) <= eps;
            if (!(same_id && close_dist)) {
                if (diverges == 0) std::cerr << "[TOPK][diverge] showing up to first 10 mismatches\n";
                std::cerr << "[TOPK][diverge] i=" << i
                          << " off_id=" << off.first << " off_d=" << off.second
                          << " on_id=" << on.first << " on_d=" << on.second << "\n";
                if (++diverges >= 10) break;
            }
        }
        if (diverges == 0) {
            std::cerr << "[TOPK][ok] online heap matches offline sort for top-" << topN << " (pre-rerank)\n";
        }
    }


    // Apply exact reranking if requested
    if ((params.use_exact_rerank || dbg_force_rerank) && !results.empty()) {
        // Determine reranking size
        // Strategy: by default, rerank the entire candidate pool (pool_k) for maximum accuracy.
        // This matches FAISS-style refine-on-shortlist behavior and fixes cases where the true NN
        // sits outside the small k*3 head of the ADC shortlist. Keep configurability via params.rerank_k.
        const std::size_t rerank_k = params.rerank_k > 0 ?
            std::min<std::size_t>(params.rerank_k, results.size()) :
            std::min<std::size_t>(pool_k, results.size());

        // Only rerank top-rerank_k candidates
        const std::size_t candidates_to_rerank = std::min(rerank_k, results.size());

        // Store cluster assignments for each candidate to avoid redundant search
        std::unordered_map<std::uint64_t, std::uint32_t> id_to_cluster;

        // First pass: find cluster assignments for candidates
        for (std::size_t i = 0; i < candidates_to_rerank; ++i) {
            const std::uint64_t id = results[i].first;

            // Search for ID in inverted lists (could optimize with reverse index)
            for (std::uint32_t cluster_id = 0; cluster_id < state_.params.nlist; ++cluster_id) {
                const auto& list = inverted_lists_[cluster_id];
                for (const auto& entry : list) {
                    if (entry.id == id) {
                        id_to_cluster[id] = cluster_id;
                        goto next_candidate;
                    }
                }
            }
            next_candidate:;
        }

        // Second pass: reconstruct and compute exact distances
        for (std::size_t i = 0; i < candidates_to_rerank; ++i) {
            const std::uint64_t id = results[i].first;

            auto cluster_it = id_to_cluster.find(id);
            if (cluster_it == id_to_cluster.end()) {
                continue; // Shouldn't happen, but be safe
            }

            const std::uint32_t cluster_id = cluster_it->second;
            const auto& list = inverted_lists_[cluster_id];
            const float* centroid = state_.coarse_centroids[cluster_id];

            // Find the entry
            for (const auto& entry : list) {
                if (entry.id == id) {
                    // Reconstruct the vector
                    std::vector<float> reconstructed(state_.dim);

                    // Decode PQ code to residual
                    std::vector<float> residual_code(state_.dim);
                    state_.pq->decode(entry.code.data(), 1, residual_code.data());

                    // Apply inverse rotation if OPQ was used
                    const float* residual_ptr = residual_code.data();
                    std::vector<float> residual_orig;
                    if (state_.params.use_opq && !state_.rotation_matrix.empty()) {
                        residual_orig.resize(state_.dim);
                        apply_rotation_T(residual_code.data(), residual_orig.data(), 1);
                        residual_ptr = residual_orig.data();
                    }

                    // Add centroid to get full vector
                    for (std::size_t d = 0; d < state_.dim; ++d) {
                        reconstructed[d] = centroid[d] + residual_ptr[d];
                    }

                    // Compute exact L2 distance
                    float exact_dist = 0.0f;
                    for (std::size_t d = 0; d < state_.dim; ++d) {
                        float diff = query[d] - reconstructed[d];
                        exact_dist += diff * diff;
                    }

                    // Update distance with exact value
                    results[i].second = exact_dist;
                    break;
                }
            }


        }

        // Re-sort by exact distances (only the reranked portion)
        if (candidates_to_rerank > 1) {
            std::sort(results.begin(), results.begin() + candidates_to_rerank,
                     [](const auto& a, const auto& b) { return a.second < b.second; });
        }

        // Trim to requested k
        if (results.size() > params.k) {
            results.resize(params.k);
        }
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
    // Projection coarse assigner always enables ANN telemetry
    stats.ann_enabled = impl_->state_.params.use_centroid_ann || (impl_->state_.params.coarse_assigner == CoarseAssigner::Projection);
    stats.ann_assignments = impl_->ann_assignments_.load(std::memory_order_relaxed);
    stats.ann_validated = impl_->ann_validated_.load(std::memory_order_relaxed);
    stats.ann_mismatches = impl_->ann_mismatches_.load(std::memory_order_relaxed);
    // KD-tree instrumentation
    stats.kd_nodes_pushed = impl_->kd_nodes_pushed_.load(std::memory_order_relaxed);
    stats.kd_nodes_popped = impl_->kd_nodes_popped_.load(std::memory_order_relaxed);
    stats.kd_leaves_scanned = impl_->kd_leaves_scanned_.load(std::memory_order_relaxed);

    // Timing telemetry (enabled if verbose or timings flag or env VESPER_TIMING)
    const bool env_timing = [](){ auto v = vesper::core::safe_getenv("VESPER_TIMING"); return v && !v->empty() && ((*v)[0] != '0'); }();
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
    bool use_v11 = false;
    if (auto e = vesper::core::safe_getenv("VESPER_IVFPQ_SAVE_V11")) { use_v11 = (!e->empty() && (*e)[0] == '1'); }
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
            // Write header fields individually to avoid struct padding issues
            write_and_hash(&hdr.type, sizeof(hdr.type));
            write_and_hash(&hdr.unc, sizeof(hdr.unc));
            write_and_hash(&hdr.comp, sizeof(hdr.comp));
            write_and_hash(&hdr.shash, sizeof(hdr.shash));
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
    bool want_mmap = false; if (auto e = vesper::core::safe_getenv("VESPER_IVFPQ_LOAD_MMAP")) want_mmap = (!e->empty() && (*e)[0]=='1');
#if defined(_WIN32)
    const bool mmap_supported = false; // Temporarily disable mmap on Windows; fallback to streaming for stability
#else
    const bool mmap_supported = true;
#endif
    if (want_mmap && is_v11 && mmap_supported) {
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
            SectionHdr hdr{};
            // Read header fields individually to avoid struct padding issues
            if (!read_and_hash(&hdr.type, sizeof(hdr.type))) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading section type", "ivf_pq.load"});
            if (!read_and_hash(&hdr.unc, sizeof(hdr.unc))) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading section unc size", "ivf_pq.load"});
            if (!read_and_hash(&hdr.comp, sizeof(hdr.comp))) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading section comp size", "ivf_pq.load"});
            if (!read_and_hash(&hdr.shash, sizeof(hdr.shash))) return std::vesper_unexpected(error{ error_code::io_failed, "Failed reading section hash", "ivf_pq.load"});
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
    if (auto e = vesper::core::safe_getenv("VESPER_IVFPQ_LOAD_STREAM_CENTROIDS")) {
        if (!e->empty() && (*e)[0] == '0') stream_centroids = false;
        else if (!e->empty() && (*e)[0] == '1') stream_centroids = true;
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


// --- Debug explain hooks (placed before namespace close) ---

auto IvfPqIndex::Impl::debug_explain_adc_rank(const float* query, std::uint64_t target_id) const
    -> std::expected<std::pair<std::size_t, float>, core::error> {
    using core::error;
    using core::error_code;

    const bool dbg = ([](){ auto v = vesper::core::safe_getenv("VESPER_IVFPQ_DEBUG"); return v && !v->empty() && ((*v)[0] != '0'); })();
    if (!dbg) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "debug_explain_adc_rank requires VESPER_IVFPQ_DEBUG=1",
            "ivf_pq"
        });
    }

    if (!state_.trained || n_vectors_ == 0) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Index must be trained and non-empty",
            "ivf_pq"
        });
    }

    const std::uint32_t nlist = state_.params.nlist;
    const std::size_t dim = state_.dim;

    std::vector<float> residual_query(dim);
    std::vector<float> residual_query_rot(dim);

    // Pass 1: locate target ID and compute its ADC distance
    bool found = false;
    float target_dist = 0.0f;
    for (std::uint32_t list_idx = 0; list_idx < nlist && !found; ++list_idx) {
        const auto& list = inverted_lists_[list_idx];
        const float* centroid = state_.coarse_centroids[list_idx];
        for (std::size_t d = 0; d < dim; ++d) {
            residual_query[d] = query[d] - centroid[d];
        }
        const float* lut_query = residual_query.data();
        if (state_.params.use_opq && !state_.rotation_matrix.empty()) {
            apply_rotation(residual_query.data(), residual_query_rot.data(), 1);
            lut_query = residual_query_rot.data();
        }
        auto luts = state_.pq->compute_lookup_tables(lut_query);
        for (const auto& entry : list) {
            if (entry.id == target_id) {
                target_dist = compute_adc_distance(lut_query, entry.code, luts);
                found = true;
                break;
            }
        }
    }
    if (!found) {
        return std::vesper_unexpected(error{
            error_code::not_found,
            "Target ID not found in index",
            "ivf_pq"
        });
    }

    // Pass 2: count how many entries have ADC distance strictly less than target
    std::size_t count_less = 0;
    for (std::uint32_t list_idx = 0; list_idx < nlist; ++list_idx) {
        const auto& list = inverted_lists_[list_idx];
        const float* centroid = state_.coarse_centroids[list_idx];
        for (std::size_t d = 0; d < dim; ++d) {
            residual_query[d] = query[d] - centroid[d];
        }
        const float* lut_query = residual_query.data();
        if (state_.params.use_opq && !state_.rotation_matrix.empty()) {
            apply_rotation(residual_query.data(), residual_query_rot.data(), 1);
            lut_query = residual_query_rot.data();
        }
        auto luts = state_.pq->compute_lookup_tables(lut_query);
        for (const auto& entry : list) {
            const float dist = compute_adc_distance(lut_query, entry.code, luts);
            if (dist < target_dist) {
                ++count_less;
            }
        }
    }

    const std::size_t rank = count_less + 1; // 1-based
    return std::make_pair(rank, target_dist);
}


auto IvfPqIndex::Impl::debug_explain_centroid_rank(const float* query, std::uint64_t target_id) const
    -> std::expected<std::pair<std::size_t, std::uint32_t>, core::error> {
    using core::error;
    using core::error_code;

    const bool dbg = ([](){ auto v = vesper::core::safe_getenv("VESPER_IVFPQ_DEBUG"); return v && !v->empty() && ((*v)[0] != '0'); })();
    if (!dbg) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "debug_explain_centroid_rank requires VESPER_IVFPQ_DEBUG=1",
            "ivf_pq"
        });
    }

    if (!state_.trained || n_vectors_ == 0) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Index must be trained and non-empty",
            "ivf_pq"
        });
    }

    const std::uint32_t nlist = state_.params.nlist;
    const std::size_t dim = state_.dim;

    // 1) Locate which centroid (inverted list) holds target_id
    std::optional<std::uint32_t> assigned;
    for (std::uint32_t list_idx = 0; list_idx < nlist && !assigned.has_value(); ++list_idx) {
        const auto& list = inverted_lists_[list_idx];
        for (const auto& entry : list) {
            if (entry.id == target_id) { assigned = list_idx; break; }
        }
    }
    if (!assigned.has_value()) {
        return std::vesper_unexpected(error{ error_code::not_found, "Target ID not found in index", "ivf_pq" });
    }

    // 2) Compute L2 distances from query to all centroids and rank the assigned one
    const auto& ops = kernels::select_backend_auto();
    float assigned_dist = std::numeric_limits<float>::infinity();
    std::size_t count_less = 0;

    for (std::uint32_t c = 0; c < nlist; ++c) {
        const float dist = ops.l2_sq(std::span(query, dim), state_.coarse_centroids.get_centroid(c));
        if (c == *assigned) assigned_dist = dist;
    }
    // Count strictly smaller distances
    for (std::uint32_t c = 0; c < nlist; ++c) {
        const float dist = ops.l2_sq(std::span(query, dim), state_.coarse_centroids.get_centroid(c));
        if (dist < assigned_dist) ++count_less;
    }

    const std::size_t rank = count_less + 1; // 1-based
    return std::make_pair(rank, *assigned);
}

auto IvfPqIndex::debug_explain_centroid_rank(const float* query, std::uint64_t target_id) const
    -> std::expected<std::pair<std::size_t, std::uint32_t>, core::error> {
    return impl_->debug_explain_centroid_rank(query, target_id);
}

auto IvfPqIndex::debug_explain_adc_rank(const float* query, std::uint64_t target_id) const
    -> std::expected<std::pair<std::size_t, float>, core::error> {
    return impl_->debug_explain_adc_rank(query, target_id);
}

} // namespace vesper::index
