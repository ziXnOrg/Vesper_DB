#include "vesper/index/ivf_pq.hpp"
#include "vesper/index/kmeans_elkan.hpp"
#ifdef __x86_64__
#include "vesper/index/pq_fastscan.hpp"
#else
#include "vesper/index/pq_simple.hpp"
#endif
#include "vesper/kernels/dispatch.hpp"
#include "vesper/kernels/batch_distances.hpp"
#include "vesper/core/memory_pool.hpp"

#include <algorithm>
#include <numeric>
#include <queue>
#include <thread>
#include <fstream>
#include <cstring>

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
        std::unique_ptr<FastScanPq> pq;
        std::vector<float> rotation_matrix;  // For OPQ
    } state_;
    
    /** \brief Inverted list entry. */
    struct InvertedListEntry {
        std::uint64_t id;
        PqCode code;
    };
    
    /** \brief Inverted lists for each coarse centroid. */
    std::vector<std::vector<InvertedListEntry>> inverted_lists_;
    
    /** \brief Statistics. */
    std::size_t n_vectors_{0};
    std::mutex lists_mutex_;
    
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
    
    /** \brief Apply rotation to vectors. */
    auto apply_rotation(const float* input, float* output, std::size_t n) const -> void;
    
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
    
    // Step 2: Learn OPQ rotation if enabled
    if (params.use_opq) {
        if (auto result = learn_opq_rotation(data, n); !result.has_value()) {
            return std::vesper_unexpected(result.error());
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
    
    return {};
}

auto IvfPqIndex::Impl::train_product_quantizer(const float* data, std::size_t n)
    -> std::expected<void, core::error> {
    
    // Apply rotation if OPQ is enabled
    std::vector<float> rotated_data;
    const float* training_data = data;
    
    if (state_.params.use_opq && !state_.rotation_matrix.empty()) {
        rotated_data.resize(n * state_.dim);
        apply_rotation(data, rotated_data.data(), n);
        training_data = rotated_data.data();
    }
    
    // Configure and train FastScan PQ
    FastScanPqConfig pq_config{
        .m = state_.params.m,
        .nbits = state_.params.nbits,
        .block_size = 32
    };
    
    state_.pq = std::make_unique<FastScanPq>(pq_config);
    
    if (auto result = state_.pq->train(training_data, n, state_.dim); 
        !result.has_value()) {
        return std::vesper_unexpected(result.error());
    }
    
    return {};
}

auto IvfPqIndex::Impl::learn_opq_rotation(const float* /* data */, std::size_t /* n */)
    -> std::expected<void, core::error> {
    // OPQ rotation learning will be implemented in next phase
    // For now, use identity rotation
    const std::size_t matrix_size = state_.dim * state_.dim;
    state_.rotation_matrix.resize(matrix_size, 0.0f);
    
    // Identity matrix
    for (std::size_t i = 0; i < state_.dim; ++i) {
        state_.rotation_matrix[i * state_.dim + i] = 1.0f;
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
    for (std::size_t i = 0; i < n; ++i) {
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
    
    // Apply rotation if OPQ
    std::vector<float> rotated_data;
    const float* encode_data = data;
    
    if (state_.params.use_opq && !state_.rotation_matrix.empty()) {
        rotated_data.resize(n * state_.dim);
        apply_rotation(data, rotated_data.data(), n);
        encode_data = rotated_data.data();
    }
    
    // Encode vectors with PQ
    std::vector<std::uint8_t> all_codes(n * state_.params.m);
    state_.pq->encode(encode_data, n, all_codes.data());
    
    // Assign to coarse centroids and add to inverted lists
    std::vector<std::uint32_t> assignments(n);
    
    #pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
        const float* vec = data + i * state_.dim;
        
        // Find nearest coarse centroid
        float min_dist = std::numeric_limits<float>::max();
        std::uint32_t best_idx = 0;
        
        for (std::uint32_t c = 0; c < state_.params.nlist; ++c) {
            const auto& ops = kernels::select_backend_auto();
            const float dist = ops.l2_sq(
                std::span(vec, state_.dim),
                state_.coarse_centroids.get_centroid(c)
            );
            
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = c;
            }
        }
        
        assignments[i] = best_idx;
    }
    
    // Add to inverted lists (serialize access)
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
    
    // Apply rotation if OPQ
    std::vector<float> rotated_query;
    const float* search_query = query;
    
    if (state_.params.use_opq && !state_.rotation_matrix.empty()) {
        rotated_query.resize(state_.dim);
        apply_rotation(query, rotated_query.data(), 1);
        search_query = rotated_query.data();
    }
    
    // Find nprobe nearest coarse centroids
    auto probe_lists = find_nearest_centroids(query, params.nprobe);
    
    // Compute lookup tables for ADC
    auto luts = state_.pq->compute_lookup_tables(search_query);
    
    // Collect candidates from inverted lists
    using Candidate = std::pair<float, std::uint64_t>;
    std::priority_queue<Candidate> heap;
    
    for (const auto& [list_idx, coarse_dist] : probe_lists) {
        const auto& list = inverted_lists_[list_idx];
        
        for (const auto& entry : list) {
            const float dist = compute_adc_distance(search_query, entry.code, luts);
            
            if (heap.size() < params.k) {
                heap.emplace(dist, entry.id);
            } else if (dist < heap.top().first) {
                heap.pop();
                heap.emplace(dist, entry.id);
            }
        }
    }
    
    // Extract results in sorted order
    std::vector<std::pair<std::uint64_t, float>> results;
    results.reserve(heap.size());
    
    while (!heap.empty()) {
        const auto& [dist, id] = heap.top();
        results.emplace_back(id, dist);
        heap.pop();
    }
    
    std::reverse(results.begin(), results.end());
    
    return results;
}

auto IvfPqIndex::Impl::search_batch(const float* queries, std::size_t n_queries,
                                   const IvfPqSearchParams& params) const
    -> std::expected<std::vector<std::vector<std::pair<std::uint64_t, float>>>, core::error> {
    
    std::vector<std::vector<std::pair<std::uint64_t, float>>> results(n_queries);
    
    #pragma omp parallel for
    for (std::size_t i = 0; i < n_queries; ++i) {
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
    
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return std::vesper_unexpected(error{
            error_code::io_failed,
            "Failed to open file for writing",
            "ivf_pq"
        });
    }
    
    // Write header
    const char magic[] = "IVFPQ001";
    file.write(magic, 8);
    
    // Write parameters
    file.write(reinterpret_cast<const char*>(&impl_->state_.params), 
               sizeof(IvfPqTrainParams));
    file.write(reinterpret_cast<const char*>(&impl_->state_.dim), sizeof(std::size_t));
    file.write(reinterpret_cast<const char*>(&impl_->n_vectors_), sizeof(std::size_t));
    
    // Write coarse centroids
    const auto centroid_data = impl_->state_.coarse_centroids.data();
    const std::size_t centroid_bytes = impl_->state_.params.nlist * 
                                       impl_->state_.dim * sizeof(float);
    file.write(reinterpret_cast<const char*>(centroid_data), centroid_bytes);
    
    // Write inverted lists
    for (const auto& list : impl_->inverted_lists_) {
        const std::size_t list_size = list.size();
        file.write(reinterpret_cast<const char*>(&list_size), sizeof(std::size_t));
        
        for (const auto& entry : list) {
            file.write(reinterpret_cast<const char*>(&entry.id), sizeof(std::uint64_t));
            file.write(reinterpret_cast<const char*>(entry.code.data()), 
                      impl_->state_.params.m);
        }
    }
    
    return {};
}

auto IvfPqIndex::load(const std::string& /* path */) 
    -> std::expected<IvfPqIndex, core::error> {
    // Serialization will be fully implemented in next phase
    return IvfPqIndex();
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

} // namespace vesper::index