#include "vesper/index/kmeans_elkan.hpp"
#include "vesper/kernels/dispatch.hpp"
#include "vesper/kernels/batch_distances.hpp"
#include "vesper/core/memory_pool.hpp"
#include "vesper/core/platform_utils.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <random>
#include <execution>
#include <cmath>
#include <thread>
#include <iostream>
#include <span>


namespace vesper::index {

namespace {

/** \brief Compute squared L2 distance using SIMD kernels. */
[[gnu::hot, gnu::always_inline]]
inline auto compute_distance(const float* a, const float* b, std::size_t dim) -> float {
    const auto& ops = kernels::select_backend_auto();
    return ops.l2_sq(std::span(a, dim), std::span(b, dim));
}

/** \brief Update centroid from assigned points. */
auto update_centroid(const float* data, std::size_t dim,
                    const std::vector<std::size_t>& indices,
                    std::vector<float>& centroid) -> void {
    if (indices.empty()) return;

    std::fill(centroid.begin(), centroid.end(), 0.0f);

    // Accumulate with Kahan summation for numerical stability
    std::vector<double> sum(dim, 0.0);
    std::vector<double> c(dim, 0.0);  // Compensation

    for (std::size_t idx : indices) {
        const float* point = data + idx * dim;
        for (std::size_t d = 0; d < dim; ++d) {
            double y = point[d] - c[d];
            double t = sum[d] + y;
            c[d] = (t - sum[d]) - y;
            sum[d] = t;
        }
    }

    const double scale = 1.0 / static_cast<double>(indices.size());
    for (std::size_t d = 0; d < dim; ++d) {
        centroid[d] = static_cast<float>(sum[d] * scale);
    }
}

} // anonymous namespace

auto KmeansElkan::compute_distance_instrumented(const float* a, const float* b,
                                               std::size_t dim) -> float {
    distance_counter_.fetch_add(1, std::memory_order_relaxed);
    return compute_distance(a, b, dim);
}

auto KmeansElkan::initialize_bounds(const float* data, std::size_t n, std::size_t dim,
                                   const std::vector<std::vector<float>>& centroids)
    -> std::vector<PointBounds> {
    const std::uint32_t k = centroids.size();
    std::vector<PointBounds> bounds(n);

    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < static_cast<int>(n); ++i) {
        const float* point = data + i * dim;
        bounds[i].lower.resize(k);

        // Find nearest centroid and compute all distances
        float min_dist = std::numeric_limits<float>::max();
        std::uint32_t best_c = 0;

        for (std::uint32_t c = 0; c < k; ++c) {
            const float dist = compute_distance_instrumented(
                point, centroids[c].data(), dim);
            bounds[i].lower[c] = dist;

            if (dist < min_dist) {
                min_dist = dist;
                best_c = c;
            }
        }

        bounds[i].upper = min_dist;
        bounds[i].assignment = best_c;
        bounds[i].upper_tight = true;
    }

    return bounds;
}

auto KmeansElkan::update_centroid_distances(
    const std::vector<std::vector<float>>& centroids)
    -> std::pair<std::vector<std::vector<float>>, std::vector<float>> {

    const std::uint32_t k = centroids.size();
    const std::size_t dim = centroids[0].size();

    // Compute inter-centroid distances
    std::vector<std::vector<float>> dist(k, std::vector<float>(k, 0.0f));

    #pragma omp parallel for collapse(2) if(k > 16)
    for (int i = 0; i < static_cast<int>(k); ++i) {
        for (int j = i + 1; j < static_cast<int>(k); ++j) {
            const float d = compute_distance(
                centroids[i].data(), centroids[j].data(), dim);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }

    // Compute s values for squared-distance Elkan: s_quarter = 0.25 * min_j ||mu_i - mu_j||^2
    std::vector<float> s_quarter(k);
    for (std::uint32_t i = 0; i < k; ++i) {
        float min_d2 = std::numeric_limits<float>::max();
        for (std::uint32_t j = 0; j < k; ++j) {
            if (i != j && dist[i][j] < min_d2) {
                min_d2 = dist[i][j];
            }
        }
        s_quarter[i] = 0.25f * min_d2;
    }

    return {dist, s_quarter};
}

auto KmeansElkan::update_centroid_distances_aligned(
    const AlignedCentroidBuffer& centroids)
    -> std::pair<AlignedDistanceMatrix, std::vector<float>> {

    const std::uint32_t k = centroids.size();

    AlignedDistanceMatrix dist(k);

    // Use SIMD-optimized symmetric distance matrix computation
    kernels::compute_symmetric_distance_matrix(centroids, dist);

    // Compute s values (squared-distance Elkan): s_quarter = 0.25 * min_j ||mu_i - mu_j||^2
    std::vector<float> s_quarter(k);

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(k); ++i) {
        const auto row = dist.row(i);
        float min_d2 = std::numeric_limits<float>::max();

        // Unrolled loop for better vectorization
        std::uint32_t j = 0;
        for (; j + 4 <= k; j += 4) {
            if (j != i && j + 1 != i && j + 2 != i && j + 3 != i) {
                const float block_min = std::min(std::min(row[j], row[j+1]), std::min(row[j+2], row[j+3]));
                min_d2 = std::min(min_d2, block_min);
            } else {
                // Handle cases where i is in this block
                for (std::uint32_t jj = j; jj < j + 4; ++jj) {
                    if (jj != i) {
                        min_d2 = std::min(min_d2, row[jj]);
                    }
                }
            }
        }

        // Handle remainder
        for (; j < k; ++j) {
            if (j != i) {
                min_d2 = std::min(min_d2, row[j]);
            }
        }

        s_quarter[i] = 0.25f * min_d2;
    }

    return {std::move(dist), s_quarter};
}

auto KmeansElkan::assign_with_bounds(const float* data, std::size_t n, std::size_t dim,
                                    const std::vector<std::vector<float>>& centroids,
                                    std::vector<PointBounds>& bounds,
                                    const std::vector<float>& s_quarter)
    -> float {
    const std::uint32_t k = centroids.size();
    double total_inertia = 0.0;

    #pragma omp parallel for schedule(dynamic, 256) reduction(+:total_inertia)
    for (int i = 0; i < static_cast<int>(n); ++i) {
        const float* point = data + i * dim;
        auto& b = bounds[i];

        // Step 3a: If upper bound indicates we can skip (s_quarter with squared distances)
        if (b.upper <= s_quarter[b.assignment]) {
            // Ensure inertia uses a tight upper bound
            if (!b.upper_tight) {
                const float actual_dist = compute_distance_instrumented(
                    point, centroids[b.assignment].data(), dim);
                b.lower[b.assignment] = actual_dist;
                b.upper = actual_dist;
                b.upper_tight = true;
            }
            skip_counter_.fetch_add(k - 1, std::memory_order_relaxed);
            total_inertia += b.upper;
            continue;
        }

        // Step 3b: Tighten upper bound if needed
        if (!b.upper_tight) {
            const float actual_dist = compute_distance_instrumented(
                point, centroids[b.assignment].data(), dim);
            b.lower[b.assignment] = actual_dist;
            b.upper = actual_dist;
            b.upper_tight = true;

            if (b.upper <= s_quarter[b.assignment]) {
                skip_counter_.fetch_add(k - 1, std::memory_order_relaxed);
                total_inertia += b.upper;
                continue;
            }
        }

        // Step 3c: Check other centroids using lower bounds
        [[maybe_unused]] bool changed = false;
        for (std::uint32_t c = 0; c < k; ++c) {
            if (c == b.assignment) continue;

            // Skip if lower bound indicates this centroid is too far
            if (b.upper <= b.lower[c]) {
                skip_counter_.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            // Skip if centroid is definitely farther (squared-distance Elkan: 0.25 * ||mu_a - mu_c||^2)
            if (b.upper <= 0.25f * inter_centroid_dist_cache_[b.assignment][c]) {
                skip_counter_.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            // Must compute actual distance
            const float dist = compute_distance_instrumented(
                point, centroids[c].data(), dim);
            b.lower[c] = dist;

            if (dist < b.upper) {
                // Found closer centroid
                b.assignment = c;
                b.upper = dist;
                b.upper_tight = true;
                changed = true;
                reassignment_counter_.fetch_add(1, std::memory_order_relaxed);
            }
        }

        total_inertia += b.upper;
    }

    return static_cast<float>(total_inertia);
}

auto KmeansElkan::update_bounds(std::vector<PointBounds>& bounds,
                              const std::vector<float>& centroid_shift,
                              [[maybe_unused]] const std::vector<std::vector<float>>& inter_centroid_dist) {
    const std::uint32_t k = centroid_shift.size();

    // Convert centroid shifts (squared) to Euclidean deltas
    std::vector<float> delta(k);
    for (std::uint32_t c = 0; c < k; ++c) {
        delta[c] = std::sqrt(std::max(0.0f, centroid_shift[c]));
    }

    #pragma omp parallel for
    for (int idx = 0; idx < static_cast<int>(bounds.size()); ++idx) {
        auto& b = bounds[idx];
        // Update upper bound: U'^2 <= (sqrt(U^2) + δ_a)^2
        float u = std::sqrt(std::max(0.0f, b.upper));
        u += delta[b.assignment];
        b.upper = u * u;
        b.upper_tight = false;

        // Update lower bounds: L_j'^2 >= max(0, sqrt(L_j^2) - δ_j)^2
        for (std::uint32_t c = 0; c < k; ++c) {
            float l = std::sqrt(std::max(0.0f, b.lower[c]));
            l = std::max(0.0f, l - delta[c]);
            b.lower[c] = l * l;
        }
    }
}

auto KmeansElkan::cluster(const float* data, std::size_t n, std::size_t dim,
                         const Config& config)
    -> std::expected<KmeansResult, core::error> {
    using core::error;
    using core::error_code;

    if (n < config.k) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Not enough data points for k clusters",
            "kmeans_elkan"
        });
    }

    const auto start_time = std::chrono::steady_clock::now();

    // Reset statistics
    distance_counter_ = 0;
    skip_counter_ = 0;
    reassignment_counter_ = 0;

    // Initialize centroids using selected method
    std::vector<std::vector<float>> centroids;
    if (config.init_method == KmeansElkan::Config::InitMethod::KMeansParallel) {
        const std::uint32_t rounds = config.kmeans_parallel_rounds;
        const std::uint32_t l = config.kmeans_parallel_oversampling ?
                                config.kmeans_parallel_oversampling : 0u;
        centroids = kmeans_parallel_init(data, n, dim, config.k, rounds, l, config.seed);
    } else {
        centroids = kmeans_plusplus_init(data, n, dim, config.k, config.seed);
    }

    // Initialize bounds
    auto bounds = initialize_bounds(data, n, dim, centroids);

    // Cache inter-centroid distances
    auto [inter_dist, s_quarter] = update_centroid_distances(centroids);
    inter_centroid_dist_cache_ = inter_dist;

    float prev_inertia = std::numeric_limits<float>::max();
    std::uint32_t iter = 0;

    for (; iter < config.max_iter; ++iter) {
        // E-step: Assign points using bounds
        const float inertia = assign_with_bounds(data, n, dim, centroids, bounds, s_quarter);

        // Check convergence
        const float change = std::abs(prev_inertia - inertia) / (prev_inertia + 1e-10f);
        if (change < config.epsilon) {
            break;
        }
        prev_inertia = inertia;

        // M-step: Update centroids with pooled memory
        core::PoolScope pool_scope;
        auto cluster_indices = core::make_pooled_vector<
            core::PooledVector<std::size_t>>(config.k);
        for (auto& indices : cluster_indices) {
            indices.reserve(n / config.k + 1);  // Estimate size
        }

        for (std::size_t i = 0; i < n; ++i) {
            cluster_indices[bounds[i].assignment].push_back(i);
        }

        std::vector<float> centroid_shift(config.k);
        #pragma omp parallel for
        for (int c = 0; c < static_cast<int>(config.k); ++c) {
            std::vector<float> old_centroid = centroids[c];
            // Convert PooledVector to std::vector for the function call
            std::vector<std::size_t> indices(cluster_indices[c].begin(),
                                            cluster_indices[c].end());
            update_centroid(data, dim, indices, centroids[c]);
            centroid_shift[c] = compute_distance(
                old_centroid.data(), centroids[c].data(), dim);
        }

        // Update bounds
        update_bounds(bounds, centroid_shift, inter_dist);

        // Update inter-centroid distances
        std::tie(inter_dist, s_quarter) = update_centroid_distances(centroids);
        inter_centroid_dist_cache_ = inter_dist;
    }

    // Prepare result
    std::vector<std::uint32_t> assignments(n);
    std::vector<std::uint32_t> cluster_sizes(config.k, 0);
    for (std::size_t i = 0; i < n; ++i) {
        assignments[i] = bounds[i].assignment;
        cluster_sizes[bounds[i].assignment]++;
    }

    // Count empty clusters
    std::uint32_t empty_clusters = 0;
    for (std::uint32_t c = 0; c < config.k; ++c) {
        if (cluster_sizes[c] == 0) {
            ++empty_clusters;
        }
    }

    const auto end_time = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration<float>(end_time - start_time);

    // Update statistics
    stats_.distance_computations = distance_counter_.load();
    stats_.distance_skipped = skip_counter_.load();
    const auto attempted = stats_.distance_skipped + stats_.distance_computations;
    stats_.skip_rate = attempted ? static_cast<float>(stats_.distance_skipped) /
                                   static_cast<float>(attempted) : 0.0f;
    stats_.iterations = iter;
    stats_.final_inertia = prev_inertia;
    stats_.time_sec = duration.count();
    stats_.reassignments = reassignment_counter_.load();
    stats_.empty_clusters = empty_clusters;

    // Debug output when VESPER_IVFPQ_DEBUG environment variable is set
    if (auto v = core::safe_getenv("VESPER_IVFPQ_DEBUG"); v && !v->empty() && (*v)[0] == '1') {
        std::cerr << "[KMEANS][debug] n=" << n << " dim=" << dim << " k=" << config.k << "\n";
        std::cerr << "[KMEANS][debug] iterations=" << stats_.iterations
                  << " final_inertia=" << stats_.final_inertia << "\n";
        std::cerr << "[KMEANS][debug] distance_computations=" << stats_.distance_computations
                  << " distance_skipped=" << stats_.distance_skipped
                  << " skip_rate=" << (stats_.skip_rate * 100.0f) << "%\n";
        std::cerr << "[KMEANS][debug] reassignments=" << stats_.reassignments
                  << " empty_clusters=" << stats_.empty_clusters << "\n";
        std::cerr << "[KMEANS][debug] time_sec=" << stats_.time_sec << "\n";
    }

    return KmeansResult{
        .centroids = std::move(centroids),
        .assignments = std::move(assignments),
        .cluster_sizes = std::move(cluster_sizes),
        .inertia = prev_inertia,
        .iterations = iter,
        .time_sec = duration.count()
    };
}


auto kmeans_parallel_init(const float* data, std::size_t n, std::size_t dim,
                          std::uint32_t k, std::uint32_t rounds,
                          std::uint32_t oversampling_l,
                          std::uint32_t seed)
    -> std::vector<std::vector<float>> {
    std::mt19937_64 gen(seed);

    // 1) Pick one random center uniformly
    std::uniform_int_distribution<std::size_t> first_dist(0, n - 1);
    std::vector<std::vector<float>> centers;
    centers.reserve(static_cast<std::size_t>(k) * 4);
    const std::size_t first_idx = first_dist(gen);
    centers.emplace_back(data + first_idx * dim, data + (first_idx + 1) * dim);

    // Oversampling factor l (default 2k)
    const std::uint32_t l = oversampling_l ? oversampling_l : (2u * k);

    // 2) Rounds of parallel sampling
    for (std::uint32_t round = 0; round < rounds; ++round) {
        // Compute D^2(x) to the nearest center for all x
        std::vector<float> d2(n);
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(n); ++i) {
            const float* point = data + static_cast<std::size_t>(i) * dim;
            float best = std::numeric_limits<float>::max();
            for (const auto& c : centers) {
                const float dist2 = compute_distance(point, c.data(), dim);
                if (dist2 < best) best = dist2;
            }
            d2[static_cast<std::size_t>(i)] = best;
        }

        // Potential phi = sum D^2(x)
        const double phi = std::accumulate(d2.begin(), d2.end(), 0.0);
        if (phi <= 0.0) break;

        // Sample each point independently with p(x) = min(1, l * D^2(x) / phi)
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        for (std::size_t i = 0; i < n; ++i) {
            const double px = std::min(1.0, (static_cast<double>(l) * d2[i]) / phi);
            if (uni(gen) < px) {
                centers.emplace_back(data + i * dim, data + (i + 1) * dim);
            }
        }
    }

    // 3) Compute weights: number of points assigned to each sampled center
    const std::size_t csz = centers.size();
    if (csz == 0) {
        // Fallback: should not happen, but return trivial k-means++ init
        return kmeans_plusplus_init(data, n, dim, k, seed);
    }
    std::vector<double> weights(csz, 0.0);
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(n); ++i) {
        const float* point = data + static_cast<std::size_t>(i) * dim;
        std::size_t best_j = 0;
        float best = std::numeric_limits<float>::max();
        for (std::size_t j = 0; j < csz; ++j) {
            const float dist2 = compute_distance(point, centers[j].data(), dim);
            if (dist2 < best) { best = dist2; best_j = j; }
        }
        #pragma omp atomic
        weights[best_j] += 1.0;
    }

    // 4) Weighted k-means++ on sampled centers to pick exactly k
    // First center chosen with prob proportional to weight
    std::vector<std::vector<float>> final_centers;
    final_centers.reserve(k);

    auto choose_index_by_weight = [&](const std::vector<double>& w) -> std::size_t {
        const double total = std::accumulate(w.begin(), w.end(), 0.0);
        std::uniform_real_distribution<double> u(0.0, total);
        double r = u(gen);
        for (std::size_t j = 0; j < w.size(); ++j) {
            r -= w[j];
            if (r <= 0.0) return j;
        }
        return w.empty() ? 0 : (w.size() - 1);
    };

    // Pick first
    std::size_t first = choose_index_by_weight(weights);
    final_centers.push_back(centers[first]);

    // Distances to nearest chosen center among sampled centers
    std::vector<double> d2_to_chosen(csz, std::numeric_limits<double>::infinity());
    for (std::size_t j = 0; j < csz; ++j) {
        double d = compute_distance(centers[j].data(), final_centers[0].data(), dim);
        d2_to_chosen[j] = d;
    }

    while (final_centers.size() < k && final_centers.size() < csz) {
        // Prob proportional to weight * D^2 to nearest chosen
        std::vector<double> score(csz, 0.0);
        for (std::size_t j = 0; j < csz; ++j) {
            score[j] = weights[j] * d2_to_chosen[j];
        }
        std::size_t next = choose_index_by_weight(score);
        final_centers.push_back(centers[next]);

        // Update d2_to_chosen
        for (std::size_t j = 0; j < csz; ++j) {
            double d = compute_distance(centers[j].data(), final_centers.back().data(), dim);
            if (d < d2_to_chosen[j]) d2_to_chosen[j] = d;
        }
    }

    // If still fewer than k (edge case), pad with best-weighted remaining centers
    if (final_centers.size() < k) {
        std::vector<std::size_t> idx(csz);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b){return weights[a] > weights[b];});
        for (std::size_t t = 0; t < idx.size() && final_centers.size() < k; ++t) {
            final_centers.push_back(centers[idx[t]]);
        }
    }

    return final_centers;
}

} // namespace vesper::index
