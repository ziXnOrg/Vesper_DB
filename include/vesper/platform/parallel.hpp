#pragma once

/** \file parallel.hpp
 *  \brief Cross-platform parallelization abstractions.
 *
 * Provides portable parallelization with OpenMP fallback to serial.
 * Handles platform differences in OpenMP support and loop requirements.
 *
 * Key features:
 * - Safe parallel loops with proper index types
 * - Thread count management
 * - Reduction operations
 * - Critical sections
 */

#include <cstddef>
#include <algorithm>
#include <thread>
#include <functional>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vesper::platform {

/** \brief Get number of available threads.
 *
 * Returns OpenMP thread count if available, otherwise hardware concurrency.
 */
[[nodiscard]] inline auto get_num_threads() noexcept -> int {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return static_cast<int>(std::thread::hardware_concurrency());
#endif
}

/** \brief Set number of threads for parallel regions.
 *
 * \param num_threads Number of threads to use
 */
inline auto set_num_threads(int num_threads) noexcept -> void {
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#else
    (void)num_threads;  // No-op without OpenMP
#endif
}

/** \brief Get current thread ID.
 *
 * \return Thread ID (0-based)
 */
[[nodiscard]] inline auto get_thread_id() noexcept -> int {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;  // Single-threaded
#endif
}

/** \brief Check if in parallel region.
 *
 * \return true if executing in parallel region
 */
[[nodiscard]] inline auto in_parallel() noexcept -> bool {
#ifdef _OPENMP
    return omp_in_parallel() != 0;
#else
    return false;
#endif
}

/** \brief Parallel for loop with signed indices.
 *
 * Automatically handles OpenMP's requirement for signed loop variables.
 *
 * \param begin Starting index
 * \param end Ending index (exclusive)
 * \param func Function to call for each index
 * \param schedule OpenMP schedule type (static, dynamic, guided)
 * \param chunk_size Chunk size for scheduling
 */
template<typename Func>
inline auto parallel_for(std::size_t begin, std::size_t end, Func&& func,
                        const char* schedule = "static",
                        int chunk_size = 0) -> void {
    if (begin >= end) return;
    
#ifdef _OPENMP
    const int ibegin = static_cast<int>(begin);
    const int iend = static_cast<int>(end);
    
    if (chunk_size > 0) {
        if (std::string_view(schedule) == "static") {
            #pragma omp parallel for schedule(static, chunk_size)
            for (int i = ibegin; i < iend; ++i) {
                func(static_cast<std::size_t>(i));
            }
        } else if (std::string_view(schedule) == "dynamic") {
            #pragma omp parallel for schedule(dynamic, chunk_size)
            for (int i = ibegin; i < iend; ++i) {
                func(static_cast<std::size_t>(i));
            }
        } else {
            #pragma omp parallel for schedule(guided, chunk_size)
            for (int i = ibegin; i < iend; ++i) {
                func(static_cast<std::size_t>(i));
            }
        }
    } else {
        #pragma omp parallel for
        for (int i = ibegin; i < iend; ++i) {
            func(static_cast<std::size_t>(i));
        }
    }
#else
    // Serial fallback
    for (std::size_t i = begin; i < end; ++i) {
        func(i);
    }
#endif
}

/** \brief Parallel for loop with reduction.
 *
 * \param begin Starting index
 * \param end Ending index (exclusive)
 * \param init Initial value for reduction
 * \param func Function that returns a value to reduce
 * \param reduce Reduction operation (e.g., std::plus<>())
 * \return Reduced value
 */
template<typename T, typename Func, typename ReduceOp>
[[nodiscard]] inline auto parallel_reduce(std::size_t begin, std::size_t end,
                                          T init, Func&& func, ReduceOp&& reduce) -> T {
    if (begin >= end) return init;
    
#ifdef _OPENMP
    T result = init;
    const int ibegin = static_cast<int>(begin);
    const int iend = static_cast<int>(end);
    
    #pragma omp parallel
    {
        T local_result = init;
        #pragma omp for nowait
        for (int i = ibegin; i < iend; ++i) {
            local_result = reduce(local_result, func(static_cast<std::size_t>(i)));
        }
        
        #pragma omp critical
        {
            result = reduce(result, local_result);
        }
    }
    
    return result;
#else
    // Serial fallback
    T result = init;
    for (std::size_t i = begin; i < end; ++i) {
        result = reduce(result, func(i));
    }
    return result;
#endif
}

/** \brief Execute code in parallel region.
 *
 * \param func Function to execute in parallel
 * \param num_threads Optional thread count
 */
template<typename Func>
inline auto parallel_region(Func&& func, int num_threads = 0) -> void {
#ifdef _OPENMP
    if (num_threads > 0) {
        #pragma omp parallel num_threads(num_threads)
        {
            func();
        }
    } else {
        #pragma omp parallel
        {
            func();
        }
    }
#else
    // Serial fallback
    func();
#endif
}

/** \brief Execute code in critical section.
 *
 * Only one thread can execute this code at a time.
 *
 * \param func Function to execute exclusively
 * \param name Optional critical section name
 */
template<typename Func>
inline auto critical_section(Func&& func, [[maybe_unused]] const char* name = nullptr) -> void {
#ifdef _OPENMP
    #pragma omp critical
    {
        func();
    }
#else
    // Serial execution is already exclusive
    func();
#endif
}

/** \brief Memory barrier for all threads.
 *
 * All threads wait until everyone reaches this point.
 */
inline auto barrier() noexcept -> void {
#ifdef _OPENMP
    #pragma omp barrier
#else
    // No-op for serial execution
#endif
}

/** \brief Single execution in parallel region.
 *
 * Only one thread executes this code, others wait.
 *
 * \param func Function to execute once
 */
template<typename Func>
inline auto single_execution(Func&& func) -> void {
#ifdef _OPENMP
    #pragma omp single
    {
        func();
    }
#else
    func();
#endif
}

/** \brief Master thread execution.
 *
 * Only master thread executes, others continue.
 *
 * \param func Function for master thread
 */
template<typename Func>
inline auto master_execution(Func&& func) -> void {
#ifdef _OPENMP
    #pragma omp master
    {
        func();
    }
#else
    func();
#endif
}

/** \brief RAII guard for setting thread count.
 *
 * Restores previous thread count on destruction.
 */
class ThreadCountGuard {
public:
    explicit ThreadCountGuard(int num_threads)
        : prev_threads_(get_num_threads()) {
        set_num_threads(num_threads);
    }
    
    ~ThreadCountGuard() {
        set_num_threads(prev_threads_);
    }
    
    ThreadCountGuard(const ThreadCountGuard&) = delete;
    auto operator=(const ThreadCountGuard&) -> ThreadCountGuard& = delete;
    
private:
    int prev_threads_;
};

/** \brief Helper for chunked parallel processing.
 *
 * Divides work into chunks for better cache locality.
 */
template<typename Func>
inline auto parallel_chunks(std::size_t total_size, std::size_t chunk_size, Func&& func) -> void {
    const std::size_t num_chunks = (total_size + chunk_size - 1) / chunk_size;
    
    parallel_for(0, num_chunks, [&](std::size_t chunk_id) {
        const std::size_t begin = chunk_id * chunk_size;
        const std::size_t end = std::min(begin + chunk_size, total_size);
        func(begin, end);
    });
}

} // namespace vesper::platform