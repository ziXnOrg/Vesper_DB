#pragma once

/** \file hnsw_thread_pool.hpp
 *  \brief Thread pool for parallel HNSW construction.
 *
 * Default: simple centralized FIFO task queue.
 * Optional: define VESPER_EXPERIMENTAL_WORK_STEALING=1 to enable per-worker
 * deques with basic work-stealing (guarded, experimental).
 */

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <optional>


#ifndef VESPER_EXPERIMENTAL_WORK_STEALING
#define VESPER_EXPERIMENTAL_WORK_STEALING 0
#endif

namespace vesper::index {

/** \brief Work-stealing thread pool for HNSW operations. */
class HnswThreadPool {
public:
    /** \brief Construct thread pool with specified number of workers.
     *  Default implementation is centralized FIFO; when
     *  VESPER_EXPERIMENTAL_WORK_STEALING=1, uses per-worker deques
     *  with per-queue locks and a global pending counter.
     *
     * \param num_threads Number of worker threads (0 = hardware concurrency / 2)
     */
    explicit HnswThreadPool(std::size_t num_threads = 0)
        : stop_(false) {
        if (num_threads == 0) {
            num_threads = std::max(1u, std::thread::hardware_concurrency() / 2);
        }

#if VESPER_EXPERIMENTAL_WORK_STEALING
        queues_.resize(num_threads);
        queue_mutexes_.reserve(num_threads);
        for (std::size_t i = 0; i < num_threads; ++i) queue_mutexes_.push_back(std::make_unique<std::mutex>());
        pending_.store(0, std::memory_order_relaxed);
#endif

        workers_.reserve(num_threads);
        for (std::size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this, i] { worker_loop(i); });
        }
    }

    ~HnswThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        cv_.notify_all();

        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    /** \brief Submit task to thread pool.
     *
     * \param task Function to execute
     * \return Future for task result
     */
    template<typename Func, typename... Args>
    auto submit(Func&& func, Args&&... args)
        -> std::future<decltype(func(args...))> {
        using return_type = decltype(func(args...));

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<Func>(func), std::forward<Args>(args)...)
        );

        auto future = task->get_future();

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                throw std::runtime_error("Thread pool is stopped");
            }
#if VESPER_EXPERIMENTAL_WORK_STEALING
            auto idx = rr_.fetch_add(1, std::memory_order_relaxed) % workers_.size();
            {
                std::lock_guard<std::mutex> ql(*queue_mutexes_[idx]);
                queues_[idx].emplace_back([task] { (*task)(); });
                pending_.fetch_add(1, std::memory_order_relaxed);
            }
#else
            tasks_.emplace_back([task] { (*task)(); });
#endif
        }

        cv_.notify_one();
        return future;
    }

    /** \brief Submit batch of tasks for parallel execution.
     *
     * \param tasks Vector of tasks to execute
     */
    template<typename Func>
    auto submit_batch(const std::vector<Func>& tasks) {
        std::vector<std::future<void>> futures;
        futures.reserve(tasks.size());

        for (const auto& task : tasks) {
            futures.push_back(submit(task));
        }

        return futures;
    }

    /** \brief Execute parallel for loop.
     *
     * \param start Start index
     * \param end End index
     * \param func Function to call with index
     * \param chunk_size Size of chunks to process (0 = auto)
     */
    template<typename Func>
    auto parallel_for(std::size_t start, std::size_t end,
                     Func&& func, std::size_t chunk_size = 0) -> void {
        if (chunk_size == 0) {
            chunk_size = std::max<std::size_t>(1, (end - start) / (workers_.size() * 4));
        }

        std::vector<std::future<void>> futures;

        for (std::size_t i = start; i < end; i += chunk_size) {
            auto chunk_end = std::min(i + chunk_size, end);
            futures.push_back(submit([i, chunk_end, func] {
                for (std::size_t j = i; j < chunk_end; ++j) {
                    func(j);
                }
            }));
        }

        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.wait();
        }
    }

    /** \brief Get number of worker threads. */
    [[nodiscard]] auto num_threads() const noexcept -> std::size_t {
        return workers_.size();
    }

    /** \brief Wait for all tasks to complete. */
    auto wait_all() -> void {
        std::unique_lock<std::mutex> lock(queue_mutex_);
#if VESPER_EXPERIMENTAL_WORK_STEALING
        cv_.wait(lock, [this] {
            return pending_.load(std::memory_order_relaxed) == 0;
        });
#else
        cv_.wait(lock, [this] { return tasks_.empty(); });
#endif
    }

private:
    auto worker_loop(std::size_t worker_id) -> void {
        while (true) {
            std::function<void()> task;

            {
#if VESPER_EXPERIMENTAL_WORK_STEALING
                std::unique_lock<std::mutex> lock(queue_mutex_);
                cv_.wait(lock, [this] {
                    return stop_.load(std::memory_order_relaxed) ||
                           pending_.load(std::memory_order_relaxed) > 0;
                });
                if (stop_.load(std::memory_order_relaxed) &&
                    pending_.load(std::memory_order_relaxed) == 0) {
                    return;
                }
                lock.unlock(); // release cv mutex while operating per-queue locks

                // Prefer own queue
                {
                    std::lock_guard<std::mutex> ql(*queue_mutexes_[worker_id]);
                    auto& myq = queues_[worker_id];
                    if (!myq.empty()) {
                        task = std::move(myq.front());
                        myq.pop_front();
                        auto rem = pending_.fetch_sub(1, std::memory_order_relaxed) - 1;
                        if (rem == 0) cv_.notify_all();
                    }
                }
                if (!task) {
                    // Attempt steal from others (from back for better locality)
                    const std::size_t n = queues_.size();
                    for (std::size_t k = 1; k < n; ++k) {
                        const std::size_t victim = (worker_id + k) % n;
                        std::lock_guard<std::mutex> ql(*queue_mutexes_[victim]);
                        auto& q = queues_[victim];
                        if (!q.empty()) {
                            task = std::move(q.back());
                            q.pop_back();
                            auto rem = pending_.fetch_sub(1, std::memory_order_relaxed) - 1;
                            if (rem == 0) cv_.notify_all();
                            break;
                        }
                    }
                }
#else
                std::unique_lock<std::mutex> lock(queue_mutex_);
                cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

                if (stop_ && tasks_.empty()) {
                    return;
                }

                if (!tasks_.empty()) {
                    task = std::move(tasks_.front());
                    tasks_.pop_front();
                }
#endif
            }

            if (task) {
                task();
            }
        }
    }

    std::vector<std::thread> workers_;
#if VESPER_EXPERIMENTAL_WORK_STEALING
    std::vector<std::deque<std::function<void()>>> queues_;
    std::vector<std::unique_ptr<std::mutex>> queue_mutexes_;
    std::atomic<std::size_t> rr_{0};
    std::atomic<std::size_t> pending_{0};
#else
    std::deque<std::function<void()>> tasks_;
#endif
    std::mutex queue_mutex_;            // only for CV and centralized queue path
    std::condition_variable cv_;
    std::atomic<bool> stop_;
};

/** \brief Thread-local scratch space for HNSW operations. */
struct HnswScratchSpace {
    std::vector<float> distances;
    std::vector<std::uint32_t> candidates;
    std::vector<bool> visited;

    auto resize(std::size_t max_candidates) -> void {
        distances.resize(max_candidates);
        candidates.resize(max_candidates);
        visited.resize(max_candidates, false);
    }

    auto clear() -> void {
        std::fill(visited.begin(), visited.end(), false);
    }
};

/** \brief Thread pool with scratch spaces for HNSW. */
class HnswParallelContext {
public:
    explicit HnswParallelContext(std::size_t num_threads = 0)
        : pool_(num_threads) {
        scratch_spaces_.resize(pool_.num_threads());
        for (auto& scratch : scratch_spaces_) {
            scratch.resize(1000);  // Default size
        }
    }

    auto pool() -> HnswThreadPool& { return pool_; }
    auto scratch(std::size_t thread_id) -> HnswScratchSpace& {
        return scratch_spaces_[thread_id];
    }

private:
    HnswThreadPool pool_;
    std::vector<HnswScratchSpace> scratch_spaces_;
};

} // namespace vesper::index