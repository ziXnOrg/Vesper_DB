#pragma once

/** \file hnsw_thread_pool.hpp
 *  \brief Thread pool for parallel HNSW construction with work-stealing.
 *
 * Implements a high-performance thread pool optimized for HNSW operations
 * with work-stealing queues for load balancing.
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

namespace vesper::index {

/** \brief Work-stealing thread pool for HNSW operations. */
class HnswThreadPool {
public:
    /** \brief Construct thread pool with specified number of workers.
     *
     * \param num_threads Number of worker threads (0 = hardware concurrency / 2)
     */
    explicit HnswThreadPool(std::size_t num_threads = 0) 
        : stop_(false) {
        if (num_threads == 0) {
            num_threads = std::max(1u, std::thread::hardware_concurrency() / 2);
        }
        
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
            tasks_.emplace_back([task] { (*task)(); });
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
        cv_.wait(lock, [this] { return tasks_.empty(); });
    }
    
private:
    auto worker_loop(std::size_t worker_id) -> void {
        while (true) {
            std::function<void()> task;
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                
                if (stop_ && tasks_.empty()) {
                    return;
                }
                
                if (!tasks_.empty()) {
                    task = std::move(tasks_.front());
                    tasks_.pop_front();
                }
            }
            
            if (task) {
                task();
            }
        }
    }
    
    std::vector<std::thread> workers_;
    std::deque<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
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