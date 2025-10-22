/** \file async_io.hpp
 *  \brief Platform-agnostic async I/O abstraction for high-performance disk operations
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <expected>
#include <functional>
#include <future>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "vesper/error.hpp"
#include "vesper/platform/filesystem.hpp"

namespace vesper::io {

/**
 * \brief I/O operation types
 */
enum class IOOpType : std::uint8_t {
    READ = 0,
    WRITE = 1,
    SYNC = 2
};

/**
 * \brief I/O completion status
 */
enum class IOStatus : std::uint8_t {
    SUCCESS = 0,
    PARTIAL = 1,  // Partial completion
    FAILED = 2,   // I/O error
    CANCELLED = 3 // Request was cancelled
};

/**
 * \brief I/O request priority
 */
enum class IOPriority : std::uint8_t {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    CRITICAL = 3
};

/**
 * \brief Async I/O statistics
 */
struct AsyncIOStats {
    std::atomic<std::uint64_t> requests_submitted{0};
    std::atomic<std::uint64_t> requests_completed{0};
    std::atomic<std::uint64_t> requests_failed{0};
    std::atomic<std::uint64_t> requests_cancelled{0};
    std::atomic<std::uint64_t> bytes_read{0};
    std::atomic<std::uint64_t> bytes_written{0};
    std::atomic<std::uint64_t> total_latency_us{0};
    
    [[nodiscard]] double success_rate() const {
        auto total = requests_completed.load() + requests_failed.load();
        return total > 0 ? static_cast<double>(requests_completed.load()) / total : 0.0;
    }
    
    [[nodiscard]] double avg_latency_us() const {
        auto completed = requests_completed.load();
        return completed > 0 ? static_cast<double>(total_latency_us.load()) / completed : 0.0;
    }
    
    void reset() {
        requests_submitted = 0;
        requests_completed = 0;
        requests_failed = 0;
        requests_cancelled = 0;
        bytes_read = 0;
        bytes_written = 0;
        total_latency_us = 0;
    }
    
    // Copy constructor (needed because atomics are not copyable)
    AsyncIOStats(const AsyncIOStats& other) 
        : requests_submitted(other.requests_submitted.load())
        , requests_completed(other.requests_completed.load())
        , requests_failed(other.requests_failed.load())
        , requests_cancelled(other.requests_cancelled.load())
        , bytes_read(other.bytes_read.load())
        , bytes_written(other.bytes_written.load())
        , total_latency_us(other.total_latency_us.load()) {}
    
    // Default constructor
    AsyncIOStats() = default;
    
    // Copy assignment
    AsyncIOStats& operator=(const AsyncIOStats& other) {
        if (this != &other) {
            requests_submitted.store(other.requests_submitted.load());
            requests_completed.store(other.requests_completed.load());
            requests_failed.store(other.requests_failed.load());
            requests_cancelled.store(other.requests_cancelled.load());
            bytes_read.store(other.bytes_read.load());
            bytes_written.store(other.bytes_written.load());
            total_latency_us.store(other.total_latency_us.load());
        }
        return *this;
    }
};

/**
 * \brief I/O completion callback
 */
using IOCompletionCallback = std::function<void(IOStatus status, std::size_t bytes_transferred, std::error_code error)>;

/**
 * \brief Async I/O request
 */
class AsyncIORequest {
public:
    AsyncIORequest(IOOpType op, std::string filename, std::uint64_t offset, 
                  std::span<std::uint8_t> buffer, IOPriority priority = IOPriority::NORMAL);
    
    ~AsyncIORequest() = default;
    
    // Move-only type
    AsyncIORequest(AsyncIORequest&& other) noexcept = default;
    AsyncIORequest& operator=(AsyncIORequest&& other) noexcept = default;
    
    AsyncIORequest(const AsyncIORequest&) = delete;
    AsyncIORequest& operator=(const AsyncIORequest&) = delete;
    
    // Accessors
    [[nodiscard]] IOOpType operation() const { return operation_; }
    [[nodiscard]] const std::string& filename() const { return filename_; }
    [[nodiscard]] std::uint64_t offset() const { return offset_; }
    [[nodiscard]] std::span<std::uint8_t> buffer() const { return buffer_; }
    [[nodiscard]] IOPriority priority() const { return priority_; }
    [[nodiscard]] std::chrono::steady_clock::time_point created_at() const { return created_at_; }
    
    // Completion handling
    void set_completion_callback(IOCompletionCallback callback);
    [[nodiscard]] std::future<std::pair<IOStatus, std::size_t>> get_future();
    
    // Internal use
    void complete(IOStatus status, std::size_t bytes_transferred, std::error_code error = {});
    [[nodiscard]] std::uint64_t id() const { return id_; }
    
private:
    static std::atomic<std::uint64_t> next_id_;
    
    std::uint64_t id_;
    IOOpType operation_;
    std::string filename_;
    std::uint64_t offset_;
    std::span<std::uint8_t> buffer_;
    IOPriority priority_;
    std::chrono::steady_clock::time_point created_at_;
    
    IOCompletionCallback callback_;
    std::promise<std::pair<IOStatus, std::size_t>> promise_;
};

/**
 * \brief Async I/O queue configuration
 */
struct AsyncIOConfig {
    std::size_t max_queue_depth{128};
    std::size_t max_concurrent_ops{32};
    std::size_t io_thread_count{4};
    std::chrono::milliseconds request_timeout{5000};
    std::size_t alignment_bytes{4096};  // 4KB alignment for O_DIRECT
    bool use_direct_io{true};
    bool use_native_aio{true};  // Use io_uring/IOCP when available
};

/**
 * \brief Abstract async I/O queue interface
 */
class AsyncIOQueue {
public:
    virtual ~AsyncIOQueue() = default;
    
    /**
     * \brief Start the I/O queue
     */
    virtual void start() = 0;
    
    /**
     * \brief Stop the I/O queue
     */
    virtual void stop() = 0;
    
    /**
     * \brief Submit an I/O request
     */
    virtual auto submit(std::unique_ptr<AsyncIORequest> request) 
        -> std::expected<void, core::error> = 0;
    
    /**
     * \brief Submit multiple I/O requests
     */
    virtual auto submit_batch(std::vector<std::unique_ptr<AsyncIORequest>> requests)
        -> std::expected<void, core::error> = 0;
    
    /**
     * \brief Cancel a pending request
     */
    virtual auto cancel(std::uint64_t request_id) -> bool = 0;
    
    /**
     * \brief Cancel all pending requests
     */
    virtual void cancel_all() = 0;
    
    /**
     * \brief Get queue statistics
     */
    virtual AsyncIOStats get_stats() const = 0;
    
    /**
     * \brief Check if queue is running
     */
    virtual bool is_running() const = 0;
    
    /**
     * \brief Get current queue depth
     */
    virtual std::size_t queue_depth() const = 0;
    
protected:
    AsyncIOQueue() = default;
};

/**
 * \brief Factory for creating platform-specific async I/O queues
 */
class AsyncIOFactory {
public:
    /**
     * \brief Create the best available async I/O queue for this platform
     */
    static auto create_queue(AsyncIOConfig config = {}) 
        -> std::expected<std::unique_ptr<AsyncIOQueue>, core::error>;
    
    /**
     * \brief Check if native async I/O is available (io_uring/IOCP)
     */
    static bool is_native_async_available();
    
    /**
     * \brief Get platform-specific capabilities
     */
    static std::string get_platform_info();
    
private:
    AsyncIOFactory() = delete;
};

/**
 * \brief High-level async file reader with caching and prefetch
 */
class AsyncFileReader {
public:
    explicit AsyncFileReader(std::shared_ptr<AsyncIOQueue> queue, 
                           AsyncIOConfig config = {});
    ~AsyncFileReader();
    
    /**
     * \brief Open a file for async reading
     */
    auto open(const std::string& filename) -> std::expected<void, core::error>;
    
    /**
     * \brief Close the file
     */
    auto close() -> std::expected<void, core::error>;
    
    /**
     * \brief Read data asynchronously
     */
    auto read_async(std::uint64_t offset, std::size_t size)
        -> std::future<std::expected<std::vector<std::uint8_t>, core::error>>;
    
    /**
     * \brief Read multiple regions (scatter-gather)
     */
    auto read_batch_async(std::span<const std::pair<std::uint64_t, std::size_t>> regions)
        -> std::future<std::expected<std::vector<std::vector<std::uint8_t>>, core::error>>;
    
    /**
     * \brief Prefetch data (hint for future reads)
     */
    auto prefetch(std::uint64_t offset, std::size_t size) -> void;
    
    /**
     * \brief Get file size
     */
    [[nodiscard]] auto file_size() const -> std::expected<std::uint64_t, core::error>;
    
    /**
     * \brief Check if file is open
     */
    [[nodiscard]] bool is_open() const { return is_open_; }
    
    /**
     * \brief Get statistics
     */
    [[nodiscard]] AsyncIOStats get_stats() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    
    std::shared_ptr<AsyncIOQueue> queue_;
    AsyncIOConfig config_;
    std::string filename_;
    platform::FileHandle file_handle_;
    std::atomic<bool> is_open_{false};
    std::uint64_t file_size_{0};
    
    mutable AsyncIOStats stats_;
};

/**
 * \brief Utilities for I/O operations
 */
namespace utils {

/**
 * \brief Align size to given boundary
 */
constexpr std::size_t align_size(std::size_t size, std::size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

/**
 * \brief Check if pointer is aligned
 */
inline bool is_aligned(const void* ptr, std::size_t alignment) {
    return (reinterpret_cast<std::uintptr_t>(ptr) & (alignment - 1)) == 0;
}

/**
 * \brief Allocate aligned memory for I/O
 */
std::unique_ptr<std::uint8_t[]> allocate_aligned(std::size_t size, std::size_t alignment = 4096);

/**
 * \brief Create optimal I/O buffer
 */
std::vector<std::uint8_t> create_io_buffer(std::size_t size, std::size_t alignment = 4096);

/**
 * \brief Split large I/O into optimal chunks
 */
std::vector<std::pair<std::uint64_t, std::size_t>> split_io_request(
    std::uint64_t offset, std::size_t size, std::size_t max_chunk_size = 1024 * 1024);

} // namespace utils

} // namespace vesper::io