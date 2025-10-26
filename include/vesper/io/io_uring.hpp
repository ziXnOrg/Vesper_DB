#pragma once

#ifdef __linux__

/** \file io_uring.hpp
 *  \brief Linux io_uring support for high-performance async I/O.
 *
 * Provides zero-copy, kernel-bypass I/O with submission and completion queues.
 * Features:
 * - Batched submissions for reduced syscall overhead
 * - Vectored I/O support (readv/writev)
 * - Direct I/O with O_DIRECT for bypass of page cache
 * - Fixed file descriptors for reduced overhead
 * - IOPOLL mode for interrupt-less completions
 *
 * Thread-safety: Each io_uring instance is thread-local
 */

#include <liburing.h>
#include <vesper/core/error.hpp>
#include <vesper/span_polyfill.hpp>
#include <expected>
#include <functional>
#include <memory>
#include <vector>
#include <cstring>

namespace vesper::io {

/** \brief io_uring configuration parameters. */
struct IoUringConfig {
    std::uint32_t queue_depth = 256;      ///< Submission queue depth
    bool use_sqpoll = false;              ///< Use kernel SQ polling thread
    bool use_iopoll = false;              ///< Use busy-wait polling for completions
    bool use_fixed_files = false;         ///< Register files for reduced overhead
    std::uint32_t sqpoll_idle_ms = 1000;  ///< SQ poll thread idle time
};

/** \brief Async I/O operation result. */
struct IoResult {
    std::int32_t res;      ///< Result code (bytes transferred or -errno)
    std::uint64_t user_data; ///< User-provided context
};

/** \brief io_uring wrapper for async I/O operations. */
class IoUring {
public:
    using CompletionHandler = std::function<void(IoResult)>;

    /** \brief Initialize io_uring with configuration. */
    [[nodiscard]] static auto create(const IoUringConfig& config = {})
        -> std::expected<std::unique_ptr<IoUring>, core::error>;

    ~IoUring();

    // Non-copyable, movable
    IoUring(const IoUring&) = delete;
    IoUring& operator=(const IoUring&) = delete;
    IoUring(IoUring&&) noexcept;
    IoUring& operator=(IoUring&&) noexcept;

    /** \brief Submit async read operation.
     *
     * \param fd File descriptor
     * \param buffer Destination buffer
     * \param offset File offset
     * \param handler Completion callback
     * \return Error if submission fails
     */
    [[nodiscard]] auto submit_read(int fd, std::span<std::uint8_t> buffer,
                                   std::uint64_t offset,
                                   CompletionHandler handler)
        -> std::expected<void, core::error>;

    /** \brief Submit async write operation.
     *
     * \param fd File descriptor
     * \param buffer Source buffer
     * \param offset File offset
     * \param handler Completion callback
     * \return Error if submission fails
     */
    [[nodiscard]] auto submit_write(int fd, std::span<const std::uint8_t> buffer,
                                    std::uint64_t offset,
                                    CompletionHandler handler)
        -> std::expected<void, core::error>;

    /** \brief Submit vectored read operation.
     *
     * \param fd File descriptor
     * \param iovecs I/O vectors
     * \param offset File offset
     * \param handler Completion callback
     * \return Error if submission fails
     */
    [[nodiscard]] auto submit_readv(int fd, std::span<const iovec> iovecs,
                                    std::uint64_t offset,
                                    CompletionHandler handler)
        -> std::expected<void, core::error>;

    /** \brief Submit vectored write operation.
     *
     * \param fd File descriptor
     * \param iovecs I/O vectors
     * \param offset File offset
     * \param handler Completion callback
     * \return Error if submission fails
     */
    [[nodiscard]] auto submit_writev(int fd, std::span<const iovec> iovecs,
                                     std::uint64_t offset,
                                     CompletionHandler handler)
        -> std::expected<void, core::error>;

    /** \brief Submit fsync operation.
     *
     * \param fd File descriptor
     * \param datasync If true, only sync data (not metadata)
     * \param handler Completion callback
     * \return Error if submission fails
     */
    [[nodiscard]] auto submit_fsync(int fd, bool datasync,
                                    CompletionHandler handler)
        -> std::expected<void, core::error>;

    /** \brief Register file descriptors for fixed file operations.
     *
     * Fixed files reduce overhead by avoiding fd table lookups.
     *
     * \param fds File descriptors to register
     * \return Error if registration fails
     */
    [[nodiscard]] auto register_files(std::span<const int> fds)
        -> std::expected<void, core::error>;

    /** \brief Unregister all fixed files. */
    [[nodiscard]] auto unregister_files() -> std::expected<void, core::error>;

    /** \brief Submit all queued operations.
     *
     * \return Number of operations submitted, or error
     */
    [[nodiscard]] auto submit() -> std::expected<std::uint32_t, core::error>;

    /** \brief Wait for and process completions.
     *
     * \param min_complete Minimum completions to wait for
     * \param timeout_ms Timeout in milliseconds (-1 for infinite)
     * \return Number of completions processed, or error
     */
    [[nodiscard]] auto wait_completions(std::uint32_t min_complete = 1,
                                        std::int32_t timeout_ms = -1)
        -> std::expected<std::uint32_t, core::error>;

    /** \brief Process available completions without blocking.
     *
     * \return Number of completions processed
     */
    [[nodiscard]] auto poll_completions() -> std::uint32_t;

    /** \brief Get submission queue depth. */
    [[nodiscard]] auto queue_depth() const noexcept -> std::uint32_t {
        return queue_depth_;
    }

    /** \brief Get number of pending submissions. */
    [[nodiscard]] auto pending_submissions() const noexcept -> std::uint32_t {
        return pending_submissions_;
    }

private:
    explicit IoUring(const IoUringConfig& config);

    [[nodiscard]] auto init() -> std::expected<void, core::error>;
    auto cleanup() noexcept -> void;
    auto get_sqe() -> io_uring_sqe*;
    auto process_cqe(io_uring_cqe* cqe) -> void;

    struct CompletionData {
        CompletionHandler handler;
        std::vector<std::uint8_t> buffer_storage; // For buffer lifetime
    };

    io_uring ring_{};
    IoUringConfig config_;
    std::uint32_t queue_depth_ = 0;
    std::uint32_t pending_submissions_ = 0;
    std::unordered_map<std::uint64_t, CompletionData> completions_;
    std::uint64_t next_user_data_ = 1;
    bool initialized_ = false;
};

/** \brief Batch I/O operation for efficient submission. */
class IoUringBatch {
public:
    explicit IoUringBatch(IoUring& ring) : ring_(ring) {}

    /** \brief Add read operation to batch. */
    [[nodiscard]] auto add_read(int fd, std::span<std::uint8_t> buffer,
                                std::uint64_t offset,
                                IoUring::CompletionHandler handler)
        -> std::expected<void, core::error>;

    /** \brief Add write operation to batch. */
    [[nodiscard]] auto add_write(int fd, std::span<const std::uint8_t> buffer,
                                 std::uint64_t offset,
                                 IoUring::CompletionHandler handler)
        -> std::expected<void, core::error>;

    /** \brief Submit all batched operations. */
    [[nodiscard]] auto submit() -> std::expected<std::uint32_t, core::error>;

    /** \brief Get number of operations in batch. */
    [[nodiscard]] auto size() const noexcept -> std::size_t {
        return operations_.size();
    }

    /** \brief Clear batch without submitting. */
    auto clear() noexcept -> void {
        operations_.clear();
    }

private:
    struct Operation {
        enum Type { READ, WRITE };
        Type type;
        int fd;
        std::vector<std::uint8_t> buffer;
        std::uint64_t offset;
        IoUring::CompletionHandler handler;
    };

    IoUring& ring_;
    std::vector<Operation> operations_;
};

/** \brief Global io_uring instance manager. */
class IoUringManager {
public:
    /** \brief Get or create thread-local io_uring instance. */
    [[nodiscard]] static auto get_thread_local(const IoUringConfig& config = {})
        -> std::expected<IoUring*, core::error>;

    /** \brief Reset thread-local instance. */
    static auto reset_thread_local() noexcept -> void;

private:
    static thread_local std::unique_ptr<IoUring> thread_instance_;
};

} // namespace vesper::io

#endif // __linux__