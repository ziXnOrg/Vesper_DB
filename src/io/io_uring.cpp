#ifdef __linux__

#include "vesper/io/io_uring.hpp"
#include <sys/utsname.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <algorithm>

namespace vesper::io {

thread_local std::unique_ptr<IoUring> IoUringManager::thread_instance_;

IoUring::IoUring(const IoUringConfig& config) : config_(config) {}

IoUring::~IoUring() {
    cleanup();
}

IoUring::IoUring(IoUring&& other) noexcept
    : ring_(other.ring_),
      config_(other.config_),
      queue_depth_(other.queue_depth_),
      pending_submissions_(other.pending_submissions_),
      completions_(std::move(other.completions_)),
      next_user_data_(other.next_user_data_),
      initialized_(other.initialized_) {
    other.initialized_ = false;
}

IoUring& IoUring::operator=(IoUring&& other) noexcept {
    if (this != &other) {
        cleanup();
        ring_ = other.ring_;
        config_ = other.config_;
        queue_depth_ = other.queue_depth_;
        pending_submissions_ = other.pending_submissions_;
        completions_ = std::move(other.completions_);
        next_user_data_ = other.next_user_data_;
        initialized_ = other.initialized_;
        other.initialized_ = false;
    }
    return *this;
}

auto IoUring::create(const IoUringConfig& config)
    -> std::expected<std::unique_ptr<IoUring>, core::error> {
    auto ring = std::unique_ptr<IoUring>(new IoUring(config));
    
    if (auto result = ring->init(); !result) {
        return std::vesper_unexpected(result.error());
    }
    
    return ring;
}

auto IoUring::init() -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;
    
    // Check kernel version for io_uring support (5.1+)
    struct utsname uts;
    if (uname(&uts) == 0) {
        int major = 0, minor = 0;
        if (sscanf(uts.release, "%d.%d", &major, &minor) == 2) {
            if (major < 5 || (major == 5 && minor < 1)) {
                return std::vesper_unexpected(error{
                    error_code::unsupported,
                    "io_uring requires Linux kernel 5.1+",
                    "io_uring"
                });
            }
        }
    }
    
    // Setup io_uring parameters
    struct io_uring_params params;
    std::memset(&params, 0, sizeof(params));
    
    if (config_.use_sqpoll) {
        params.flags |= IORING_SETUP_SQPOLL;
        params.sq_thread_idle = config_.sqpoll_idle_ms;
    }
    
    if (config_.use_iopoll) {
        params.flags |= IORING_SETUP_IOPOLL;
    }
    
    // Initialize io_uring
    int ret = io_uring_queue_init_params(config_.queue_depth, &ring_, &params);
    if (ret < 0) {
        return std::vesper_unexpected(error{
            error_code::io_error,
            "Failed to initialize io_uring: " + std::string(std::strerror(-ret)),
            "io_uring"
        });
    }
    
    queue_depth_ = config_.queue_depth;
    initialized_ = true;
    
    return {};
}

auto IoUring::cleanup() noexcept -> void {
    if (initialized_) {
        // Process any remaining completions
        poll_completions();
        
        // Exit io_uring
        io_uring_queue_exit(&ring_);
        initialized_ = false;
    }
}

auto IoUring::get_sqe() -> io_uring_sqe* {
    return io_uring_get_sqe(&ring_);
}

auto IoUring::process_cqe(io_uring_cqe* cqe) -> void {
    IoResult result{cqe->res, cqe->user_data};
    
    auto it = completions_.find(cqe->user_data);
    if (it != completions_.end()) {
        if (it->second.handler) {
            it->second.handler(result);
        }
        completions_.erase(it);
    }
    
    io_uring_cqe_seen(&ring_, cqe);
}

auto IoUring::submit_read(int fd, std::span<std::uint8_t> buffer,
                          std::uint64_t offset,
                          CompletionHandler handler)
    -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;
    
    if (!initialized_) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "IoUring not initialized",
            "io_uring"
        });
    }
    
    auto* sqe = get_sqe();
    if (!sqe) {
        return std::vesper_unexpected(error{
            error_code::resource_exhausted,
            "No submission queue entries available",
            "io_uring"
        });
    }
    
    std::uint64_t user_data = next_user_data_++;
    io_uring_prep_read(sqe, fd, buffer.data(), buffer.size(), offset);
    io_uring_sqe_set_data64(sqe, user_data);
    
    completions_[user_data] = CompletionData{std::move(handler), {}};
    pending_submissions_++;
    
    return {};
}

auto IoUring::submit_write(int fd, std::span<const std::uint8_t> buffer,
                           std::uint64_t offset,
                           CompletionHandler handler)
    -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;
    
    if (!initialized_) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "IoUring not initialized",
            "io_uring"
        });
    }
    
    auto* sqe = get_sqe();
    if (!sqe) {
        return std::vesper_unexpected(error{
            error_code::resource_exhausted,
            "No submission queue entries available",
            "io_uring"
        });
    }
    
    std::uint64_t user_data = next_user_data_++;
    io_uring_prep_write(sqe, fd, buffer.data(), buffer.size(), offset);
    io_uring_sqe_set_data64(sqe, user_data);
    
    // Store buffer copy to ensure lifetime
    CompletionData data{std::move(handler), {}};
    data.buffer_storage.assign(buffer.begin(), buffer.end());
    completions_[user_data] = std::move(data);
    pending_submissions_++;
    
    return {};
}

auto IoUring::submit_readv(int fd, std::span<const iovec> iovecs,
                           std::uint64_t offset,
                           CompletionHandler handler)
    -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;
    
    if (!initialized_) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "IoUring not initialized",
            "io_uring"
        });
    }
    
    auto* sqe = get_sqe();
    if (!sqe) {
        return std::vesper_unexpected(error{
            error_code::resource_exhausted,
            "No submission queue entries available",
            "io_uring"
        });
    }
    
    std::uint64_t user_data = next_user_data_++;
    io_uring_prep_readv(sqe, fd, iovecs.data(), iovecs.size(), offset);
    io_uring_sqe_set_data64(sqe, user_data);
    
    completions_[user_data] = CompletionData{std::move(handler), {}};
    pending_submissions_++;
    
    return {};
}

auto IoUring::submit_writev(int fd, std::span<const iovec> iovecs,
                            std::uint64_t offset,
                            CompletionHandler handler)
    -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;
    
    if (!initialized_) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "IoUring not initialized",
            "io_uring"
        });
    }
    
    auto* sqe = get_sqe();
    if (!sqe) {
        return std::vesper_unexpected(error{
            error_code::resource_exhausted,
            "No submission queue entries available",
            "io_uring"
        });
    }
    
    std::uint64_t user_data = next_user_data_++;
    io_uring_prep_writev(sqe, fd, iovecs.data(), iovecs.size(), offset);
    io_uring_sqe_set_data64(sqe, user_data);
    
    completions_[user_data] = CompletionData{std::move(handler), {}};
    pending_submissions_++;
    
    return {};
}

auto IoUring::submit_fsync(int fd, bool datasync,
                           CompletionHandler handler)
    -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;
    
    if (!initialized_) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "IoUring not initialized",
            "io_uring"
        });
    }
    
    auto* sqe = get_sqe();
    if (!sqe) {
        return std::vesper_unexpected(error{
            error_code::resource_exhausted,
            "No submission queue entries available",
            "io_uring"
        });
    }
    
    std::uint64_t user_data = next_user_data_++;
    unsigned flags = datasync ? IORING_FSYNC_DATASYNC : 0;
    io_uring_prep_fsync(sqe, fd, flags);
    io_uring_sqe_set_data64(sqe, user_data);
    
    completions_[user_data] = CompletionData{std::move(handler), {}};
    pending_submissions_++;
    
    return {};
}

auto IoUring::register_files(std::span<const int> fds)
    -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;
    
    if (!initialized_) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "IoUring not initialized",
            "io_uring"
        });
    }
    
    int ret = io_uring_register_files(&ring_, fds.data(), fds.size());
    if (ret < 0) {
        return std::vesper_unexpected(error{
            error_code::io_error,
            "Failed to register files: " + std::string(std::strerror(-ret)),
            "io_uring"
        });
    }
    
    return {};
}

auto IoUring::unregister_files() -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;
    
    if (!initialized_) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "IoUring not initialized",
            "io_uring"
        });
    }
    
    int ret = io_uring_unregister_files(&ring_);
    if (ret < 0) {
        return std::vesper_unexpected(error{
            error_code::io_error,
            "Failed to unregister files: " + std::string(std::strerror(-ret)),
            "io_uring"
        });
    }
    
    return {};
}

auto IoUring::submit() -> std::expected<std::uint32_t, core::error> {
    using core::error;
    using core::error_code;
    
    if (!initialized_) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "IoUring not initialized",
            "io_uring"
        });
    }
    
    if (pending_submissions_ == 0) {
        return 0;
    }
    
    int ret = io_uring_submit(&ring_);
    if (ret < 0) {
        return std::vesper_unexpected(error{
            error_code::io_error,
            "Failed to submit operations: " + std::string(std::strerror(-ret)),
            "io_uring"
        });
    }
    
    std::uint32_t submitted = static_cast<std::uint32_t>(ret);
    pending_submissions_ -= submitted;
    
    return submitted;
}

auto IoUring::wait_completions(std::uint32_t min_complete,
                               std::int32_t timeout_ms)
    -> std::expected<std::uint32_t, core::error> {
    using core::error;
    using core::error_code;
    
    if (!initialized_) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "IoUring not initialized",
            "io_uring"
        });
    }
    
    // Setup timeout if specified
    struct __kernel_timespec ts;
    struct __kernel_timespec* pts = nullptr;
    if (timeout_ms >= 0) {
        ts.tv_sec = timeout_ms / 1000;
        ts.tv_nsec = (timeout_ms % 1000) * 1000000;
        pts = &ts;
    }
    
    // Wait for completions
    struct io_uring_cqe* cqe;
    int ret = io_uring_wait_cqe_timeout(&ring_, &cqe, pts);
    
    if (ret < 0 && ret != -ETIME && ret != -EINTR) {
        return std::vesper_unexpected(error{
            error_code::io_error,
            "Failed to wait for completions: " + std::string(std::strerror(-ret)),
            "io_uring"
        });
    }
    
    // Process all available completions
    std::uint32_t completed = 0;
    unsigned head;
    io_uring_for_each_cqe(&ring_, head, cqe) {
        process_cqe(cqe);
        completed++;
    }
    io_uring_cq_advance(&ring_, completed);
    
    return completed;
}

auto IoUring::poll_completions() -> std::uint32_t {
    if (!initialized_) {
        return 0;
    }
    
    struct io_uring_cqe* cqe;
    std::uint32_t completed = 0;
    unsigned head;
    
    io_uring_for_each_cqe(&ring_, head, cqe) {
        process_cqe(cqe);
        completed++;
    }
    
    if (completed > 0) {
        io_uring_cq_advance(&ring_, completed);
    }
    
    return completed;
}

// IoUringBatch implementation

auto IoUringBatch::add_read(int fd, std::span<std::uint8_t> buffer,
                            std::uint64_t offset,
                            IoUring::CompletionHandler handler)
    -> std::expected<void, core::error> {
    Operation op{Operation::READ, fd, {}, offset, std::move(handler)};
    op.buffer.assign(buffer.begin(), buffer.end());
    operations_.push_back(std::move(op));
    return {};
}

auto IoUringBatch::add_write(int fd, std::span<const std::uint8_t> buffer,
                             std::uint64_t offset,
                             IoUring::CompletionHandler handler)
    -> std::expected<void, core::error> {
    Operation op{Operation::WRITE, fd, {}, offset, std::move(handler)};
    op.buffer.assign(buffer.begin(), buffer.end());
    operations_.push_back(std::move(op));
    return {};
}

auto IoUringBatch::submit() -> std::expected<std::uint32_t, core::error> {
    for (auto& op : operations_) {
        std::expected<void, core::error> result;
        
        if (op.type == Operation::READ) {
            result = ring_.submit_read(op.fd, op.buffer, op.offset, std::move(op.handler));
        } else {
            result = ring_.submit_write(op.fd, op.buffer, op.offset, std::move(op.handler));
        }
        
        if (!result) {
            return std::vesper_unexpected(result.error());
        }
    }
    
    auto submitted = ring_.submit();
    if (submitted) {
        operations_.clear();
    }
    
    return submitted;
}

// IoUringManager implementation

auto IoUringManager::get_thread_local(const IoUringConfig& config)
    -> std::expected<IoUring*, core::error> {
    if (!thread_instance_) {
        auto ring = IoUring::create(config);
        if (!ring) {
            return std::vesper_unexpected(ring.error());
        }
        thread_instance_ = std::move(ring.value());
    }
    
    return thread_instance_.get();
}

auto IoUringManager::reset_thread_local() noexcept -> void {
    thread_instance_.reset();
}

} // namespace vesper::io

#endif // __linux__