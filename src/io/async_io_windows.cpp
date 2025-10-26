/** \file async_io_windows.cpp
 *  \brief Windows IOCP implementation of async I/O
 */

#ifdef _WIN32

#include "vesper/io/async_io.hpp"
#include <algorithm>
#include <expected>
#include <thread>
#include <unordered_map>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace vesper::io {

/**
 * \brief OVERLAPPED wrapper for async operations
 */
struct IOCPOverlapped : OVERLAPPED {
    AsyncIORequest* request;
    std::chrono::steady_clock::time_point start_time;
    
    IOCPOverlapped(AsyncIORequest* req) : request(req) {
        ZeroMemory(static_cast<OVERLAPPED*>(this), sizeof(OVERLAPPED));
        start_time = std::chrono::steady_clock::now();
    }
};

/**
 * \brief Windows IOCP-based async I/O queue
 */
class IOCPAsyncIOQueue : public AsyncIOQueue {
public:
    explicit IOCPAsyncIOQueue(AsyncIOConfig config) 
        : config_(config) {
        
        // Create I/O completion port
        iocp_handle_ = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 
                                            static_cast<DWORD>(config_.io_thread_count));
        if (iocp_handle_ == nullptr) {
            throw std::system_error(GetLastError(), std::system_category(), 
                                   "Failed to create I/O completion port");
        }
    }
    
    ~IOCPAsyncIOQueue() {
        stop();
        if (iocp_handle_ != nullptr) {
            CloseHandle(iocp_handle_);
        }
    }
    
    void start() override {
        if (running_.exchange(true)) {
            return;  // Already running
        }
        
        // Start I/O completion threads
        workers_.reserve(config_.io_thread_count);
        for (std::size_t i = 0; i < config_.io_thread_count; ++i) {
            workers_.emplace_back(&IOCPAsyncIOQueue::worker_thread, this);
        }
    }
    
    void stop() override {
        if (!running_.exchange(false)) {
            return;  // Already stopped
        }
        
        // Signal all worker threads to stop
        for (std::size_t i = 0; i < config_.io_thread_count; ++i) {
            PostQueuedCompletionStatus(iocp_handle_, 0, 0, nullptr);
        }
        
        // Wait for worker threads
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
        
        // Cancel all pending requests
        cancel_all();
        
        // Close all file handles
        std::lock_guard lock(files_mutex_);
        for (auto& [filename, handle] : file_handles_) {
            if (handle != INVALID_HANDLE_VALUE) {
                CloseHandle(handle);
            }
        }
        file_handles_.clear();
    }
    
    auto submit(std::unique_ptr<AsyncIORequest> request) 
        -> std::expected<void, core::error> override {
        
        if (!running_) {
            return std::expected<void, core::error>{std::unexpect, core::error{
                core::error_code::precondition_failed,
                "AsyncIOQueue is not running",
                "iocp_queue.submit"
            }};
        }
        
        if (current_queue_depth_.load() >= config_.max_queue_depth) {
            return std::expected<void, core::error>{std::unexpect, core::error{
                core::error_code::resource_exhausted,
                "Queue depth limit exceeded",
                "iocp_queue.submit"
            }};
        }
        
        auto result = submit_request(std::move(request));
        if (result.has_value()) {
            stats_.requests_submitted.fetch_add(1);
        }
        
        return result;
    }
    
    auto submit_batch(std::vector<std::unique_ptr<AsyncIORequest>> requests)
        -> std::expected<void, core::error> override {
        
        // Submit all requests - partial failure is acceptable
        std::size_t submitted = 0;
        for (auto& request : requests) {
            if (submit(std::move(request)).has_value()) {
                submitted++;
            }
        }
        
        if (submitted == 0) {
            return std::expected<void, core::error>{std::unexpect, core::error{
                core::error_code::internal,
                "Failed to submit any requests",
                "iocp_queue.submit_batch"
            }};
        }
        
        return {};
    }
    
    auto cancel(std::uint64_t request_id) -> bool override {
        std::lock_guard lock(pending_mutex_);
        
        auto it = pending_requests_.find(request_id);
        if (it == pending_requests_.end()) {
            return false;
        }
        
        // Cancel the I/O operation
        auto overlapped = it->second;
        HANDLE file_handle = get_file_handle(overlapped->request->filename());
        
        if (file_handle != INVALID_HANDLE_VALUE) {
            CancelIoEx(file_handle, overlapped);
        }
        
        return true;
    }
    
    void cancel_all() override {
        std::lock_guard lock(pending_mutex_);
        
        for (auto& [request_id, overlapped] : pending_requests_) {
            HANDLE file_handle = get_file_handle(overlapped->request->filename());
            if (file_handle != INVALID_HANDLE_VALUE) {
                CancelIoEx(file_handle, overlapped);
            }
        }
        
        pending_requests_.clear();
    }
    
    AsyncIOStats get_stats() const override {
        return stats_;
    }
    
    bool is_running() const override {
        return running_.load();
    }
    
    std::size_t queue_depth() const override {
        return current_queue_depth_.load();
    }
    
private:
    auto submit_request(std::unique_ptr<AsyncIORequest> request) 
        -> std::expected<void, core::error> {
        
        // Get or create file handle
        HANDLE file_handle = get_file_handle(request->filename());
        if (file_handle == INVALID_HANDLE_VALUE) {
            return std::expected<void, core::error>{std::unexpect, core::error{
                core::error_code::io_failed,
                "Failed to open file: " + request->filename(),
                "iocp_queue.submit_request"
            }};
        }
        
        // Create overlapped structure
        auto overlapped = std::make_unique<IOCPOverlapped>(request.get());
        overlapped->Offset = static_cast<DWORD>(request->offset() & 0xFFFFFFFF);
        overlapped->OffsetHigh = static_cast<DWORD>(request->offset() >> 32);
        
        // Track the request
        std::uint64_t request_id = request->id();
        {
            std::lock_guard lock(pending_mutex_);
            pending_requests_[request_id] = overlapped.get();
        }
        
        BOOL result = FALSE;
        DWORD bytes_transferred = 0;
        
        try {
            switch (request->operation()) {
                case IOOpType::READ: {
                    result = ReadFile(file_handle, 
                                    request->buffer().data(),
                                    static_cast<DWORD>(request->buffer().size()),
                                    &bytes_transferred,
                                    overlapped.get());
                    break;
                }
                case IOOpType::WRITE: {
                    result = WriteFile(file_handle,
                                     request->buffer().data(),
                                     static_cast<DWORD>(request->buffer().size()),
                                     &bytes_transferred,
                                     overlapped.get());
                    break;
                }
                case IOOpType::SYNC: {
                    result = FlushFileBuffers(file_handle);
                    break;
                }
            }
            
            DWORD error = GetLastError();
            
            if (!result && error != ERROR_IO_PENDING) {
                // Immediate failure
                {
                    std::lock_guard lock(pending_mutex_);
                    pending_requests_.erase(request_id);
                }
                
                return std::expected<void, core::error>{std::unexpect, core::error{
                    core::error_code::io_failed,
                    "I/O operation failed immediately",
                    "iocp_queue.submit_request"
                }};
            }
            
            // Success or pending - keep request alive
            current_queue_depth_.fetch_add(1);
            active_requests_.emplace(request_id, std::move(request));
            overlapped.release();  // Will be cleaned up in completion handler
            
            return {};
            
        } catch (...) {
            std::lock_guard lock(pending_mutex_);
            pending_requests_.erase(request_id);
            throw;
        }
    }
    
    HANDLE get_file_handle(const std::string& filename) {
        std::lock_guard lock(files_mutex_);
        
        auto it = file_handles_.find(filename);
        if (it != file_handles_.end()) {
            return it->second;
        }
        
        // Open file with async I/O flags
        DWORD flags = FILE_FLAG_OVERLAPPED;
        if (config_.use_direct_io) {
            flags |= FILE_FLAG_NO_BUFFERING;
        }
        
        HANDLE handle = CreateFileA(filename.c_str(),
                                   GENERIC_READ,
                                   FILE_SHARE_READ,
                                   nullptr,
                                   OPEN_EXISTING,
                                   flags,
                                   nullptr);
        
        if (handle == INVALID_HANDLE_VALUE) {
            return INVALID_HANDLE_VALUE;
        }
        
        // Associate with I/O completion port
        if (CreateIoCompletionPort(handle, iocp_handle_, 0, 0) == nullptr) {
            CloseHandle(handle);
            return INVALID_HANDLE_VALUE;
        }
        
        file_handles_[filename] = handle;
        return handle;
    }
    
    void worker_thread() {
        while (running_) {
            DWORD bytes_transferred = 0;
            ULONG_PTR completion_key = 0;
            IOCPOverlapped* overlapped = nullptr;
            
            BOOL result = GetQueuedCompletionStatus(
                iocp_handle_,
                &bytes_transferred,
                &completion_key,
                reinterpret_cast<OVERLAPPED**>(&overlapped),
                1000  // 1 second timeout
            );
            
            if (!result && overlapped == nullptr) {
                // Timeout or shutdown signal
                continue;
            }
            
            if (overlapped == nullptr) {
                // Shutdown signal
                break;
            }
            
            // Process completion
            handle_completion(overlapped, result, bytes_transferred);
        }
    }
    
    void handle_completion(IOCPOverlapped* overlapped, BOOL success, DWORD bytes_transferred) {
        std::unique_ptr<IOCPOverlapped> overlapped_guard(overlapped);
        
        auto request_id = overlapped->request->id();
        
        // Remove from tracking
        {
            std::lock_guard lock(pending_mutex_);
            pending_requests_.erase(request_id);
        }
        
        std::unique_ptr<AsyncIORequest> request;
        {
            std::lock_guard lock(active_mutex_);
            auto it = active_requests_.find(request_id);
            if (it != active_requests_.end()) {
                request = std::move(it->second);
                active_requests_.erase(it);
            }
        }
        
        current_queue_depth_.fetch_sub(1);
        
        if (!request) {
            return;  // Request was cancelled
        }
        
        // Calculate latency
        auto end_time = std::chrono::steady_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - overlapped->start_time);
        stats_.total_latency_us.fetch_add(latency.count());
        
        // Determine status
        IOStatus status;
        std::error_code error_code;
        
        if (success) {
            status = IOStatus::SUCCESS;
            stats_.requests_completed.fetch_add(1);
            
            if (request->operation() == IOOpType::READ) {
                stats_.bytes_read.fetch_add(bytes_transferred);
            } else if (request->operation() == IOOpType::WRITE) {
                stats_.bytes_written.fetch_add(bytes_transferred);
            }
            
        } else {
            DWORD error = GetLastError();
            if (error == ERROR_OPERATION_ABORTED) {
                status = IOStatus::CANCELLED;
                stats_.requests_cancelled.fetch_add(1);
            } else {
                status = IOStatus::FAILED;
                stats_.requests_failed.fetch_add(1);
                error_code = std::error_code(error, std::system_category());
            }
        }
        
        // Complete the request
        request->complete(status, bytes_transferred, error_code);
    }
    
    AsyncIOConfig config_;
    std::atomic<bool> running_{false};
    
    // Windows-specific handles
    HANDLE iocp_handle_{nullptr};
    
    // File handles
    std::mutex files_mutex_;
    std::unordered_map<std::string, HANDLE> file_handles_;
    
    // Worker threads
    std::vector<std::thread> workers_;
    
    // Request tracking
    std::mutex pending_mutex_;
    std::unordered_map<std::uint64_t, IOCPOverlapped*> pending_requests_;
    
    std::mutex active_mutex_;
    std::unordered_map<std::uint64_t, std::unique_ptr<AsyncIORequest>> active_requests_;
    
    std::atomic<std::size_t> current_queue_depth_{0};
    
    // Statistics
    mutable AsyncIOStats stats_;
};

// AsyncIOFactory Windows implementation

auto AsyncIOFactory::create_queue(AsyncIOConfig config) 
    -> std::expected<std::unique_ptr<AsyncIOQueue>, core::error> {
    try {
        return std::make_unique<IOCPAsyncIOQueue>(config);
    } catch (const std::exception& e) {
        return std::expected<std::unique_ptr<AsyncIOQueue>, core::error>{std::unexpect, core::error{
            core::error_code::internal,
            std::string("Failed to create IOCP queue: ") + e.what(),
            "async_io_factory.create_queue"
        }};
    }
}

bool AsyncIOFactory::is_native_async_available() {
    return true;  // IOCP is always available on Windows
}

std::string AsyncIOFactory::get_platform_info() {
    return "Windows IOCP";
}

} // namespace vesper::io

#endif // _WIN32