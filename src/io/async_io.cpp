/** \file async_io.cpp
 *  \brief Common async I/O implementation
 */

#include "vesper/io/async_io.hpp"
#include <cstring>
#include <memory>

#ifdef _WIN32
#include <malloc.h>
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#endif

namespace vesper::io {

// AsyncIORequest implementation

std::atomic<std::uint64_t> AsyncIORequest::next_id_{1};

AsyncIORequest::AsyncIORequest(IOOpType op, std::string filename, std::uint64_t offset,
                              std::span<std::uint8_t> buffer, IOPriority priority)
    : id_(next_id_.fetch_add(1))
    , operation_(op)
    , filename_(std::move(filename))
    , offset_(offset)
    , buffer_(buffer)
    , priority_(priority)
    , created_at_(std::chrono::steady_clock::now()) {
}

void AsyncIORequest::set_completion_callback(IOCompletionCallback callback) {
    callback_ = std::move(callback);
}

std::future<std::pair<IOStatus, std::size_t>> AsyncIORequest::get_future() {
    return promise_.get_future();
}

void AsyncIORequest::complete(IOStatus status, std::size_t bytes_transferred, std::error_code error) {
    if (callback_) {
        callback_(status, bytes_transferred, error);
    }
    
    promise_.set_value({status, bytes_transferred});
}

// AsyncFileReader implementation

class AsyncFileReader::Impl {
public:
    explicit Impl(std::shared_ptr<AsyncIOQueue> queue, AsyncIOConfig config)
        : queue_(std::move(queue)), config_(config) {}
    
    auto read_async(std::uint64_t offset, std::size_t size)
        -> std::future<std::expected<std::vector<std::uint8_t>, core::error>> {
        
        auto promise = std::make_shared<std::promise<std::expected<std::vector<std::uint8_t>, core::error>>>();
        auto future = promise->get_future();
        
        // Align size if needed
        std::size_t aligned_size = utils::align_size(size, config_.alignment_bytes);
        auto buffer = utils::create_io_buffer(aligned_size, config_.alignment_bytes);
        
        // Create I/O request
        auto request = std::make_unique<AsyncIORequest>(
            IOOpType::READ, filename_, offset, std::span<std::uint8_t>(buffer), IOPriority::NORMAL);
        
        // Set completion callback
        request->set_completion_callback([promise, buffer = std::move(buffer), size](
            IOStatus status, std::size_t bytes_transferred, std::error_code error) mutable {
            
            if (status == IOStatus::SUCCESS) {
                // Resize to actual requested size
                buffer.resize(std::min(size, bytes_transferred));
                promise->set_value(std::move(buffer));
            } else {
                promise->set_value(std::unexpected(core::error{
                    core::error_code::io_failed,
                    "Read operation failed",
                    "async_file_reader.read_async"
                }));
            }
        });
        
        // Submit request
        auto submit_result = queue_->submit(std::move(request));
        if (!submit_result.has_value()) {
            promise->set_value(std::unexpected(submit_result.error()));
        }
        
        return future;
    }
    
    auto read_batch_async(std::span<const std::pair<std::uint64_t, std::size_t>> regions)
        -> std::future<std::expected<std::vector<std::vector<std::uint8_t>>, core::error>> {
        
        auto promise = std::make_shared<std::promise<std::expected<std::vector<std::vector<std::uint8_t>>, core::error>>>();
        auto future = promise->get_future();
        
        if (regions.empty()) {
            promise->set_value(std::vector<std::vector<std::uint8_t>>{});
            return future;
        }
        
        // Shared state for batch completion
        struct BatchState {
            std::mutex mutex;
            std::vector<std::vector<std::uint8_t>> results;
            std::size_t completed{0};
            std::size_t total;
            bool failed{false};
            
            explicit BatchState(std::size_t n) : results(n), total(n) {}
        };
        
        auto batch_state = std::make_shared<BatchState>(regions.size());
        std::vector<std::unique_ptr<AsyncIORequest>> requests;
        
        // Create requests for each region
        for (std::size_t i = 0; i < regions.size(); ++i) {
            auto [offset, size] = regions[i];
            
            std::size_t aligned_size = utils::align_size(size, config_.alignment_bytes);
            auto buffer = utils::create_io_buffer(aligned_size, config_.alignment_bytes);
            
            auto request = std::make_unique<AsyncIORequest>(
                IOOpType::READ, filename_, offset, std::span<std::uint8_t>(buffer), IOPriority::NORMAL);
            
            // Capture the index and buffer in the callback
            request->set_completion_callback([promise, batch_state, i, buffer = std::move(buffer), size](
                IOStatus status, std::size_t bytes_transferred, std::error_code error) mutable {
                
                std::lock_guard lock(batch_state->mutex);
                
                if (status == IOStatus::SUCCESS && !batch_state->failed) {
                    buffer.resize(std::min(size, bytes_transferred));
                    batch_state->results[i] = std::move(buffer);
                } else if (!batch_state->failed) {
                    batch_state->failed = true;
                }
                
                batch_state->completed++;
                
                if (batch_state->completed == batch_state->total) {
                    if (batch_state->failed) {
                        promise->set_value(std::unexpected(core::error{
                            core::error_code::io_failed,
                            "Batch read operation failed",
                            "async_file_reader.read_batch_async"
                        }));
                    } else {
                        promise->set_value(std::move(batch_state->results));
                    }
                }
            });
            
            requests.push_back(std::move(request));
        }
        
        // Submit all requests
        auto submit_result = queue_->submit_batch(std::move(requests));
        if (!submit_result.has_value()) {
            promise->set_value(std::unexpected(submit_result.error()));
        }
        
        return future;
    }
    
    std::string filename_;
    std::shared_ptr<AsyncIOQueue> queue_;
    AsyncIOConfig config_;
};

AsyncFileReader::AsyncFileReader(std::shared_ptr<AsyncIOQueue> queue, AsyncIOConfig config)
    : impl_(std::make_unique<Impl>(std::move(queue), config))
    , queue_(impl_->queue_)
    , config_(config) {}

AsyncFileReader::~AsyncFileReader() {
    if (is_open_) {
        close();
    }
}

auto AsyncFileReader::open(const std::string& filename) -> std::expected<void, core::error> {
    if (is_open_) {
        return std::unexpected(core::error{
            core::error_code::invalid_state,
            "File already open",
            "async_file_reader.open"
        });
    }
    
    filename_ = filename;
    impl_->filename_ = filename;
    
#ifdef _WIN32
    HANDLE handle = CreateFileA(filename.c_str(),
                               GENERIC_READ,
                               FILE_SHARE_READ,
                               nullptr,
                               OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL,
                               nullptr);
    
    if (handle == INVALID_HANDLE_VALUE) {
        return std::unexpected(core::error{
            core::error_code::io_failed,
            "Failed to open file: " + filename,
            "async_file_reader.open"
        });
    }
    
    LARGE_INTEGER size;
    if (!GetFileSizeEx(handle, &size)) {
        CloseHandle(handle);
        return std::unexpected(core::error{
            core::error_code::io_failed,
            "Failed to get file size",
            "async_file_reader.open"
        });
    }
    
    file_size_ = static_cast<std::uint64_t>(size.QuadPart);
    file_handle_ = platform::FileHandle(handle);
    
#else
    int fd = ::open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        return std::unexpected(core::error{
            core::error_code::io_failed,
            "Failed to open file: " + filename,
            "async_file_reader.open"
        });
    }
    
    struct stat st;
    if (fstat(fd, &st) == -1) {
        ::close(fd);
        return std::unexpected(core::error{
            core::error_code::io_failed,
            "Failed to get file size",
            "async_file_reader.open"
        });
    }
    
    file_size_ = static_cast<std::uint64_t>(st.st_size);
    file_handle_ = platform::FileHandle(fd);
#endif
    
    is_open_ = true;
    return {};
}

auto AsyncFileReader::close() -> std::expected<void, core::error> {
    if (!is_open_) {
        return {};
    }
    
    file_handle_.close();
    is_open_ = false;
    file_size_ = 0;
    filename_.clear();
    
    return {};
}

auto AsyncFileReader::read_async(std::uint64_t offset, std::size_t size)
    -> std::future<std::expected<std::vector<std::uint8_t>, core::error>> {
    
    if (!is_open_) {
        std::promise<std::expected<std::vector<std::uint8_t>, core::error>> promise;
        promise.set_value(std::unexpected(core::error{
            core::error_code::invalid_state,
            "File not open",
            "async_file_reader.read_async"
        }));
        return promise.get_future();
    }
    
    return impl_->read_async(offset, size);
}

auto AsyncFileReader::read_batch_async(std::span<const std::pair<std::uint64_t, std::size_t>> regions)
    -> std::future<std::expected<std::vector<std::vector<std::uint8_t>>, core::error>> {
    
    if (!is_open_) {
        std::promise<std::expected<std::vector<std::vector<std::uint8_t>>, core::error>> promise;
        promise.set_value(std::unexpected(core::error{
            core::error_code::invalid_state,
            "File not open",
            "async_file_reader.read_batch_async"
        }));
        return promise.get_future();
    }
    
    return impl_->read_batch_async(regions);
}

auto AsyncFileReader::prefetch(std::uint64_t offset, std::size_t size) -> void {
    // Simple prefetch implementation - just issue read into dummy buffer
    if (is_open_) {
        auto future = read_async(offset, size);
        // Fire and forget - the cache will benefit from this
    }
}

auto AsyncFileReader::file_size() const -> std::expected<std::uint64_t, core::error> {
    if (!is_open_) {
        return std::unexpected(core::error{
            core::error_code::invalid_state,
            "File not open",
            "async_file_reader.file_size"
        });
    }
    
    return file_size_;
}

AsyncIOStats AsyncFileReader::get_stats() const {
    return stats_;
}

// Utilities implementation

namespace utils {

std::unique_ptr<std::uint8_t[]> allocate_aligned(std::size_t size, std::size_t alignment) {
#ifdef _WIN32
    void* ptr = _aligned_malloc(size, alignment);
    if (!ptr) {
        throw std::bad_alloc();
    }
    return std::unique_ptr<std::uint8_t[]>(static_cast<std::uint8_t*>(ptr));
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        throw std::bad_alloc();
    }
    return std::unique_ptr<std::uint8_t[]>(static_cast<std::uint8_t*>(ptr));
#endif
}

std::vector<std::uint8_t> create_io_buffer(std::size_t size, std::size_t alignment) {
    std::size_t aligned_size = align_size(size, alignment);
    std::vector<std::uint8_t> buffer;
    
    // Reserve extra space for alignment
    buffer.reserve(aligned_size + alignment);
    buffer.resize(aligned_size);
    
    // Check if the buffer is properly aligned (most std::vector implementations align to 16 bytes)
    if (!is_aligned(buffer.data(), alignment)) {
        // Fallback: allocate aligned memory and copy
        auto aligned_buffer = allocate_aligned(aligned_size, alignment);
        buffer.assign(aligned_buffer.get(), aligned_buffer.get() + aligned_size);
    }
    
    return buffer;
}

std::vector<std::pair<std::uint64_t, std::size_t>> split_io_request(
    std::uint64_t offset, std::size_t size, std::size_t max_chunk_size) {
    
    std::vector<std::pair<std::uint64_t, std::size_t>> chunks;
    
    while (size > 0) {
        std::size_t chunk_size = std::min(size, max_chunk_size);
        chunks.emplace_back(offset, chunk_size);
        offset += chunk_size;
        size -= chunk_size;
    }
    
    return chunks;
}

} // namespace utils

} // namespace vesper::io