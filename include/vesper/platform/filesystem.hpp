#pragma once

/** \file filesystem.hpp
 *  \brief Cross-platform filesystem operations.
 *
 * Provides portable file I/O operations with proper error handling
 * and platform-specific optimizations.
 *
 * Key features:
 * - File sync operations (fsync/FlushFileBuffers)
 * - Memory-mapped file support
 * - Directory operations
 * - Path manipulation utilities
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <system_error>
#include <filesystem>
#include <optional>

#ifdef _WIN32
#define NOMINMAX  // Prevent Windows.h from defining min/max macros
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#else
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

namespace vesper::platform {

/** \brief File handle wrapper for RAII.
 *
 * Automatically closes file on destruction.
 */
class FileHandle {
public:
#ifdef _WIN32
    using native_handle_type = HANDLE;
#else
    using native_handle_type = int;
#endif

    FileHandle() noexcept = default;
    
    explicit FileHandle(native_handle_type handle) noexcept 
        : handle_(handle) {}
    
    ~FileHandle() {
        close();
    }
    
    // Move operations
    FileHandle(FileHandle&& other) noexcept
        : handle_(other.handle_) {
        other.handle_ = invalid_handle();
    }
    
    auto operator=(FileHandle&& other) noexcept -> FileHandle& {
        if (this != &other) {
            close();
            handle_ = other.handle_;
            other.handle_ = invalid_handle();
        }
        return *this;
    }
    
    // Delete copy operations
    FileHandle(const FileHandle&) = delete;
    auto operator=(const FileHandle&) -> FileHandle& = delete;
    
    [[nodiscard]] auto get() const noexcept -> native_handle_type {
        return handle_;
    }
    
    [[nodiscard]] auto is_valid() const noexcept -> bool {
        return handle_ != invalid_handle();
    }
    
    auto release() noexcept -> native_handle_type {
        auto h = handle_;
        handle_ = invalid_handle();
        return h;
    }
    
    auto close() noexcept -> void {
        if (is_valid()) {
#ifdef _WIN32
            ::CloseHandle(handle_);
#else
            ::close(handle_);
#endif
            handle_ = invalid_handle();
        }
    }
    
private:
    [[nodiscard]] static constexpr auto invalid_handle() noexcept -> native_handle_type {
#ifdef _WIN32
        return INVALID_HANDLE_VALUE;
#else
        return -1;
#endif
    }
    
    native_handle_type handle_ = invalid_handle();
};

/** \brief Open file with platform-specific flags.
 *
 * \param path File path
 * \param write_mode true for write access, false for read-only
 * \param create true to create file if it doesn't exist
 * \param direct_io true to bypass OS cache (O_DIRECT/FILE_FLAG_NO_BUFFERING)
 * \return File handle or error
 */
[[nodiscard]] inline auto open_file(const std::filesystem::path& path,
                                    bool write_mode = false,
                                    bool create = false,
                                    bool direct_io = false) 
    -> std::optional<FileHandle> {
    
#ifdef _WIN32
    DWORD access = write_mode ? (GENERIC_READ | GENERIC_WRITE) : GENERIC_READ;
    DWORD share = FILE_SHARE_READ;
    DWORD creation = create ? CREATE_ALWAYS : OPEN_EXISTING;
    DWORD flags = FILE_ATTRIBUTE_NORMAL;
    
    if (direct_io) {
        flags |= FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH;
    }
    
    HANDLE h = ::CreateFileW(
        path.c_str(),
        access,
        share,
        nullptr,
        creation,
        flags,
        nullptr
    );
    
    if (h == INVALID_HANDLE_VALUE) {
        return std::nullopt;
    }
    
    return FileHandle(h);
#else
    int flags = write_mode ? O_RDWR : O_RDONLY;
    if (create) {
        flags |= O_CREAT | O_TRUNC;
    }
    if (direct_io) {
#ifdef O_DIRECT
        flags |= O_DIRECT;
#endif
    }
    
    int fd = ::open(path.c_str(), flags, 0644);
    if (fd == -1) {
        return std::nullopt;
    }
    
    return FileHandle(fd);
#endif
}

/** \brief Sync file to disk.
 *
 * Ensures all data and metadata are written to disk.
 *
 * \param handle File handle
 * \return true on success
 */
inline auto sync_file(const FileHandle& handle) noexcept -> bool {
    if (!handle.is_valid()) {
        return false;
    }
    
#ifdef _WIN32
    return ::FlushFileBuffers(handle.get()) != 0;
#else
    return ::fsync(handle.get()) == 0;
#endif
}

/** \brief Sync only file data (not metadata).
 *
 * Faster than full sync but doesn't update metadata.
 *
 * \param handle File handle
 * \return true on success
 */
inline auto sync_file_data(const FileHandle& handle) noexcept -> bool {
    if (!handle.is_valid()) {
        return false;
    }
    
#ifdef _WIN32
    // Windows doesn't distinguish data-only sync
    return ::FlushFileBuffers(handle.get()) != 0;
#else
#ifdef __APPLE__
    // macOS uses F_FULLFSYNC for reliable sync
    return ::fcntl(handle.get(), F_FULLFSYNC) == 0;
#else
    // Linux has fdatasync for data-only sync
    return ::fdatasync(handle.get()) == 0;
#endif
#endif
}

/** \brief Get file size.
 *
 * \param handle File handle
 * \return File size in bytes or 0 on error
 */
[[nodiscard]] inline auto get_file_size(const FileHandle& handle) noexcept -> std::uint64_t {
    if (!handle.is_valid()) {
        return 0;
    }
    
#ifdef _WIN32
    LARGE_INTEGER size;
    if (::GetFileSizeEx(handle.get(), &size)) {
        return static_cast<std::uint64_t>(size.QuadPart);
    }
    return 0;
#else
    struct stat st;
    if (::fstat(handle.get(), &st) == 0) {
        return static_cast<std::uint64_t>(st.st_size);
    }
    return 0;
#endif
}

/** \brief Read from file.
 *
 * \param handle File handle
 * \param buffer Buffer to read into
 * \param size Number of bytes to read
 * \param offset File offset (nullopt for current position)
 * \return Number of bytes read or 0 on error
 */
inline auto read_file(const FileHandle& handle,
                     void* buffer,
                     std::size_t size,
                     std::optional<std::uint64_t> offset = std::nullopt) noexcept -> std::size_t {
    if (!handle.is_valid() || !buffer || size == 0) {
        return 0;
    }
    
#ifdef _WIN32
    OVERLAPPED overlapped = {};
    OVERLAPPED* ol_ptr = nullptr;
    
    if (offset) {
        overlapped.Offset = static_cast<DWORD>(*offset);
        overlapped.OffsetHigh = static_cast<DWORD>(*offset >> 32);
        ol_ptr = &overlapped;
    }
    
    DWORD bytes_read = 0;
    if (::ReadFile(handle.get(), buffer, static_cast<DWORD>(size), 
                   &bytes_read, ol_ptr)) {
        return bytes_read;
    }
    return 0;
#else
    if (offset) {
        return ::pread(handle.get(), buffer, size, static_cast<off_t>(*offset));
    } else {
        return ::read(handle.get(), buffer, size);
    }
#endif
}

/** \brief Write to file.
 *
 * \param handle File handle
 * \param buffer Buffer to write from
 * \param size Number of bytes to write
 * \param offset File offset (nullopt for current position)
 * \return Number of bytes written or 0 on error
 */
inline auto write_file(const FileHandle& handle,
                      const void* buffer,
                      std::size_t size,
                      std::optional<std::uint64_t> offset = std::nullopt) noexcept -> std::size_t {
    if (!handle.is_valid() || !buffer || size == 0) {
        return 0;
    }
    
#ifdef _WIN32
    OVERLAPPED overlapped = {};
    OVERLAPPED* ol_ptr = nullptr;
    
    if (offset) {
        overlapped.Offset = static_cast<DWORD>(*offset);
        overlapped.OffsetHigh = static_cast<DWORD>(*offset >> 32);
        ol_ptr = &overlapped;
    }
    
    DWORD bytes_written = 0;
    if (::WriteFile(handle.get(), buffer, static_cast<DWORD>(size),
                    &bytes_written, ol_ptr)) {
        return bytes_written;
    }
    return 0;
#else
    if (offset) {
        return ::pwrite(handle.get(), buffer, size, static_cast<off_t>(*offset));
    } else {
        return ::write(handle.get(), buffer, size);
    }
#endif
}

/** \brief Memory-mapped file region.
 *
 * RAII wrapper for memory-mapped files.
 */
class MappedFile {
public:
    MappedFile() noexcept = default;
    
    MappedFile(FileHandle&& file, std::size_t size, bool write_access = false)
        : file_(std::move(file))
        , size_(size)
        , write_access_(write_access) {
        
        if (!file_.is_valid() || size == 0) {
            return;
        }
        
#ifdef _WIN32
        DWORD protect = write_access ? PAGE_READWRITE : PAGE_READONLY;
        mapping_ = ::CreateFileMappingW(
            file_.get(),
            nullptr,
            protect,
            static_cast<DWORD>(size >> 32),
            static_cast<DWORD>(size),
            nullptr
        );
        
        if (!mapping_) {
            return;
        }
        
        DWORD access = write_access ? FILE_MAP_WRITE : FILE_MAP_READ;
        data_ = ::MapViewOfFile(
            mapping_,
            access,
            0, 0,
            size
        );
#else
        int prot = PROT_READ;
        if (write_access) {
            prot |= PROT_WRITE;
        }
        
        data_ = ::mmap(
            nullptr,
            size,
            prot,
            MAP_SHARED,
            file_.get(),
            0
        );
        
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
        }
#endif
    }
    
    ~MappedFile() {
        unmap();
    }
    
    // Move operations
    MappedFile(MappedFile&& other) noexcept
        : file_(std::move(other.file_))
        , data_(other.data_)
        , size_(other.size_)
        , write_access_(other.write_access_)
#ifdef _WIN32
        , mapping_(other.mapping_)
#endif
    {
        other.data_ = nullptr;
        other.size_ = 0;
#ifdef _WIN32
        other.mapping_ = nullptr;
#endif
    }
    
    auto operator=(MappedFile&& other) noexcept -> MappedFile& {
        if (this != &other) {
            unmap();
            file_ = std::move(other.file_);
            data_ = other.data_;
            size_ = other.size_;
            write_access_ = other.write_access_;
#ifdef _WIN32
            mapping_ = other.mapping_;
            other.mapping_ = nullptr;
#endif
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // Delete copy operations
    MappedFile(const MappedFile&) = delete;
    auto operator=(const MappedFile&) -> MappedFile& = delete;
    
    [[nodiscard]] auto data() noexcept -> void* { return data_; }
    [[nodiscard]] auto data() const noexcept -> const void* { return data_; }
    [[nodiscard]] auto size() const noexcept -> std::size_t { return size_; }
    [[nodiscard]] auto is_valid() const noexcept -> bool { return data_ != nullptr; }
    
    /** \brief Sync mapped region to disk. */
    auto sync() noexcept -> bool {
        if (!data_) return false;
        
#ifdef _WIN32
        return ::FlushViewOfFile(data_, size_) != 0;
#else
        return ::msync(data_, size_, MS_SYNC) == 0;
#endif
    }
    
private:
    auto unmap() noexcept -> void {
        if (data_) {
#ifdef _WIN32
            ::UnmapViewOfFile(data_);
            if (mapping_) {
                ::CloseHandle(mapping_);
            }
#else
            ::munmap(data_, size_);
#endif
            data_ = nullptr;
        }
    }
    
    FileHandle file_;
    void* data_ = nullptr;
    std::size_t size_ = 0;
    bool write_access_ = false;
#ifdef _WIN32
    HANDLE mapping_ = nullptr;
#endif
};

/** \brief Create memory-mapped file.
 *
 * \param path File path
 * \param size Size to map (0 for entire file)
 * \param write_access true for read-write, false for read-only
 * \return Mapped file or nullopt on error
 */
[[nodiscard]] inline auto map_file(const std::filesystem::path& path,
                                   std::size_t size = 0,
                                   bool write_access = false)
    -> std::optional<MappedFile> {
    
    auto file = open_file(path, write_access, false, false);
    if (!file) {
        return std::nullopt;
    }
    
    if (size == 0) {
        size = get_file_size(*file);
        if (size == 0) {
            return std::nullopt;
        }
    }
    
    MappedFile mapped(std::move(*file), size, write_access);
    if (!mapped.is_valid()) {
        return std::nullopt;
    }
    
    return mapped;
}

/** \brief Lock file for exclusive access.
 *
 * \param handle File handle
 * \param offset Starting offset
 * \param length Number of bytes to lock
 * \return true on success
 */
inline auto lock_file(const FileHandle& handle,
                      std::uint64_t offset = 0,
                      std::uint64_t length = 0) noexcept -> bool {
    if (!handle.is_valid()) {
        return false;
    }
    
#ifdef _WIN32
    OVERLAPPED overlapped = {};
    overlapped.Offset = static_cast<DWORD>(offset);
    overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
    
    return ::LockFileEx(
        handle.get(),
        LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
        0,
        static_cast<DWORD>(length),
        static_cast<DWORD>(length >> 32),
        &overlapped
    ) != 0;
#else
    struct flock fl = {};
    fl.l_type = F_WRLCK;
    fl.l_whence = SEEK_SET;
    fl.l_start = static_cast<off_t>(offset);
    fl.l_len = static_cast<off_t>(length);
    
    return ::fcntl(handle.get(), F_SETLK, &fl) != -1;
#endif
}

/** \brief Unlock file.
 *
 * \param handle File handle
 * \param offset Starting offset
 * \param length Number of bytes to unlock
 * \return true on success
 */
inline auto unlock_file(const FileHandle& handle,
                        std::uint64_t offset = 0,
                        std::uint64_t length = 0) noexcept -> bool {
    if (!handle.is_valid()) {
        return false;
    }
    
#ifdef _WIN32
    OVERLAPPED overlapped = {};
    overlapped.Offset = static_cast<DWORD>(offset);
    overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
    
    return ::UnlockFileEx(
        handle.get(),
        0,
        static_cast<DWORD>(length),
        static_cast<DWORD>(length >> 32),
        &overlapped
    ) != 0;
#else
    struct flock fl = {};
    fl.l_type = F_UNLCK;
    fl.l_whence = SEEK_SET;
    fl.l_start = static_cast<off_t>(offset);
    fl.l_len = static_cast<off_t>(length);
    
    return ::fcntl(handle.get(), F_SETLK, &fl) != -1;
#endif
}

} // namespace vesper::platform