#ifdef __linux__

#include <catch2/catch_test_macros.hpp>
#include "vesper/io/io_uring.hpp"
#include <cstring>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

using namespace vesper::io;

TEST_CASE("IoUring basic operations", "[io_uring]") {
    // Skip test if io_uring not available
    auto uring_result = IoUring::create();
    if (!uring_result) {
        SKIP("io_uring not available on this system");
    }
    
    auto& uring = *uring_result.value();
    
    SECTION("Queue depth") {
        REQUIRE(uring.queue_depth() >= 16);
    }
    
    SECTION("Async read/write") {
        // Create temporary file
        char tmp_name[] = "/tmp/vesper_test_XXXXXX";
        int fd = mkstemp(tmp_name);
        REQUIRE(fd >= 0);
        
        // Write test data
        const std::string test_data = "Hello, io_uring!";
        std::vector<std::uint8_t> write_buffer(test_data.begin(), test_data.end());
        
        bool write_completed = false;
        auto write_result = uring.submit_write(fd, write_buffer, 0,
            [&](IoResult result) {
                write_completed = true;
                REQUIRE(result.res == static_cast<int>(write_buffer.size()));
            });
        REQUIRE(write_result.has_value());
        
        auto submit_result = uring.submit();
        REQUIRE(submit_result.has_value());
        REQUIRE(submit_result.value() == 1);
        
        auto wait_result = uring.wait_completions();
        REQUIRE(wait_result.has_value());
        REQUIRE(wait_result.value() >= 1);
        REQUIRE(write_completed);
        
        // Read test data back
        std::vector<std::uint8_t> read_buffer(write_buffer.size());
        bool read_completed = false;
        
        auto read_result = uring.submit_read(fd, read_buffer, 0,
            [&](IoResult result) {
                read_completed = true;
                REQUIRE(result.res == static_cast<int>(read_buffer.size()));
            });
        REQUIRE(read_result.has_value());
        
        submit_result = uring.submit();
        REQUIRE(submit_result.has_value());
        
        wait_result = uring.wait_completions();
        REQUIRE(wait_result.has_value());
        REQUIRE(read_completed);
        
        // Verify data
        REQUIRE(std::memcmp(read_buffer.data(), write_buffer.data(), 
                           read_buffer.size()) == 0);
        
        // Cleanup
        close(fd);
        unlink(tmp_name);
    }
    
    SECTION("Vectored I/O") {
        char tmp_name[] = "/tmp/vesper_test_XXXXXX";
        int fd = mkstemp(tmp_name);
        REQUIRE(fd >= 0);
        
        // Prepare iovecs
        std::string part1 = "Hello ";
        std::string part2 = "io_uring ";
        std::string part3 = "world!";
        
        std::vector<iovec> iovecs = {
            {const_cast<char*>(part1.data()), part1.size()},
            {const_cast<char*>(part2.data()), part2.size()},
            {const_cast<char*>(part3.data()), part3.size()}
        };
        
        bool completed = false;
        auto result = uring.submit_writev(fd, iovecs, 0,
            [&](IoResult res) {
                completed = true;
                std::size_t total_size = part1.size() + part2.size() + part3.size();
                REQUIRE(res.res == static_cast<int>(total_size));
            });
        REQUIRE(result.has_value());
        
        auto submit_result = uring.submit();
        REQUIRE(submit_result.has_value());
        
        auto wait_result = uring.wait_completions();
        REQUIRE(wait_result.has_value());
        REQUIRE(completed);
        
        // Cleanup
        close(fd);
        unlink(tmp_name);
    }
    
    SECTION("Fsync") {
        char tmp_name[] = "/tmp/vesper_test_XXXXXX";
        int fd = mkstemp(tmp_name);
        REQUIRE(fd >= 0);
        
        // Write some data
        const char* data = "test data";
        write(fd, data, strlen(data));
        
        bool completed = false;
        auto result = uring.submit_fsync(fd, false,
            [&](IoResult res) {
                completed = true;
                REQUIRE(res.res == 0);
            });
        REQUIRE(result.has_value());
        
        auto submit_result = uring.submit();
        REQUIRE(submit_result.has_value());
        
        auto wait_result = uring.wait_completions();
        REQUIRE(wait_result.has_value());
        REQUIRE(completed);
        
        // Cleanup
        close(fd);
        unlink(tmp_name);
    }
}

TEST_CASE("IoUringBatch operations", "[io_uring]") {
    auto uring_result = IoUring::create();
    if (!uring_result) {
        SKIP("io_uring not available on this system");
    }
    
    auto& uring = *uring_result.value();
    IoUringBatch batch(uring);
    
    SECTION("Batch multiple operations") {
        char tmp_name[] = "/tmp/vesper_batch_XXXXXX";
        int fd = mkstemp(tmp_name);
        REQUIRE(fd >= 0);
        
        // Add multiple write operations to batch
        std::vector<std::vector<std::uint8_t>> buffers;
        std::vector<bool> completed(3, false);
        
        for (int i = 0; i < 3; ++i) {
            std::string data = "Data " + std::to_string(i);
            buffers.emplace_back(data.begin(), data.end());
            
            auto result = batch.add_write(fd, buffers.back(), i * 100,
                [&completed, i](IoResult res) {
                    completed[i] = true;
                    REQUIRE(res.res > 0);
                });
            REQUIRE(result.has_value());
        }
        
        REQUIRE(batch.size() == 3);
        
        // Submit batch
        auto submit_result = batch.submit();
        REQUIRE(submit_result.has_value());
        REQUIRE(submit_result.value() == 3);
        
        // Wait for all completions
        auto wait_result = uring.wait_completions(3);
        REQUIRE(wait_result.has_value());
        REQUIRE(wait_result.value() == 3);
        
        for (bool c : completed) {
            REQUIRE(c);
        }
        
        // Cleanup
        close(fd);
        unlink(tmp_name);
    }
}

TEST_CASE("IoUringManager thread-local instances", "[io_uring]") {
    auto instance1 = IoUringManager::get_thread_local();
    if (!instance1) {
        SKIP("io_uring not available on this system");
    }
    
    auto instance2 = IoUringManager::get_thread_local();
    REQUIRE(instance2.has_value());
    
    // Should return same instance
    REQUIRE(instance1.value() == instance2.value());
    
    // Reset and get new instance
    IoUringManager::reset_thread_local();
    auto instance3 = IoUringManager::get_thread_local();
    REQUIRE(instance3.has_value());
    REQUIRE(instance3.value() != instance1.value());
}

#endif // __linux__