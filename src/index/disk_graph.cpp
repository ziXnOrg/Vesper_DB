/** \file disk_graph.cpp
 *  \brief Implementation of DiskANN-style graph index (Vamana algorithm).
 */

#include "vesper/index/disk_graph.hpp"
#include "vesper/index/product_quantizer.hpp"
#include "vesper/cache/lru_cache.hpp"
#include "vesper/io/async_io.hpp"
#include <expected>  // For std::expected and std::vesper_unexpected

// Windows fix for std::max
#ifdef _WIN32
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif
#endif

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <queue>
#include <random>
#include <thread>
#include <unordered_set>
#include <filesystem>
#include <numeric>

// Platform-specific includes for memory-mapped files
#ifndef _WIN32
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

// Now we have std::vesper_unexpected from the expected header

namespace vesper::index {

template<typename T>
using span = std::span<T>;

namespace {

/** \brief Distance comparison for min-heap (smaller distances first). */
struct DistanceComparator {
    bool operator()(const std::pair<float, std::uint32_t>& a,
                   const std::pair<float, std::uint32_t>& b) const {
        return a.first > b.first; // min-heap
    }
};

/** \brief Distance comparison for max-heap (larger distances first). */
struct ReverseDistanceComparator {
    bool operator()(const std::pair<float, std::uint32_t>& a,
                   const std::pair<float, std::uint32_t>& b) const {
        return a.first < b.first; // max-heap
    }
};

/** \brief Compute L2 distance between two vectors. */
inline float compute_l2_distance(const float* a, const float* b, std::size_t dim) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

/** \brief Compute L2 distance between two vectors (exposed for standalone functions). */
inline float compute_l2_distance_global(const float* a, const float* b, std::size_t dim) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

} // anonymous namespace

/** \brief Implementation class for DiskGraphIndex. */
class DiskGraphIndex::Impl {
public:
    explicit Impl(std::size_t dim, std::size_t cache_size_mb = 1024) 
        : dimension_(dim)
        , pq_(nullptr)
        , num_nodes_(0)
        , cache_size_mb_(cache_size_mb)
        , vectors_mmap_(nullptr)
        , graph_mmap_(nullptr)
        , vectors_file_size_(0)
        , graph_file_size_(0) {
        
        const std::size_t cache_bytes = cache_size_mb * 1024 * 1024;
        neighbor_cache_ = std::make_unique<cache::GraphNodeCache>(
            cache_bytes, 16, std::nullopt, nullptr
        );
        vector_cache_ = std::make_unique<cache::VectorCache>(
            cache_bytes, 16, std::nullopt, nullptr
        );
        
        // Initialize async I/O
        io::AsyncIOConfig io_config;
        io_config.io_thread_count = 4;
        io_config.max_queue_depth = 256;
        io_config.use_direct_io = true;
        
        auto queue_result = io::AsyncIOFactory::create_queue(io_config);
        if (queue_result.has_value()) {
            async_io_queue_ = std::move(queue_result.value());
            async_io_queue_->start();
        }
    }

    ~Impl() {
        close_files();
        close_mmap();
        if (async_io_queue_) {
            async_io_queue_->stop();
        }
    }
    
    void close_mmap() {
        // Unmap memory-mapped files
        if (vectors_mmap_) {
#ifdef _WIN32
            UnmapViewOfFile(vectors_mmap_);
            vectors_mmap_ = nullptr;
#else
            munmap(vectors_mmap_, vectors_file_size_);
            vectors_mmap_ = nullptr;
#endif
        }
        
        if (graph_mmap_) {
#ifdef _WIN32
            UnmapViewOfFile(graph_mmap_);
            graph_mmap_ = nullptr;
#else
            munmap(graph_mmap_, graph_file_size_);
            graph_mmap_ = nullptr;
#endif
        }
        
#ifdef _WIN32
        if (vectors_mapping_handle_) {
            CloseHandle(vectors_mapping_handle_);
            vectors_mapping_handle_ = nullptr;
        }
        if (graph_mapping_handle_) {
            CloseHandle(graph_mapping_handle_);
            graph_mapping_handle_ = nullptr;
        }
        if (vectors_file_handle_ && vectors_file_handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(vectors_file_handle_);
            vectors_file_handle_ = nullptr;
        }
        if (graph_file_handle_ && graph_file_handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(graph_file_handle_);
            graph_file_handle_ = nullptr;
        }
#else
        if (vectors_fd_ >= 0) {
            close(vectors_fd_);
            vectors_fd_ = -1;
        }
        if (graph_fd_ >= 0) {
            close(graph_fd_);
            graph_fd_ = -1;
        }
#endif
    }
    
    auto open_mmap_write(const std::string& vectors_path, const std::string& graph_path,
                        std::size_t vector_size, std::size_t graph_size)
        -> std::expected<void, core::error> {
        
        vectors_file_size_ = vector_size;
        graph_file_size_ = graph_size;
        
#ifdef _WIN32
        // Create/open files for writing
        vectors_file_handle_ = CreateFileA(
            vectors_path.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            0,
            nullptr,
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            nullptr
        );
        
        if (vectors_file_handle_ == INVALID_HANDLE_VALUE) {
            return std::unexpected(core::error{
                core::error_code::io_error,
                "Failed to create vectors file",
                "disk_graph"
            });
        }
        
        // Set file size
        LARGE_INTEGER size;
        size.QuadPart = vectors_file_size_;
        if (!SetFilePointerEx(vectors_file_handle_, size, nullptr, FILE_BEGIN) ||
            !SetEndOfFile(vectors_file_handle_)) {
            close_mmap();
            return std::unexpected(core::error{
                core::error_code::io_error,
                "Failed to set vectors file size",
                "disk_graph"
            });
        }
        
        // Create file mapping
        vectors_mapping_handle_ = CreateFileMappingA(
            vectors_file_handle_,
            nullptr,
            PAGE_READWRITE,
            static_cast<DWORD>(vectors_file_size_ >> 32),
            static_cast<DWORD>(vectors_file_size_ & 0xFFFFFFFF),
            nullptr
        );
        
        if (!vectors_mapping_handle_) {
            close_mmap();
            return std::unexpected(core::error{
                core::error_code::io_error,
                "Failed to create vectors file mapping",
                "disk_graph"
            });
        }
        
        // Map view of file
        vectors_mmap_ = MapViewOfFile(
            vectors_mapping_handle_,
            FILE_MAP_ALL_ACCESS,
            0, 0,
            vectors_file_size_
        );
        
        if (!vectors_mmap_) {
            close_mmap();
            return std::unexpected(core::error{
                core::error_code::io_error,
                "Failed to map vectors file",
                "disk_graph"
            });
        }
        
        // Same for graph file
        graph_file_handle_ = CreateFileA(
            graph_path.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            0,
            nullptr,
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            nullptr
        );
        
        if (graph_file_handle_ == INVALID_HANDLE_VALUE) {
            close_mmap();
            return std::unexpected(core::error{
                core::error_code::io_error,
                "Failed to create graph file",
                "disk_graph"
            });
        }
        
        size.QuadPart = graph_file_size_;
        if (!SetFilePointerEx(graph_file_handle_, size, nullptr, FILE_BEGIN) ||
            !SetEndOfFile(graph_file_handle_)) {
            close_mmap();
            return std::unexpected(core::error{
                core::error_code::io_error,
                "Failed to set graph file size",
                "disk_graph"
            });
        }
        
        graph_mapping_handle_ = CreateFileMappingA(
            graph_file_handle_,
            nullptr,
            PAGE_READWRITE,
            static_cast<DWORD>(graph_file_size_ >> 32),
            static_cast<DWORD>(graph_file_size_ & 0xFFFFFFFF),
            nullptr
        );
        
        if (!graph_mapping_handle_) {
            close_mmap();
            return std::unexpected(core::error{
                core::error_code::io_error,
                "Failed to create graph file mapping",
                "disk_graph"
            });
        }
        
        graph_mmap_ = MapViewOfFile(
            graph_mapping_handle_,
            FILE_MAP_ALL_ACCESS,
            0, 0,
            graph_file_size_
        );
        
        if (!graph_mmap_) {
            close_mmap();
            return std::unexpected(core::error{
                core::error_code::io_error,
                "Failed to map graph file",
                "disk_graph"
            });
        }
#else
        // POSIX implementation
        vectors_fd_ = open(vectors_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (vectors_fd_ < 0) {
            return std::unexpected(core::error{
                core::error_code::io_error,
                "Failed to create vectors file",
                "disk_graph"
            });
        }
        
        // Set file size
        if (ftruncate(vectors_fd_, vectors_file_size_) != 0) {
            close_mmap();
            return std::unexpected(core::error{
                core::error_code::io_error,
                "Failed to set vectors file size",
                "disk_graph"
            });
        }
        
        // Memory map the file
        vectors_mmap_ = mmap(nullptr, vectors_file_size_,
                           PROT_READ | PROT_WRITE, MAP_SHARED,
                           vectors_fd_, 0);
        
        if (vectors_mmap_ == MAP_FAILED) {
            vectors_mmap_ = nullptr;
            close_mmap();
            return std::unexpected(core::error{
                core::error_code::io_error,
                "Failed to map vectors file",
                "disk_graph"
            });
        }
        
        // Same for graph file
        graph_fd_ = open(graph_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (graph_fd_ < 0) {
            close_mmap();
            return std::unexpected(core::error{
                core::error_code::io_error,
                "Failed to create graph file",
                "disk_graph"
            });
        }
        
        if (ftruncate(graph_fd_, graph_file_size_) != 0) {
            close_mmap();
            return std::unexpected(core::error{
                core::error_code::io_error,
                "Failed to set graph file size",
                "disk_graph"
            });
        }
        
        graph_mmap_ = mmap(nullptr, graph_file_size_,
                         PROT_READ | PROT_WRITE, MAP_SHARED,
                         graph_fd_, 0);
        
        if (graph_mmap_ == MAP_FAILED) {
            graph_mmap_ = nullptr;
            close_mmap();
            return std::unexpected(core::error{
                core::error_code::io_error,
                "Failed to map graph file",
                "disk_graph"
            });
        }
#endif
        
        return {};
    }

    auto build(span<const float> vectors, const VamanaBuildParams& params)
        -> std::expected<VamanaBuildStats, core::error> {
        
        const auto start_time = std::chrono::steady_clock::now();
        
        // Validate parameters
        if (vectors.size() % dimension_ != 0) {
            return std::vesper_unexpected(core::error{
                core::error_code::config_invalid,
                "Vector count not divisible by dimension",
                "disk_graph.build"
            });
        }

        const std::size_t n = vectors.size() / dimension_;
        if (n == 0) {
            return std::vesper_unexpected(core::error{
                core::error_code::config_invalid,
                "Empty vector set",
                "disk_graph.build"
            });
        }

        num_nodes_ = static_cast<std::uint32_t>(n);
        build_params_ = params;
        
        // Store vectors
        vectors_.resize(vectors.size());
        std::memcpy(vectors_.data(), vectors.data(), vectors.size() * sizeof(float));
        
        // Initialize graph with empty adjacency lists
        graph_.resize(num_nodes_);
        
        // Initialize PQ if enabled
        if (params.use_pq) {
            init_product_quantizer(vectors.data(), n);
        }

        // Phase 1: Build initial random graph
        build_initial_graph(vectors.data(), n);
        
        // Phase 2: Iterative refinement with RobustPrune
        for (std::uint32_t iter = 0; iter < params.max_iters; ++iter) {
            refine_graph(vectors.data(), n, iter == params.max_iters - 1);
        }

        // Compute entry points (high-degree nodes)
        compute_entry_points();

        // Collect statistics
        VamanaBuildStats stats;
        stats.num_nodes = num_nodes_;
        stats.total_edges = 0;
        stats.max_degree = 0;
        
        for (const auto& neighbors : graph_) {
            stats.total_edges += neighbors.size();
            stats.max_degree = (std::max)(stats.max_degree, 
                                         static_cast<std::uint32_t>(neighbors.size()));
        }
        
        stats.avg_degree = static_cast<float>(stats.total_edges) / num_nodes_;
        
        const auto end_time = std::chrono::steady_clock::now();
        stats.build_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        
        build_stats_ = stats;
        return stats;
    }

    auto search(span<const float> query, const VamanaSearchParams& params) const
        -> std::expected<std::vector<std::pair<float, std::uint32_t>>, core::error> {
        
        if (query.size() != dimension_) {
            return std::vesper_unexpected(core::error{
                core::error_code::config_invalid,
                "Query dimension mismatch",
                "disk_graph.search"
            });
        }

        if (num_nodes_ == 0) {
            return std::vector<std::pair<float, std::uint32_t>>{};
        }

        // Initialize search with entry points
        std::vector<std::uint32_t> init_points = entry_points_;
        if (init_points.empty()) {
            // Use random starting points if no entry points
            std::mt19937 rng(42);
            std::uniform_int_distribution<std::uint32_t> dist(0, num_nodes_ - 1);
            for (std::uint32_t i = 0; i < params.init_list_size && i < num_nodes_; ++i) {
                init_points.push_back(dist(rng));
            }
        }

        // Perform beam search
        auto candidates = beam_search(query.data(), init_points, params);
        
        // Return top-k results
        std::vector<std::pair<float, std::uint32_t>> results;
        std::uint32_t count = 0;
        while (!candidates.empty() && count < params.k) {
            results.push_back(candidates.top());
            candidates.pop();
            count++;
        }
        
        // Reverse to get ascending order
        std::reverse(results.begin(), results.end());
        
        return results;
    }
    
    auto load_vector_async(std::uint32_t node_id) const 
        -> std::future<std::vector<float>> {
        
        auto promise = std::make_shared<std::promise<std::vector<float>>>();
        auto future = promise->get_future();
        
        // Check if we have vectors in memory
        if (!vectors_.empty()) {
            if (node_id < num_nodes_) {
                std::vector<float> vec(dimension_);
                std::memcpy(vec.data(), &vectors_[node_id * dimension_], 
                          dimension_ * sizeof(float));
                promise->set_value(std::move(vec));
            } else {
                promise->set_value(std::vector<float>{});
            }
            return future;
        }
        
        // Check if we have memory-mapped vectors
        if (vectors_mmap_ && node_id < num_nodes_) {
            std::vector<float> vec(dimension_);
            const float* src = static_cast<const float*>(vectors_mmap_) + node_id * dimension_;
            std::memcpy(vec.data(), src, dimension_ * sizeof(float));
            promise->set_value(std::move(vec));
            return future;
        }
        
        // Use async I/O to load from disk
        if (vector_file_path_.empty() || !async_io_queue_ || node_id >= num_nodes_) {
            promise->set_value(std::vector<float>{});
            return future;
        }
        
        std::size_t offset = node_id * dimension_ * sizeof(float);
        std::size_t read_size = dimension_ * sizeof(float);
        auto buffer = std::make_unique<std::uint8_t[]>(read_size);
        auto buffer_span = std::span<std::uint8_t>(buffer.get(), read_size);
        
        auto request = std::make_unique<io::AsyncIORequest>(
            io::IOOpType::READ,
            vector_file_path_,
            offset,
            buffer_span,
            io::IOPriority::HIGH
        );
        
        request->set_completion_callback(
            [promise, buffer = std::move(buffer), dim = dimension_](
                io::IOStatus status,
                std::size_t bytes_transferred,
                std::error_code error) {
                
                if (status == io::IOStatus::SUCCESS) {
                    std::vector<float> vec(dim);
                    std::memcpy(vec.data(), buffer.get(), bytes_transferred);
                    promise->set_value(std::move(vec));
                } else {
                    promise->set_value(std::vector<float>{});
                }
            }
        );
        
        async_io_queue_->submit(std::move(request));
        return future;
    }

    auto save(const std::string& path) const -> std::expected<void, core::error> {
        namespace fs = std::filesystem;
        
        // Create directory if it doesn't exist
        fs::create_directories(path);
        
        // Calculate sizes for disk files
        std::size_t vectors_size = num_nodes_ * dimension_ * sizeof(float);
        
        // Calculate graph size (header + offsets + adjacency data)
        std::size_t graph_data_size = 0;
        for (const auto& neighbors : graph_) {
            graph_data_size += neighbors.size() * sizeof(std::uint32_t);
        }
        std::size_t graph_size = sizeof(std::uint32_t) * 2 +  // num_nodes, dimension
                                num_nodes_ * sizeof(std::uint64_t) * 2 +  // offsets and counts
                                graph_data_size;
        
        // Open memory-mapped files for writing
        auto mmap_result = const_cast<Impl*>(this)->open_mmap_write(
            path + "/vectors.bin",
            path + "/graph.adj",
            vectors_size,
            graph_size
        );
        
        if (!mmap_result) {
            return mmap_result;
        }
        
        // Write vectors to memory-mapped file
        if (vectors_.size() > 0) {
            std::memcpy(vectors_mmap_, vectors_.data(), vectors_size);
        }
        
        // Write graph structure to memory-mapped file
        std::uint8_t* graph_ptr = static_cast<std::uint8_t*>(graph_mmap_);
        
        // Write header
        *reinterpret_cast<std::uint32_t*>(graph_ptr) = num_nodes_;
        graph_ptr += sizeof(std::uint32_t);
        *reinterpret_cast<std::uint32_t*>(graph_ptr) = static_cast<std::uint32_t>(dimension_);
        graph_ptr += sizeof(std::uint32_t);
        
        // Write offsets and adjacency lists
        std::uint64_t current_offset = sizeof(std::uint32_t) * 2 + 
                                      num_nodes_ * sizeof(std::uint64_t) * 2;
        
        // First pass: write offsets
        std::uint64_t* offset_ptr = reinterpret_cast<std::uint64_t*>(graph_ptr);
        std::uint32_t* count_ptr = reinterpret_cast<std::uint32_t*>(graph_ptr + num_nodes_ * sizeof(std::uint64_t));
        
        for (std::uint32_t i = 0; i < num_nodes_; ++i) {
            offset_ptr[i] = current_offset;
            count_ptr[i] = static_cast<std::uint32_t>(graph_[i].size());
            current_offset += graph_[i].size() * sizeof(std::uint32_t);
        }
        
        // Second pass: write adjacency data
        std::uint8_t* adj_data_ptr = graph_ptr + sizeof(std::uint64_t) * num_nodes_ + 
                                    sizeof(std::uint32_t) * num_nodes_;
        
        for (std::uint32_t i = 0; i < num_nodes_; ++i) {
            if (!graph_[i].empty()) {
                std::memcpy(adj_data_ptr, graph_[i].data(), 
                          graph_[i].size() * sizeof(std::uint32_t));
                adj_data_ptr += graph_[i].size() * sizeof(std::uint32_t);
            }
        }
        
        // Sync memory-mapped files to disk
#ifdef _WIN32
        FlushViewOfFile(vectors_mmap_, vectors_file_size_);
        FlushViewOfFile(graph_mmap_, graph_file_size_);
#else
        msync(vectors_mmap_, vectors_file_size_, MS_SYNC);
        msync(graph_mmap_, graph_file_size_, MS_SYNC);
#endif
        
        // Store file paths for future use
        const_cast<Impl*>(this)->graph_file_path_ = path + "/graph.adj";
        const_cast<Impl*>(this)->vector_file_path_ = path + "/vectors.bin";
        
        // Save metadata
        std::string meta_path = path + "/graph.meta";
        std::ofstream meta_file(meta_path, std::ios::binary);
        if (!meta_file) {
            return std::vesper_unexpected(core::error{
                core::error_code::io_failed,
                "Failed to create metadata file",
                "disk_graph.save"
            });
        }
        
        // Write metadata
        meta_file.write(reinterpret_cast<const char*>(&dimension_), sizeof(dimension_));
        meta_file.write(reinterpret_cast<const char*>(&num_nodes_), sizeof(num_nodes_));
        meta_file.write(reinterpret_cast<const char*>(&build_params_.degree), 
                       sizeof(build_params_.degree));
        meta_file.write(reinterpret_cast<const char*>(&build_params_.alpha), 
                       sizeof(build_params_.alpha));
        
        // Write flag for whether vectors are stored
        bool has_vectors = !vectors_.empty();
        meta_file.write(reinterpret_cast<const char*>(&has_vectors), sizeof(has_vectors));
        
        // Save adjacency lists
        std::string adj_path = path + "/graph.adj";
        std::ofstream adj_file(adj_path, std::ios::binary);
        if (!adj_file) {
            return std::vesper_unexpected(core::error{
                core::error_code::io_failed,
                "Failed to create adjacency file",
                "disk_graph.save"
            });
        }
        
        for (const auto& neighbors : graph_) {
            std::uint32_t degree = static_cast<std::uint32_t>(neighbors.size());
            adj_file.write(reinterpret_cast<const char*>(&degree), sizeof(degree));
            adj_file.write(reinterpret_cast<const char*>(neighbors.data()), 
                          degree * sizeof(std::uint32_t));
        }
        
        // Save vectors if available
        if (has_vectors) {
            std::string vec_path = path + "/vectors.bin";
            std::ofstream vec_file(vec_path, std::ios::binary);
            if (!vec_file) {
                return std::vesper_unexpected(core::error{
                    core::error_code::io_failed,
                    "Failed to create vectors file",
                    "disk_graph.save"
                });
            }
            
            // Write vectors contiguously
            vec_file.write(reinterpret_cast<const char*>(vectors_.data()),
                          vectors_.size() * sizeof(float));
        }
        
        // Save entry points
        std::string entry_path = path + "/graph.entry";
        std::ofstream entry_file(entry_path, std::ios::binary);
        if (!entry_file) {
            return std::vesper_unexpected(core::error{
                core::error_code::io_failed,
                "Failed to create entry points file",
                "disk_graph.save"
            });
        }
        
        std::uint32_t num_entry = static_cast<std::uint32_t>(entry_points_.size());
        entry_file.write(reinterpret_cast<const char*>(&num_entry), sizeof(num_entry));
        entry_file.write(reinterpret_cast<const char*>(entry_points_.data()),
                        num_entry * sizeof(std::uint32_t));
        
        return {};
    }

    auto load(const std::string& path) -> std::expected<void, core::error> {
        namespace fs = std::filesystem;
        
        // Store file paths for async I/O
        graph_file_path_ = path + "/graph.adj";
        vector_file_path_ = path + "/vectors.bin";
        
        // Load metadata
        std::string meta_path = path + "/graph.meta";
        std::ifstream meta_file(meta_path, std::ios::binary);
        if (!meta_file) {
            return std::vesper_unexpected(core::error{
                core::error_code::io_failed,
                "Failed to open metadata file",
                "disk_graph.load"
            });
        }
        
        std::size_t loaded_dim;
        meta_file.read(reinterpret_cast<char*>(&loaded_dim), sizeof(loaded_dim));
        if (loaded_dim != dimension_) {
            return std::vesper_unexpected(core::error{
                core::error_code::config_invalid,
                "Dimension mismatch",
                "disk_graph.load"
            });
        }
        
        meta_file.read(reinterpret_cast<char*>(&num_nodes_), sizeof(num_nodes_));
        meta_file.read(reinterpret_cast<char*>(&build_params_.degree), 
                      sizeof(build_params_.degree));
        meta_file.read(reinterpret_cast<char*>(&build_params_.alpha), 
                      sizeof(build_params_.alpha));
        
        // Check if vectors are stored
        bool has_vectors = false;
        if (meta_file.peek() != EOF) {
            meta_file.read(reinterpret_cast<char*>(&has_vectors), sizeof(has_vectors));
        }
        
        // Load adjacency lists
        std::string adj_path = path + "/graph.adj";
        std::ifstream adj_file(adj_path, std::ios::binary);
        if (!adj_file) {
            return std::vesper_unexpected(core::error{
                core::error_code::io_failed,
                "Failed to open adjacency file",
                "disk_graph.load"
            });
        }
        
        graph_.clear();
        graph_.resize(num_nodes_);
        
        for (std::uint32_t i = 0; i < num_nodes_; ++i) {
            std::uint32_t degree;
            adj_file.read(reinterpret_cast<char*>(&degree), sizeof(degree));
            graph_[i].resize(degree);
            adj_file.read(reinterpret_cast<char*>(graph_[i].data()), 
                         degree * sizeof(std::uint32_t));
        }
        
        // Load entry points
        std::string entry_path = path + "/graph.entry";
        std::ifstream entry_file(entry_path, std::ios::binary);
        if (!entry_file) {
            return std::vesper_unexpected(core::error{
                core::error_code::io_failed,
                "Failed to open entry points file",
                "disk_graph.load"
            });
        }
        
        std::uint32_t num_entry;
        entry_file.read(reinterpret_cast<char*>(&num_entry), sizeof(num_entry));
        entry_points_.resize(num_entry);
        entry_file.read(reinterpret_cast<char*>(entry_points_.data()),
                       num_entry * sizeof(std::uint32_t));
        
        // Load vectors if available
        if (has_vectors) {
            std::string vec_path = path + "/vectors.bin";
            std::ifstream vec_file(vec_path, std::ios::binary);
            if (vec_file) {
                vectors_.resize(num_nodes_ * dimension_);
                vec_file.read(reinterpret_cast<char*>(vectors_.data()),
                             vectors_.size() * sizeof(float));
            }
        }
        
        return {};
    }

    auto size() const -> std::uint32_t { return num_nodes_; }
    auto dimension() const -> std::size_t { return dimension_; }
    auto build_stats() const -> VamanaBuildStats { return build_stats_; }
    auto io_stats() const -> IOStats { 
        IOStats stats;
        stats.reads = io_stats_.reads.load();
        stats.read_bytes = io_stats_.read_bytes.load();
        stats.cache_hits = io_stats_.cache_hits.load();
        stats.cache_misses = io_stats_.cache_misses.load();
        stats.prefetch_hits = io_stats_.prefetch_hits.load();
        return stats;
    }

private:
    void init_product_quantizer(const float* vectors, std::size_t n) {
        // Initialize PQ with training vectors
        PqTrainParams pq_params;
        pq_params.m = build_params_.pq_m;
        pq_params.nbits = build_params_.pq_bits;
        
        pq_ = std::make_unique<ProductQuantizer>();
        
        // Train PQ on a subset of vectors
        std::size_t train_size = (std::min)(n, static_cast<std::size_t>(100000));
        auto result = pq_->train(vectors, train_size, dimension_, pq_params);
        if (!result) {
            // Handle error - for now just skip PQ
            pq_.reset();
            return;
        }
        
        // Encode all vectors
        pq_codes_.resize(n * pq_->code_size());
        for (std::size_t i = 0; i < n; ++i) {
            pq_->encode_one(vectors + i * dimension_, 
                           pq_codes_.data() + i * pq_->code_size());
        }
    }

    void build_initial_graph(const float* vectors, std::size_t n) {
        std::mt19937 rng(build_params_.seed);
        std::uniform_int_distribution<std::uint32_t> dist(0, num_nodes_ - 1);
        
        // Initialize each node with random neighbors
        for (std::uint32_t i = 0; i < num_nodes_; ++i) {
            std::unordered_set<std::uint32_t> neighbors;
            
            // Add random neighbors
            while (neighbors.size() < build_params_.degree && 
                   neighbors.size() < num_nodes_ - 1) {
                std::uint32_t neighbor = dist(rng);
                if (neighbor != i) {
                    neighbors.insert(neighbor);
                }
            }
            
            graph_[i].assign(neighbors.begin(), neighbors.end());
            std::sort(graph_[i].begin(), graph_[i].end());
        }
    }

    void refine_graph(const float* vectors, std::size_t n, bool final_iteration) {
        // For each node, search for better neighbors and apply RobustPrune
        for (std::uint32_t i = 0; i < num_nodes_; ++i) {
            const float* query = vectors + i * dimension_;
            
            // Search for candidates using current graph
            auto candidates = greedy_search_internal(
                query, {i}, build_params_.L, vectors);
            
            // Apply RobustPrune
            float alpha = final_iteration ? build_params_.alpha : 1.0f;
            auto pruned = robust_prune_internal(
                i, candidates, build_params_.degree, alpha, vectors);
            
            // Update adjacency list
            graph_[i] = pruned;
            
            // Add reverse edges (make graph undirected)
            for (std::uint32_t neighbor : pruned) {
                auto& rev_neighbors = graph_[neighbor];
                if (std::find(rev_neighbors.begin(), rev_neighbors.end(), i) 
                    == rev_neighbors.end()) {
                    rev_neighbors.push_back(i);
                    
                    // Prune reverse neighbor list if too large
                    if (rev_neighbors.size() > build_params_.degree) {
                        std::vector<std::pair<float, std::uint32_t>> rev_candidates;
                        for (std::uint32_t nb : rev_neighbors) {
                            float dist = compute_l2_distance(
                                vectors + neighbor * dimension_,
                                vectors + nb * dimension_,
                                dimension_);
                            rev_candidates.push_back({dist, nb});
                        }
                        
                        auto pruned_rev = robust_prune_internal(
                            neighbor, rev_candidates, build_params_.degree, 
                            alpha, vectors);
                        graph_[neighbor] = pruned_rev;
                    }
                }
            }
        }
    }

    void compute_entry_points() {
        // Select nodes with highest degree as entry points
        std::vector<std::pair<std::uint32_t, std::uint32_t>> degree_pairs;
        for (std::uint32_t i = 0; i < num_nodes_; ++i) {
            degree_pairs.push_back({static_cast<std::uint32_t>(graph_[i].size()), i});
        }
        
        std::sort(degree_pairs.rbegin(), degree_pairs.rend());
        
        entry_points_.clear();
        std::uint32_t num_entry = (std::min)(static_cast<std::uint32_t>(50), 
                                          num_nodes_ / 100);
        for (std::uint32_t i = 0; i < num_entry && i < degree_pairs.size(); ++i) {
            entry_points_.push_back(degree_pairs[i].second);
        }
    }

    auto greedy_search_internal(
        const float* query,
        const std::vector<std::uint32_t>& init_points,
        std::uint32_t L,
        const float* vectors) const
        -> std::vector<std::pair<float, std::uint32_t>> {
        
        std::priority_queue<std::pair<float, std::uint32_t>,
                           std::vector<std::pair<float, std::uint32_t>>,
                           DistanceComparator> candidates;
        
        std::priority_queue<std::pair<float, std::uint32_t>,
                           std::vector<std::pair<float, std::uint32_t>>,
                           ReverseDistanceComparator> w;
        
        std::unordered_set<std::uint32_t> visited;
        
        // Initialize with entry points
        for (std::uint32_t id : init_points) {
            float dist = compute_l2_distance(query, vectors + id * dimension_, dimension_);
            candidates.push({dist, id});
            w.push({dist, id});
            visited.insert(id);
        }
        
        // Beam search
        while (!candidates.empty()) {
            auto [current_dist, current_id] = candidates.top();
            candidates.pop();
            
            if (current_dist > w.top().first) {
                break; // All remaining candidates are worse
            }
            
            // Check neighbors (cache temporarily disabled)
            const auto& neighbors = graph_[current_id]; // get_neighbors_cached(current_id);
            for (std::uint32_t neighbor : neighbors) {
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    
                    float dist = compute_l2_distance(
                        query, vectors + neighbor * dimension_, dimension_);
                    
                    if (dist < w.top().first || w.size() < L) {
                        candidates.push({dist, neighbor});
                        w.push({dist, neighbor});
                        
                        if (w.size() > L) {
                            w.pop();
                        }
                    }
                }
            }
        }
        
        // Extract results
        std::vector<std::pair<float, std::uint32_t>> results;
        while (!w.empty()) {
            results.push_back(w.top());
            w.pop();
        }
        
        std::reverse(results.begin(), results.end());
        return results;
    }

    auto robust_prune_internal(
        std::uint32_t node_id,
        std::vector<std::pair<float, std::uint32_t>>& candidates,
        std::uint32_t degree,
        float alpha,
        const float* vectors) const
        -> std::vector<std::uint32_t> {
        
        // Sort candidates by distance
        std::sort(candidates.begin(), candidates.end());
        
        std::vector<std::uint32_t> pruned;
        std::vector<bool> selected(candidates.size(), false);
        
        // Greedy selection with RobustPrune
        for (std::size_t i = 0; i < candidates.size() && pruned.size() < degree; ++i) {
            if (selected[i]) continue;
            
            auto [dist_i, id_i] = candidates[i];
            if (id_i == node_id) continue; // Skip self
            
            bool should_select = true;
            
            // Check if this candidate is dominated by already selected neighbors
            for (std::size_t j = 0; j < i; ++j) {
                if (!selected[j]) continue;
                
                auto [dist_j, id_j] = candidates[j];
                
                // Compute distance between candidates
                float inter_dist = compute_l2_distance(
                    vectors + id_i * dimension_,
                    vectors + id_j * dimension_,
                    dimension_);
                
                // Check domination condition with alpha
                if (inter_dist < alpha * dist_i) {
                    should_select = false;
                    break;
                }
            }
            
            if (should_select) {
                pruned.push_back(id_i);
                selected[i] = true;
            }
        }
        
        return pruned;
    }

    auto beam_search(
        const float* query,
        const std::vector<std::uint32_t>& init_points,
        const VamanaSearchParams& params) const
        -> std::priority_queue<std::pair<float, std::uint32_t>,
                              std::vector<std::pair<float, std::uint32_t>>,
                              ReverseDistanceComparator> {
        
        // Priority queue for candidates (min-heap by distance)
        std::priority_queue<std::pair<float, std::uint32_t>,
                           std::vector<std::pair<float, std::uint32_t>>,
                           DistanceComparator> candidates;
        
        // Priority queue for results (max-heap to keep k-nearest)
        std::priority_queue<std::pair<float, std::uint32_t>,
                           std::vector<std::pair<float, std::uint32_t>>,
                           ReverseDistanceComparator> w;
        
        std::unordered_set<std::uint32_t> visited;
        
        // Initialize with entry points
        for (std::uint32_t id : init_points) {
            // Try to get vector from cache first
            auto cached_vec = vector_cache_->get(id);
            
            float dist;
            if (cached_vec.has_value()) {
                // Use cached vector
                dist = compute_l2_distance(query, cached_vec.value().data(), dimension_);
                io_stats_.cache_hits.fetch_add(1);
            } else {
                // Load vector from disk if available
                if (!vector_file_path_.empty()) {
                    // Synchronous load for now (async would be better for batch)
                    std::ifstream file(vector_file_path_, std::ios::binary);
                    if (file) {
                        file.seekg(id * dimension_ * sizeof(float));
                        std::vector<float> vec(dimension_);
                        file.read(reinterpret_cast<char*>(vec.data()), dimension_ * sizeof(float));
                        
                        if (file.good()) {
                            dist = compute_l2_distance(query, vec.data(), dimension_);
                            // Cache the loaded vector
                            vector_cache_->put(id, vec, vec.size() * sizeof(float));
                            io_stats_.reads.fetch_add(1);
                            io_stats_.read_bytes.fetch_add(dimension_ * sizeof(float));
                        } else {
                            // Skip this point if can't load
                            continue;
                        }
                    } else {
                        // No vector file, skip
                        continue;
                    }
                } else {
                    // No vector data available
                    continue;
                }
                io_stats_.cache_misses.fetch_add(1);
            }
            
            candidates.push({dist, id});
            w.push({dist, id});
            visited.insert(id);
        }
        
        // Beam search with L candidates
        while (!candidates.empty()) {
            auto [current_dist, current_id] = candidates.top();
            candidates.pop();
            
            if (current_dist > w.top().first) {
                break; // Prune search
            }
            
            // Get neighbors from cache or disk
            auto cached_neighbors = neighbor_cache_->get(current_id);
            std::vector<std::uint32_t> neighbors;
            
            if (cached_neighbors.has_value()) {
                neighbors = cached_neighbors.value();
                io_stats_.cache_hits.fetch_add(1);
            } else {
                // Use in-memory graph if available
                if (current_id < graph_.size()) {
                    neighbors = graph_[current_id];
                    neighbor_cache_->put(current_id, neighbors, neighbors.size() * sizeof(std::uint32_t));
                }
                io_stats_.cache_misses.fetch_add(1);
            }
            
            // Check each neighbor
            for (std::uint32_t neighbor : neighbors) {
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    
                    // Get neighbor vector
                    auto neighbor_vec = vector_cache_->get(neighbor);
                    float dist;
                    
                    if (neighbor_vec.has_value()) {
                        dist = compute_l2_distance(query, neighbor_vec.value().data(), dimension_);
                    } else if (!vector_file_path_.empty()) {
                        // Load from disk
                        std::ifstream file(vector_file_path_, std::ios::binary);
                        if (file) {
                            file.seekg(neighbor * dimension_ * sizeof(float));
                            std::vector<float> vec(dimension_);
                            file.read(reinterpret_cast<char*>(vec.data()), dimension_ * sizeof(float));
                            
                            if (file.good()) {
                                dist = compute_l2_distance(query, vec.data(), dimension_);
                                vector_cache_->put(neighbor, vec, vec.size() * sizeof(float));
                                io_stats_.reads.fetch_add(1);
                                io_stats_.read_bytes.fetch_add(dimension_ * sizeof(float));
                            } else {
                                continue;
                            }
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    }
                    
                    // Update candidates and results
                    if (dist < w.top().first || w.size() < params.L) {
                        candidates.push({dist, neighbor});
                        w.push({dist, neighbor});
                        
                        // Keep only L best candidates
                        if (w.size() > params.L) {
                            w.pop();
                        }
                    }
                }
            }
        }
        
        return w;
    }

    void close_files() {
        // Close any open file handles
        // Wait for pending I/O requests to complete
        if (async_io_queue_) {
            while (pending_io_requests_.load() > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    }
    
    // Async load neighbors for a node
    auto load_neighbors_async(std::uint32_t node_id) const 
        -> std::expected<void, core::error> {
        
        if (!async_io_queue_ || graph_file_path_.empty()) {
            return std::vesper_unexpected(core::error{
                core::error_code::precondition_failed,
                "Async I/O not initialized",
                "disk_graph.load_neighbors_async"
            });
        }
        
        // Check cache first
        auto cached = neighbor_cache_->get(node_id);
        if (cached.has_value()) {
            return {};
        }
        
        // Calculate file offset for this node's neighbors
        // Assuming fixed-size neighbor lists for simplicity
        std::size_t offset = node_id * (sizeof(std::uint32_t) * (build_params_.degree + 1));
        std::size_t size = sizeof(std::uint32_t) * (build_params_.degree + 1);
        
        // Create buffer for read operation
        auto buffer = std::make_shared<std::vector<std::uint8_t>>(size);
        
        // Create async read request
        auto request = std::make_unique<io::AsyncIORequest>(
            io::IOOpType::READ,
            graph_file_path_,
            offset,
            std::span<std::uint8_t>(buffer->data(), buffer->size())
        );
        
        // Set completion handler
        auto completion_handler = [this, node_id, buffer](io::IOStatus status, 
                                                          std::size_t bytes_transferred,
                                                          const std::error_code& error) mutable {
            if (status == io::IOStatus::SUCCESS && bytes_transferred > 0) {
                // Parse the neighbor data from buffer
                const std::uint8_t* data_ptr = buffer->data();
                
                // First 4 bytes: number of neighbors
                std::uint32_t num_neighbors = 0;
                if (bytes_transferred >= sizeof(std::uint32_t)) {
                    std::memcpy(&num_neighbors, data_ptr, sizeof(std::uint32_t));
                    data_ptr += sizeof(std::uint32_t);
                    
                    // Validate neighbor count
                    if (num_neighbors > 0 && num_neighbors <= config_.max_degree) {
                        std::vector<std::uint32_t> neighbors;
                        neighbors.reserve(num_neighbors);
                        
                        // Read neighbor IDs (4 bytes each)
                        std::size_t expected_size = sizeof(std::uint32_t) + num_neighbors * sizeof(std::uint32_t);
                        if (bytes_transferred >= expected_size) {
                            for (std::uint32_t i = 0; i < num_neighbors; ++i) {
                                std::uint32_t neighbor_id = 0;
                                std::memcpy(&neighbor_id, data_ptr + i * sizeof(std::uint32_t), sizeof(std::uint32_t));
                                neighbors.push_back(neighbor_id);
                            }
                            
                            // Update neighbor cache
                            const_cast<Impl*>(this)->neighbor_cache_->put(node_id, std::move(neighbors));
                        }
                    }
                }
                
                // Update statistics
                io_stats_.reads.fetch_add(1);
                io_stats_.read_bytes.fetch_add(bytes_transferred);
            } else if (status == io::IOStatus::FAILED) {
                // Log error but don't crash - use empty neighbor list
                const_cast<Impl*>(this)->neighbor_cache_->put(node_id, std::vector<std::uint32_t>{});
            }
            
            const_cast<Impl*>(this)->pending_io_requests_.fetch_sub(1);
        };
        
        const_cast<Impl*>(this)->pending_io_requests_.fetch_add(1);
        return async_io_queue_->submit(std::move(request));
    }
    
    // Async load vector data for a node
    auto load_vector_async(std::uint32_t node_id) const 
        -> std::expected<void, core::error> {
        
        if (!async_io_queue_ || vector_file_path_.empty()) {
            return std::vesper_unexpected(core::error{
                core::error_code::precondition_failed,
                "Async I/O not initialized",
                "disk_graph.load_vector_async"
            });
        }
        
        // Check cache first
        auto cached = vector_cache_->get(node_id);
        if (cached.has_value()) {
            return {};
        }
        
        // Calculate file offset for this vector
        std::size_t offset = node_id * dimension_ * sizeof(float);
        std::size_t size = dimension_ * sizeof(float);
        
        // Create buffer for read operation
        auto buffer = std::make_shared<std::vector<std::uint8_t>>(size);
        
        // Create async read request
        auto request = std::make_unique<io::AsyncIORequest>(
            io::IOOpType::READ,
            vector_file_path_,
            offset,
            std::span<std::uint8_t>(buffer->data(), buffer->size())
        );
        
        // Set completion handler
        auto completion_handler = [this, node_id, buffer, dimension = dimension_](io::IOStatus status,
                                                                                  std::size_t bytes_transferred,
                                                                                  const std::error_code& error) mutable {
            if (status == io::IOStatus::SUCCESS && bytes_transferred == dimension * sizeof(float)) {
                // Parse the vector data from buffer
                std::vector<float> vector(dimension);
                
                // Convert bytes to floats
                const float* float_ptr = reinterpret_cast<const float*>(buffer->data());
                std::memcpy(vector.data(), float_ptr, dimension * sizeof(float));
                
                // Update vector cache
                const_cast<Impl*>(this)->vector_cache_->put(node_id, std::move(vector));
                
                // Update statistics
                io_stats_.reads.fetch_add(1);
                io_stats_.read_bytes.fetch_add(bytes_transferred);
            } else if (status == io::IOStatus::FAILED) {
                // Log error but don't crash - use zero vector as fallback
                std::vector<float> zero_vector(dimension, 0.0f);
                const_cast<Impl*>(this)->vector_cache_->put(node_id, std::move(zero_vector));
            }
            
            const_cast<Impl*>(this)->pending_io_requests_.fetch_sub(1);
        };
        
        const_cast<Impl*>(this)->pending_io_requests_.fetch_add(1);
        return async_io_queue_->submit(std::move(request));
    }
    
    // Get neighbors (cache temporarily disabled)
    /*
    auto get_neighbors_cached(std::uint32_t node_id) const 
        -> const std::vector<std::uint32_t>& {
        // Cache disabled for compilation
        return graph_[node_id];
    }
    */
    
    // Get vector (cache temporarily disabled)
    /*
    auto get_vector_cached(std::uint32_t node_id, const float* vectors) const
        -> std::vector<float> {
        // Cache disabled for compilation
        std::vector<float> vec(dimension_);
        std::memcpy(vec.data(), vectors + node_id * dimension_, 
                   dimension_ * sizeof(float));
        return vec;
    }
    */

private:
    std::size_t dimension_;
    std::uint32_t num_nodes_;
    VamanaBuildParams build_params_;
    VamanaBuildStats build_stats_;
    mutable IOStats io_stats_;
    
    // Disk-based storage with memory-mapped files
#ifdef _WIN32
    void* vectors_file_handle_{nullptr};
    void* graph_file_handle_{nullptr};
    void* vectors_mapping_handle_{nullptr};
    void* graph_mapping_handle_{nullptr};
#else
    int vectors_fd_{-1};
    int graph_fd_{-1};
#endif
    void* vectors_mmap_;
    void* graph_mmap_;
    std::size_t vectors_file_size_;
    std::size_t graph_file_size_;
    
    // Graph adjacency list offsets (for variable-length neighbors)
    std::vector<std::pair<std::uint64_t, std::uint32_t>> graph_offsets_;  // offset, count
    
    // In-memory structures for building (converted to disk format after build)
    std::vector<std::vector<std::uint32_t>> graph_;
    std::vector<float> vectors_;
    
    // Entry points for search
    std::vector<std::uint32_t> entry_points_;
    
    // Product quantizer
    std::unique_ptr<ProductQuantizer> pq_;
    std::vector<std::uint8_t> pq_codes_;
    
    // LRU Cache for graph nodes and vectors
    mutable std::unique_ptr<cache::GraphNodeCache> neighbor_cache_;
    mutable std::unique_ptr<cache::VectorCache> vector_cache_;
    std::size_t cache_size_mb_;
    
    // Async I/O support
    std::unique_ptr<io::AsyncIOQueue> async_io_queue_;
    std::string graph_file_path_;
    std::string vector_file_path_;
    std::atomic<std::uint64_t> pending_io_requests_{0};
};

// Public interface implementation

DiskGraphIndex::DiskGraphIndex(std::size_t dim)
    : impl_(std::make_unique<Impl>(dim)) {
}

DiskGraphIndex::~DiskGraphIndex() = default;

auto DiskGraphIndex::build(span<const float> vectors, const VamanaBuildParams& params)
    -> std::expected<VamanaBuildStats, core::error> {
    return impl_->build(vectors, params);
}

auto DiskGraphIndex::search(span<const float> query, const VamanaSearchParams& params) const
    -> std::expected<std::vector<std::pair<float, std::uint32_t>>, core::error> {
    return impl_->search(query, params);
}

auto DiskGraphIndex::save(const std::string& path) const
    -> std::expected<void, core::error> {
    return impl_->save(path);
}

auto DiskGraphIndex::load(const std::string& path)
    -> std::expected<void, core::error> {
    return impl_->load(path);
}

auto DiskGraphIndex::size() const -> std::uint32_t {
    return impl_->size();
}

auto DiskGraphIndex::dimension() const -> std::size_t {
    return impl_->dimension();
}

auto DiskGraphIndex::build_stats() const -> VamanaBuildStats {
    return impl_->build_stats();
}

auto DiskGraphIndex::io_stats() const -> IOStats {
    return impl_->io_stats();
}

// Standalone algorithm implementations

auto robust_prune(
    std::uint32_t node_id,
    std::vector<std::pair<float, std::uint32_t>>& candidates,
    std::uint32_t degree,
    float alpha,
    const float* vectors,
    std::size_t dim)
    -> std::vector<std::uint32_t> {
    
    // Sort candidates by distance
    std::sort(candidates.begin(), candidates.end());
    
    std::vector<std::uint32_t> pruned;
    std::vector<bool> selected(candidates.size(), false);
    
    for (std::size_t i = 0; i < candidates.size() && pruned.size() < degree; ++i) {
        if (selected[i]) continue;
        
        auto [dist_i, id_i] = candidates[i];
        if (id_i == node_id) continue;
        
        bool should_select = true;
        
        for (std::size_t j = 0; j < i; ++j) {
            if (!selected[j]) continue;
            
            auto [dist_j, id_j] = candidates[j];
            
            float inter_dist = compute_l2_distance_global(
                vectors + id_i * dim,
                vectors + id_j * dim,
                dim);
            
            if (inter_dist < alpha * dist_i) {
                should_select = false;
                break;
            }
        }
        
        if (should_select) {
            pruned.push_back(id_i);
            selected[i] = true;
        }
    }
    
    return pruned;
}

auto greedy_search(
    span<const float> query,
    const std::vector<std::uint32_t>& entry_points,
    const std::vector<std::vector<std::uint32_t>>& graph,
    std::uint32_t L,
    const float* vectors,
    std::size_t dim)
    -> std::vector<std::pair<float, std::uint32_t>> {
    
    std::priority_queue<std::pair<float, std::uint32_t>,
                       std::vector<std::pair<float, std::uint32_t>>,
                       DistanceComparator> candidates;
    
    std::priority_queue<std::pair<float, std::uint32_t>,
                       std::vector<std::pair<float, std::uint32_t>>,
                       ReverseDistanceComparator> w;
    
    std::unordered_set<std::uint32_t> visited;
    
    for (std::uint32_t id : entry_points) {
        float dist = compute_l2_distance_global(query.data(), vectors + id * dim, dim);
        candidates.push({dist, id});
        w.push({dist, id});
        visited.insert(id);
    }
    
    while (!candidates.empty()) {
        auto [current_dist, current_id] = candidates.top();
        candidates.pop();
        
        if (current_dist > w.top().first) {
            break;
        }
        
        for (std::uint32_t neighbor : graph[current_id]) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                
                float dist = compute_l2_distance_global(query.data(), vectors + neighbor * dim, dim);
                
                if (dist < w.top().first || w.size() < L) {
                    candidates.push({dist, neighbor});
                    w.push({dist, neighbor});
                    
                    if (w.size() > L) {
                        w.pop();
                    }
                }
            }
        }
    }
    
    std::vector<std::pair<float, std::uint32_t>> results;
    while (!w.empty()) {
        results.push_back(w.top());
        w.pop();
    }
    
    std::reverse(results.begin(), results.end());
    return results;
}

} // namespace vesper::index