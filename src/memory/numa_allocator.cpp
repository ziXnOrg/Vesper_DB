#include "vesper/memory/numa_allocator.hpp"
#include <algorithm>
#include <cstring>
#include <mutex>

#ifdef __linux__
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#include <windows.h>
#include <memoryapi.h>
#endif

namespace vesper::memory {

std::vector<std::unique_ptr<NumaAllocator>> NumaAllocatorPool::allocators_;
std::mutex NumaAllocatorPool::mutex_;

// NumaTopology implementation

auto NumaTopology::detect() -> std::expected<std::unique_ptr<NumaTopology>, core::error> {
    auto topology = std::unique_ptr<NumaTopology>(new NumaTopology());
    
    if (auto result = topology->init(); !result) {
        return std::vesper_unexpected(result.error());
    }
    
    return topology;
}

auto NumaTopology::init() -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;

#ifdef __linux__
    // Check if NUMA is available
    if (numa_available() < 0) {
        // Single node system
        numa_available_ = false;
        
        NumaNode node;
        node.id = 0;
        node.memory_size = static_cast<std::size_t>(sysconf(_SC_PHYS_PAGES)) * 
                          static_cast<std::size_t>(sysconf(_SC_PAGE_SIZE));
        node.memory_free = static_cast<std::size_t>(sysconf(_SC_AVPHYS_PAGES)) * 
                          static_cast<std::size_t>(sysconf(_SC_PAGE_SIZE));
        
        // Add all CPUs to single node
        int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
        for (int i = 0; i < num_cpus; ++i) {
            node.cpus.push_back(i);
        }
        
        node.distances = {10}; // Local distance
        nodes_.push_back(std::move(node));
        total_memory_ = nodes_[0].memory_size;
        
        return {};
    }
    
    numa_available_ = true;
    int max_node = numa_max_node();
    
    // Discover all nodes
    for (int n = 0; n <= max_node; ++n) {
        // Check if node exists
        if (numa_node_size(n, nullptr) < 0) {
            continue;
        }
        
        NumaNode node;
        node.id = n;
        
        // Get memory info
        long long size;
        long long free;
        numa_node_size64(n, &size);
        node.memory_size = static_cast<std::size_t>(size);
        
        // Estimate free memory (not directly available)
        node.memory_free = node.memory_size / 2; // Conservative estimate
        
        // Get CPUs on this node
        struct bitmask* cpus = numa_allocate_cpumask();
        if (numa_node_to_cpus(n, cpus) == 0) {
            for (unsigned int i = 0; i < cpus->size; ++i) {
                if (numa_bitmask_isbitset(cpus, i)) {
                    node.cpus.push_back(i);
                }
            }
        }
        numa_free_cpumask(cpus);
        
        // Get distances to other nodes
        for (int m = 0; m <= max_node; ++m) {
            int dist = numa_distance(n, m);
            node.distances.push_back(dist > 0 ? dist : 20);
        }
        
        nodes_.push_back(std::move(node));
        total_memory_ += node.memory_size;
    }
    
    if (nodes_.empty()) {
        return std::vesper_unexpected(core::error{
            core::error_code::internal,
            "No NUMA nodes detected",
            "numa"
        });
    }
    
#elif defined(_WIN32)
    // Windows NUMA support
    numa_available_ = false;
    
    ULONG highest_node = 0;
    if (!GetNumaHighestNodeNumber(&highest_node)) {
        // Single node system
        NumaNode node;
        node.id = 0;
        
        MEMORYSTATUSEX mem_info;
        mem_info.dwLength = sizeof(mem_info);
        GlobalMemoryStatusEx(&mem_info);
        
        node.memory_size = mem_info.ullTotalPhys;
        node.memory_free = mem_info.ullAvailPhys;
        
        SYSTEM_INFO sys_info;
        GetSystemInfo(&sys_info);
        for (DWORD i = 0; i < sys_info.dwNumberOfProcessors; ++i) {
            node.cpus.push_back(i);
        }
        
        node.distances = {10};
        nodes_.push_back(std::move(node));
        total_memory_ = node.memory_size;
        
        return {};
    }
    
    numa_available_ = true;
    
    for (ULONG n = 0; n <= highest_node; ++n) {
        NumaNode node;
        node.id = n;
        
        // Get processor mask for node
        GROUP_AFFINITY affinity;
        if (GetNumaNodeProcessorMaskEx(static_cast<USHORT>(n), &affinity)) {
            for (int i = 0; i < 64; ++i) {
                if (affinity.Mask & (1ULL << i)) {
                    node.cpus.push_back(i);
                }
            }
        }
        
        // Windows doesn't provide per-node memory info easily
        // Use total system memory divided by number of nodes as estimate
        MEMORYSTATUSEX mem_info;
        mem_info.dwLength = sizeof(mem_info);
        GlobalMemoryStatusEx(&mem_info);
        
        node.memory_size = mem_info.ullTotalPhys / (highest_node + 1);
        node.memory_free = mem_info.ullAvailPhys / (highest_node + 1);
        
        // Simple distance matrix (10 for local, 20 for remote)
        for (ULONG m = 0; m <= highest_node; ++m) {
            node.distances.push_back(n == m ? 10 : 20);
        }
        
        nodes_.push_back(std::move(node));
        total_memory_ += node.memory_size;
    }
    
#else
    // No NUMA support on this platform
    numa_available_ = false;
    
    NumaNode node;
    node.id = 0;
    node.memory_size = 1024ULL * 1024 * 1024 * 16; // 16GB default
    node.memory_free = node.memory_size / 2;
    node.cpus = {0};
    node.distances = {10};
    
    nodes_.push_back(std::move(node));
    total_memory_ = node.memory_size;
#endif
    
    return {};
}

auto NumaTopology::current_node() const noexcept -> std::uint32_t {
#ifdef __linux__
    if (numa_available_) {
        int node = numa_node_of_cpu(sched_getcpu());
        if (node >= 0 && node < static_cast<int>(nodes_.size())) {
            return static_cast<std::uint32_t>(node);
        }
    }
#elif defined(_WIN32)
    if (numa_available_) {
        PROCESSOR_NUMBER proc_num;
        GetCurrentProcessorNumberEx(&proc_num);
        
        USHORT node_num;
        if (GetNumaProcessorNodeEx(&proc_num, &node_num)) {
            return static_cast<std::uint32_t>(node_num);
        }
    }
#endif
    return 0;
}

auto NumaTopology::distance(std::uint32_t from, std::uint32_t to) const noexcept
    -> std::uint32_t {
    if (from >= nodes_.size() || to >= nodes_.size()) {
        return 100; // Max distance for invalid nodes
    }
    
    const auto& node = nodes_[from];
    if (to < node.distances.size()) {
        return node.distances[to];
    }
    
    return 20; // Default remote distance
}

// NumaAllocator implementation

NumaAllocator::NumaAllocator(const NumaConfig& config) : config_(config) {}

NumaAllocator::~NumaAllocator() = default;

NumaAllocator::NumaAllocator(NumaAllocator&& other) noexcept
    : config_(std::move(other.config_))
    , topology_(std::move(other.topology_))
    , total_allocated_(other.total_allocated_.load())
    , total_deallocated_(other.total_deallocated_.load())
    , current_usage_(other.current_usage_.load())
    , peak_usage_(other.peak_usage_.load()) {
    // Manually move per_node_usage_ (array of atomics)
    num_nodes_ = other.num_nodes_;
    if (num_nodes_ > 0) {
        per_node_usage_ = std::make_unique<std::atomic<std::size_t>[]>(num_nodes_);
        for (size_t i = 0; i < num_nodes_; ++i) {
            per_node_usage_[i].store(other.per_node_usage_[i].load());
            other.per_node_usage_[i].store(0);
        }
    }
    other.num_nodes_ = 0;
    // Reset other's atomics to zero
    other.total_allocated_.store(0);
    other.total_deallocated_.store(0);
    other.current_usage_.store(0);
    other.peak_usage_.store(0);
}

NumaAllocator& NumaAllocator::operator=(NumaAllocator&& other) noexcept {
    if (this != &other) {
        config_ = std::move(other.config_);
        topology_ = std::move(other.topology_);
        total_allocated_.store(other.total_allocated_.load());
        total_deallocated_.store(other.total_deallocated_.load());
        current_usage_.store(other.current_usage_.load());
        peak_usage_.store(other.peak_usage_.load());
        
        // Manually move per_node_usage_ (array of atomics)
        num_nodes_ = other.num_nodes_;
        if (num_nodes_ > 0) {
            per_node_usage_ = std::make_unique<std::atomic<std::size_t>[]>(num_nodes_);
            for (size_t i = 0; i < num_nodes_; ++i) {
                per_node_usage_[i].store(other.per_node_usage_[i].load());
                other.per_node_usage_[i].store(0);
            }
        } else {
            per_node_usage_.reset();
        }
        other.num_nodes_ = 0;
        
        // Reset other's atomics to zero
        other.total_allocated_.store(0);
        other.total_deallocated_.store(0);
        other.current_usage_.store(0);
        other.peak_usage_.store(0);
    }
    return *this;
}

auto NumaAllocator::create(const NumaConfig& config)
    -> std::expected<std::unique_ptr<NumaAllocator>, core::error> {
    auto allocator = std::unique_ptr<NumaAllocator>(new NumaAllocator(config));
    
    if (auto result = allocator->init(); !result) {
        return std::vesper_unexpected(result.error());
    }
    
    return allocator;
}

auto NumaAllocator::init() -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;
    
    // Initialize topology
    auto topology = NumaTopology::detect();
    if (!topology) {
        return std::vesper_unexpected(topology.error());
    }
    topology_ = std::move(topology.value());
    
    // Initialize per-node statistics
    num_nodes_ = topology_->num_nodes();
    if (num_nodes_ > 0) {
        per_node_usage_ = std::make_unique<std::atomic<std::size_t>[]>(num_nodes_);
        for (std::size_t i = 0; i < num_nodes_; ++i) {
            per_node_usage_[i].store(0);
        }
    }
    
    return {};
}

auto NumaAllocator::allocate(std::size_t size)
    -> std::expected<void*, core::error> {
    using core::error;
    using core::error_code;
    
    if (size == 0) {
        return nullptr;
    }
    
    void* ptr = nullptr;
    
#ifdef __linux__
    if (topology_->is_numa_available()) {
        switch (config_.policy) {
        case NumaPolicy::LOCAL:
            ptr = numa_alloc_local(size);
            break;
            
        case NumaPolicy::INTERLEAVE:
            if (config_.nodes.empty()) {
                ptr = numa_alloc_interleaved(size);
            } else {
                struct bitmask* mask = numa_allocate_nodemask();
                for (auto node : config_.nodes) {
                    numa_bitmask_setbit(mask, node);
                }
                ptr = numa_alloc_interleaved_subset(size, mask);
                numa_free_nodemask(mask);
            }
            break;
            
        case NumaPolicy::PREFERRED:
            if (!config_.nodes.empty()) {
                ptr = numa_alloc_onnode(size, config_.nodes[0]);
            } else {
                ptr = numa_alloc_local(size);
            }
            break;
            
        case NumaPolicy::BIND:
            if (!config_.nodes.empty()) {
                ptr = numa_alloc_onnode(size, config_.nodes[0]);
            } else {
                ptr = numa_alloc_local(size);
            }
            break;
        }
        
        if (!ptr && errno == ENOMEM) {
            // Try huge pages if configured
            if (config_.use_1gb_pages) {
                ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | (30 << MAP_HUGE_SHIFT),
                          -1, 0);
            } else if (config_.use_huge_pages) {
                ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                          -1, 0);
            }
            
            if (ptr == MAP_FAILED) {
                ptr = nullptr;
            }
        }
    } else {
        // Fallback to regular allocation
        if (config_.use_huge_pages) {
            ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                      -1, 0);
            if (ptr == MAP_FAILED) {
                ptr = nullptr;
            }
        } else {
            ptr = std::aligned_alloc(config_.alignment, size);
        }
    }
    
#elif defined(_WIN32)
    // Windows allocation
    DWORD alloc_type = MEM_COMMIT | MEM_RESERVE;
    if (config_.use_huge_pages) {
        alloc_type |= MEM_LARGE_PAGES;
    }
    
    if (topology_->is_numa_available() && !config_.nodes.empty()) {
        ptr = VirtualAllocExNuma(GetCurrentProcess(), nullptr, size,
                                 alloc_type, PAGE_READWRITE, config_.nodes[0]);
    } else {
        ptr = VirtualAlloc(nullptr, size, alloc_type, PAGE_READWRITE);
    }
    
#else
    // Generic allocation
    ptr = std::aligned_alloc(config_.alignment, size);
#endif
    
    if (!ptr) {
        return std::vesper_unexpected(core::error{
            core::error_code::out_of_memory,
            "Failed to allocate " + std::to_string(size) + " bytes",
            "numa"
        });
    }
    
    // Update statistics
    total_allocated_.fetch_add(size, std::memory_order_relaxed);
    current_usage_.fetch_add(size, std::memory_order_relaxed);
    
    std::size_t current = current_usage_.load(std::memory_order_relaxed);
    std::size_t peak = peak_usage_.load(std::memory_order_relaxed);
    while (current > peak && !peak_usage_.compare_exchange_weak(peak, current)) {
        // Retry
    }
    
    // Update per-node usage
    std::uint32_t node = topology_->current_node();
    if (node < num_nodes_ && per_node_usage_) {
        per_node_usage_[node].fetch_add(size, std::memory_order_relaxed);
    }
    
    return ptr;
}

auto NumaAllocator::allocate_aligned(std::size_t size, std::size_t alignment)
    -> std::expected<void*, core::error> {
    using core::error;
    using core::error_code;
    
    if (size == 0) {
        return nullptr;
    }
    
    // Ensure alignment is power of 2
    if (alignment & (alignment - 1)) {
        return std::vesper_unexpected(core::error{
            core::error_code::invalid_argument,
            "Alignment must be power of 2",
            "numa"
        });
    }
    
    // Allocate with extra space for alignment
    std::size_t alloc_size = size + alignment - 1;
    auto result = allocate(alloc_size);
    if (!result) {
        return result;
    }
    
    void* ptr = result.value();
    void* aligned = reinterpret_cast<void*>(
        (reinterpret_cast<std::uintptr_t>(ptr) + alignment - 1) & ~(alignment - 1)
    );
    
    return aligned;
}

auto NumaAllocator::allocate_on_node(std::size_t size, std::uint32_t node)
    -> std::expected<void*, core::error> {
    using core::error;
    using core::error_code;
    
    if (size == 0) {
        return nullptr;
    }
    
    if (node >= topology_->num_nodes()) {
        return std::vesper_unexpected(core::error{
            core::error_code::invalid_argument,
            "Invalid NUMA node: " + std::to_string(node),
            "numa"
        });
    }
    
    void* ptr = nullptr;
    
#ifdef __linux__
    if (topology_->is_numa_available()) {
        ptr = numa_alloc_onnode(size, node);
    } else {
        ptr = std::aligned_alloc(config_.alignment, size);
    }
    
#elif defined(_WIN32)
    if (topology_->is_numa_available()) {
        ptr = VirtualAllocExNuma(GetCurrentProcess(), nullptr, size,
                                 MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE, node);
    } else {
        ptr = VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    }
    
#else
    ptr = std::aligned_alloc(config_.alignment, size);
#endif
    
    if (!ptr) {
        return std::vesper_unexpected(core::error{
            core::error_code::out_of_memory,
            "Failed to allocate on node " + std::to_string(node),
            "numa"
        });
    }
    
    // Update statistics
    total_allocated_.fetch_add(size, std::memory_order_relaxed);
    current_usage_.fetch_add(size, std::memory_order_relaxed);
    
    if (node < num_nodes_ && per_node_usage_) {
        per_node_usage_[node].fetch_add(size, std::memory_order_relaxed);
    }
    
    return ptr;
}

auto NumaAllocator::deallocate(void* ptr, std::size_t size) noexcept -> void {
    if (!ptr) return;
    
#ifdef __linux__
    if (topology_->is_numa_available()) {
        numa_free(ptr, size);
    } else if (config_.use_huge_pages) {
        munmap(ptr, size);
    } else {
        std::free(ptr);
    }
    
#elif defined(_WIN32)
    VirtualFree(ptr, 0, MEM_RELEASE);
    
#else
    std::free(ptr);
#endif
    
    // Update statistics
    total_deallocated_.fetch_add(size, std::memory_order_relaxed);
    current_usage_.fetch_sub(size, std::memory_order_relaxed);
    
    // Update per-node usage (approximate - we don't track which node)
    std::uint32_t node = topology_->current_node();
    if (node < num_nodes_ && per_node_usage_) {
        per_node_usage_[node].fetch_sub(
            (std::min)(size, per_node_usage_[node].load(std::memory_order_relaxed)),
            std::memory_order_relaxed
        );
    }
}

auto NumaAllocator::migrate(void* ptr, std::size_t size, std::uint32_t node)
    -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;
    
    if (!ptr || size == 0) {
        return {};
    }
    
    if (node >= topology_->num_nodes()) {
        return std::vesper_unexpected(core::error{
            core::error_code::invalid_argument,
            "Invalid NUMA node: " + std::to_string(node),
            "numa"
        });
    }
    
#ifdef __linux__
    if (topology_->is_numa_available()) {
        // Set memory policy for pages
        unsigned long nodemask = 1UL << node;
        if (mbind(ptr, size, MPOL_BIND, &nodemask, sizeof(nodemask) * 8,
                  MPOL_MF_MOVE | MPOL_MF_STRICT) < 0) {
            return std::vesper_unexpected(core::error{
                core::error_code::io_error,
                "Failed to migrate memory: " + std::string(std::strerror(errno)),
                "numa"
            });
        }
    }
#endif
    
    return {};
}

auto NumaAllocator::touch_pages(void* ptr, std::size_t size) noexcept -> void {
    if (!ptr || size == 0) return;
    
    // Touch each page to fault it in
    const std::size_t page_size = 4096;
    volatile char* p = static_cast<volatile char*>(ptr);
    
    for (std::size_t i = 0; i < size; i += page_size) {
        p[i] = p[i];
    }
    
    // Touch last byte if not page-aligned
    if (size % page_size != 0) {
        p[size - 1] = p[size - 1];
    }
}

auto NumaAllocator::get_stats() const noexcept -> Stats {
    Stats stats;
    stats.total_allocated = total_allocated_.load(std::memory_order_relaxed);
    stats.total_deallocated = total_deallocated_.load(std::memory_order_relaxed);
    stats.current_usage = current_usage_.load(std::memory_order_relaxed);
    stats.peak_usage = peak_usage_.load(std::memory_order_relaxed);
    
    if (per_node_usage_) {
        for (std::size_t i = 0; i < num_nodes_; ++i) {
            stats.per_node_usage.push_back(per_node_usage_[i].load(std::memory_order_relaxed));
        }
    }
    
    return stats;
}

// ThreadAffinity implementation

auto ThreadAffinity::bind_to_node(std::uint32_t node)
    -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;
    
#ifdef __linux__
    if (numa_available() >= 0) {
        struct bitmask* mask = numa_allocate_nodemask();
        numa_bitmask_setbit(mask, node);
        
        if (numa_run_on_node_mask(mask) < 0) {
            numa_free_nodemask(mask);
            return std::vesper_unexpected(core::error{
                core::error_code::io_error,
                "Failed to bind to node: " + std::string(std::strerror(errno)),
                "numa"
            });
        }
        
        numa_free_nodemask(mask);
    }
    
#elif defined(_WIN32)
    ULONG highest_node = 0;
    if (GetNumaHighestNodeNumber(&highest_node) && node <= highest_node) {
        GROUP_AFFINITY affinity;
        if (GetNumaNodeProcessorMaskEx(static_cast<USHORT>(node), &affinity)) {
            if (!SetThreadGroupAffinity(GetCurrentThread(), &affinity, nullptr)) {
                return std::vesper_unexpected(core::error{
                    core::error_code::io_error,
                    "Failed to bind to node",
                    "numa"
                });
            }
        }
    }
#endif
    
    return {};
}

auto ThreadAffinity::bind_to_cpus(std::span<const std::uint32_t> cpus)
    -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;
    
    if (cpus.empty()) {
        return {};
    }
    
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    
    for (auto cpu : cpus) {
        CPU_SET(cpu, &cpuset);
    }
    
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) < 0) {
        return std::vesper_unexpected(core::error{
            core::error_code::io_error,
            "Failed to set CPU affinity: " + std::string(std::strerror(errno)),
            "numa"
        });
    }
    
#elif defined(_WIN32)
    DWORD_PTR mask = 0;
    for (auto cpu : cpus) {
        if (cpu < 64) {
            mask |= (1ULL << cpu);
        }
    }
    
    if (!SetThreadAffinityMask(GetCurrentThread(), mask)) {
        return std::vesper_unexpected(core::error{
            core::error_code::io_error,
            "Failed to set CPU affinity",
            "numa"
        });
    }
#endif
    
    return {};
}

auto ThreadAffinity::current_cpu() noexcept -> std::uint32_t {
#ifdef __linux__
    return static_cast<std::uint32_t>(sched_getcpu());
#elif defined(_WIN32)
    return GetCurrentProcessorNumber();
#else
    return 0;
#endif
}

auto ThreadAffinity::reset() noexcept -> void {
#ifdef __linux__
    if (numa_available() >= 0) {
        numa_run_on_node_mask(numa_all_nodes_ptr);
    }
#elif defined(_WIN32)
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)-1);
#endif
}

// NumaAllocatorPool implementation

auto NumaAllocatorPool::get_for_node(std::uint32_t node, const NumaConfig& config)
    -> std::expected<NumaAllocator*, core::error> {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (node >= allocators_.size()) {
        allocators_.resize(node + 1);
    }
    
    if (!allocators_[node]) {
        auto allocator = NumaAllocator::create(config);
        if (!allocator) {
            return std::vesper_unexpected(allocator.error());
        }
        allocators_[node] = std::move(allocator.value());
    }
    
    return allocators_[node].get();
}

auto NumaAllocatorPool::get_local(const NumaConfig& config)
    -> std::expected<NumaAllocator*, core::error> {
    auto topology = NumaTopology::detect();
    if (!topology) {
        return std::vesper_unexpected(topology.error());
    }
    
    std::uint32_t node = topology.value()->current_node();
    return get_for_node(node, config);
}

auto NumaAllocatorPool::reset() noexcept -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    allocators_.clear();
}

} // namespace vesper::memory