# Vesper Operator Guide

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Performance Tuning](#performance-tuning)
4. [Monitoring](#monitoring)
5. [Maintenance](#maintenance)
6. [Troubleshooting](#troubleshooting)
7. [Disaster Recovery](#disaster-recovery)
8. [Security](#security)
9. [Best Practices](#best-practices)

## Installation

### System Requirements

#### Minimum Requirements
- **CPU**: x86-64 with AVX2 support (2013+ Intel Haswell, AMD Excavator)
- **RAM**: 4 GB minimum
- **Storage**: 10 GB free space + data size
- **OS**: Linux (kernel 5.1+), Windows 10/11, macOS 11+

#### Recommended Requirements
- **CPU**: x86-64 with AVX-512 support (Intel Skylake-X/Ice Lake or newer)
- **RAM**: 16 GB or more
- **Storage**: NVMe SSD with 2x data size free space
- **OS**: Linux with kernel 5.10+ (for io_uring support)

### Building from Source

#### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libnuma-dev \
    liburing-dev \
    libtbb-dev

# RHEL/CentOS/Fedora
sudo dnf install -y \
    gcc-c++ \
    cmake \
    git \
    numactl-devel \
    liburing-devel \
    tbb-devel

# macOS
brew install cmake git tbb
```

#### Build Steps

```bash
# Clone repository
git clone https://github.com/vesper-arch/vesper.git
cd vesper

# Create build directory
mkdir build && cd build

# Configure with Release mode
cmake -DCMAKE_BUILD_TYPE=Release \
      -DVESPER_ENABLE_TESTS=ON \
      -DVESPER_ENABLE_BENCH=ON \
      ..

# Build (use all cores)
cmake --build . -j$(nproc)

# Run tests
ctest --output-on-failure

# Install
sudo cmake --install .
```

### Docker Installation

```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libnuma-dev \
    liburing-dev

WORKDIR /vesper
COPY . .

RUN mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    cmake --build . -j$(nproc)

CMD ["./build/vesper_server"]
```

## Configuration

### Configuration File

Create `vesper.yaml`:

```yaml
# Data storage configuration
storage:
  data_dir: /var/lib/vesper/data
  wal_dir: /var/lib/vesper/wal
  max_wal_size_mb: 1024
  sync_interval_ms: 100
  compression: true
  encryption:
    enabled: false
    key_file: /etc/vesper/master.key

# Memory configuration
memory:
  cache_size_mb: 4096
  numa_aware: true
  huge_pages: true
  arena_size_mb: 256

# Performance settings
performance:
  num_threads: 0  # 0 = auto-detect
  io_threads: 4
  batch_size: 1000
  prefetch_distance: 32

# Index defaults
index:
  hnsw:
    M: 16
    ef_construction: 200
    max_M: 16
  ivf_pq:
    nlist: 1024
    nprobe: 10
    nbits: 8
  disk_graph:
    R: 64
    L: 128
    cache_fraction: 0.1

# Monitoring
monitoring:
  metrics_enabled: true
  metrics_port: 9090
  log_level: info
  log_file: /var/log/vesper/vesper.log

# Network (if using server mode)
network:
  bind_address: 0.0.0.0
  port: 8080
  max_connections: 1000
  keepalive_ms: 60000
```

### Environment Variables

```bash
# Override configuration
export VESPER_DATA_DIR=/data/vesper
export VESPER_CACHE_SIZE_MB=8192
export VESPER_NUM_THREADS=16

# Performance tuning
export VESPER_KERNEL_BACKEND=avx512  # or: avx2, scalar, auto
export VESPER_SIMD_MASK=2            # 0=scalar, 1=avx2, 2=avx512


# IVF-PQ exact rerank controls (opt-in overrides)
# Enable exact reranking and control shortlist sizing without code changes
export VESPER_USE_EXACT_RERANK=true    # or: false/0/1
export VESPER_RERANK_K=200             # 0 = auto via heuristic
export VESPER_RERANK_ALPHA=2.0         # cand_k = max(k, alpha * k * log2(1+nprobe))
export VESPER_RERANK_CEIL=2000         # 0 = no cap on adaptive shortlist

# Test sweep utilities
export VESPER_RERANK_AUTO=1            # convenience: force rerank_k=0 (auto)
export VESPER_SWEEP_NQ=500             # limit queries processed in sweep tests


# Performance smoke test (integration/performance_test.cpp)
export VESPER_PERF_SMOKE_FORCE_SYNTH=1  # Force synthetic 20kx64 dataset with 200 queries for the smoke test

# NUMA settings
export VESPER_NUMA_NODES=0,1         # Bind to specific nodes
export VESPER_NUMA_INTERLEAVE=true   # Interleave memory

# Debug settings
export VESPER_LOG_LEVEL=debug
export VESPER_ENABLE_PROFILING=1
```

## Performance Tuning

### CPU Optimization

#### SIMD Acceleration

```bash
# Check CPU features
cat /proc/cpuinfo | grep -E "avx2|avx512" | head -1

# Force specific backend
export VESPER_KERNEL_BACKEND=avx512

# Verify backend selection
./vesper_bench_simd --benchmark_filter=".*Auto.*"
```

#### NUMA Optimization

```bash
# Check NUMA topology
numactl --hardware

# Bind to specific node
numactl --cpunodebind=0 --membind=0 ./vesper_server

# Interleave memory across nodes
numactl --interleave=all ./vesper_server
```

#### Thread Affinity

```bash
# Set CPU affinity
taskset -c 0-15 ./vesper_server

# Isolate CPUs for Vesper
echo "isolcpus=8-15" >> /etc/default/grub
update-grub && reboot
```

### Memory Optimization

#### Huge Pages

```bash
# Enable transparent huge pages
echo always > /sys/kernel/mm/transparent_hugepage/enabled

# Configure 2MB huge pages
echo 2048 > /proc/sys/vm/nr_hugepages

# Configure 1GB huge pages (boot parameter)
# Add to /etc/default/grub:
# GRUB_CMDLINE_LINUX="hugepagesz=1G hugepages=16"
```

#### Memory Allocation

```bash
# Disable memory overcommit
echo 2 > /proc/sys/vm/overcommit_memory

# Set swappiness low
echo 10 > /proc/sys/vm/swappiness

# Increase memory map limit
echo 655360 > /proc/sys/vm/max_map_count
```

### Storage Optimization

#### File System

```bash
# Use XFS or ext4 with optimizations
mkfs.xfs -f -d agcount=16 /dev/nvme0n1

# Mount with optimizations
mount -o noatime,nodiratime,nobarrier /dev/nvme0n1 /var/lib/vesper

# For ext4
mount -o noatime,nodiratime,data=writeback /dev/sda1 /var/lib/vesper
```

#### I/O Scheduler

```bash
# Use none (noop) for NVMe
echo none > /sys/block/nvme0n1/queue/scheduler

# Use deadline for SSD
echo deadline > /sys/block/sda/queue/scheduler

# Increase read-ahead
echo 4096 > /sys/block/nvme0n1/queue/read_ahead_kb
```

#### io_uring (Linux 5.1+)

```bash
# Check io_uring support
ls /sys/class/misc/uring*

# Increase io_uring entries
echo 32768 > /proc/sys/fs/io_uring_setup_sqe_entries
```

### Index Tuning

#### HNSW Parameters

```yaml
# High recall configuration
hnsw:
  M: 32                # More connections
  ef_construction: 400  # Better graph quality
  ef_search: 200       # Higher search quality

# Low latency configuration
hnsw:
  M: 12                # Fewer connections
  ef_construction: 100  # Faster build
  ef_search: 50        # Faster search
```

#### IVF-PQ Parameters

```yaml
# Memory optimized
ivf_pq:
  nlist: 4096          # More clusters
  m: 64               # More subquantizers
  nbits: 4            # Fewer bits (more compression)

# Speed optimized
ivf_pq:
  nlist: 256          # Fewer clusters
  m: 8                # Fewer subquantizers
  nbits: 8            # More bits (less compression)
```

## Monitoring

### Metrics

#### Prometheus Integration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'vesper'
    static_configs:
      - targets: ['localhost:9090']
```

#### Key Metrics

```bash
# Query latency percentiles
vesper_query_latency_seconds{quantile="0.5"}
vesper_query_latency_seconds{quantile="0.95"}
vesper_query_latency_seconds{quantile="0.99"}

# Throughput
rate(vesper_queries_total[1m])
rate(vesper_insertions_total[1m])

# Index statistics
vesper_index_vectors_total
vesper_index_segments_count
vesper_index_memory_bytes

# System resources
vesper_memory_usage_bytes
vesper_cpu_usage_percent
vesper_disk_usage_bytes
```

### Logging

#### Log Levels

```bash
# Set log level
export VESPER_LOG_LEVEL=debug  # trace, debug, info, warn, error

# Log format
export VESPER_LOG_FORMAT=json  # text, json

# Log rotation
cat > /etc/logrotate.d/vesper << EOF
/var/log/vesper/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
EOF
```

#### Structured Logging

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "info",
  "component": "index",
  "message": "Search completed",
  "query_id": "abc123",
  "latency_ms": 2.5,
  "results": 10,
  "recall": 0.95
}
```

### Health Checks

```bash
# Basic health check
curl http://localhost:8080/health

# Detailed status
curl http://localhost:8080/status

# Response format
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "collections": 5,
  "total_vectors": 1000000,
  "memory_usage_mb": 2048,
  "query_rate": 150.5
}
```

## Maintenance

### Backup and Restore

#### Snapshot Creation

```bash
# Create snapshot via API
curl -X POST http://localhost:8080/collections/my_collection/snapshot

# Create snapshot via CLI
vesper-cli snapshot create my_collection --name backup_$(date +%Y%m%d)

# Automated snapshots
cat > /etc/cron.d/vesper-backup << EOF
0 2 * * * vesper vesper-cli snapshot create-all --prefix daily_
0 3 * * 0 vesper vesper-cli snapshot create-all --prefix weekly_
EOF
```

#### Restore Procedures

```bash
# List available snapshots
vesper-cli snapshot list my_collection

# Restore from snapshot
vesper-cli snapshot restore my_collection --name backup_20240115

# Point-in-time recovery
vesper-cli snapshot restore my_collection --timestamp "2024-01-15 10:30:00"
```

### Compaction

```bash
# Manual compaction
vesper-cli compact my_collection

# Automatic compaction configuration
cat >> vesper.yaml << EOF
compaction:
  auto_enabled: true
  trigger_segments: 10
  trigger_size_mb: 1024
  schedule: "0 3 * * *"  # 3 AM daily
EOF
```

### Index Rebuilding

```bash
# Rebuild specific index
vesper-cli index rebuild my_collection --type hnsw

# Optimize index parameters
vesper-cli index optimize my_collection --auto

# Rebalance shards
vesper-cli index rebalance my_collection
```

## Troubleshooting

### Common Issues

#### High Memory Usage

```bash
# Check memory breakdown
cat /proc/$(pidof vesper_server)/status | grep -E "Vm|Rss"

# Reduce cache size
export VESPER_CACHE_SIZE_MB=2048

# Enable memory profiling
export VESPER_ENABLE_PROFILING=1
./vesper_server &
kill -USR1 $(pidof vesper_server)  # Dump memory profile
```

#### Slow Queries

```bash
# Enable query profiling
export VESPER_QUERY_PROFILE=1

# Check slow query log
tail -f /var/log/vesper/slow_queries.log

# Analyze query plan
vesper-cli explain "SELECT * FROM collection WHERE ..."
```

#### Corruption Detection

```bash
# Verify data integrity
vesper-cli verify my_collection --deep

# Check WAL integrity
vesper-cli wal verify --dir /var/lib/vesper/wal

# Repair corrupted segments
vesper-cli repair my_collection --force
```

### Debug Tools

```bash
# Enable debug logging
export VESPER_LOG_LEVEL=trace

# Core dump analysis
ulimit -c unlimited
echo "/tmp/core-%e-%p-%t" > /proc/sys/kernel/core_pattern

# Trace system calls
strace -c -p $(pidof vesper_server)

# Profile CPU usage
perf record -g -p $(pidof vesper_server) -- sleep 10
perf report

# Memory leak detection
valgrind --leak-check=full ./vesper_server
```

## Disaster Recovery

### Backup Strategy

```yaml
# backup-policy.yaml
policies:
  - name: hourly
    schedule: "0 * * * *"
    retention: 24

  - name: daily
    schedule: "0 2 * * *"
    retention: 7
    compress: true

  - name: weekly
    schedule: "0 3 * * 0"
    retention: 4
    compress: true
    encrypt: true

  - name: monthly
    schedule: "0 4 1 * *"
    retention: 12
    compress: true
    encrypt: true
    offsite: true
```

### Recovery Procedures

#### Complete System Failure

```bash
# 1. Install fresh Vesper instance
./install.sh

# 2. Restore configuration
cp /backup/vesper.yaml /etc/vesper/

# 3. Restore latest snapshot
vesper-cli snapshot restore-all --from /backup/latest/

# 4. Replay WAL if needed
vesper-cli wal replay --from /backup/wal/

# 5. Verify integrity
vesper-cli verify --all

# 6. Resume operations
systemctl start vesper
```

#### Partial Data Loss

```bash
# Identify corrupted segments
vesper-cli check my_collection --report

# Remove corrupted segments
vesper-cli segment remove my_collection --corrupted

# Rebuild from healthy segments
vesper-cli index rebuild my_collection --partial

# Verify and compact
vesper-cli compact my_collection --verify
```

## Security

### Access Control

```yaml
# security.yaml
authentication:
  enabled: true
  method: jwt  # or: basic, oauth2, ldap

authorization:
  enabled: true
  default_policy: deny

  roles:
    - name: admin
      permissions: ["*"]

    - name: writer
      permissions: ["collection:write", "collection:read"]

    - name: reader
      permissions: ["collection:read"]
```

### Encryption

#### At Rest

```bash
# Generate master key
openssl rand -base64 32 > /etc/vesper/master.key
chmod 600 /etc/vesper/master.key

# Enable encryption
cat >> vesper.yaml << EOF
encryption:
  at_rest:
    enabled: true
    algorithm: xchacha20-poly1305
    key_file: /etc/vesper/master.key
EOF
```

#### In Transit

```bash
# Generate certificates
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Configure TLS
cat >> vesper.yaml << EOF
tls:
  enabled: true
  cert_file: /etc/vesper/cert.pem
  key_file: /etc/vesper/key.pem
  min_version: "1.2"
EOF
```

### Audit Logging

```yaml
audit:
  enabled: true
  log_file: /var/log/vesper/audit.log
  events:
    - authentication
    - authorization
    - data_access
    - configuration_change
    - admin_action
```

## Best Practices

### Capacity Planning

```bash
# Estimate storage requirements
# Storage = (vector_size * num_vectors * replication_factor) +
#          (index_overhead * 1.5) + (wal_size * 2)

# Example: 1M vectors, 768 dimensions
# Raw: 768 * 4 bytes * 1M = 3 GB
# Indexed: 3 GB * 1.5 = 4.5 GB
# Total: ~10 GB with overhead
```

### Production Checklist

- [ ] Enable authentication and authorization
- [ ] Configure TLS for network connections
- [ ] Set up automated backups
- [ ] Configure monitoring and alerting
- [ ] Tune kernel parameters
- [ ] Set resource limits (ulimit)
- [ ] Enable audit logging
- [ ] Document recovery procedures
- [ ] Test disaster recovery plan
- [ ] Configure log rotation
- [ ] Set up health checks
- [ ] Implement rate limiting
- [ ] Configure firewall rules
- [ ] Review security settings
- [ ] Performance baseline established

### Scaling Guidelines

#### Vertical Scaling
- Add more CPU cores for parallel queries
- Increase RAM for larger cache
- Use faster storage (NVMe > SSD > HDD)

#### Horizontal Scaling
- Shard by collection
- Read replicas for query load
- Separate index types to different nodes
- Use load balancer for distribution

### Maintenance Windows

```bash
# Graceful shutdown
systemctl stop vesper  # Waits for pending operations

# Maintenance mode
vesper-cli maintenance enable
# Perform maintenance tasks
vesper-cli maintenance disable

# Rolling updates (with replicas)
for node in node1 node2 node3; do
  ssh $node "systemctl stop vesper && upgrade.sh && systemctl start vesper"
  sleep 60  # Wait for recovery
done
```