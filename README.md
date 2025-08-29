# Vesper

**Crash-Safe, Embeddable Vector Search Engine**

*Ultra-fast vector search for edge, offline, and privacy-sensitive applications*

## Overview

Vesper brings enterprise-grade vector search directly to where your data lives—delivering uncompromising speed, durability, and privacy for on-device and air-gapped environments. Built with C++20 and optimized for CPU-only deployment, Vesper eliminates the need for cloud dependencies while maintaining production-ready performance.

## Why Choose Vesper?

### 🚀 **Performance Without Compromise**
- **Sub-20ms latency** on commodity hardware
- **SIMD-accelerated kernels** (AVX2/AVX-512)
- **Multiple index strategies** for different use cases
- **Blazing-fast metadata filtering** with Roaring bitmaps

### 🛡️ **Enterprise-Grade Reliability**
- **Crash-safe by design** with checksummed WAL
- **Atomic snapshot publishing** for data integrity
- **Deterministic persistence** for mission-critical applications
- **Optional at-rest encryption** (XChaCha20-Poly1305, AES-GCM)

### 🌐 **Deploy Anywhere**
- **Pure C++20** with stable C ABI
- **Cross-platform support** (Linux, macOS, Windows)
- **No GPU requirements** - runs on any modern CPU
- **Embeddable library** - integrate seamlessly

## Perfect For

### 🎯 **Edge AI & IoT**
- Offline RAG workflows
- Real-time similarity search
- Resource-constrained environments
- Field robotics and embedded systems

### 🏥 **Regulated Industries**
- Healthcare data processing
- Financial services compliance
- Government and defense applications
- Data sovereignty requirements

### 🔒 **Privacy-First Applications**
- Local-only processing
- Air-gapped deployments
- GDPR/HIPAA compliance
- Zero cloud dependencies

## Key Features

### Advanced Indexing
- **IVF-PQ/OPQ**: Compact, SSD-friendly indexing
- **HNSW**: High-performance in-memory segments
- **Disk-graph**: DiskANN-style for billion-scale datasets
- **Pluggable index families** per collection

### Smart Filtering
- **Early metadata filtering** during traversal
- **Roaring bitmap optimization** for AND/intersection operations
- **Hybrid search capabilities**
- **Custom filter predicates**

### Production Ready
- **Crash-safe persistence** with strict fsync discipline
- **Atomic operations** for data consistency
- **Memory-mapped file support**
- **Comprehensive error handling**

## Quick Start

```bash
# Prerequisites and build instructions
see docs/SETUP.md

# Sample schemas and examples
cd experiments/

# Run validation tests
see experiments/VALIDATION.md
```

## Performance Targets

| Metric | Target | Notes |
|--------|-----------|-------|
| **Latency** | p50: 1-3ms, p99: 10-20ms | 128-1536D vectors |
| **Quality** | recall@10 ≈ 0.95 | Tunable precision |
| **Recovery** | Seconds to minutes | WAL replay/snapshot restore |
| **Throughput** | CPU-bound scaling | No GPU required |

## Technical Architecture

- **[System Design](blueprint.md)**: High-level architecture and data models
- **[API Documentation](api-notes.md)**: Out-of-code documentation
- **[Performance Analysis](benchmark-plan.md)**: Detailed benchmarking methodology
- **[Security Model](threat-model.md)**: Assets, adversaries, and controls

## Platform Support

### Operating Systems
- ✅ Linux (primary)
- ✅ macOS 
- ✅ Windows

### Compiler Requirements
- **GCC**: 12+ 
- **Clang**: 15+ / AppleClang 15+
- **MSVC**: 19.36+

### CPU Requirements
- **Baseline**: AVX2 for optimal performance
- **Fallback**: Scalar operations supported
- **Runtime dispatch**: Automatic CPU feature detection

## Development & Contributing

We follow a **deterministic, tests-first** development approach:

- **AI-assisted development** with fixed parameters (temperature=0.0)
- **Comprehensive test coverage** before feature implementation
- **Prompt-driven architecture** for consistency

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for:
- Branching strategy
- Pull request workflow
- CI/CD pipeline
- Schema validation gates

## Roadmap

| Phase | Focus | Timeline |
|-------|--------|----------|
| **Phase 1** | Core engine & basic indexing | Q1 2025 |
| **Phase 2** | Advanced features & optimization | Q2 2025 |
| **Phase 3** | Enterprise features & scaling | Q3 2025 |

Detailed roadmap: **[prompt-dev-roadmap.md](prompt-dev-roadmap.md)**

## Security & Privacy

### Built-in Protection
- 🔒 **No network I/O by default**
- 🔐 **Optional strong encryption at rest**
- 📁 **Local-only file operations**
- 🛡️ **Strict durability guarantees**

### Compliance Ready
- **GDPR compliant** - data never leaves your infrastructure
- **HIPAA friendly** - secure local processing
- **SOC 2 compatible** - comprehensive audit trails

Full security analysis: **[threat-model.md](threat-model.md)**

## License

**Apache License 2.0** - see [LICENSE](LICENSE) for details.

## Get Involved

- 💬 **[Start a Discussion](../../discussions)** - Ask questions, share ideas
- 🐛 **[Report Issues](../../issues)** - Bug reports and feature requests  
- 📖 **[Documentation](docs/)** - Setup guides and tutorials
- 🤝 **[Contributing](CONTRIBUTING.md)** - Join our development community

---

**Ready to bring vector search to the edge?** Start with our [Quick Start Guide](docs/SETUP.md) or explore the [sample applications](experiments/).
