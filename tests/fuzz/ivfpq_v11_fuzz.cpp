#include <vesper/index/ivf_pq.hpp>

#include <cstdint>
#include <vector>
#include <string>
#include <string_view>
#include <random>
#include <filesystem>
#include <fstream>
#include <optional>
#include <cstring>

namespace fs = std::filesystem;
using vesper::index::IvfPqIndex;

namespace {

// Simple FNV-1a 64-bit
static inline std::uint64_t fnv64(const void* ptr, size_t nbytes) {
    const auto* p = static_cast<const std::uint8_t*>(ptr);
    std::uint64_t h = 1469598103934665603ull; constexpr std::uint64_t F = 1099511628211ull;
    for (size_t i=0;i<nbytes;++i){ h^=p[i]; h*=F; }
    return h;
}

struct SectionHdr { std::uint32_t type; std::uint64_t unc; std::uint64_t comp; std::uint64_t shash; };

struct Blueprint {
    // Parameters kept small for speed
    std::uint32_t dim{16};
    std::uint32_t m{4};
    std::uint32_t nbits{8};
    std::uint32_t nlist{32};
    std::uint32_t avg_list{2};
    bool bad_trailer{false};
    bool bad_shash{false};
    bool lists_mismatch{false};
    bool truncate_inverted{false};
    bool set_metadata{false};
    bool metadata_invalid_utf8{false};
    bool mmap{false};
};

static Blueprint blueprint_from_input(const std::uint8_t* data, size_t size) {
    Blueprint bp{};
    if (size == 0) return bp;
    auto rnd = [&](int lo, int hi)->int{ if (size==0) return lo; static size_t idx=0; int v = data[idx%size]; idx++; return lo + (v % (hi-lo+1)); };
    bp.dim = 8u * (1u + (rnd(0,5))); // 8,16,24,32,40,48
    bp.m = static_cast<std::uint32_t>(std::max(1, rnd(1, 6))); // 1..6
    if (bp.dim % bp.m != 0) { // ensure divisible
        bp.m = 1;
    }
    bp.nbits = 8;
    bp.nlist = static_cast<std::uint32_t>(std::max(1, rnd(1, 128)));
    bp.avg_list = static_cast<std::uint32_t>(std::max(1, rnd(1, 4)));
    // toggles
    unsigned mask = (size>0? data[0] : 0);
    bp.bad_trailer = (mask & 0x1) != 0;
    bp.bad_shash = (mask & 0x2) != 0;
    bp.lists_mismatch = (mask & 0x4) != 0;
    bp.truncate_inverted = (mask & 0x8) != 0;
    bp.set_metadata = (mask & 0x10) != 0;
    bp.metadata_invalid_utf8 = (mask & 0x20) != 0;
    bp.mmap = (mask & 0x40) != 0;
    return bp;
}

static std::string make_metadata(const std::uint8_t* data, size_t size, bool invalid_utf8) {
    if (!invalid_utf8) {
        // Small valid JSON
        return std::string("{\"a\":1,\"b\":\"x\"}");
    }
    // Insert some invalid UTF-8 bytes
    std::string s("{\"x\": \"");
    s.push_back(static_cast<char>(0xC0)); // overlong lead
    s.push_back('x');
    s += "\"}";
    return s;
}

static std::vector<std::uint8_t> build_v11(const Blueprint& bp, const std::uint8_t* data, size_t size) {
    std::vector<std::uint8_t> out;
    auto append = [&](const void* ptr, size_t n){ auto* b = static_cast<const std::uint8_t*>(ptr); out.insert(out.end(), b, b+n); };

    char magic[8] = {'I','V','F','P','Q','v','1','1'};
    append(magic, 8);
    std::uint16_t maj=1, min=1; append(&maj,2); append(&min,2);
    std::uint32_t flags=0; append(&flags,4);

    std::uint32_t dim=bp.dim, nlist=bp.nlist, m=bp.m, nbits=bp.nbits, dsub= (m? dim/m : 0);
    std::uint64_t nvec = static_cast<std::uint64_t>(nlist) * bp.avg_list;
    std::uint32_t code_size=m; std::uint64_t build_ts=0;
    append(&dim,4); append(&nlist,4); append(&m,4); append(&nbits,4); append(&dsub,4);
    append(&nvec,8); append(&code_size,4); append(&build_ts,8);

    std::string meta;
    if (bp.set_metadata) meta = make_metadata(data,size,bp.metadata_invalid_utf8);
    std::uint32_t meta_len = static_cast<std::uint32_t>(meta.size());
    append(&meta_len,4);
    if (meta_len) append(meta.data(), meta.size());

    auto append_section = [&](std::uint32_t type, const void* payload, size_t bytes, bool corrupt_hash=false){
        SectionHdr hdr{type, static_cast<std::uint64_t>(bytes), static_cast<std::uint64_t>(bytes), 0};
        hdr.shash = fnv64(payload, bytes);
        if (corrupt_hash && hdr.shash) hdr.shash ^= 0x1ull;
        append(&hdr, sizeof(hdr));
        append(payload, bytes);
    };

    // Centroids [nlist x dim]
    std::vector<float> centroids(static_cast<size_t>(nlist) * dim);
    for (size_t i=0;i<centroids.size();++i) centroids[i] = static_cast<float>((i%13)-6) * 0.01f;
    append_section(1, centroids.data(), centroids.size()*sizeof(float), bp.bad_shash);

    // Codebooks [m*ksub x dsub]
    const size_t ksub = (1u << nbits);
    std::vector<float> codebooks(static_cast<size_t>(m) * ksub * dsub);
    for (auto &v: codebooks) v = 0.0f;
    append_section(2, codebooks.data(), codebooks.size()*sizeof(float), false);

    // Inverted lists
    std::vector<std::uint8_t> inv;
    auto put32=[&](std::uint32_t v){ size_t s=inv.size(); inv.resize(s+4); std::memcpy(inv.data()+s,&v,4);} ;
    auto put64=[&](std::uint64_t v){ size_t s=inv.size(); inv.resize(s+8); std::memcpy(inv.data()+s,&v,8);} ;

    std::uint32_t lists_count = bp.lists_mismatch ? (nlist + 1) : nlist;
    put32(lists_count);
    std::uint64_t cur_id = 0;
    std::vector<std::uint8_t> code(m, 0);
    for (std::uint32_t li=0; li<nlist; ++li) {
        std::uint64_t sz = bp.avg_list;
        put64(sz);
        for (std::uint64_t i=0;i<sz;++i) {
            put64(cur_id++);
            // write m bytes
            size_t s=inv.size(); inv.resize(s + m);
            std::memset(inv.data()+s, static_cast<int>(i+li), m);
        }
    }
    if (bp.truncate_inverted && inv.size() > 8) inv.resize(inv.size()-5);
    append_section(3, inv.data(), inv.size(), false);

    // Trailer
    const char tail[4] = {'C','H','K','S'};
    std::uint64_t ch = fnv64(out.data(), out.size());
    append(tail, 4);
    if (bp.bad_trailer) ch ^= 0x2ull;
    append(&ch, 8);

    return out;
}

static void write_file(const fs::path& p, const std::vector<std::uint8_t>& bytes) {
    fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, size_t size) {
    Blueprint bp = blueprint_from_input(data, size);

    // Randomize mmap path via env var
#ifdef _WIN32
    _putenv_s("VESPER_IVFPQ_LOAD_MMAP", bp.mmap ? "1" : "0");
#else
    setenv("VESPER_IVFPQ_LOAD_MMAP", bp.mmap ? "1" : "0", 1);
#endif

    // Build a small v1.1 index file (possibly corrupted per blueprint toggles)
    auto bytes = build_v11(bp, data, size);

    // Create temp dir
    fs::path dir = fs::temp_directory_path() / fs::path("vesper_fuzz_ivfpq_v11");
    // Use process-unique subdir name from pointer value to avoid collisions
    std::uintptr_t tag = reinterpret_cast<std::uintptr_t>(data) ^ static_cast<std::uintptr_t>(size);
    dir /= std::to_string(tag);
    fs::create_directories(dir);

    // Write file
    write_file(dir / "ivfpq.bin", bytes);

    // Exercise loader; ignore outcome but must not crash/UB
    auto lr = IvfPqIndex::load(dir.string());
    if (lr.has_value()) {
        // Light-touch use of the index to exercise paths
        auto stats = lr->get_stats();
        (void)stats.n_vectors; // suppress unused warning
    }

    // Cleanup: best-effort
    std::error_code ec{};
    fs::remove_all(dir, ec);
    return 0;
}

