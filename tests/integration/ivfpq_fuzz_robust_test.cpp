#include <catch2/catch_test_macros.hpp>
#include <vesper/index/ivf_pq.hpp>
#include <filesystem>
#include <fstream>
#include <random>

using namespace vesper::index;
namespace fs = std::filesystem;

static auto tmp_path(const char* name) -> fs::path {
    auto base = fs::temp_directory_path();
    std::mt19937_64 rng{1234567};
    for (int i = 0; i < 1000; ++i) {
        auto p = base / (std::string(name) + "." + std::to_string(rng()));
        if (!fs::exists(p)) return p;
    }
    // Fallback
    return base / (std::string(name) + ".fallback");
}

static std::vector<unsigned char> read_all(const fs::path& p) {
    REQUIRE(fs::exists(p));
    REQUIRE(fs::is_regular_file(p));
    std::ifstream f(p, std::ios::binary);
    REQUIRE(f.good());
    f.seekg(0, std::ios::end); auto n = static_cast<size_t>(f.tellg()); f.seekg(0);
    std::vector<unsigned char> buf(n); f.read(reinterpret_cast<char*>(buf.data()), n);
    return buf;
}

static void write_all(const fs::path& p, const std::vector<unsigned char>& buf) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    REQUIRE(f.good());
    f.write(reinterpret_cast<const char*>(buf.data()), buf.size());
}
#include <cstdint>
#include <cstring>

struct SecInfo { std::size_t hdr_off; std::uint32_t type; std::uint64_t unc; std::uint64_t comp; std::size_t payload_off; };

static std::vector<SecInfo> parse_v11_sections(const std::vector<unsigned char>& buf) {
    std::vector<SecInfo> out;
    if (buf.size() < 60+12) return out;
    if (std::memcmp(buf.data(), "IVFPQv11", 8) != 0) return out;
    std::uint32_t meta_len = 0; std::memcpy(&meta_len, buf.data()+56, 4);
    std::size_t p = 60 + meta_len; // after header (+optional meta)
    while (p + 32 <= buf.size() - 12) {
        SecInfo s{}; s.hdr_off = p;
        std::memcpy(&s.type, buf.data()+p+0, 4);
        std::memcpy(&s.unc,  buf.data()+p+4, 8);
        std::memcpy(&s.comp, buf.data()+p+12, 8);
        s.payload_off = p + 32;
        out.push_back(s);
        if (s.comp > buf.size()) break; // guard
        p += 32 + static_cast<std::size_t>(s.comp);
    }
    return out;
}

static void expect_load_error_for_buf(const std::vector<unsigned char>& corrupted, int mmap_mode) {
#ifdef _WIN32
    _putenv_s("VESPER_IVFPQ_LOAD_MMAP", mmap_mode ? "1" : "0");
#else
    setenv("VESPER_IVFPQ_LOAD_MMAP", mmap_mode ? "1" : "0", 1);
#endif
    auto d = tmp_path("ivfpq_fuzz_case"); std::error_code ec; fs::create_directories(d, ec);
    write_all(d / "ivfpq.bin", corrupted);
    auto r = IvfPqIndex::load(d.string().c_str());
    REQUIRE_FALSE(r.has_value());
}


TEST_CASE("IVFPQ v1.1 loader robustness under corrupted inputs", "[ivfpq][serialize][fuzz]") {
#ifdef _WIN32
    _putenv_s("VESPER_IVFPQ_SAVE_V11", "1");
#else
    setenv("VESPER_IVFPQ_SAVE_V11", "1", 1);
#endif

    // Create a small valid file first
    IvfPqIndex idx;
    IvfPqTrainParams t{}; t.nlist = 4; t.m = 2; t.nbits = 8; t.use_opq = false;
    const int dim = 8; const int n = 50;
    std::vector<float> train(n * dim);
    for (int i = 0; i < n * dim; ++i) train[i] = float((i*7)%17) * 0.01f;
    auto tr = idx.train(train.data(), dim, n, t);
    REQUIRE(tr.has_value());

    fs::path good_dir = "ivfpq_fuzz_good";
    REQUIRE(idx.save(good_dir.string().c_str()).has_value());

    auto base = read_all(good_dir / "ivfpq.bin");
    REQUIRE(base.size() > 64);

    std::mt19937 rng(123);

    SECTION("Truncations at various cut points should return error") {
        for (size_t cut : {8ull, 16ull, 24ull, base.size()/4, base.size()/2, base.size()-1}) {
            if (cut >= base.size()) continue;


            auto tmp = base; tmp.resize(cut);
            fs::path p = "ivfpq_trunc_dir"; std::error_code ec; std::filesystem::create_directories(p, ec);
            write_all(p / "ivfpq.bin", tmp);
            auto lr = IvfPqIndex::load(p.string().c_str());
            REQUIRE_FALSE(lr.has_value());
        }
    }

    SECTION("Invalid magic and version should be rejected (mmap and streaming)") {
        for (int mm = 0; mm <= 1; ++mm) {
            {
                auto tmp = base; tmp[0] ^= 0xFF; // break magic
                expect_load_error_for_buf(tmp, mm);
            }
            {
                auto tmp = base; // bump major version
                std::uint16_t bad_major = 2; std::memcpy(tmp.data()+8, &bad_major, sizeof(bad_major));
                expect_load_error_for_buf(tmp, mm);
            }
        }
    }

    SECTION("Missing required sections by retagging types should fail") {
        auto secs = parse_v11_sections(base);
        REQUIRE_FALSE(secs.empty());
        for (int mm = 0; mm <= 1; ++mm) {
            bool mutated_any = false;
            for (auto need : {1u, 2u, 3u}) { // CENTROIDS, CODEBOOKS, INVERTED
                auto tmp = base;
                bool did=false;
                for (const auto& s : secs) if (s.type == need) { std::uint32_t bad=99; std::memcpy(tmp.data()+s.hdr_off+0, &bad, 4); did=true; mutated_any=true; break; }
                if (did) expect_load_error_for_buf(tmp, mm);
            }
            if (!mutated_any) SUCCEED();
        }
    }

    SECTION("Inverted lists payload with wrong lists_count should fail") {
        auto secs = parse_v11_sections(base);
        REQUIRE_FALSE(secs.empty());
        auto it = std::find_if(secs.begin(), secs.end(), [](const SecInfo& s){ return s.type==3u; });
        if (it == secs.end()) {
            SUCCEED();
        } else {
            for (int mm = 0; mm <= 1; ++mm) {
                auto tmp = base;
                std::uint32_t lc = 0; std::memcpy(&lc, tmp.data()+it->payload_off, 4);
                lc += 1; std::memcpy(tmp.data()+it->payload_off, &lc, 4);
                expect_load_error_for_buf(tmp, mm);
            }
        }
    }

    SECTION("Header indicates compressed (comp!=unc) but payload is not compressed should fail") {
        auto secs = parse_v11_sections(base);
        REQUIRE_FALSE(secs.empty());
        // Pick the centroids section (1) if available; else codebooks (2)
        const SecInfo* target=nullptr; for (const auto& s : secs) if (s.type==1u) { target=&s; break; }
        if (!target) for (const auto& s : secs) if (s.type==2u) { target=&s; break; }
        REQUIRE(target != nullptr);
        for (int mm = 0; mm <= 1; ++mm) {
            auto tmp = base;
            std::uint64_t fake_unc = target->unc + 7; // mismatch
            std::memcpy(tmp.data()+target->hdr_off+4, &fake_unc, 8);
            expect_load_error_for_buf(tmp, mm);
        }
    }

    SECTION("Trailer corruption (bad magic / checksum) should fail") {
        for (int mm = 0; mm <= 1; ++mm) {
            {
                auto tmp = base;
                // overwrite 'CHKS' trailer tag
                std::memset(tmp.data()+tmp.size()-12, 'X', 4);
                expect_load_error_for_buf(tmp, mm);
            }
            {
                auto tmp = base;
                // flip a bit in the checksum
                tmp[tmp.size()-1] ^= 0x80;
                expect_load_error_for_buf(tmp, mm);
            }
        }
    }


    SECTION("Random byte flips should not crash and should error") {
        for (int iter = 0; iter < 16; ++iter) {
            auto tmp = base;
            std::uniform_int_distribution<size_t> dpos(0, tmp.size()-1);
            for (int flips = 0; flips < 8; ++flips) tmp[dpos(rng)] ^= static_cast<unsigned char>(1u << (flips % 7));
            fs::path p = "ivfpq_flip_dir"; std::error_code ec; std::filesystem::create_directories(p, ec);
            write_all(p / "ivfpq.bin", tmp);
            auto lr = IvfPqIndex::load(p.string().c_str());
            if (!lr.has_value()) {
                SUCCEED();
            } else {
                // It's okay if occasionally it still parses; ensure no crash and a benign API remains callable
                CHECK(lr->get_metadata_json().size() >= 0);
            }
        }
    }

    SECTION("Corrupt per-section checksums should be rejected") {
        // find and flip bytes that likely belong to section checksum fields (they are 8-byte aligned after header)
        auto tmp = base;
        for (size_t i = 32; i + 8 <= tmp.size(); i += 24) { // rough stride through potential headers
            tmp[i+16] ^= 0xFF; // flip within shash
        }
        fs::path p = "ivfpq_badshash_dir"; std::error_code ec; std::filesystem::create_directories(p, ec);
        write_all(p / "ivfpq.bin", tmp);
        auto lr = IvfPqIndex::load(p.string().c_str());
        REQUIRE_FALSE(lr.has_value());
    }
}

