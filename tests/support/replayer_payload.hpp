#pragma once

/** \file replayer_payload.hpp
 *  \brief Test-only helpers to encode/decode toy WAL payloads and apply to a tiny in-memory index.
 *
 * Binary schema (little-endian):
 *  uint8 opcode; uint8[3] reserved=0; uint64 id; uint32 dim; uint32 ntags;
 *  if opcode==1 (UPSERT_VEC_TAGS): float32 vec[dim]; then ntags times { uint16 klen; uint16 vlen; bytes k; bytes v }
 *  if opcode==2 (DELETE_DOC): dim=0, ntags=0 and no body
 */

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace test_support {

enum : std::uint8_t { UPSERT_VEC_TAGS = 1, DELETE_DOC = 2 };

struct Doc {
  std::vector<float> vec;
  std::unordered_map<std::string, std::string> tags;
};
using ToyIndex = std::unordered_map<std::uint64_t, Doc>;

inline void le_store16(std::uint8_t* p, std::uint16_t v){ std::memcpy(p, &v, 2); }
inline void le_store32(std::uint8_t* p, std::uint32_t v){ std::memcpy(p, &v, 4); }
inline void le_store64(std::uint8_t* p, std::uint64_t v){ std::memcpy(p, &v, 8); }

inline std::vector<std::uint8_t> make_upsert(std::uint64_t id, std::span<const float> vec,
  std::initializer_list<std::pair<std::string,std::string>> tags)
{
  std::uint32_t dim = static_cast<std::uint32_t>(vec.size());
  // compute size
  std::size_t sz = 1 + 3 + 8 + 4 + 4 + dim*4;
  for (const auto& kv : tags) sz += 2 + 2 + kv.first.size() + kv.second.size();
  std::vector<std::uint8_t> out(sz);
  std::uint8_t* p = out.data();
  *p++ = UPSERT_VEC_TAGS; *p++=0; *p++=0; *p++=0;
  le_store64(p, id); p+=8;
  le_store32(p, dim); p+=4;
  le_store32(p, static_cast<std::uint32_t>(tags.size())); p+=4;
  // vec
  for (std::size_t i=0;i<dim;++i){ std::memcpy(p, &vec[i], 4); p+=4; }
  // tags
  for (const auto& kv : tags) {
    le_store16(p, static_cast<std::uint16_t>(kv.first.size())); p+=2;
    le_store16(p, static_cast<std::uint16_t>(kv.second.size())); p+=2;
    std::memcpy(p, kv.first.data(), kv.first.size()); p+=kv.first.size();
    std::memcpy(p, kv.second.data(), kv.second.size()); p+=kv.second.size();
  }
  return out;
}

inline std::vector<std::uint8_t> make_delete(std::uint64_t id){
  std::vector<std::uint8_t> out(1+3+8+4+4);
  std::uint8_t* p = out.data();
  *p++ = DELETE_DOC; *p++=0; *p++=0; *p++=0;
  le_store64(p, id); p+=8;
  le_store32(p, 0); p+=4; // dim
  le_store32(p, 0); p+=4; // ntags
  return out;
}

inline void apply_frame_payload(std::span<const std::uint8_t> pl, ToyIndex& idx){
  const std::uint8_t* p = pl.data();
  const std::uint8_t* e = pl.data() + pl.size();
  if (e - p < 1+3+8+4+4) return;
  std::uint8_t op = *p++; p+=3;
  std::uint64_t id; std::memcpy(&id, p, 8); p+=8;
  std::uint32_t dim; std::memcpy(&dim, p, 4); p+=4;
  std::uint32_t ntags; std::memcpy(&ntags, p, 4); p+=4;
  if (op == UPSERT_VEC_TAGS) {
    if (e - p < static_cast<std::ptrdiff_t>(dim)*4) return;
    Doc d; d.vec.resize(dim);
    for (std::size_t i=0;i<dim;++i){ std::memcpy(&d.vec[i], p, 4); p+=4; }
    for (std::size_t i=0;i<ntags;++i){
      if (e - p < 2+2) return; std::uint16_t kl, vl; std::memcpy(&kl, p, 2); p+=2; std::memcpy(&vl, p, 2); p+=2;
      if (e - p < kl + vl) return; std::string k(reinterpret_cast<const char*>(p), kl); p+=kl; std::string v(reinterpret_cast<const char*>(p), vl); p+=vl;
      d.tags.emplace(std::move(k), std::move(v));
    }
    idx[id] = std::move(d);
  } else if (op == DELETE_DOC) {
    idx.erase(id);
  }
}

} // namespace test_support

