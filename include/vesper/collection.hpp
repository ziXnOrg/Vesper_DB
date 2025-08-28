#pragma once

/**
 * \file collection.hpp
 * \brief Public C++ API: collections, search parameters, and results.
 *
 * Thread-safety: read operations are thread-safe; mutating operations require a single writer
 * per collection (see ADR-0004). No exceptions are thrown along hot paths; errors are
 * propagated via std::expected.
 *
 * Ownership & lifetime:
 * - Callers own input buffers (query, vectors); pointers must remain valid for the duration
 *   of the call. The library does not retain pointers beyond the call boundary.
 * - Returned containers (e.g., search results) are value-owned by the caller.
 *
 * Complexity (typical): see blueprint.md §11 for planner details and per-index behavior.
 */

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "vesper/error.hpp"

namespace vesper {

/** \brief Parameters controlling search behavior. */
struct search_params {
  /** metric name: "l2" | "ip" | "cosine" */
  std::string metric;
  /** top-k to return */
  std::uint32_t k{10};
  /** target recall guidance to the planner */
  float target_recall{0.95f};
  /** IVF probing depth */
  std::uint32_t nprobe{8};
  /** Graph beam width */
  std::uint32_t ef_search{64};
  /** Optional exact re-rank size */
  std::uint32_t rerank{0};
};

/** \brief One search hit. */
struct search_result {
  std::uint64_t id{};   /**< document/vector id */
  float score{};        /**< distance or similarity depending on metric */
};

struct filter_expr; // fwd-decl

/** \brief A collection of vectors and metadata stored on disk. */
class collection {
public:
  /**
   * \brief Open or create a collection at a filesystem path.
   * \param path Filesystem directory path (UTF-8)
   * \return collection on success; error otherwise
   * Preconditions: parent directory exists and is writable.
   */
  static auto open(const std::string& path) -> std::expected<collection, core::error>;

  /**
   * \brief Insert or upsert a vector (with optional metadata).
   * \param id user-supplied id (64-bit). If 0, an id may be auto-assigned in future revisions.
   * \param vec pointer to float32 buffer of length dim
   * \param dim vector dimensionality
   * \return success or error
   * Preconditions: vec != nullptr, dim > 0; finite values; collection open for writes.
   */
  auto insert(std::uint64_t id, const float* vec, std::size_t dim /*, metadata TBD*/)
      -> std::expected<void, core::error>;

  /** \brief Remove (tombstone) a vector by id. */
  auto remove(std::uint64_t id) -> std::expected<void, core::error>;

  /**
   * \brief Search a query vector with optional filter.
   * \param query pointer to float32 buffer (length dim)
   * \param dim vector dimensionality
   * \param p search parameters
   * \param filter optional filter expression (may be null)
   * \return top-k results on success; error otherwise
   * Thread-safety: safe for concurrent calls across threads.
   */
  auto search(const float* query, std::size_t dim, const search_params& p,
              const filter_expr* filter)
      -> std::expected<std::vector<search_result>, core::error>;

  /** \brief Seal the current mutable segment; non-blocking to readers. */
  auto seal_segment() -> std::expected<void, core::error>;
  /** \brief Compact sealed segments; crash-safe staged publish. */
  auto compact() -> std::expected<void, core::error>;
  /** \brief Create a point-in-time snapshot. */
  auto snapshot() -> std::expected<void, core::error>;
  /** \brief Recover by loading snapshot and replaying WAL. */
  auto recover() -> std::expected<void, core::error>;
};

} // namespace vesper

