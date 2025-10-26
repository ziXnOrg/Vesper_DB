# Batch 1 — Index Manager + Query Planner: Thread-safety, Determinism, Plan Integration

Status: COMPLETE (H1, H2, H3)

## Overview
Batch 1 focuses on integrating the QueryPlanner with IndexManager, making the planner thread-safe, and adding a determinism “frozen mode” to support reproducible runs.

- H1: Apply tuned plan.config produced by QueryPlanner
- H2: Make QueryPlanner thread-safe (read-mostly; counters via atomics)
- H3: Determinism mode (frozen mode) with environment toggle

Cross references:
- Code: src/index/query_planner.cpp, include/vesper/index/index_manager.hpp
- Tests: tests/unit/index_manager_planner_tests.cpp

## H1 — Apply tuned plan.config
Goal: Ensure IndexManager honors planner-produced parameters when use_query_planner=true.

Key changes
- index_manager.cpp: apply QueryPlanner::QueryPlan::config as the effective search configuration when planner is enabled.
- Helper extracted for clarity (apply_query_plan()).

Validation
- Added 4 tests:
  - Planner config applied for HNSW
  - Planner config applied for IVF-PQ
  - Planner config applied for DiskANN
  - Planner rerank params are applied (IVF-PQ)
- All H1 tests pass.

## H2 — Thread-safety
Goal: Safe concurrent plan() and update_stats() without data races.

Synchronization design
- Atomics: plans_generated_, plans_executed_ (std::atomic<uint64_t>, acq_rel semantics)
- Shared state: cost model, history, aggregates guarded by std::shared_mutex
  - plan(): shared_lock (read-only)
  - update_stats(): unique_lock (write)
  - get_stats(): shared_lock (read) + atomic counters

Validation
- Added 3 concurrency tests:
  - planner concurrent plan calls
  - planner concurrent plan and update_stats
  - planner property-based randomized concurrent ops (TSan)
- All H2 tests pass.

## H3 — Determinism mode (frozen)
Goal: Optional mode to make planning fully deterministic and disable adaptive tuning for reproducible runs.

Mechanism
- Environment variable: VESPER_PLANNER_FROZEN
  - "1" → enable frozen mode; other values/unset → disabled
  - Accessed via vesper::core::safe_getenv("VESPER_PLANNER_FROZEN")
  - Read once in QueryPlanner::Impl constructor; cached as const bool (frozen_mode_)
- Behavior when frozen:
  - plan(): uses fixed deterministic logic and stable tie-breaking (no adaptive reads)
  - update_stats(): early return after incrementing counter; skips adaptive aggregates
  - get_stats(): counters reflect usage; avg aggregates remain unchanged
- Tie-breaking policy (stable, deterministic): HNSW > IVF_PQ > DiskANN
  - Matches IndexManager::get_active_indexes() order

Performance
- Frozen mode avoids locking in update_stats() (no shared-state writes), slightly reducing overhead on heavy update paths.

Tests (H3)
- planner deterministic plan with fixed seed (n=50k; Hybrid)
- planner frozen mode no adaptive updates (n=5k; Manual HNSW)
- planner deterministic index selection order (n=50k; Hybrid)

Acceptance criteria
- Frozen mode test passes: aggregates unchanged; plans stable across updates; counters increment
- Deterministic plan with fixed seed passes: two planners produce identical plans repeatedly
- Deterministic index selection order passes: stable index choice across iterations

Runtime considerations
- Debug + MSVC + Hybrid at 50k vectors is slow (minutes). Tests intentionally reduced from 200k→50k for RED/CI viability.

Results (latest local)
- All 10 planner tests pass (4 H1 + 3 H2 + 3 H3)
- No new warnings introduced by H3 (existing third‑party croaring C4244 warnings unchanged)

Usage examples
- Windows PowerShell:
  - $env:VESPER_PLANNER_FROZEN = "1"
  - .\index_manager_planner_tests.exe '[planner][determinism]' --durations yes --order lex
- POSIX shells:
  - export VESPER_PLANNER_FROZEN=1
  - ./index_manager_planner_tests '[planner][determinism]' --durations yes --order lex

Commands (developer convenience)
- Build tests (Debug):
  - cmake --build build --config Debug --target index_manager_planner_tests -j
- Run tests:
  - build/Debug/index_manager_planner_tests.exe '[index_manager][planner]' --durations yes --order lex
  - build/Debug/index_manager_planner_tests.exe '[planner][concurrency]' --durations yes --order lex
  - build/Debug/index_manager_planner_tests.exe '[planner][determinism]' --durations yes --order lex

Notes
- Determinism relies on stable ordering from IndexManager::get_active_indexes()
- safe_getenv() centralizes environment handling and is required by coding standards
- The frozen flag is intentionally cached for thread-safety and to avoid per-call env reads

