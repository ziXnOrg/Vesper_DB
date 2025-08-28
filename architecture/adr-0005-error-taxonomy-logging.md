# ADR-0005: Error Taxonomy and Logging Levels

Status: Proposed
Date: 2025-08-28

## Context
We need a consistent, portable error model and logging guidance that supports crash-safety, reproducibility, and performance diagnostics without exceptions on hot paths.

## Decision
- Use `std::expected<T, Error>` for recoverable failures on hot paths.
- Define an error taxonomy with stable codes and categories: `IO`, `Config`, `DataIntegrity`, `Precondition`, `Resource`, `NotFound`, `Unavailable`, `Cancelled`, `Internal`.
- Map to logging levels: `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR`.
- Structure logs as JSON lines for machine parsing; include `code`, `message`, `component`, and `context` fields.

## Error code shape (sketch)
```cpp
enum class error_code : uint32_t {
  ok = 0,
  io_failed = 1001,
  io_eof = 1002,
  config_invalid = 2001,
  data_integrity = 3001,
  precondition_failed = 4001,
  resource_exhausted = 5001,
  not_found = 6001,
  unavailable = 7001,
  cancelled = 8001,
  internal = 9001,
};
```

## Logging policy
- TRACE: perf counters, kernel iterations (disabled by default)
- DEBUG: detailed planner/storage traces under sampling

## Traceability to blueprint.md
- §15 Observability & Tooling → structured logs, counters, histograms
- §16 Testing & Verification → error handling in tests and gates

- INFO: lifecycle events (open, seal, snapshot publish)
- WARN: transient issues with retries/backoff
- ERROR: non-recoverable failures; include remediation hints

## Consequences
- Portable and testable without exceptions
- Stable error codes enable programmatic handling and telemetry aggregation

## References
- `docs/blueprint.md#12-concurrency-correctness`, `#16-testing-verification`
- `docs/CODING_STANDARDS.md`

