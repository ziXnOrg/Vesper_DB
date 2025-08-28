# Error Taxonomy Reference

This reference summarizes error codes, categories, typical causes, and remediation. See ADR-0005 and include/vesper/error.hpp.

| Code | Name | Category | Typical causes | Remediation |
|---|---|---|---|---|
| 0 | ok | - | success | n/a |
| 1001 | io_failed | IO | fsync/rename failure; permission denied | retry with backoff; check permissions/disk |
| 1002 | io_eof | IO | truncated read; end of file | treat as boundary; validate length |
| 2001 | config_invalid | Config | bad parameter; unknown metric | validate inputs; provide defaults |
| 3001 | data_integrity | DataIntegrity | checksum mismatch; corrupt WAL frame | reject frame; continue recovery |
| 4001 | precondition_failed | Precondition | null pointer; dim==0; non-finite values | validate preconditions; return error |
| 5001 | resource_exhausted | Resource | OOM; file descriptors exhausted | reduce concurrency; increase limits |
| 6001 | not_found | NotFound | id missing; file missing | return 404-like error; ensure id exists |
| 7001 | unavailable | Unavailable | transient fs or system error | retry later; backoff |
| 8001 | cancelled | Cancelled | user cancelled operation | propagate cancellation cleanly |
| 9001 | internal | Internal | invariant violation; unexpected state | log + fail fast in debug; return error |

Notes
- Error codes are stable and part of the ABI contract; extend by adding new values at the end of ranges.
- Logging policy: JSON lines with {code, message, component, context}; see ADR-0005.

