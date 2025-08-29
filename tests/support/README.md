# Test support helpers

- wal_replay_helpers.* provides a deterministic utility used by WAL replay tests to:
  1) apply baseline (LSN ≤ cutoff) via per-file scan in ascending sequence order
  2) apply post-cutoff (LSN > cutoff) via recover_replay(dir, ...)

Prefer using this helper in WAL replay tests to avoid duplicating baseline+replay plumbing.

