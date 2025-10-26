# ADR-0002: Storage Engine (WAL, Snapshots, Recovery)

Status: Proposed
Date: 2025-08-28

## Context
We require strong crash-safety and fast recovery without external services. Writes must be durable; publishes must be atomic; recovery must be deterministic and idempotent.

## Decision
Implement an append-only WAL with checksummed frames and explicit commits; immutable segments are published via atomic rename with parent directory fsync. Snapshots provide point-in-time recovery.

## Consequences
- Group commit with `fdatasync()` meets durability
- Recovery scans WAL verifying checksums, replays idempotently
- Segment compaction is staged then atomically published


## Traceability to blueprint.md
- §6.1 Files & Layout → directory structure
- §6.2 WAL Format → framing, checksums, fsync policy
- §6.3 Snapshots & Compaction → publish semantics and staging

## References
- `docs/blueprint.md#6-storage-engine-persistence-crash-safety`

