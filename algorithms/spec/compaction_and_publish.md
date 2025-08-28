# Compaction and Snapshot Publish — Pseudocode Spec

## Compaction
```
Input: sealed segments S, tombstones T
Output: new segment S'

1: merge postings & tombstones
2: rebuild filter bitmaps
3: rewrite index files (family-specific)
4: stage to temp path; fsync each file
5: atomic rename to publish; fsync parent dir
```

## Recovery
```
1: load latest snapshot, then WAL replay to LSN
2: ignore torn/truncated frames (checksum/length)
3: reconstruction is idempotent
```

## Failure modes
- Power loss mid-publish → safe due to staging + atomic rename
- Corrupted WAL frame → ignored due to checksum; recovery continues


## Preconditions / Postconditions
- Preconditions: all staged files written and fsynced
- Postconditions: publish via atomic rename; parent dir fsynced

## Edge cases & fallbacks
- Partial publish due to crash → on startup, ignore temp paths and keep last atomically published snapshot

## References
- docs/blueprint.md §6.3 Snapshots & Compaction

