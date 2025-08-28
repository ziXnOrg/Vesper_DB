
# Vesper Threat Model & Security Notes (v0)

## Assets
- Vector data, metadata, manifests, WAL logs, snapshots, encryption keys.

## Adversaries
- Local attacker with filesystem access.
- Disk corruption / partial writes / power loss.
- Misconfiguration (nonce reuse, weak keys).

## Controls
- **Integrity:** per‑block CRC32C/xxHash64; AEAD tags if encryption enabled; fail‑closed read paths.
- **Confidentiality:** optional at‑rest encryption via XChaCha20‑Poly1305 Secretstream (libsodium). Envelope keys via OS KMS/TPM/HSM; Argon2id for passphrases.
- **Durability:** POSIX `fsync` on WAL commit, atomic `rename()` on publish; parent dir `fsync` to persist metadata.
- **Recovery:** WAL checksums and length framing; ignore torn/truncated frames; idempotent replay.
- **Supply chain:** reproducible builds; vendored crypto; FIPS‑friendly mode (AES‑GCM) with strict nonce management.
- **Telemetry privacy:** opt‑in only; no network IO by default.

## Validation
- Fault‑injection tests (power‑fail), bit‑flip simulations, fuzzing of parsers, KATs for crypto.
