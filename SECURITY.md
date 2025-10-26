# Security Policy

Thank you for helping keep Vesper and its users safe.

## Reporting a vulnerability
- Email: colin@teraflux.app
- Please include a detailed description, steps to reproduce, affected versions/OS/compilers, and any PoC or logs.
- Do not create public issues for sensitive reports.

## Scope
- Vesper code in this repository (source, docs, build scripts).
- Supply chain concerns related to our build and release process.

## Safe disclosure & handling
- We will acknowledge reports within 3 business days and provide a timeline for remediation when possible.
- We prefer coordinated disclosure; we will request a reasonable embargo until a fix is available.

## Key security areas (reference)
- Crash‑safety and durability (WAL integrity, fsync discipline)
- At‑rest encryption (XChaCha20‑Poly1305; optional AES‑GCM for FIPS mode)
- No network IO by default; telemetry is opt‑in only
- Prompt security for any generated artifacts (see prompt-blueprint.md)

## Credits
- Please let us know how you’d like to be credited. We maintain a SECURITY-ACKS section in release notes when applicable.

