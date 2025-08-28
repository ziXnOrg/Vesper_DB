# Contributing to Vesper

We use a deterministic, tests‑first, prompt‑first workflow. All prompt‑generated artifacts must be reproducible and evaluated. Please read this document before opening PRs.

## Branching and PR flow
- Create feature branches off `main`.
- Keep PRs small and atomic; one logical change per PR.
- Link each PR to a BACKLOG item; reference roadmap phase labels (P0–P9).
- Require green CI and at least one reviewer approval (two for core modules).

## Commit messages
- Short, imperative subject (≤ 72 chars), optional scope: `core: fix wal checksum on rollover`.
- Body includes rationale and links to issues/backlog.

## Deterministic prompt workflow (authoritative)
- Decoding must be fixed: `temperature=0.0`, `top_p=1.0`, fixed `seed`, `n=1`.
- Attach to the PR: the exact prompt(s), decoding params, and model/version hash.
- Output must conform to requested schemas (no extra prose). Use stop sequences as needed.
- See `prompt-blueprint.md` and `prompt-dev-roadmap.md` for eval gates and sequencing.

## Schema and CI gates
- Schemas (to be added in Phase 0) must validate.
- CI stages (lint → build → tests → evals) must pass; regressions fail the PR.
- Prompt changes must include eval diffs and pass robustness gates where applicable.

## Local checks
- Format and lint: `.clang-format`, `.clang-tidy` (see Coding Standards).
- Build and test: follow `docs/SETUP.md`.
- Run unit/property tests locally before opening PRs.

## Code review policy
- Staff‑level quality bar for core paths; performance and correctness first.
- Reviewers may request micro‑benchmarks for hot paths; add evidence.

## Security disclosure
- Please report vulnerabilities privately to: security@example.com (placeholder; update before first release).

## Licensing and CLA/DCO
- License is TBD; if a CLA/DCO is required, link and instructions will be added prior to first release.

