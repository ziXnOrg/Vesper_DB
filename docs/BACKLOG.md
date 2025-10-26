# Initial Backlog (Phases P0–P9)

This document is the single text index of issues/epics aligned with `prompt-dev-roadmap.md`. Create GitHub Issues from these items and link back here.

## Phase 0 — Bootstrap (NOW)
- [ ] Repo skeleton and directory contracts
  - Acceptance: repo.yaml drafted; directories and invariants documented
- [ ] Schemas for prompt artifacts (prompt.manifest.json, experiment.yaml, eval.report.json)
  - Acceptance: JSON Schemas validate; examples provided
- [ ] CI scaffolding (lint → build → tests → evals)
  - Acceptance: pipeline runs on PR; placeholders for eval jobs present
- [ ] CONTRIBUTING and style configs (.clang-format, .clang-tidy, .editorconfig)
  - Acceptance: local linters run clean on repo

## Phase 1 — Architecture (NOW)
- [ ] ADR set (adr-*.md) capturing key decisions
  - Acceptance: ADRs reference spec pack; reviewers sign‑off
- [ ] Component map and non‑functional SLOs
  - Acceptance: Mermaid diagram + textual LLD; measurable SLOs

## Phase 2 — Algorithms Spec (NOW)
- [ ] Deterministic pseudocode for IVF‑PQ/OPQ, HNSW, Disk‑graph, filters
  - Acceptance: pre/post‑conditions, complexity, edge cases
- [ ] Test fixtures/oracles and numerical tolerances
  - Acceptance: seeds and ULP bounds defined

## Phase 3 — API & ABI (SEQ)
- [ ] Public headers (include/vesper/*.hpp) and stable C ABI (vesper_c.h)
  - Acceptance: headers compile with -pedantic; ABI free of STL
- [ ] Error taxonomy (errors.md)
  - Acceptance: codes, remediation, logging levels

## Phase 4 — Tests First (SEQ)
- [ ] Unit, property, fuzz, and micro‑bench suites
  - Acceptance: tests compile/run; CI green

## Phase 5 — Implementation (SEQ)
- [ ] Module implementations to satisfy tests and SLOs
  - Acceptance: tests + benches pass; SLOs met

## Phase 6 — Docs (SEQ)
- [ ] API reference (Doxygen) and Operator guides
  - Acceptance: examples compile; links resolve

## Phase 7 — Evals & Safety (SEQ)
- [ ] HELM/PromptBench/GAIA runbooks and policies
  - Acceptance: gates configured; regressions blocked

## Phase 8 — Performance & Serving (LATER)
- [ ] Serving configs and load tests (optional)
  - Acceptance: p50/p99 and cost recorded

## Phase 9 — Release (SEQ)
- [ ] Reproducible builds, SBOM, signatures, CHANGELOG
  - Acceptance: artifacts signed; SPDX SBOM present

## Labels and ownership
- Labels: `phase:P0`..`phase:P9`, `area:storage`, `area:index`, `area:simd`, `area:api`, `area:docs`, `kind:bug`, `kind:feature`.
- Owners: TBD; assign in GitHub issues.

## How to create issues
- Use the templates under `.github/ISSUE_TEMPLATE/` (if present) or copy sections above into a new issue.

