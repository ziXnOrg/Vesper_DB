## Prompt Title
Augment Code Global User Guidelines — Deep Research and Modernization (Framework‑Agnostic, Production‑Ready)

## Role and Mode
You are GPT‑5 Pro operating in Deep Research mode as a principal engineer and staff architect. Your job is to modernize and strengthen the Global User Guidelines for Augment Code, making them state‑of‑the‑art for agentic software development and production engineering across languages and platforms. You must perform comprehensive web research, synthesize best practices, and return a single consolidated, implementation‑ready guidelines document with evidence and rationale.

Important:
- Do not assume access to local files or prior context.
- Perform web research for every claim that is not obviously standard; cite reputable sources with links.
- Keep the guidelines framework‑agnostic and vendor‑neutral (do not tailor to a specific product or repository).
- Treat agentic/LLM workflows as first‑class citizens integrated with traditional engineering rigor.

## Embedded Baseline (to be enhanced by research)
Below is the initial guideline baseline you must refine and enhance. Use it as the starting point—strengthen, correct, and extend it with web‑sourced evidence and modern practices.

0) Scope and Intent
- Apply to planning, design, coding, testing, reviews, CI/CD, and release.
- Default to conservative, auditable, deterministic behavior; favor correctness, security, reliability.
- Prioritize reproducibility, evidence‑backed decisions, and cross‑platform compatibility.
- Prefer header‑only when practical, runtime capability detection, and safe‑by‑default configurations.

1) Research and Planning Methodology
- Context acquisition: read docs/requirements/ADRs; list stakeholders, constraints, acceptance criteria; surface unknowns/risks; review CI/CD and quality gates.
- First principles and alternatives: ≥2 viable approaches (≥3 for architectural); analyze complexity, performance, determinism, security, reliability, maintainability, cross‑platform; record selection criteria and rejected options.
- ADR‑driven workflow: author ADRs for architectural/cross‑cutting changes (context, options, decision, consequences, compatibility/migration, references). Keep versioned and link in PRs.
- Implementation planning: define modules/files, interfaces/contracts, error model, invariants, determinism requirements; plan tests/benchmarks/verification before coding; specify perf budgets and memory bounds; document risks/mitigations/rollback; include cross‑platform matrix (Windows/Linux/macOS).

2) Code Quality Standards
- General: strong typing; explicit ownership/lifetimes; no “magic”; immutability‑by‑default where practical; explicit results/errors with actionable messages; strict input validation; avoid global state; testable, composable, minimal APIs; header‑only where feasible with runtime capability detection.
- Performance: zero‑waste (avoid needless alloc/copies/indirection); cache‑aware layouts; record Big‑O for hot paths; document non‑obvious costs; SIMD backends with runtime selection; memory‑bound awareness.
- Concurrency: prefer message passing; if shared memory, document synchronization and thread‑safety; explicit memory ordering; avoid relaxed in control paths; define linearization points; safe reclamation (epochs/hazard pointers) where applicable.
- Cross‑platform: runtime feature detection with safe guards; scalar fallback end‑to‑end; test matrix (Windows MSVC 2022, Linux GCC/Clang, macOS Apple Clang); prevent illegal‑instruction hazards.
- API Design: stable, versioned APIs; deprecate with timelines; document purpose, I/O, error model, side effects, complexity, examples; constexpr helpers where appropriate.
- Documentation: comment rationale/invariants/security & perf caveats; no TODOs/PLACEHOLDERs in production; Doxygen‑style docs for public APIs.

3) Testing and Verification Requirements
- Coverage and scope: unit, integration, property‑based, end‑to‑end; coverage gates ≥85% overall, ≥90% for core algorithms/concurrency; include negative paths (malformed/bounds/timeouts/OOM/concurrency edges); validate cross‑platform matrix.
- Determinism & reproducibility: fixed seeds; mock/virtual time; hermetic tests; stable ordering/snapshots; define determinism acceptance (bit‑exact integers; bounded epsilon/ULP for FP); golden files for serialization; backend parity (SIMD vs scalar must match).
- Property‑based testing: use language‑appropriate frameworks; encode invariants/bounds; enable shrinking; capture minimal counterexamples.
- Performance testing: stable scenarios and manifests; record hardware/toolchain; block/flag >5% regression (CPU time, peak memory, critical latency) without approved rationale; variance‑bounds per OS (Linux ≤0.10, macOS ≤0.20, Windows ≤0.25).

4) CI/CD and Quality Gates
- Pipeline: Lint → Build (Debug+Release) → Unit/Integration/Property → Sanitizers → Coverage → Benchmarks → Cross‑Platform Matrix.
- Zero warnings; treat as errors; fail CI on violations. Multi‑platform builds: ubuntu‑latest, windows‑latest, macos‑latest (minimum).
- Static analysis and sanitizers: enable comprehensive analyzers (e.g., Clang‑Tidy) and sanitizers (ASan/TSan/UBSan). No new violations without rationale; justify any suppression.
- Performance regression detection: automated benchmarks with NDJSON output + schema validation; platform‑specific tolerance; baseline refresh procedures.
- Policy/capability checks: validate runtime capability detection; environment overrides; force‑scalar mode.

5) Security and Reliability Practices
- Threat modeling: define trust boundaries/attacker models/abuse cases for external interfaces; supply chain security (pin/lockfiles, CVE scans, checksums/signatures).
- Input/data handling: strict validation/sanitization; safe parsing; fail closed; least privilege; never log secrets.
- Reliability/resilience: crash‑safety (atomic or recoverable state changes); long‑running ops with timeouts/backoff/retry/checkpoints/graceful shutdown/resource bounds.
- Observability: structured logs, metrics, traces; correlation IDs; capture environment manifests (toolchain/driver versions, seeds) for reproducibility.

6) PR and Commit Standards
- PR description includes: research summary with references; architecture mapping & interface contracts; first principles (invariants/budgets); alternatives with trade‑offs; decision rationale (ADR link where applicable); tests/benchmarks added with coverage impact; cross‑platform verification; performance impact; rollback plan if risk non‑trivial.
- Conformance checklist: style compliance; zero warnings; sanitizers pass; static analysis clean; coverage gates met; cross‑platform builds pass; performance regressions ≤5% or rationale approved; determinism verified (goldens, backend parity); security review for interface changes; docs updated.
- Artifacts: coverage reports (by platform); benchmark outputs (NDJSON) + manifests; golden file results; cross‑platform build logs; perf regression analysis.

7) Determinism and Reproducibility Policy
- When determinism is required: stable iteration orders; fixed seeds; avoid wall‑clock/time dependence; prefer integer/fixed‑point for critical logic; document epsilon/ULP bounds for FP; snapshot/hash cadence and schema; SHA‑256 manifests for byte‑exact reproducibility.
- Cross‑platform determinism: backend parity (SIMD vs scalar); compiler‑agnostic (GCC/Clang/MSVC); platform‑agnostic (Windows/Linux/macOS) where feasible.

8) Performance and Benchmarking
- Budgets: define latency/throughput/memory/power budgets; micro & macro benchmarks with stable inputs; record HW/OS/toolchain manifests; report medians and dispersion; consider thread affinity.
- Regression detection: automate in CI with platform‑specific tolerances; NDJSON schema validation; capture environment manifests.

9) Documentation Standards
- API documentation: purpose, I/O, error model, side effects; hot‑path complexity; examples; constraints/invariants; determinism/precision modes; known limitations; cross‑platform notes.
- Implementation docs: comprehensive plans with validation matrices; runtime dispatch architecture docs; performance characteristics and optimization notes.

10) Agentic AI/Augment Agent Behavior & Execution Policy
- Task management: start with investigation; maintain minimal tasklist for multi‑file work; one task IN_PROGRESS; batch updates; summarize progress/next steps; phased implementation with clear validation gates.
- Safe‑by‑default verification: run tests/linters/builds/benches that are safe and do not modify external state; do not install dependencies/deploy/run destructive commands without permission.
- Cross‑platform awareness: always consider Windows/Linux/macOS; test runtime capability detection/fallback paths; validate environment override mechanisms.
- Evidence‑first communication: summarize what was run, where, exit codes, key logs; attach artifacts; provide cross‑platform validation results; propose minimal fixes with platform‑specific considerations.

11) Exceptions and Governance
- If constraints cannot be met: document trade‑offs, mitigation, follow‑up plan with owners/timelines; for urgent patches, record a stability waiver (scope, risk, rollback) and schedule cleanup with platform verification.
- Continuous improvement: regularly audit gates/thresholds and raise standards over time; maintain baseline refresh procedures for performance and compatibility testing.

## Research Mandate (You must perform these web searches and synthesis)
For each item below, perform web research, collect 6–12 authoritative sources (standards, well‑maintained OSS, vendor docs, academic/industry reports), and synthesize actionable enhancements to the baseline above. Cite sources inline [like this]. Prioritize recency (2023–2025) and durability.

A) Agentic Systems and Orchestration (state of the art)
- Web search: agentic multi‑agent orchestration patterns 2024 2025
- Web search: hierarchical vs graph vs swarm agents best practices
- Web search: LLM tool‑use safety policies and execution sandboxes
- Web search: agent memory/context management CRDT event sourcing OT comparison
- Web search: retrieval‑augmented generation reliability patterns production
- Web search: evaluation harnesses for agents (task‑oriented evals, reproducibility)
Deliver: concrete patterns, failure modes, governance hooks, and implementation checklists suitable for production.

B) Testing, Determinism, and Reproducibility for AI Systems
- Web search: determinism in ML/LLM inference temperature sampling reproducibility
- Web search: golden file testing serialization schemas versioning
- Web search: cross‑platform floating point reproducibility techniques
- Web search: property‑based testing frameworks by language 2024 2025
- Web search: flaky test mitigation CI practices
Deliver: refined determinism policy, language‑specific frameworks, FP tolerance guidance, golden‑file schemas, and CI tactics.

C) CI/CD for Agentic Software
- Web search: CI CD best practices 2025 cross platform matrix
- Web search: static analysis and sanitizers modern guidance C++ Rust Python TS
- Web search: performance regression detection NDJSON schema benchmarks
- Web search: OpenTelemetry logs metrics traces standard updates 2024 2025
- Web search: supply chain security SLSA SBOM 2025
Deliver: concrete pipeline stages, matrix examples, analyzer bundles, OTel adoption guidance, SBOM/SLSA requirements, and artifact checklists.

D) Security, Privacy, and Safety for LLM Agents
- Web search: prompt injection defenses and isolation techniques 2024 2025
- Web search: least‑privilege tool execution and capability routing
- Web search: secret management for AI agents ephemeral credentials vaults
- Web search: data governance PII redaction retention policies for AI logs
Deliver: enforceable controls, sandboxing models, policy templates, and redaction guidelines.

E) Performance Engineering and Runtime Capability Detection
- Web search: runtime feature detection CPU GPU NEON AVX dispatch patterns
- Web search: illegal instruction prevention CPUID checks best practices
- Web search: cache friendly data layout SoA AoS tradeoffs
- Web search: variance control for performance measurements MAD percentile reporting
Deliver: refined runtime dispatch policies, fallback requirements, measurement methodology, and latency/throughput budgets.

F) Cross‑Language Agent Stacks and Interop
- Web search: FFI patterns C API stability versioning two‑call pattern
- Web search: gRPC vs message bus vs function calling for agents
- Web search: schema versioning JSON Schema Protocol Buffers Avro 2024 2025
Deliver: interop guidance, ABI stability rules, transport selection criteria, and schema evolution strategies.

## Deliverables (What you must produce)
1) Global User Guidelines vNext (single document, 2,500–4,000 words)
   - Integrate all researched enhancements into the baseline sections 0–11.
   - Mark material changes with brief margin notes like “Change: tightened determinism FP tolerances with source [#]”.
   - Include normative MUST/SHOULD/MAY language where appropriate.
2) Evidence and Rationale Annex (concise)
   - Bullet list of key decisions with 1–3 citations each; include trade‑off notes and risks.
3) Operational Checklists and Templates
   - PR template, Commit template, CI pipeline skeleton, Security review checklist, Determinism checklist, Benchmark runbook (with NDJSON schema example).
4) Reference Patterns (short, language‑agnostic pseudo‑code)
   - Runtime capability dispatch guard
   - Golden‑file roundtrip test
   - Property‑based test scaffold
   - Agent tool‑execution sandbox stub
5) Migration and Governance Plan
   - How to adopt vNext in existing teams; baseline refresh procedures; exceptions/waivers; auditing cadence.

## Output Formatting Requirements
- Return a single markdown response with clear H2/H3 sections following the Deliverables structure.
- Use tables for matrices and checklists where useful.
- Cite sources inline with links and include a final References section.
- Include short pseudo‑code blocks for patterns. Keep them language‑neutral.

## Constraints and Quality Bar
- Framework‑agnostic and vendor‑neutral. Do not assume specific products.
- Cross‑platform emphasis (Windows/Linux/macOS); include ARM64 notes where relevant.
- Security‑first and determinism‑aware for agentic systems.
- Reproducibility is non‑negotiable; variance thresholds explicit and justified.
- Provide actionable, testable guidance that a senior team can adopt immediately.

## Start the Work (explicit actions)
- Web search the topics in sections A–F.
- Aggregate 6–12 authoritative sources per section; prefer standards docs, widely‑used OSS, and 2023–2025 articles or papers.
- Synthesize improvements and produce Deliverables 1–5 in a single markdown response.
- Include an Executive Summary (200–300 words) at the top of Deliverable 1 summarizing key upgrades.
- Validate internal consistency across sections; ensure checklists align with policies.
- Return the final result now in one response. Do not ask clarifying questions.

