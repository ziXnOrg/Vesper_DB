# Validation Guide (Phase 0)

This folder contains JSON Schemas and minimal examples for Vesper’s prompt artifacts and evals.

## Layout
```
experiments/
  schema/
    prompt.manifest.schema.json
    eval.report.schema.json
    experiment.schema.json
  examples/
    prompt.manifest.json
    eval.report.json
    experiment.yaml
```

## Validation (local)
Use any JSON Schema Draft 2020‑12 validator. Examples using `jq` + `spectral` or `ajv` are shown below; these can be run later once tooling is installed.

- Validate prompt manifest
  - ajv: `ajv validate -s experiments/schema/prompt.manifest.schema.json -d experiments/examples/prompt.manifest.json`
- Validate eval report
  - ajv: `ajv validate -s experiments/schema/eval.report.schema.json -d experiments/examples/eval.report.json`
- Validate experiment config (YAML)
  - Convert YAML to JSON then validate: `yq -o=json e experiments/examples/experiment.yaml | ajv validate -s experiments/schema/experiment.schema.json -d -`

## Notes and invariants
- Determinism is enforced in schemas: decoding must be `{temperature: 0.0, top_p: 1.0, n: 1}` and include a fixed integer `seed`.
- Paths should be relative to the repo root unless they start with `file://` or `http(s)://`.
- Timestamps use RFC3339 (JSON Schema `date-time`).
- `blueprint.md` must appear in `spec_refs` for prompt manifests (checked in CI/tooling).

## CI (Phase 0+)
In CI, add jobs to:
- Validate all manifest/report/experiment files changed in a PR.
- Enforce invariants not expressible in pure JSON Schema (e.g., `metrics >= gates ⇒ passed`).
- Upload validation logs as artifacts.

