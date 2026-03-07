# ProtectAI Baseline Spec

## Goal
Add a third baseline in `parapet-runner` that reproduces a ProtectAI-style data recipe and compares it directly against:
- random baseline
- mirror/hybrid runs

This is a runner feature (experiment orchestration + reporting), not a data curation feature.

## v2 Plan Linkage
This spec is tracked in:
- `parapet/schema/eval/attack_types/v2/plan.md`:
  - Scope (runner baseline expansion)
  - Experiment Matrix (required `protectai_size_matched`)
  - Success Gates (baseline-coverage reporting gate)
  - Deliverables (runner baseline fields and recipe artifacts)
  - Execution Sequence (runner implementation before Tier 1 matrix)

## Scope
- Runner can execute a `protectai` baseline recipe as a first-class baseline mode.
- Runner emits comparable metrics and deltas for:
  - `random`
  - `pg2`
  - `protectai`
- Runner records recipe identity and dataset fingerprint in `RunManifest`.

## Non-Goals
- Reimplementing the full ProtectAI training stack.
- Changing classifier architecture for this baseline.
- Mixing ProtectAI baseline data into mirror training by default.

## Design

### 1. Baseline Type Expansion
Add baseline type enum/literal:
- `pg2` (existing)
- `random` (existing)
- `protectai` (new)

### 2. ProtectAI Recipe Contract
Add a runner-side recipe contract:
- dataset allowlist
- per-source caps
- label mapping rules
- contamination denylist
- seed

Materialized recipe output:
- `train.jsonl`
- `val.jsonl`
- `holdout.jsonl`
- `recipe_manifest.json`

### 3. Size-Matched Variant
Runner supports:
- `protectai_repro` (native size)
- `protectai_size_matched` (same size as target run, e.g., 24k)

Both must use fixed seed and deterministic ordering.

### 4. Manifest Additions
`RunManifest` adds:
- `baseline_family` (`random|pg2|protectai_repro|protectai_size_matched`)
- `baseline_recipe_hash`
- `baseline_data_hash`
- `baseline_data_size`

### 5. Reporting
Error analysis and summary output include pairwise deltas:
- candidate vs random
- candidate vs pg2
- candidate vs protectai_size_matched

## Execution Order
1. Resolve canonical holdout (unchanged).
2. Materialize chosen baseline dataset recipe.
3. Train with existing trainer pipeline.
4. Calibrate threshold on val only.
5. Evaluate on holdout.
6. Persist baseline manifest + delta report.

## Acceptance Criteria
1. Same seed + recipe yields identical baseline data hash.
2. `protectai_size_matched` row count exactly equals target run size.
3. Holdout remains unchanged across all baselines.
4. Run manifest contains baseline recipe/hash fields.
5. Integration test validates three-baseline compare output.

## Tests
- unit: recipe parser, label mapping, deterministic sampling
- unit: manifest baseline fields
- integration: full run with `protectai_size_matched`
- regression: existing `pg2` baseline path unchanged
