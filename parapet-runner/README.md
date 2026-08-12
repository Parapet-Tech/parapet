# parapet-runner

Experiment orchestration for prompt injection classifier training.

## Status

**Runner wiring complete.**

What works today: DI-based experiment runner, threshold calibration, optional baseline adapters, runtime identity collection, semantic parity hashing, concrete split/trainer/evaluator wiring, and CLI entrypoint.

## Architecture

```
CurationManifest (from parapet-data)
    |
    v
SplitResolver --- locates train/val/holdout JSONL splits
    |
    v
Trainer --------- calls train_l1_specialist.py, produces model artifact
    |
    v
ThresholdCalibrator --- grid search over val split (F1 maximization)
    |                    ONLY touches val, never holdout
    v
Evaluator ------- runs parapet-eval on holdout with calibrated threshold
    |
    v
BaselineProvider --- runs optional comparison baselines on the same holdout
    |
    v
ErrorAnalyzer --- writes YAML error summary with deltas
    |
    v
RunManifest (immutable experiment receipt)
```

All boxes above are **Protocol interfaces**. The runner owns orchestration; adapters own execution. Every dependency is injected via `ExperimentDependencies`.

## Modules

### config.py -- Training configuration

- **TrainConfig**: Hyperparameters (ngram_range, max_features, min_df, C, class_weight), threshold policy, mode validation.
  - `iteration` mode: cv_folds=0, max_features <= 15K (fast feedback)
  - `final` mode: cv_folds >= 3, max_features >= 25K (publication quality)
- **ThresholdPolicy**: FIXED (use explicit value) or CALIBRATE_F1 (grid search on val).
- `to_train_script_args()`: Renders config to CLI args for the training script.

### manifest.py -- Reproducibility contracts

- **EvalResult**: f1, precision, recall, FP/FN counts, threshold, holdout size.
- **RuntimeIdentity**: git_sha, script hashes, model ID, env_hash (lockfile-first).
- **RunManifest**: Complete experiment receipt. Validates delta consistency.
- **CurationManifest**: Imported from parapet-data when available, falls back to minimal stub.
- **compute_semantic_parity_hash()**: Delegates to parapet-data's canonical implementation when installed. Strict validation -- rejects malformed per-cell dicts (missing keys, unknown keys, string backfill_sources).

### baseline.py -- Baseline adapters

- **PG2BaselineRunner**: Runs the legacy Prompt Guard 2 comparison path via injectable `CommandExecutor`.
- **parse_eval_result_json()**: Flexible JSON parser with recursive key lookup for eval output.
- Subprocess boundary fully injectable for testing.

### temporal_adapter.py -- Claim-bearing event adapter

- **TemporalEventAssembler**: Protocol for the routing-owned join between
  lab-produced raw-envelope events and frozen detector/CDF observations.
- **RawEnvelopeEvent**: exact scored payload plus raw envelope role, original
  source order, construction labels, frozen generator/mechanism/surface strata,
  and source receipt reference.
- **DetectorObservation**: payload-hash-bound D_eval score, surprise,
  score-context, fallback level, and detector provenance.
- **P3TemporalAdapterPins**: explicit index, D_eval, reference-CDF, source-code,
  and contract pins. The frozen P3 `alpha=1`, `n_c>=50`, and fallback order are
  validated fail-closed.
- Output is payload-free `TemporalEvent` JSONL plus a deterministic construction
  receipt. The adapter preserves honest missingness and does not parse source-
  specific traces, run D_eval, select a cohort, or make a performance claim.
- **temporal_adapter_io.py** owns deterministic JSONL loading/writing, sanitized
  validation diagnostics, input/output hashes, and the payload-free receipt.

### runner.py -- Experiment orchestration

**8 Protocol interfaces:**

| Protocol | Method | Concrete impl |
|----------|--------|---------------|
| SplitResolver | `resolve()` | ManifestSplitResolver |
| Trainer | `train()` | TrainScriptTrainer |
| Evaluator | `evaluate()` | ParapetEvalEvaluator |
| BaselineProvider | `run()` | ParapetEvalPG2BaselineProvider |
| ThresholdCalibrator | `calibrate()` | F1GridSearchCalibrator |
| ErrorAnalyzer | `write()` | YamlErrorAnalyzer |
| RuntimeIdentityProvider | `collect()` | RuntimeIdentityCollector |
| ArtifactVerifier | `verify()` | OutputHashVerifier |

**ExperimentRunner.run_experiment()** orchestrates the full pipeline:
1. Verify curation artifact hash (optional)
2. Resolve splits from CurationManifest
3. Train model
4. Calibrate threshold on val (if CALIBRATE_F1)
5. Evaluate on holdout
6. Run enabled comparison baselines on the same holdout
7. Compute metric deltas
8. Write error analysis YAML
9. Compute semantic parity hash
10. Return RunManifest

## Cross-package contract

parapet-runner imports `compute_semantic_hash` and `CellFillRecord` from parapet-data when available. A contract test (`test_semantic_hash_matches_parapet_data_contract`) proves the runner's hash equals the data package's hash for identical input. CI should gate on this.

## Install

```
pip install -e parapet/parapet-runner
```

Requires Python >= 3.10, pydantic >= 2.7, pyyaml >= 6.0.

## Test

```
cd parapet/parapet-runner
python -m pytest -q
```

Run the targeted temporal tests with:

```bash
python -m pytest tests/test_temporal_adapter.py tests/test_temporal.py -q
```

## Checkpointed ablations

Long-running mix ablations can be run with incremental checkpointing:

```
cd parapet
python parapet-runner/scripts/v2_mix_ablation.py \
  --ratios 100:0,70:30,50:50,30:70,0:100 \
  --seeds 42,43,44 \
  --threshold -0.5 \
  --continue-on-error
```

Artifacts are written incrementally under `parapet-runner/runs/mirror_v2_mix_ablation/`:
- one folder per cell (`m{mirror}_n{non}_s{seed}`) with `status.json`, logs, and `result.json`
- aggregate files regenerated after every completed cell: `results.jsonl`, `summary.csv`, `summary.md`

## Usage

Run from `parapet/parapet-runner/`:

```bash
python -m parapet_runner.runner run \
  --workspace-root <absolute-path-to-parapet/> \
  --curation-manifest <absolute-path-to-manifest.json> \
  --train-config configs/iteration_v1_calibrated.yaml \
  --output-dir runs/my_run \
  --parapet-eval-bin <absolute-path-to-parapet-eval.exe> \
  --skip-recompile
```

| Flag | Required | Notes |
|------|----------|-------|
| `--workspace-root` | yes | Absolute path to `parapet/`. All default relative paths resolve from here. |
| `--curation-manifest` | yes | Absolute path to `manifest.json` produced by `parapet-data curate`. |
| `--train-config` | yes | Path to a TrainConfig YAML. Can be relative to cwd. |
| `--output-dir` | yes | Where run artifacts go. Can be relative to cwd. |
| `--parapet-eval-bin` | yes | Absolute path to compiled `parapet-eval` binary. |
| `--skip-recompile` | no | Skip Rust binary rebuild (use when iterating on data, not weights). |
| `--pg2-mode on` | no | Enable the legacy PG2 comparison baseline. Off by default. |
| `--protectai-mode protectai_size_matched` | no | Enable ProtectAI baseline. Off by default. |
| `--random-mode on` | no | Enable random-sample baseline. Off by default. |

See `parapet-data/README.md` for the full end-to-end workflow (spec generation, curation, then runner).

### Temporal scorer

`temporal-score` emits a generic diagnostic receipt by default. Supply
`--metadata-json` with a validated `TemporalReceiptMetadata` envelope to emit
the mechanically distinct `p3_temporal_validation` receipt:

```bash
python -m parapet_runner.runner temporal-score \
  --events-jsonl runs/<run>/temporal-events.jsonl \
  --metadata-json runs/<run>/temporal-receipt-metadata.json \
  --output-receipt runs/<run>/temporal-receipt.json \
  --k-u-b <frozen-value> \
  --k-u-c-strict <frozen-value> \
  --k-u-c-broad <frozen-value> \
  --peak-alert-level <frozen-value>
```

The P3 envelope fails closed unless calibration/evaluation/registration
references, CI and float-policy hashes, trajectory strata, and scorer
thresholds agree. Delta intervals use deterministic paired cell-level bootstrap
resampling. Supplying the envelope does not authorize an empirical run; the
frozen cohort and owning-team artifacts must already exist.

Validate a written receipt and rerun its mechanical P3 recomputation gates with:

```bash
python -m parapet_runner.runner temporal-validate \
  --receipt-json runs/<run>/temporal-receipt.json
```

### Temporal event adapter

After owning teams have produced exact raw-envelope event JSONL, frozen D_eval/CDF
observation JSONL, and an explicit pins file, assemble scorer input with:

```bash
python -m parapet_runner.runner temporal-adapt \
  --raw-events-jsonl runs/<run>/raw-envelope-events.jsonl \
  --detector-observations-jsonl runs/<run>/detector-observations.jsonl \
  --pins-json runs/<run>/temporal-adapter-pins.json \
  --output-events-jsonl runs/<run>/temporal-events.jsonl \
  --output-receipt runs/<run>/temporal-adapter-receipt.json
```

The command fails closed on unsafe trajectory paths, non-contiguous trajectory
blocks, source-order drift, payload-hash mismatch, missing/extra detector rows,
label or cell drift, unknown CDF fallback levels, and frozen-pin drift. Event
payload text is never written to the output or receipt.
