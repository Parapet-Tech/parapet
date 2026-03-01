# parapet-runner

Experiment orchestration for prompt injection classifier training.

## Status

**Runner wiring complete.**

What works today: DI-based experiment runner, threshold calibration, PG2 baseline adapter, runtime identity collection, semantic parity hashing, concrete split/trainer/evaluator wiring, and CLI entrypoint.

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
BaselineProvider --- runs PG2 on same holdout for comparison
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

### baseline.py -- PG2 baseline adapter

- **PG2BaselineRunner**: Runs PromptGuard 2 evaluation via injectable `CommandExecutor`.
- **parse_eval_result_json()**: Flexible JSON parser with recursive key lookup for eval output.
- Subprocess boundary fully injectable for testing.

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
6. Run PG2 baseline on same holdout
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

23 passed.

## What's next

1. **Run a full experiment** -- `python -m parapet_runner.runner run --curation-manifest manifest.json --train-config train.json --output-dir ./runs/`
2. **Add Makefile orchestration targets** -- `make run-fast`, `make run-final`, `make compare`.
