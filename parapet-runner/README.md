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

## Resource budgets

Production `run` training, evaluation, baselines and builds, and the `eval`
command use the resource guard. Defaults: one locked Parapet job per host,
12 GiB total tracked RSS, 4 GiB available headroom, 32 processes,
2 workers, 4 hours, niceness 10, polling every 2 seconds. The host lock is
Parapet-only; unrelated applications can still consume resources.

```sh
python -m parapet_runner budget-run --profile train --wall 2h --max-rss 16G --max-workers 4 -- python scripts/train_l2b.py --help
python -m parapet_runner budget-run --show-config
```

Resolution: CLI > environment > profile > YAML defaults > built-in defaults.
Sizes accept `512M`, `12G`; durations accept `90m`, `4h`.
`PARAPET_BUDGET_CONFIG` selects an optional local config (default
`~/.config/parapet/budget.yaml`); keep this user file untracked. Example:

```yaml
defaults:
  max_rss_bytes: 12G
  min_free_bytes: 4G
  max_processes: 32
  max_workers: 2
  wall_seconds: 4h
  poll_seconds: 2s
  nice: 10
profiles:
  quick:
    wall_seconds: 90m
    max_rss_bytes: 512M
  train:
    max_workers: 2
  eval:
    max_rss_bytes: 4G
```

Select with `--profile` or `PARAPET_BUDGET_PROFILE`. Environment overrides:
`PARAPET_BUDGET_MAX_RSS_GB`, `PARAPET_BUDGET_MIN_FREE_GB`,
`PARAPET_BUDGET_EXPECTED_RSS_GB` (bare values mean GiB),
`PARAPET_BUDGET_MAX_PROCESSES`, `PARAPET_BUDGET_MAX_WORKERS`,
`PARAPET_BUDGET_WALL`, `PARAPET_BUDGET_NICE`, `PARAPET_BUDGET_POLL`.
CLI equivalents: `--max-rss`, `--min-free`, `--expected-rss`, `--max-processes`,
`--max-workers`, `--wall`, `--nice`, `--poll`.
`--show-config` prints resolved values and sources without starting a job.
`PARAPET_JOB_LOCK` overrides `~/.cache/parapet/job.lock`; a held lock reports
its PID and executable basename in owner-only (0600) metadata. The lease covers
the entire `run`, `eval`, or `budget-run` job through final output emission.
Only `budget-run` accepts `--no-lock`, an operator escape hatch that is not
mechanically read-only and is recorded in receipts. Caps are exported to OpenMP, BLAS, MKL, Accelerate and Loky.

`max_rss_bytes` is an enforced runtime kill cap, never a preflight reservation.
Preflight requires available memory >= `min_free_bytes` (4 GiB by default).
Optionally set `expected_rss_bytes` in defaults/profiles, `--expected-rss 6G`,
or `PARAPET_BUDGET_EXPECTED_RSS_GB=6` to require headroom plus expected RSS.
Expected RSS is unset by default. On macOS, available memory is the sum of
free + inactive + speculative + purgeable pages times the page size.

Raise limits for a single command:

```sh
python -m parapet_runner budget-run --max-rss 24G --wall 8h --max-workers 4 -- python train.py
PARAPET_BUDGET_MAX_RSS_GB=24 PARAPET_BUDGET_WALL=8h python -m parapet_runner budget-run -- python train.py
```

Accounting sums the union of the process group and PPID descendants, including
children calling `setsid`. Observed PIDs remain tracked until exit even after
reparenting. On breach the guard re-enumerates, sends SIGTERM to the group and
tracked PIDs, waits up to 5 seconds, re-enumerates, then sends SIGKILL. The wait
is shortened at 1.5 times the wall limit. Poll sleeps cannot cross the wall
limit. Brief spikes between polls may be missed. A double-fork daemon that
reparents to PID 1 between polls can escape tracking: macOS has no reliable
user-space hook without cgroups. This is a known limitation, not a guarantee;
the live double-fork regression is a non-strict xfail.

`BudgetedExecutor.run` implements `CommandExecutor`; `run_with_receipt` returns
both result and receipt, also retained on the executor. Run manifests store
optional `runtime.budget_receipts` with peaks, wall time, resolved budget,
reason, lock bypass and containment limitation. `budget-run` prints its receipt
to stdout on its own line; human diagnostics go to stderr with a trailing newline.
Configuration errors name the source and fix and exit 2 without a traceback.
Exit codes: RSS 120, process cap 121, wall 122, preflight 123, lock 124;
otherwise preserve child exit status (signals map to 128 + signal on CLI).
Live tests require `ps` and skip with an explicit reason if denied.

Job receipts include `cpu_user_s` and `cpu_sys_s`, measured using the
`resource.getrusage(RUSAGE_CHILDREN)` delta around the job. This counts only
reaped descendants, not reparented orphans. Run manifests include a final job
receipt; `eval` writes `budget_receipt.json` before releasing its lease.
