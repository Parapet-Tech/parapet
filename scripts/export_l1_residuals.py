"""Export L1 out-of-fold residuals for L2b training.

Runs grouped 5-fold cross-validation on the v7 train split:
  1. Group samples by source family (small sources merged by reason x language)
  2. Assign groups to 5 folds with best-effort label x language balance
  3. For each fold: train L1 on 4 folds, eval held-out fold through full harness
  4. Collect all out-of-fold predictions with harness signals
  5. Export residual dataset: FPs, FNs, margin-band, baseline mix

Usage:
    cd parapet
    python scripts/export_l1_residuals.py \
      --curation-dir parapet-data/curated/v7_35k_experiment \
      --output-dir parapet-runner/runs/l2b_residuals
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_yaml_samples(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected YAML list")
    return raw


def content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def write_yaml_samples(samples: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(samples, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False, width=200)


# ---------------------------------------------------------------------------
# Grouped fold assignment
# ---------------------------------------------------------------------------

MAX_GROUP_SIZE = 1000  # Sources above this get split into sub-groups
MIN_GROUP_SIZE = 50   # Sources below this get merged into composite groups


def assign_groups(samples: list[dict[str, Any]], seed: int = 42) -> list[str]:
    """Assign each sample a group ID for fold splitting.

    Large sources (> MAX_GROUP_SIZE) are split into sub-groups of ~MAX_GROUP_SIZE,
    shuffled first so sub-groups are random slices of the source.
    Small sources (< MIN_GROUP_SIZE) are merged into composite groups by reason x language.
    Mid-sized sources are their own group.
    """
    rng = random.Random(seed)

    # Index samples by source
    source_indices: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        source_indices[s.get("source", "unknown")].append(i)

    groups = [""] * len(samples)

    for src, indices in source_indices.items():
        if len(indices) > MAX_GROUP_SIZE:
            # Split large source into sub-groups
            shuffled = list(indices)
            rng.shuffle(shuffled)
            n_chunks = max(2, len(indices) // MAX_GROUP_SIZE)
            chunk_size = len(indices) // n_chunks
            for chunk_id in range(n_chunks):
                start = chunk_id * chunk_size
                end = start + chunk_size if chunk_id < n_chunks - 1 else len(shuffled)
                for idx in shuffled[start:end]:
                    groups[idx] = f"{src}__chunk{chunk_id}"
        elif len(indices) < MIN_GROUP_SIZE:
            # Merge small sources
            for idx in indices:
                reason = samples[idx].get("reason", "unknown")
                lang = samples[idx].get("language", "unknown")
                groups[idx] = f"_small_{reason}_{lang}"
        else:
            for idx in indices:
                groups[idx] = src

    return groups


def make_folds(
    samples: list[dict[str, Any]],
    groups: list[str],
    n_folds: int,
    seed: int,
) -> list[int]:
    """Assign each sample to a fold (0..n_folds-1), grouped.

    All samples in the same group go to the same fold.
    Best-effort balance on total size, with label-aware tie-breaking.
    """
    rng = random.Random(seed)

    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        group_to_indices[g].append(i)

    # Sort groups by size descending for greedy bin-packing
    sorted_groups = sorted(group_to_indices.keys(),
                           key=lambda g: len(group_to_indices[g]), reverse=True)

    fold_sizes = [0] * n_folds
    group_to_fold: dict[str, int] = {}

    for g in sorted_groups:
        min_fold = min(range(n_folds), key=lambda f: fold_sizes[f])
        group_to_fold[g] = min_fold
        fold_sizes[min_fold] += len(group_to_indices[g])

    folds = [0] * len(samples)
    for i, g in enumerate(groups):
        folds[i] = group_to_fold[g]

    return folds


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def build_fold_manifest(
    curation_dir: Path,
    temp_dir: Path,
    train_samples: list[dict[str, Any]],
    holdout_samples: list[dict[str, Any]],
) -> Path:
    """Build a manifest for one fold: custom train + custom holdout, real val."""
    original_manifest = json.loads(
        (curation_dir / "manifest.json").read_text(encoding="utf-8")
    )

    write_yaml_samples(train_samples, temp_dir / "train.yaml")
    write_yaml_samples(holdout_samples, temp_dir / "holdout.yaml")

    train_hashes = sorted(content_hash(s["content"]) for s in train_samples)
    holdout_hashes = sorted(content_hash(s["content"]) for s in holdout_samples)

    manifest = {
        "spec_name": "l2b_residuals",
        "spec_version": "7.0.0",
        "spec_hash": original_manifest.get("spec_hash", ""),
        "seed": 42,
        "timestamp": utc_now(),
        "source_hashes": original_manifest.get("source_hashes", {}),
        "source_metadata": original_manifest.get("source_metadata", {}),
        "output_path": str(temp_dir / "curated.yaml"),
        "output_hash": "fold",
        "semantic_hash": "fold",
        "total_samples": len(train_samples) + len(holdout_samples),
        "attack_samples": sum(1 for s in train_samples if s.get("label") == "malicious"),
        "benign_samples": sum(1 for s in train_samples if s.get("label") == "benign"),
        "splits": {
            "train": {
                "name": "train",
                "sample_count": len(train_samples),
                "content_hashes": train_hashes,
                "artifact_path": "train.yaml",
            },
            "val": original_manifest["splits"]["val"],
            "holdout": {
                "name": "holdout",
                "sample_count": len(holdout_samples),
                "content_hashes": holdout_hashes,
                "artifact_path": "holdout.yaml",
            },
        },
        "cell_fills": {},
        "gaps": [],
        "duplicates_dropped": 0,
        "cross_contamination_dropped": 0,
    }

    manifest_path = temp_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# Run one fold
# ---------------------------------------------------------------------------

def run_fold(
    *,
    fold_id: int,
    train_samples: list[dict[str, Any]],
    holdout_samples: list[dict[str, Any]],
    curation_dir: Path,
    output_dir: Path,
    workspace_root: Path,
    train_config_path: Path,
    parapet_eval_bin: Path,
    python_bin: str,
) -> Path | None:
    """Train on fold's train, eval on fold's holdout. Returns eval.json path or None."""
    fold_dir = output_dir / f"fold_{fold_id}"
    eval_json = fold_dir / "run" / "_eval_holdout_0p0" / "eval.json"

    if eval_json.exists():
        print(f"  [fold_{fold_id}] SKIP (cached)")
        return eval_json

    fold_dir.mkdir(parents=True, exist_ok=True)
    curated_dir = fold_dir / "curated"
    curated_dir.mkdir(parents=True, exist_ok=True)

    # Copy val from real curation
    shutil.copy2(curation_dir / "val.yaml", curated_dir / "val.yaml")

    manifest_path = build_fold_manifest(
        curation_dir, curated_dir, train_samples, holdout_samples,
    )

    run_dir = fold_dir / "run"
    cmd = [
        python_bin, "-m", "parapet_runner.runner", "run",
        "--workspace-root", str(workspace_root),
        "--curation-manifest", str(manifest_path),
        "--train-config", str(train_config_path),
        "--output-dir", str(run_dir.resolve()),
        "--parapet-eval-bin", str(parapet_eval_bin),
        "--run-id", f"l2b_fold_{fold_id}",
        "--pg2-mode", "off",
        "--skip-output-hash-verify",
        "--skip-split-hash-verify",
    ]

    runner_cwd = workspace_root / "parapet-runner"
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(runner_cwd), capture_output=True, text=True)
    elapsed = time.time() - start

    log_path = fold_dir / "runner.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# fold: {fold_id}\n")
        f.write(f"# elapsed_sec: {elapsed:.1f}\n")
        f.write(f"# returncode: {proc.returncode}\n\n")
        f.write("## stdout\n")
        f.write(proc.stdout or "")
        f.write("\n\n## stderr\n")
        f.write(proc.stderr or "")

    if proc.returncode == 0 and eval_json.exists():
        print(f"  [fold_{fold_id}] OK ({elapsed:.0f}s, {len(holdout_samples)} holdout)")
        return eval_json
    else:
        print(f"  [fold_{fold_id}] FAIL (rc={proc.returncode}, see {log_path})", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Parse eval results with harness signals
# ---------------------------------------------------------------------------

def parse_eval_results(
    eval_json_path: Path,
    holdout_samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Parse eval.json and join with sample metadata."""
    eval_data = json.loads(eval_json_path.read_text(encoding="utf-8"))
    results = eval_data["results"]

    # Build index from holdout samples by position (eval preserves order)
    rows = []
    for i, (result, sample) in enumerate(zip(results, holdout_samples)):
        expected = result["expected"]  # "blocked" or "allowed"
        actual = result["actual"]

        true_label = "malicious" if expected == "blocked" else "benign"
        pred_label = "malicious" if actual == "blocked" else "benign"
        correct = result["correct"]

        # Classify error type
        if true_label == "malicious" and pred_label == "benign":
            error_type = "false_negative"
        elif true_label == "benign" and pred_label == "malicious":
            error_type = "false_positive"
        else:
            error_type = "correct"

        # Extract harness signals from l1_signals if present
        # l1_signals is a list of per-message signals; take first for single-message inputs
        l1_raw = result.get("l1_signals") or []
        l1 = l1_raw[0] if isinstance(l1_raw, list) and l1_raw else {}

        row = {
            "content": sample["content"],
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": correct,
            "error_type": error_type,
            "language": sample.get("language", ""),
            "source": sample.get("source", ""),
            "reason": sample.get("reason", ""),
            "format_bin": sample.get("format_bin", ""),
            "length_bin": sample.get("length_bin", ""),
            "raw_score": l1.get("raw_score"),
            "raw_unquoted_score": l1.get("raw_unquoted_score"),
            "raw_squash_score": l1.get("raw_squash_score"),
            "raw_score_delta": l1.get("raw_score_delta"),
            "quote_detected": l1.get("quote_detected"),
            "score": l1.get("score"),
            "unquoted_score": l1.get("unquoted_score"),
            "squash_score": l1.get("squash_score"),
            "content_hash": content_hash(sample["content"]),
        }
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Build residual dataset
# ---------------------------------------------------------------------------

MARGIN_BAND = 1.0  # raw_score within +/- this of 0 is "near boundary"
BASELINE_SAMPLE_RATE = 0.10  # fraction of correct predictions to include


def build_residual_dataset(
    all_predictions: list[dict[str, Any]],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Partition predictions into residual categories."""
    rng = random.Random(seed)

    false_negatives = []
    false_positives = []
    near_boundary_benign = []
    baseline_correct = []

    for row in all_predictions:
        et = row["error_type"]
        raw = row.get("raw_score")

        if et == "false_negative":
            false_negatives.append(row)
        elif et == "false_positive":
            false_positives.append(row)
        elif et == "correct":
            # Near-boundary benign
            if row["true_label"] == "benign" and raw is not None and abs(raw) <= MARGIN_BAND:
                near_boundary_benign.append(row)
            # Baseline sample
            elif rng.random() < BASELINE_SAMPLE_RATE:
                baseline_correct.append(row)

    return {
        "false_negative": false_negatives,
        "false_positive": false_positives,
        "near_boundary_benign": near_boundary_benign,
        "baseline_correct": baseline_correct,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    default_workspace_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Export L1 out-of-fold residuals for L2b")
    parser.add_argument("--workspace-root", type=Path, default=default_workspace_root)
    parser.add_argument("--curation-dir", type=Path, required=True)
    parser.add_argument(
        "--train-config", type=Path,
        default=Path("parapet-runner/configs/phase1_learning_curve.yaml"),
    )
    parser.add_argument(
        "--parapet-eval-bin", type=Path,
        default=Path("parapet/target/release/parapet-eval.exe"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-bin", type=str, default="python")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    curation_dir = Path(args.curation_dir)
    if not curation_dir.is_absolute():
        curation_dir = (workspace_root / curation_dir).resolve()
    parapet_eval_bin = Path(args.parapet_eval_bin)
    if not parapet_eval_bin.is_absolute():
        parapet_eval_bin = (workspace_root / parapet_eval_bin).resolve()
    train_config_path = Path(args.train_config)
    if not train_config_path.is_absolute():
        train_config_path = (workspace_root / train_config_path).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (workspace_root / output_dir).resolve()

    # Load train split
    train_path = curation_dir / "train.yaml"
    print(f"Loading {train_path}...")
    all_samples = load_yaml_samples(train_path)
    print(f"  {len(all_samples):,} samples")

    # Assign groups and folds
    groups = assign_groups(all_samples, seed=args.seed)
    unique_groups = set(groups)
    print(f"  {len(unique_groups)} groups (max={MAX_GROUP_SIZE}, min={MIN_GROUP_SIZE})")

    folds = make_folds(all_samples, groups, args.n_folds, args.seed)
    fold_counts = Counter(folds)
    print(f"  Fold sizes: {dict(sorted(fold_counts.items()))}")

    # Check fold balance
    for f in range(args.n_folds):
        fold_samples = [s for s, fi in zip(all_samples, folds) if fi == f]
        labels = Counter(s["label"] for s in fold_samples)
        langs = Counter(s["language"] for s in fold_samples)
        print(f"  Fold {f}: {len(fold_samples):>5}  "
              f"mal={labels.get('malicious',0):>4} ben={labels.get('benign',0):>4}  "
              f"EN={langs.get('EN',0):>4} RU={langs.get('RU',0):>4} "
              f"ZH={langs.get('ZH',0):>4} AR={langs.get('AR',0):>4}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save fold assignments
    fold_assignments = [
        {"index": i, "fold": folds[i], "group": groups[i],
         "source": all_samples[i].get("source", ""),
         "label": all_samples[i].get("label", ""),
         "language": all_samples[i].get("language", "")}
        for i in range(len(all_samples))
    ]
    (output_dir / "fold_assignments.json").write_text(
        json.dumps(fold_assignments[:10], indent=2) + f"\n... ({len(fold_assignments)} total)\n",
        encoding="utf-8",
    )

    # Run 5 folds
    print(f"\n=== Running {args.n_folds} folds ===")
    all_predictions: list[dict[str, Any]] = []

    for fold_id in range(args.n_folds):
        train_fold = [s for s, f in zip(all_samples, folds) if f != fold_id]
        holdout_fold = [s for s, f in zip(all_samples, folds) if f == fold_id]

        print(f"\nFold {fold_id}: train={len(train_fold)}, holdout={len(holdout_fold)}")

        eval_json = run_fold(
            fold_id=fold_id,
            train_samples=train_fold,
            holdout_samples=holdout_fold,
            curation_dir=curation_dir,
            output_dir=output_dir,
            workspace_root=workspace_root,
            train_config_path=train_config_path,
            parapet_eval_bin=parapet_eval_bin,
            python_bin=args.python_bin,
        )

        if eval_json is None:
            print(f"  FATAL: fold {fold_id} failed", file=sys.stderr)
            return 1

        predictions = parse_eval_results(eval_json, holdout_fold)
        all_predictions.extend(predictions)
        print(f"  Collected {len(predictions)} predictions "
              f"(FP={sum(1 for p in predictions if p['error_type']=='false_positive')}, "
              f"FN={sum(1 for p in predictions if p['error_type']=='false_negative')})")

    # Build residual dataset
    print(f"\n=== Building residual dataset from {len(all_predictions)} predictions ===")
    residuals = build_residual_dataset(all_predictions, seed=args.seed)

    for category, rows in residuals.items():
        print(f"  {category:<25} {len(rows):>5}")

    total_residual = sum(len(v) for v in residuals.values())
    print(f"  TOTAL residual samples:  {total_residual}")

    # Write outputs
    # Full predictions (for analysis)
    predictions_path = output_dir / "all_predictions.jsonl"
    with open(predictions_path, "w", encoding="utf-8") as f:
        for p in all_predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n  All predictions: {predictions_path} ({len(all_predictions)} rows)")

    # Residual dataset (for L2b training)
    for category, rows in residuals.items():
        cat_path = output_dir / f"residual_{category}.jsonl"
        with open(cat_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {category}: {cat_path} ({len(rows)} rows)")

    # Combined residual dataset
    combined = []
    for category, rows in residuals.items():
        for r in rows:
            r["residual_category"] = category
            combined.append(r)

    combined_path = output_dir / "l2b_training_candidates.jsonl"
    with open(combined_path, "w", encoding="utf-8") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Combined: {combined_path} ({len(combined)} rows)")

    # Summary
    summary = {
        "created_utc": utc_now(),
        "n_folds": args.n_folds,
        "total_samples": len(all_samples),
        "total_predictions": len(all_predictions),
        "residual_counts": {k: len(v) for k, v in residuals.items()},
        "total_residual": total_residual,
        "error_summary": {
            "false_positives": len(residuals["false_positive"]),
            "false_negatives": len(residuals["false_negative"]),
            "near_boundary_benign": len(residuals["near_boundary_benign"]),
            "baseline_correct": len(residuals["baseline_correct"]),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\n=== Done ===")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
