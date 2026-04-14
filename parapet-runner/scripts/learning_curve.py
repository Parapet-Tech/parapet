"""Phase 1: Learning curve estimation for L1 optimal SVM.

Trains the same model family on nested subsets of the training data to
estimate the size frontier: recall(n) = R_inf - a * n^-b.

Each cell:
  1. Subsamples the train split (stratified by label x language)
  2. Writes a temporary curated directory with the subsampled train + original val/holdout
  3. Invokes parapet-runner to train, calibrate, and evaluate on the search split
  4. Records per-seed and per-language metrics

The script re-uses the same runner invocation pattern as l1_hparam_sweep.py.

Example:
    cd parapet
    python parapet-runner/scripts/learning_curve.py \
      --curation-dir parapet-data/curated/v6_25k_experiment \
      --output-dir parapet-runner/runs/phase1_learning_curve
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
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class CurveCell:
    size: int
    seed: int

    @property
    def tag(self) -> str:
        return f"n{self.size}_s{self.seed}"


def load_yaml_samples(path: Path) -> list[dict[str, Any]]:
    """Load samples from a YAML split file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected YAML list")
    return raw


def content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def stratified_subsample(
    samples: list[dict[str, Any]],
    target_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Subsample stratified by label x language, preserving proportions."""
    if target_size >= len(samples):
        return list(samples)

    rng = random.Random(seed)

    # Group by (label, language)
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        key = (sample.get("label", "unknown"), sample.get("language", "EN"))
        strata[key].append(sample)

    # Allocate proportionally
    total = len(samples)
    result: list[dict[str, Any]] = []
    remaining = target_size

    sorted_keys = sorted(strata.keys())
    allocations: dict[tuple[str, str], int] = {}

    for key in sorted_keys:
        group = strata[key]
        alloc = max(1, round(target_size * len(group) / total))
        alloc = min(alloc, len(group), remaining)
        allocations[key] = alloc
        remaining -= alloc

    # Distribute any remainder to the largest strata
    if remaining > 0:
        for key in sorted(sorted_keys, key=lambda k: len(strata[k]), reverse=True):
            can_add = min(remaining, len(strata[key]) - allocations[key])
            if can_add > 0:
                allocations[key] += can_add
                remaining -= can_add
            if remaining <= 0:
                break

    for key in sorted_keys:
        group = list(strata[key])
        rng.shuffle(group)
        result.extend(group[: allocations[key]])

    rng.shuffle(result)
    return result


def write_yaml_samples(samples: list[dict[str, Any]], path: Path) -> None:
    """Write samples to YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(samples, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False, width=200)


def build_minimal_manifest(
    curation_dir: Path,
    temp_dir: Path,
    train_count: int,
) -> Path:
    """Build a minimal manifest.json pointing to the temp train and original val/holdout."""
    original_manifest = json.loads(
        (curation_dir / "manifest.json").read_text(encoding="utf-8")
    )

    # Compute content hashes for the subsampled train
    train_samples = load_yaml_samples(temp_dir / "train.yaml")
    train_hashes = sorted(content_hash(s["content"]) for s in train_samples)

    manifest = {
        "spec_name": original_manifest.get("spec_name", "learning_curve"),
        "spec_version": original_manifest.get("spec_version", "0.0.0"),
        "spec_hash": original_manifest.get("spec_hash", ""),
        "seed": original_manifest.get("seed", 42),
        "timestamp": utc_now(),
        "source_hashes": original_manifest.get("source_hashes", {}),
        "source_metadata": original_manifest.get("source_metadata", {}),
        "output_path": str(temp_dir / "curated.yaml"),
        "output_hash": "subsampled",
        "semantic_hash": "subsampled",
        "total_samples": train_count,
        "attack_samples": sum(1 for s in train_samples if s.get("label") in ("malicious", "attack")),
        "benign_samples": sum(1 for s in train_samples if s.get("label") == "benign"),
        "splits": {
            "train": {
                "name": "train",
                "sample_count": len(train_samples),
                "content_hashes": train_hashes,
                "artifact_path": "train.yaml",
            },
            "val": original_manifest["splits"]["val"],
            "holdout": original_manifest["splits"]["holdout"],
        },
        "cell_fills": original_manifest.get("cell_fills", {}),
        "gaps": original_manifest.get("gaps", []),
        "duplicates_dropped": 0,
        "cross_contamination_dropped": 0,
    }

    manifest_path = temp_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_path


def run_logged(cmd: list[str], *, cwd: Path, log_path: Path) -> tuple[int, float]:
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    elapsed = time.time() - start
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# cwd: {cwd}\n")
        f.write(f"# cmd: {' '.join(cmd)}\n")
        f.write(f"# start_utc: {utc_now()}\n")
        f.write(f"# elapsed_sec: {elapsed:.3f}\n")
        f.write(f"# returncode: {proc.returncode}\n\n")
        f.write("## stdout\n")
        f.write(proc.stdout or "")
        f.write("\n\n## stderr\n")
        f.write(proc.stderr or "")
    return proc.returncode, elapsed


def parse_run_result(run_manifest_path: Path, *, cell: CurveCell) -> dict[str, Any]:
    payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    eval_result = payload["eval_result"]
    return {
        "size": cell.size,
        "seed": cell.seed,
        "tag": cell.tag,
        "status": "completed",
        "threshold": eval_result["threshold"],
        "f1": eval_result["f1"],
        "precision": eval_result["precision"],
        "recall": eval_result["recall"],
        "false_positives": eval_result["false_positives"],
        "false_negatives": eval_result["false_negatives"],
        "holdout_size": eval_result["holdout_size"],
        "run_id": payload.get("run_id"),
        "run_manifest": str(run_manifest_path),
    }


def write_aggregates(output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []

    for cell_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        if cell_dir.name.startswith("_"):
            continue
        result_path = cell_dir / "result.json"
        if result_path.exists():
            rows.append(json.loads(result_path.read_text(encoding="utf-8")))

    rows.sort(key=lambda r: (r["size"], r["seed"]))

    # JSONL
    results_jsonl = output_dir / "results.jsonl"
    results_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    # CSV
    summary_csv = output_dir / "summary.csv"
    fieldnames = [
        "tag", "size", "seed", "f1", "precision", "recall",
        "false_positives", "false_negatives", "threshold", "holdout_size",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    # Summary by size
    by_size: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_size[row["size"]].append(row)

    lines = [
        "# Phase 1: Learning Curve Results",
        "",
        f"- updated_utc: {utc_now()}",
        f"- completed_cells: {len(rows)}",
        "",
        "## Results by Size",
        "",
        "| size | seeds | mean_recall | std_recall | mean_precision | mean_f1 |",
        "|-----:|------:|------------:|-----------:|---------------:|--------:|",
    ]
    for size in sorted(by_size.keys()):
        group = by_size[size]
        recalls = [r["recall"] for r in group]
        precisions = [r["precision"] for r in group]
        f1s = [r["f1"] for r in group]
        mean_r = sum(recalls) / len(recalls)
        std_r = (sum((x - mean_r) ** 2 for x in recalls) / len(recalls)) ** 0.5
        mean_p = sum(precisions) / len(precisions)
        mean_f = sum(f1s) / len(f1s)
        lines.append(
            f"| {size:,} | {len(group)} | {mean_r:.4f} | {std_r:.4f} | {mean_p:.4f} | {mean_f:.4f} |"
        )

    summary_md = output_dir / "summary.md"
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    default_workspace_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Phase 1: Learning curve estimation")
    parser.add_argument("--workspace-root", type=Path, default=default_workspace_root)
    parser.add_argument(
        "--curation-dir",
        type=Path,
        required=True,
        help="Directory containing the re-curated v6 corpus (with 70/12/18 splits)",
    )
    parser.add_argument(
        "--base-train-config",
        type=Path,
        default=Path("parapet-runner/configs/iteration_v1_calibrated.yaml"),
    )
    parser.add_argument(
        "--parapet-eval-bin",
        type=Path,
        default=Path("parapet/target/release/parapet-eval.exe"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-bin", type=str, default="python")
    parser.add_argument(
        "--sizes",
        type=str,
        default="1000,2000,5000,10000,15000,20000",
        help="Comma-separated training subset sizes",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44,45,46",
        help="Comma-separated random seeds",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    runner_cwd = workspace_root / "parapet-runner"
    curation_dir = Path(args.curation_dir)
    if not curation_dir.is_absolute():
        curation_dir = (workspace_root / curation_dir).resolve()

    parapet_eval_bin = Path(args.parapet_eval_bin)
    if not parapet_eval_bin.is_absolute():
        parapet_eval_bin = (workspace_root / parapet_eval_bin).resolve()

    base_config_path = Path(args.base_train_config)
    if not base_config_path.is_absolute():
        base_config_path = (workspace_root / base_config_path).resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (workspace_root / output_dir).resolve()

    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Load train split once
    train_path = curation_dir / "train.yaml"
    if not train_path.exists():
        print(f"ERROR: train.yaml not found at {train_path}", file=sys.stderr)
        return 1

    print(f"Loading train split from {train_path}...")
    all_train_samples = load_yaml_samples(train_path)
    print(f"  {len(all_train_samples):,} training samples loaded")

    # Add full_train as a size point
    full_size = len(all_train_samples)
    effective_sizes = [s for s in sizes if s < full_size] + [full_size]
    print(f"  Size ladder: {effective_sizes}")
    print(f"  Seeds: {seeds}")

    # Verify val and holdout exist
    val_path = curation_dir / "val.yaml"
    holdout_path = curation_dir / "holdout.yaml"
    for p in [val_path, holdout_path]:
        if not p.exists():
            print(f"ERROR: {p.name} not found at {p}", file=sys.stderr)
            return 1

    # Build cells
    cells = [
        CurveCell(size=size, seed=seed)
        for size in effective_sizes
        for seed in seeds
    ]
    if args.limit > 0:
        cells = cells[: args.limit]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write sweep manifest
    sweep_manifest = {
        "created_utc": utc_now(),
        "workspace_root": str(workspace_root),
        "curation_dir": str(curation_dir),
        "base_train_config": str(base_config_path),
        "parapet_eval_bin": str(parapet_eval_bin),
        "full_train_size": full_size,
        "sizes": effective_sizes,
        "seeds": seeds,
        "cells": [{"tag": c.tag, "size": c.size, "seed": c.seed} for c in cells],
    }
    (output_dir / "sweep_manifest.json").write_text(
        json.dumps(sweep_manifest, indent=2), encoding="utf-8"
    )

    print(f"\nOutput: {output_dir}")
    print(f"Cells: {len(cells)}")

    failures = 0
    for idx, cell in enumerate(cells, start=1):
        cell_dir = output_dir / cell.tag
        result_path = cell_dir / "result.json"
        status_path = cell_dir / "status.json"

        if result_path.exists():
            print(f"[{idx}/{len(cells)}] SKIP completed {cell.tag}")
            continue

        if status_path.exists() and not args.rerun_failed:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "failed":
                print(f"[{idx}/{len(cells)}] SKIP failed {cell.tag} (use --rerun-failed)")
                continue

        cell_dir.mkdir(parents=True, exist_ok=True)

        # Subsample
        subset = stratified_subsample(all_train_samples, cell.size, cell.seed)
        actual_size = len(subset)

        # Write temp curated directory
        temp_curated = cell_dir / "curated"
        temp_curated.mkdir(parents=True, exist_ok=True)
        write_yaml_samples(subset, temp_curated / "train.yaml")

        # Symlink or copy val and holdout from the real curation
        for split_name in ("val.yaml", "holdout.yaml"):
            src = curation_dir / split_name
            dst = temp_curated / split_name
            if dst.exists():
                dst.unlink()
            shutil.copy2(src, dst)

        # Build minimal manifest
        manifest_path = build_minimal_manifest(curation_dir, temp_curated, actual_size)

        status_payload = {
            "tag": cell.tag,
            "status": "running",
            "started_utc": utc_now(),
            "size": cell.size,
            "actual_size": actual_size,
            "seed": cell.seed,
        }
        (cell_dir / "status.json").write_text(
            json.dumps(status_payload, indent=2), encoding="utf-8"
        )

        cell_run_dir = cell_dir / "run"
        cmd = [
            args.python_bin,
            "-m",
            "parapet_runner.runner",
            "run",
            "--workspace-root",
            str(workspace_root),
            "--curation-manifest",
            str(manifest_path),
            "--train-config",
            str(base_config_path),
            "--output-dir",
            str(cell_run_dir.resolve()),
            "--parapet-eval-bin",
            str(parapet_eval_bin),
            "--run-id",
            f"lc_{cell.tag}",
            "--pg2-mode",
            "off",
            "--skip-output-hash-verify",
            "--skip-split-hash-verify",
        ]

        print(f"[{idx}/{len(cells)}] RUN {cell.tag} (n={actual_size})")
        rc, elapsed = run_logged(cmd, cwd=runner_cwd, log_path=cell_dir / "runner.log")
        run_manifest_path = cell_run_dir / "run_manifest.json"

        if rc == 0 and run_manifest_path.exists():
            result_payload = parse_run_result(run_manifest_path, cell=cell)
            result_payload["actual_size"] = actual_size
            result_payload["elapsed_sec"] = round(elapsed, 3)
            result_payload["completed_utc"] = utc_now()
            result_path.write_text(
                json.dumps(result_payload, indent=2), encoding="utf-8"
            )
            (cell_dir / "status.json").write_text(
                json.dumps({
                    "tag": cell.tag,
                    "status": "completed",
                    "elapsed_sec": round(elapsed, 3),
                    "size": cell.size,
                    "actual_size": actual_size,
                    "seed": cell.seed,
                }, indent=2),
                encoding="utf-8",
            )
            print(
                f"           F1={result_payload['f1']:.4f} "
                f"P={result_payload['precision']:.4f} "
                f"R={result_payload['recall']:.4f} "
                f"thr={result_payload['threshold']}"
            )
        else:
            failures += 1
            error_text = (
                f"runner failed (rc={rc}); see {cell_dir / 'runner.log'}"
                if rc != 0
                else f"run_manifest missing: {run_manifest_path}"
            )
            (cell_dir / "status.json").write_text(
                json.dumps({
                    "tag": cell.tag,
                    "status": "failed",
                    "elapsed_sec": round(elapsed, 3),
                    "error": error_text,
                    "size": cell.size,
                    "seed": cell.seed,
                }, indent=2),
                encoding="utf-8",
            )
            print(f"           FAIL {error_text}", file=sys.stderr)
            if not args.continue_on_error:
                write_aggregates(output_dir)
                return 1

        write_aggregates(output_dir)

    write_aggregates(output_dir)
    if failures:
        print(f"\nLearning curve completed with {failures} failed cell(s)", file=sys.stderr)
        return 1
    print("\nLearning curve completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
