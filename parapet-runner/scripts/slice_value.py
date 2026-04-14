"""Phase 2: Marginal Slice Value — Where Data Helps Most.

Measures value(s) = d_metric / d_rows for each slice to answer the sourcing
question: which slices are worth finding new data for?

Design:
  - Subsplit the train set 90/10 (base / reserve), stratified by label x language
  - Train a base model on the 90% base with 3 seeds → baseline metrics
  - For each slice: augment base with reserve rows from that slice → delta metrics
  - value(s) = mean(delta) / topup_count

The 90/10 subsplit is necessary because Phase 1 showed N*=full_train. We sacrifice
~10% of training data to create a clean topup pool. The base model at ~15.7K is
still in the plateau region of the learning curve.

Example:
    cd parapet
    python parapet-runner/scripts/slice_value.py \
      --curation-dir parapet-data/curated/v6_25k_experiment \
      --output-dir parapet-runner/runs/phase2_slice_value
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


def classify_slice(row: dict[str, Any]) -> list[str]:
    """Return all slice tags a row belongs to."""
    label = row.get("label", "")
    lang = row.get("language", "EN")
    reason = row.get("reason", "")
    source = row.get("source", "")

    label_tag = "attack" if label == "malicious" else "benign"
    tags = [f"{lang}_{label_tag}"]

    if reason == "obfuscation" and label == "malicious":
        tags.append("obfuscated")
    if reason == "indirect_injection" and label == "malicious":
        tags.append("indirect")
    if "discussion" in source.lower() or "wildjailbreak" in source.lower():
        if label == "benign":
            tags.append("discussion_benign")

    return tags


def stratified_subsplit(
    samples: list[dict[str, Any]],
    reserve_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split samples into base (1 - reserve_ratio) and reserve (reserve_ratio),
    stratified by label x language."""
    rng = random.Random(seed)

    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        key = (s.get("label", "?"), s.get("language", "?"))
        strata[key].append(s)

    base: list[dict[str, Any]] = []
    reserve: list[dict[str, Any]] = []

    for key in sorted(strata.keys()):
        group = list(strata[key])
        rng.shuffle(group)
        n_reserve = max(1, round(len(group) * reserve_ratio))
        reserve.extend(group[:n_reserve])
        base.extend(group[n_reserve:])

    rng.shuffle(base)
    rng.shuffle(reserve)
    return base, reserve


def build_minimal_manifest(
    curation_dir: Path,
    temp_dir: Path,
) -> Path:
    """Build a minimal manifest.json for the runner."""
    original_manifest = json.loads(
        (curation_dir / "manifest.json").read_text(encoding="utf-8")
    )

    train_samples = load_yaml_samples(temp_dir / "train.yaml")
    train_hashes = sorted(content_hash(s["content"]) for s in train_samples)

    manifest = {
        "spec_name": original_manifest.get("spec_name", "slice_value"),
        "spec_version": original_manifest.get("spec_version", "0.0.0"),
        "spec_hash": original_manifest.get("spec_hash", ""),
        "seed": original_manifest.get("seed", 42),
        "timestamp": utc_now(),
        "source_hashes": original_manifest.get("source_hashes", {}),
        "source_metadata": original_manifest.get("source_metadata", {}),
        "output_path": str(temp_dir / "curated.yaml"),
        "output_hash": "subsampled",
        "semantic_hash": "subsampled",
        "total_samples": len(train_samples),
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


def parse_run_result(run_manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    return payload["eval_result"]


def run_cell(
    *,
    tag: str,
    train_samples: list[dict[str, Any]],
    curation_dir: Path,
    output_dir: Path,
    workspace_root: Path,
    train_config_path: Path,
    parapet_eval_bin: Path,
    python_bin: str,
) -> dict[str, Any] | None:
    """Run one training + eval cell. Returns eval_result dict or None on failure."""
    cell_dir = output_dir / tag
    result_path = cell_dir / "result.json"

    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    cell_dir.mkdir(parents=True, exist_ok=True)
    temp_curated = cell_dir / "curated"
    temp_curated.mkdir(parents=True, exist_ok=True)

    write_yaml_samples(train_samples, temp_curated / "train.yaml")
    for split_name in ("val.yaml", "holdout.yaml"):
        src = curation_dir / split_name
        dst = temp_curated / split_name
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)

    manifest_path = build_minimal_manifest(curation_dir, temp_curated)

    cell_run_dir = cell_dir / "run"
    cmd = [
        python_bin, "-m", "parapet_runner.runner", "run",
        "--workspace-root", str(workspace_root),
        "--curation-manifest", str(manifest_path),
        "--train-config", str(train_config_path),
        "--output-dir", str(cell_run_dir.resolve()),
        "--parapet-eval-bin", str(parapet_eval_bin),
        "--run-id", f"sv_{tag}",
        "--pg2-mode", "off",
        "--skip-output-hash-verify",
        "--skip-split-hash-verify",
    ]

    runner_cwd = workspace_root / "parapet-runner"
    rc, elapsed = run_logged(cmd, cwd=runner_cwd, log_path=cell_dir / "runner.log")
    run_manifest_path = cell_run_dir / "run_manifest.json"

    if rc == 0 and run_manifest_path.exists():
        eval_result = parse_run_result(run_manifest_path)
        result = {
            "tag": tag,
            "status": "completed",
            "elapsed_sec": round(elapsed, 3),
            "train_size": len(train_samples),
            **eval_result,
        }
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    else:
        error = f"runner failed (rc={rc}); see {cell_dir / 'runner.log'}"
        (cell_dir / "status.json").write_text(
            json.dumps({"tag": tag, "status": "failed", "error": error}, indent=2),
            encoding="utf-8",
        )
        print(f"  FAIL {tag}: {error}", file=sys.stderr)
        return None


SLICES = [
    "EN_attack", "EN_benign",
    "RU_attack", "RU_benign",
    "ZH_attack", "ZH_benign",
    "AR_attack", "AR_benign",
    "discussion_benign", "obfuscated", "indirect",
]


def write_summary(output_dir: Path, baseline: dict, slice_results: dict) -> None:
    """Write Phase 2 summary with slice values."""
    lines = [
        "# Phase 2: Marginal Slice Value Results",
        "",
        f"- updated_utc: {utc_now()}",
        f"- baseline_recall: {baseline['mean_recall']:.4f}",
        f"- baseline_precision: {baseline['mean_precision']:.4f}",
        f"- baseline_f1: {baseline['mean_f1']:.4f}",
        "",
        "## Slice Values",
        "",
        "| slice | topup_rows | mean_d_recall | mean_d_precision | mean_d_f1 | value_pp_per_1k | signal |",
        "|-------|----------:|-------------:|----------------:|---------:|----------------:|--------|",
    ]

    ranked = sorted(
        slice_results.items(),
        key=lambda kv: abs(kv[1].get("value_recall_pp_per_1k", 0)),
        reverse=True,
    )

    for slice_name, sr in ranked:
        topup = sr.get("topup_count", 0)
        dr = sr.get("mean_delta_recall", 0)
        dp = sr.get("mean_delta_precision", 0)
        df = sr.get("mean_delta_f1", 0)
        val = sr.get("value_recall_pp_per_1k", 0)
        signal = "strong" if abs(val) > 0.5 else ("weak" if abs(val) > 0.1 else "none")
        lines.append(
            f"| {slice_name:<20} | {topup:>4} | {dr:>+.4f} | {dp:>+.4f} | {df:>+.4f} | {val:>+.3f} | {signal} |"
        )

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    default_workspace_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Phase 2: Marginal slice value")
    parser.add_argument("--workspace-root", type=Path, default=default_workspace_root)
    parser.add_argument("--curation-dir", type=Path, required=True)
    parser.add_argument(
        "--base-train-config", type=Path,
        default=Path("parapet-runner/configs/phase1_learning_curve.yaml"),
    )
    parser.add_argument(
        "--parapet-eval-bin", type=Path,
        default=Path("parapet/target/release/parapet-eval.exe"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-bin", type=str, default="python")
    parser.add_argument("--reserve-ratio", type=float, default=0.10)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    curation_dir = Path(args.curation_dir)
    if not curation_dir.is_absolute():
        curation_dir = (workspace_root / curation_dir).resolve()
    parapet_eval_bin = Path(args.parapet_eval_bin)
    if not parapet_eval_bin.is_absolute():
        parapet_eval_bin = (workspace_root / parapet_eval_bin).resolve()
    train_config_path = Path(args.base_train_config)
    if not train_config_path.is_absolute():
        train_config_path = (workspace_root / train_config_path).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (workspace_root / output_dir).resolve()

    # Load and subsplit train data
    train_path = curation_dir / "train.yaml"
    print(f"Loading train split from {train_path}...")
    all_train = load_yaml_samples(train_path)
    print(f"  {len(all_train):,} total training samples")

    base_samples, reserve_samples = stratified_subsplit(
        all_train, args.reserve_ratio, seed=99,
    )
    print(f"  Base: {len(base_samples):,}, Reserve: {len(reserve_samples):,}")

    # Index reserve by slice
    reserve_by_slice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reserve_samples:
        for tag in classify_slice(row):
            reserve_by_slice[tag].append(row)

    print("\nReserve inventory:")
    for s in SLICES:
        print(f"  {s:<25} {len(reserve_by_slice.get(s, [])):>5}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment manifest
    manifest = {
        "created_utc": utc_now(),
        "base_size": len(base_samples),
        "reserve_size": len(reserve_samples),
        "reserve_ratio": args.reserve_ratio,
        "slices": SLICES,
        "reserve_by_slice": {s: len(reserve_by_slice.get(s, [])) for s in SLICES},
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # Step 1: Train baseline (base only, single deterministic run)
    print("\n=== Baseline (base only) ===")
    print(f"  [baseline]", end=" ", flush=True)
    baseline_result = run_cell(
        tag="baseline",
        train_samples=base_samples,
        curation_dir=curation_dir,
        output_dir=output_dir,
        workspace_root=workspace_root,
        train_config_path=train_config_path,
        parapet_eval_bin=parapet_eval_bin,
        python_bin=args.python_bin,
    )
    if not baseline_result:
        print("ERROR: baseline failed", file=sys.stderr)
        return 1

    baseline = {
        "mean_recall": baseline_result["recall"],
        "mean_precision": baseline_result["precision"],
        "mean_f1": baseline_result["f1"],
    }
    print(f"F1={baseline['mean_f1']:.4f} R={baseline['mean_recall']:.4f} "
          f"P={baseline['mean_precision']:.4f}")

    # Step 2: Topup experiments (one deterministic run per slice)
    slice_results: dict[str, dict] = {}

    for slice_name in SLICES:
        topup_pool = reserve_by_slice.get(slice_name, [])
        if not topup_pool:
            print(f"\n=== {slice_name}: SKIP (0 reserve rows) ===")
            slice_results[slice_name] = {
                "topup_count": 0,
                "mean_delta_recall": 0,
                "mean_delta_precision": 0,
                "mean_delta_f1": 0,
                "value_recall_pp_per_1k": 0,
                "skipped": True,
            }
            continue

        topup_count = len(topup_pool)
        augmented = base_samples + topup_pool
        print(f"\n=== {slice_name}: +{topup_count} rows (total {len(augmented)}) ===")

        tag = f"topup_{slice_name}"
        print(f"  [{tag}]", end=" ", flush=True)
        result = run_cell(
            tag=tag,
            train_samples=augmented,
            curation_dir=curation_dir,
            output_dir=output_dir,
            workspace_root=workspace_root,
            train_config_path=train_config_path,
            parapet_eval_bin=parapet_eval_bin,
            python_bin=args.python_bin,
        )

        if result:
            dr = result["recall"] - baseline["mean_recall"]
            dp = result["precision"] - baseline["mean_precision"]
            df = result["f1"] - baseline["mean_f1"]
            value = (dr * 100) / (topup_count / 1000) if topup_count > 0 else 0

            slice_results[slice_name] = {
                "topup_count": topup_count,
                "recall": result["recall"],
                "precision": result["precision"],
                "f1": result["f1"],
                "mean_delta_recall": dr,
                "mean_delta_precision": dp,
                "mean_delta_f1": df,
                "value_recall_pp_per_1k": value,
            }
            print(f"F1={result['f1']:.4f} R={result['recall']:.4f} P={result['precision']:.4f} "
                  f"d_recall={dr:+.4f} value={value:+.2f}pp/1K")
        elif not args.continue_on_error:
            return 1

        write_summary(output_dir, baseline, slice_results)

    # Final summary
    write_summary(output_dir, baseline, slice_results)
    (output_dir / "slice_results.json").write_text(
        json.dumps({"baseline": baseline, "slices": slice_results}, indent=2),
        encoding="utf-8",
    )

    print("\n=== Phase 2 Complete ===")
    print(f"Results: {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
