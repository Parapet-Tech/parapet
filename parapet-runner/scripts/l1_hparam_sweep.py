"""Checkpointed L1 hyperparameter sweep over parapet-runner train configs.

This is intended for fast-but-valid model-family iteration. Each sweep cell:

1. materializes a concrete TrainConfig YAML
2. runs `python -m parapet_runner.runner run`
3. stores immutable artifacts under one cell directory
4. updates aggregate CSV/JSONL/Markdown summaries

Important:
- Leave recompilation enabled for weight sweeps. The runner installs trained
  weights into the Rust crate only when recompilation is active.
- PG2 is optional and off by default for the sweep. Re-enable it for the final
  winner comparison if needed.

Example:
  cd parapet
  python parapet-runner/scripts/l1_hparam_sweep.py \
    --curation-manifest parapet-data/curated/v5_5k_r8_ablation_hashfix/manifest.json \
    --output-dir parapet-runner/runs/v5_5k_r8_hparam_sweep_stage1
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SweepCell:
    c: float
    max_features: int
    min_df: int
    class_weight: str
    ngram_min: int
    ngram_max: int
    seed: int

    @property
    def tag(self) -> str:
        return (
            f"c{slug_float(self.c)}"
            f"_f{self.max_features}"
            f"_df{self.min_df}"
            f"_cw{self.class_weight}"
            f"_ng{self.ngram_min}-{self.ngram_max}"
            f"_s{self.seed}"
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slug_float(value: float) -> str:
    text = f"{value:.6g}"
    text = text.replace("-", "m").replace(".", "p")
    return text


def parse_float_grid(raw: str) -> list[float]:
    values: list[float] = []
    for part in (item.strip() for item in raw.split(",")):
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("expected at least one float")
    return values


def parse_int_grid(raw: str) -> list[int]:
    values: list[int] = []
    for part in (item.strip() for item in raw.split(",")):
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_text_grid(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one value")
    return values


def parse_ngram_grid(raw: str) -> list[tuple[int, int]]:
    values: list[tuple[int, int]] = []
    for part in (item.strip() for item in raw.split(",")):
        if not part:
            continue
        toks = part.split(":")
        if len(toks) != 2:
            raise ValueError(f"invalid ngram pair '{part}' (expected MIN:MAX)")
        n_min = int(toks[0])
        n_max = int(toks[1])
        if n_min <= 0 or n_max <= 0:
            raise ValueError(f"invalid ngram pair '{part}' (must be positive)")
        if n_min > n_max:
            raise ValueError(f"invalid ngram pair '{part}' (MIN > MAX)")
        values.append((n_min, n_max))
    if not values:
        raise ValueError("expected at least one ngram pair")
    return values


def resolve_path(path: Path, *, base_dir: Path, must_exist: bool = False) -> Path:
    if path.is_absolute():
        return path.resolve()

    cwd_candidate = (Path.cwd() / path).resolve()
    base_candidate = (base_dir / path).resolve()
    if must_exist:
        if cwd_candidate.exists():
            return cwd_candidate
        if base_candidate.exists():
            return base_candidate
    return cwd_candidate


def load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected YAML mapping")
    return dict(raw)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def build_cells(
    *,
    c_grid: list[float],
    max_features_grid: list[int],
    min_df_grid: list[int],
    class_weight_grid: list[str],
    ngram_grid: list[tuple[int, int]],
    seed_grid: list[int],
) -> list[SweepCell]:
    cells = [
        SweepCell(
            c=c,
            max_features=max_features,
            min_df=min_df,
            class_weight=class_weight,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
            seed=seed,
        )
        for c, max_features, min_df, class_weight, (ngram_min, ngram_max), seed in itertools.product(
            c_grid,
            max_features_grid,
            min_df_grid,
            class_weight_grid,
            ngram_grid,
            seed_grid,
        )
    ]
    return sorted(cells, key=lambda cell: cell.tag)


def materialize_train_config(base_config: dict[str, Any], cell: SweepCell) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    config["c"] = cell.c
    config["max_features"] = cell.max_features
    config["min_df"] = cell.min_df
    config["class_weight"] = cell.class_weight
    config["ngram_min"] = cell.ngram_min
    config["ngram_max"] = cell.ngram_max
    config["seed"] = cell.seed
    return config


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


def parse_run_result(run_manifest_path: Path, *, cell: SweepCell) -> dict[str, Any]:
    payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    eval_result = payload["eval_result"]
    baseline_results = payload.get("baseline_results") or {}
    pg2 = baseline_results.get("pg2")
    row = {
        **asdict(cell),
        "tag": cell.tag,
        "run_id": payload.get("run_id"),
        "status": "completed",
        "threshold": eval_result["threshold"],
        "f1": eval_result["f1"],
        "precision": eval_result["precision"],
        "recall": eval_result["recall"],
        "false_positives": eval_result["false_positives"],
        "false_negatives": eval_result["false_negatives"],
        "holdout_size": eval_result["holdout_size"],
        "semantic_parity_hash": payload.get("semantic_parity_hash"),
        "run_manifest": str(run_manifest_path),
        "pg2_f1": pg2["f1"] if pg2 else None,
        "pg2_precision": pg2["precision"] if pg2 else None,
        "pg2_recall": pg2["recall"] if pg2 else None,
    }
    return row


def write_aggregates(output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for cell_dir in sorted(path for path in output_dir.iterdir() if path.is_dir()):
        if cell_dir.name.startswith("_"):
            continue
        result_path = cell_dir / "result.json"
        status_path = cell_dir / "status.json"
        if result_path.exists():
            rows.append(json.loads(result_path.read_text(encoding="utf-8")))
        elif status_path.exists():
            failures.append(json.loads(status_path.read_text(encoding="utf-8")))

    rows.sort(key=lambda row: (-float(row["f1"]), row["tag"]))
    failures.sort(key=lambda row: row["tag"])

    results_jsonl = output_dir / "results.jsonl"
    results_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    summary_csv = output_dir / "summary.csv"
    fieldnames = [
        "tag",
        "status",
        "c",
        "max_features",
        "min_df",
        "class_weight",
        "ngram_min",
        "ngram_max",
        "seed",
        "f1",
        "precision",
        "recall",
        "false_positives",
        "false_negatives",
        "threshold",
        "holdout_size",
        "pg2_f1",
        "run_id",
        "semantic_parity_hash",
        "run_manifest",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    best = rows[0] if rows else None
    summary_md = output_dir / "summary.md"
    lines = [
        "# L1 Hyperparameter Sweep",
        "",
        f"- updated_utc: {utc_now()}",
        f"- completed_cells: {len(rows)}",
        f"- failed_cells: {len(failures)}",
    ]
    if best:
        lines.extend(
            [
                f"- best_tag: `{best['tag']}`",
                f"- best_f1: `{best['f1']:.4f}`",
                f"- best_precision: `{best['precision']:.4f}`",
                f"- best_recall: `{best['recall']:.4f}`",
                f"- best_threshold: `{best['threshold']}`",
            ]
        )
    lines.extend(["", "## Top Results", "", "| tag | F1 | P | R | FP | FN | threshold |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in rows[:10]:
        lines.append(
            f"| `{row['tag']}` | {row['f1']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['false_positives']} | {row['false_negatives']} | "
            f"{row['threshold']} |"
        )
    if failures:
        lines.extend(["", "## Failed Cells", ""])
        for row in failures:
            lines.append(f"- `{row['tag']}`: {row.get('error', 'failed')}")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    default_workspace_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Checkpointed L1 hyperparameter sweep")
    parser.add_argument("--workspace-root", type=Path, default=default_workspace_root)
    parser.add_argument("--curation-manifest", type=Path, required=True)
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("parapet-runner/runs/l1_hparam_sweep"),
    )
    parser.add_argument("--python-bin", type=str, default="python")
    parser.add_argument(
        "--calibration-thresholds",
        type=str,
        default="-1.5,-1.0,-0.5,0.0,0.5,1.0",
    )
    parser.add_argument("--pg2-mode", choices=["off", "on"], default="off")
    parser.add_argument(
        "--c-grid",
        type=str,
        default="0.03,0.1,0.3,1.0",
        help="Comma-separated C values",
    )
    parser.add_argument(
        "--max-features-grid",
        type=str,
        default="5000,10000,15000",
        help="Comma-separated max_features values",
    )
    parser.add_argument("--min-df-grid", type=str, default=None)
    parser.add_argument("--class-weight-grid", type=str, default=None)
    parser.add_argument("--ngram-grid", type=str, default=None, help="Comma-separated MIN:MAX pairs")
    parser.add_argument("--seed-grid", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument(
        "--skip-recompile",
        action="store_true",
        help="Unsafe for weight sweeps; only use if the eval binary already has the target weights installed",
    )
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    runner_cwd = workspace_root / "parapet-runner"
    curation_manifest = resolve_path(Path(args.curation_manifest), base_dir=workspace_root, must_exist=True)
    base_train_config_path = resolve_path(
        Path(args.base_train_config),
        base_dir=workspace_root,
        must_exist=True,
    )
    parapet_eval_bin = resolve_path(Path(args.parapet_eval_bin), base_dir=workspace_root, must_exist=True)
    output_dir = resolve_path(Path(args.output_dir), base_dir=workspace_root)

    base_config = load_yaml(base_train_config_path)
    min_df_default = int(base_config.get("min_df", 5))
    class_weight_default = str(base_config.get("class_weight", "none"))
    ngram_default = (int(base_config.get("ngram_min", 3)), int(base_config.get("ngram_max", 5)))
    seed_default = int(base_config.get("seed", 42))

    cells = build_cells(
        c_grid=parse_float_grid(args.c_grid),
        max_features_grid=parse_int_grid(args.max_features_grid),
        min_df_grid=parse_int_grid(args.min_df_grid) if args.min_df_grid else [min_df_default],
        class_weight_grid=parse_text_grid(args.class_weight_grid) if args.class_weight_grid else [class_weight_default],
        ngram_grid=parse_ngram_grid(args.ngram_grid) if args.ngram_grid else [ngram_default],
        seed_grid=parse_int_grid(args.seed_grid) if args.seed_grid else [seed_default],
    )

    if args.limit > 0:
        cells = cells[: args.limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_manifest = {
        "created_utc": utc_now(),
        "workspace_root": str(workspace_root),
        "curation_manifest": str(curation_manifest),
        "base_train_config": str(base_train_config_path),
        "parapet_eval_bin": str(parapet_eval_bin),
        "pg2_mode": args.pg2_mode,
        "calibration_thresholds": args.calibration_thresholds,
        "skip_recompile": args.skip_recompile,
        "cells": [asdict(cell) | {"tag": cell.tag} for cell in cells],
    }
    write_json(output_dir / "sweep_manifest.json", sweep_manifest)

    print(f"Sweep output: {output_dir}")
    print(f"Cells: {len(cells)}")

    failures = 0
    for idx, cell in enumerate(cells, start=1):
        cell_dir = output_dir / cell.tag
        result_path = cell_dir / "result.json"
        status_path = cell_dir / "status.json"
        cell_run_dir = cell_dir / "run"

        if result_path.exists():
            print(f"[{idx}/{len(cells)}] SKIP completed {cell.tag}")
            continue

        if status_path.exists() and not args.rerun_failed:
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))
            if status_payload.get("status") == "failed":
                print(f"[{idx}/{len(cells)}] SKIP failed {cell.tag} (use --rerun-failed)")
                continue

        cell_dir.mkdir(parents=True, exist_ok=True)
        cell_config = materialize_train_config(base_config, cell)
        write_yaml(cell_dir / "train_config.yaml", cell_config)
        write_json(cell_dir / "cell.json", asdict(cell) | {"tag": cell.tag})

        status_payload = {
            "tag": cell.tag,
            "status": "running",
            "started_utc": utc_now(),
            **asdict(cell),
        }
        write_json(status_path, status_payload)

        cmd = [
            args.python_bin,
            "-m",
            "parapet_runner.runner",
            "run",
            "--workspace-root",
            str(workspace_root),
            "--curation-manifest",
            str(curation_manifest),
            "--train-config",
            str((cell_dir / "train_config.yaml").resolve()),
            "--output-dir",
            str(cell_run_dir.resolve()),
            "--parapet-eval-bin",
            str(parapet_eval_bin),
            "--run-id",
            f"sweep_{cell.tag}",
            "--pg2-mode",
            args.pg2_mode,
            f"--calibration-thresholds={args.calibration_thresholds}",
        ]
        if args.skip_recompile:
            cmd.append("--skip-recompile")

        print(f"[{idx}/{len(cells)}] RUN {cell.tag}")
        rc, elapsed = run_logged(cmd, cwd=runner_cwd, log_path=cell_dir / "runner.log")
        run_manifest_path = cell_run_dir / "run_manifest.json"

        if rc == 0 and run_manifest_path.exists():
            result_payload = parse_run_result(run_manifest_path, cell=cell)
            result_payload["elapsed_sec"] = round(elapsed, 3)
            result_payload["completed_utc"] = utc_now()
            write_json(result_path, result_payload)
            write_json(
                status_path,
                {
                    "tag": cell.tag,
                    "status": "completed",
                    "elapsed_sec": round(elapsed, 3),
                    **asdict(cell),
                },
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
            write_json(
                status_path,
                {
                    "tag": cell.tag,
                    "status": "failed",
                    "elapsed_sec": round(elapsed, 3),
                    "error": error_text,
                    **asdict(cell),
                },
            )
            print(f"           FAIL {error_text}", file=sys.stderr)
            if not args.continue_on_error:
                write_aggregates(output_dir)
                return 1

        write_aggregates(output_dir)

    write_aggregates(output_dir)
    if failures:
        print(f"Sweep completed with {failures} failed cell(s)", file=sys.stderr)
        return 1
    print("Sweep completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
