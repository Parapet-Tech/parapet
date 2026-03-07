"""Run checkpointed mirror-v2 mix ablations and persist results incrementally.

This script is designed for long runs where interruptions are expected.
Each completed cell writes an immutable `result.json`. Aggregate artifacts
(`results.jsonl`, `summary.csv`, `summary.md`) are regenerated after every
successful cell so partial progress is never lost.

Example:
  cd parapet
  python parapet-runner/scripts/v2_mix_ablation.py \
    --ratios 100:0,70:30,50:50,30:70,0:100 \
    --seeds 42,43,44 \
    --threshold -0.5 \
    --continue-on-error
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RatioSpec:
    mirror_pct: int
    non_mirror_pct: int

    @property
    def label(self) -> str:
        return f"{self.mirror_pct}:{self.non_mirror_pct}"

    @property
    def tag(self) -> str:
        return f"m{self.mirror_pct}_n{self.non_mirror_pct}"


@dataclass(frozen=True)
class EvalTarget:
    name: str
    dataset_dir: Path
    source: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_list_int(raw: str) -> list[int]:
    out: list[int] = []
    for part in (s.strip() for s in raw.split(",")):
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("expected at least one integer")
    return out


def parse_ratios(raw: str) -> list[RatioSpec]:
    specs: list[RatioSpec] = []
    for part in (s.strip() for s in raw.split(",")):
        if not part:
            continue
        toks = part.split(":")
        if len(toks) != 2:
            raise ValueError(f"invalid ratio '{part}' (expected A:B)")
        a = int(toks[0])
        b = int(toks[1])
        if a < 0 or b < 0:
            raise ValueError(f"invalid ratio '{part}' (negative values not allowed)")
        if a + b == 0:
            raise ValueError(f"invalid ratio '{part}' (A+B must be > 0)")
        specs.append(RatioSpec(mirror_pct=a, non_mirror_pct=b))
    if not specs:
        raise ValueError("expected at least one ratio")
    return specs


def norm_label(raw: Any) -> str:
    val = str(raw or "").strip().lower()
    if val in {"malicious", "attack", "positive"}:
        return "malicious"
    if val in {"benign", "negative"}:
        return "benign"
    raise ValueError(f"unsupported label '{raw}'")


def load_structured(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid json: {exc}") from exc
        return rows
    if suffix in {".yaml", ".yml"}:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"{path}: expected top-level YAML list")
        return [dict(row) for row in loaded]
    raise ValueError(f"unsupported file type: {path}")


def write_yaml_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(rows, allow_unicode=True, sort_keys=False, width=200),
        encoding="utf-8",
    )


def split_label_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    attacks: list[dict[str, Any]] = []
    benign: list[dict[str, Any]] = []
    for row in rows:
        label = norm_label(row.get("label"))
        out_row = dict(row)
        out_row["label"] = label
        if label == "malicious":
            attacks.append(out_row)
        else:
            benign.append(out_row)
    return attacks, benign


def content_hash(row: dict[str, Any]) -> str:
    text = str(row.get("content") or "")
    return sha256_text(text)


def dedup_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        h = content_hash(row)
        if h in seen:
            continue
        seen.add(h)
        out.append(row)
    return out


def filter_excluding(rows: list[dict[str, Any]], blocked_hashes: set[str]) -> tuple[list[dict[str, Any]], int]:
    out = []
    dropped = 0
    for row in rows:
        if content_hash(row) in blocked_hashes:
            dropped += 1
            continue
        out.append(row)
    return out, dropped


def sample_mix(
    *,
    mirror_rows: list[dict[str, Any]],
    non_rows: list[dict[str, Any]],
    ratio: RatioSpec,
    target_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    total = ratio.mirror_pct + ratio.non_mirror_pct
    mirror_n = (target_count * ratio.mirror_pct) // total
    non_n = target_count - mirror_n
    if mirror_n > len(mirror_rows):
        raise ValueError(
            f"ratio {ratio.label} needs {mirror_n} mirror rows but pool has {len(mirror_rows)}"
        )
    if non_n > len(non_rows):
        raise ValueError(
            f"ratio {ratio.label} needs {non_n} non-mirror rows but pool has {len(non_rows)}"
        )
    rng = random.Random(seed)
    picked_mirror = rng.sample(mirror_rows, mirror_n) if mirror_n > 0 else []
    picked_non = rng.sample(non_rows, non_n) if non_n > 0 else []
    combined = picked_mirror + picked_non
    rng.shuffle(combined)
    return combined, {
        "mirror": mirror_n,
        "non_mirror": non_n,
        "total": len(combined),
    }


def source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("source") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def run_logged(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    ok_returncodes: set[int] | None = None,
) -> float:
    allowed = ok_returncodes if ok_returncodes is not None else {0}
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
    if proc.returncode not in allowed:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return elapsed


def parse_eval_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    layers = payload.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError(f"{path}: missing layers[0]")
    layer = layers[0]
    return {
        "tp": int(layer.get("tp", 0)),
        "fp": int(layer.get("fp", 0)),
        "fn": int(layer.get("fn_count", 0)),
        "tn": int(layer.get("tn", 0)),
        "total": int(layer.get("total", 0)),
        "precision": float(layer.get("precision", 0.0)),
        "recall": float(layer.get("recall", 0.0)),
        "f1": float(layer.get("f1", 0.0)),
        "fpr": (
            float(layer.get("fp", 0)) / float(layer.get("fp", 0) + layer.get("tn", 0))
            if int(layer.get("fp", 0)) + int(layer.get("tn", 0)) > 0
            else 0.0
        ),
    }


def collect_results(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for result_path in sorted(output_root.glob("*/result.json")):
        rows.append(json.loads(result_path.read_text(encoding="utf-8")))
    rows.sort(key=lambda r: (r["ratio"]["mirror_pct"], r["ratio"]["non_mirror_pct"], r["seed"]))
    return rows


def write_aggregate_files(output_root: Path) -> None:
    results = collect_results(output_root)

    jsonl_path = output_root / "results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    csv_path = output_root / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "cell_id",
                "ratio",
                "seed",
                "threshold",
                "holdout_f1",
                "holdout_precision",
                "holdout_recall",
                "holdout_fp",
                "holdout_fn",
                "neutral_fpr",
                "neutral_fp",
                "attack_recall",
                "attack_fn",
                "completed_at",
            ]
        )
        for row in results:
            holdout = row["metrics"]["holdout"]
            neutral = row["metrics"]["tough_neutral"]
            attack = row["metrics"]["tough_attack"]
            writer.writerow(
                [
                    row["cell_id"],
                    row["ratio"]["label"],
                    row["seed"],
                    row["threshold"],
                    f"{holdout['f1']:.6f}",
                    f"{holdout['precision']:.6f}",
                    f"{holdout['recall']:.6f}",
                    holdout["fp"],
                    holdout["fn"],
                    f"{neutral['fpr']:.6f}",
                    neutral["fp"],
                    f"{attack['recall']:.6f}",
                    attack["fn"],
                    row["completed_at"],
                ]
            )

    md_path = output_root / "summary.md"
    lines = []
    lines.append("# v2 mix ablation summary")
    lines.append("")
    lines.append(f"- updated_utc: {utc_now()}")
    lines.append(f"- completed_cells: {len(results)}")
    lines.append("")
    lines.append(
        "| cell | ratio | seed | thr | holdout F1 | holdout P | holdout R | neutral FPR | attack R |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in results:
        holdout = row["metrics"]["holdout"]
        neutral = row["metrics"]["tough_neutral"]
        attack = row["metrics"]["tough_attack"]
        lines.append(
            "| "
            f"{row['cell_id']} | "
            f"{row['ratio']['label']} | "
            f"{row['seed']} | "
            f"{row['threshold']:+.2f} | "
            f"{holdout['f1']:.4f} | "
            f"{holdout['precision']:.4f} | "
            f"{holdout['recall']:.4f} | "
            f"{neutral['fpr']:.4f} | "
            f"{attack['recall']:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Checkpointed v2 mirror:non-mirror ablation runner")
    parser.add_argument("--ratios", default="100:0,70:30,50:50,30:70,0:100")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--threshold", type=float, default=-0.5)
    parser.add_argument("--target-per-class", type=int, default=0, help="0 = auto from mirror pool")
    parser.add_argument("--max-cells", type=int, default=0, help="0 = all")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="rerun even if result.json exists")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("parapet-runner/runs/mirror_v2_mix_ablation"),
    )
    parser.add_argument(
        "--mirror-attacks",
        type=Path,
        default=Path("parapet-runner/runs/mirror_v2_tier1_full/_prepared/train/attacks.yaml"),
    )
    parser.add_argument(
        "--mirror-benign",
        type=Path,
        default=Path("parapet-runner/runs/mirror_v2_tier1_full/_prepared/train/benign.yaml"),
    )
    parser.add_argument(
        "--non-mirror-train",
        type=Path,
        default=Path("parapet-runner/runs/mirror_v2_tier1_full/_baseline_random/data/train.yaml"),
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path("scripts/train_l1_specialist.py"),
    )
    parser.add_argument(
        "--weights-target",
        type=Path,
        default=Path("parapet/src/layers/l1_weights_generalist_clean_1to1_51042.rs"),
    )
    parser.add_argument(
        "--parapet-eval",
        type=Path,
        default=Path("parapet/target/release/parapet-eval.exe"),
    )
    parser.add_argument(
        "--eval-config-base",
        type=Path,
        default=Path("schema/eval/eval_config_l1_only.yaml"),
    )
    args = parser.parse_args()

    parapet_root = Path(__file__).resolve().parents[2]
    output_root = (parapet_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    ratios = parse_ratios(args.ratios)
    seeds = parse_list_int(args.seeds)
    max_cells = args.max_cells if args.max_cells > 0 else None

    mirror_attacks_path = (parapet_root / args.mirror_attacks).resolve()
    mirror_benign_path = (parapet_root / args.mirror_benign).resolve()
    non_mirror_train_path = (parapet_root / args.non_mirror_train).resolve()
    train_script = (parapet_root / args.train_script).resolve()
    weights_target = (parapet_root / args.weights_target).resolve()
    eval_bin = (parapet_root / args.parapet_eval).resolve()
    eval_config_base = (parapet_root / args.eval_config_base).resolve()
    rust_crate_dir = (parapet_root / "parapet").resolve()

    targets = [
        EvalTarget(
            name="holdout",
            dataset_dir=(parapet_root / "parapet-runner/runs/mirror_v2_tier1_full/_eval_holdout_m0p5/dataset").resolve(),
            source="holdout",
        ),
        EvalTarget(
            name="tough_neutral",
            dataset_dir=(parapet_root / "schema/eval/challenges/tough_neutral_v1").resolve(),
            source="tough_neutral_mirror_v2_novel",
        ),
        EvalTarget(
            name="tough_attack",
            dataset_dir=(parapet_root / "schema/eval/challenges/tough_attack_v1").resolve(),
            source="tough_attack_mirror_v2_novel",
        ),
    ]

    mirror_attacks, _ = split_label_rows(dedup_rows(load_structured(mirror_attacks_path)))
    _, mirror_benign = split_label_rows(dedup_rows(load_structured(mirror_benign_path)))
    non_raw = dedup_rows(load_structured(non_mirror_train_path))
    non_attacks_raw, non_benign_raw = split_label_rows(non_raw)

    mirror_attack_hashes = {content_hash(row) for row in mirror_attacks}
    mirror_benign_hashes = {content_hash(row) for row in mirror_benign}
    non_attacks, dropped_non_attacks = filter_excluding(non_attacks_raw, mirror_attack_hashes)
    non_benign, dropped_non_benign = filter_excluding(non_benign_raw, mirror_benign_hashes)

    auto_target = min(len(mirror_attacks), len(mirror_benign))
    target_per_class = args.target_per_class if args.target_per_class > 0 else auto_target

    run_spec = {
        "created_at": utc_now(),
        "ratios": [r.label for r in ratios],
        "seeds": seeds,
        "threshold": args.threshold,
        "target_per_class": target_per_class,
        "mirror_pools": {"attacks": len(mirror_attacks), "benign": len(mirror_benign)},
        "non_mirror_pools": {
            "attacks": len(non_attacks),
            "benign": len(non_benign),
            "dropped_overlap_attacks": dropped_non_attacks,
            "dropped_overlap_benign": dropped_non_benign,
        },
        "paths": {
            "mirror_attacks": str(mirror_attacks_path),
            "mirror_benign": str(mirror_benign_path),
            "non_mirror_train": str(non_mirror_train_path),
            "train_script": str(train_script),
            "weights_target": str(weights_target),
            "eval_bin": str(eval_bin),
            "eval_config_base": str(eval_config_base),
        },
    }
    write_json(output_root / "run_spec.json", run_spec)

    cells: list[tuple[RatioSpec, int]] = []
    for ratio in ratios:
        for seed in seeds:
            cells.append((ratio, seed))
    if max_cells is not None:
        cells = cells[:max_cells]

    print(
        f"loaded pools | mirror atk={len(mirror_attacks)} ben={len(mirror_benign)} | "
        f"non atk={len(non_attacks)} ben={len(non_benign)} | target/class={target_per_class}",
        file=sys.stderr,
    )
    print(f"planned cells: {len(cells)}", file=sys.stderr)

    completed = 0
    failed = 0

    for ratio, seed in cells:
        cell_id = f"{ratio.tag}_s{seed}"
        cell_dir = output_root / cell_id
        result_path = cell_dir / "result.json"

        if result_path.exists() and not args.force:
            print(f"skip existing cell: {cell_id}", file=sys.stderr)
            completed += 1
            continue

        cell_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            cell_dir / "status.json",
            {
                "cell_id": cell_id,
                "status": "running",
                "updated_at": utc_now(),
                "ratio": ratio.label,
                "seed": seed,
            },
        )

        try:
            atk_rows, atk_mix = sample_mix(
                mirror_rows=mirror_attacks,
                non_rows=non_attacks,
                ratio=ratio,
                target_count=target_per_class,
                seed=seed * 17 + 101,
            )
            ben_rows, ben_mix = sample_mix(
                mirror_rows=mirror_benign,
                non_rows=non_benign,
                ratio=ratio,
                target_count=target_per_class,
                seed=seed * 17 + 503,
            )

            composition = {
                "cell_id": cell_id,
                "ratio": {"mirror_pct": ratio.mirror_pct, "non_mirror_pct": ratio.non_mirror_pct},
                "seed": seed,
                "target_per_class": target_per_class,
                "counts": {
                    "attacks": atk_mix,
                    "benign": ben_mix,
                    "total_rows": len(atk_rows) + len(ben_rows),
                },
                "source_counts": {
                    "attacks": source_counts(atk_rows),
                    "benign": source_counts(ben_rows),
                },
            }

            if args.dry_run:
                write_json(cell_dir / "composition.json", composition)
                write_json(
                    cell_dir / "status.json",
                    {
                        "cell_id": cell_id,
                        "status": "dry_run_prepared",
                        "updated_at": utc_now(),
                        "ratio": ratio.label,
                        "seed": seed,
                    },
                )
                print(f"dry-run prepared cell: {cell_id}", file=sys.stderr)
                continue

            data_dir = cell_dir / "data"
            train_attacks_path = data_dir / "attacks.yaml"
            train_benign_path = data_dir / "benign.yaml"
            write_yaml_rows(train_attacks_path, atk_rows)
            write_yaml_rows(train_benign_path, ben_rows)
            composition["hashes"] = {
                "attacks_yaml_sha256": sha256_file(train_attacks_path),
                "benign_yaml_sha256": sha256_file(train_benign_path),
            }
            write_json(cell_dir / "composition.json", composition)

            model_dir = cell_dir / "model"
            logs_dir = cell_dir / "logs"
            eval_dir = cell_dir / "eval"
            model_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)
            eval_dir.mkdir(parents=True, exist_ok=True)

            model_weights = model_dir / "l1_weights_generalist.rs"
            model_holdout = model_dir / "train_holdout.yaml"
            train_cmd = [
                sys.executable,
                str(train_script),
                "--specialist",
                "generalist",
                "--attack-files",
                str(train_attacks_path),
                "--benign-files",
                str(train_benign_path),
                "--analyzer",
                "char_wb",
                "--ngram-min",
                "3",
                "--ngram-max",
                "5",
                "--max-features",
                "10000",
                "--min-df",
                "5",
                "--c",
                "0.1",
                "--class-weight",
                "none",
                "--max-iter",
                "100000",
                "--cv-folds",
                "0",
                "--cv-max-samples",
                "120000",
                "--prune-threshold",
                "0.05",
                "--seed",
                str(seed),
                "--apply-l0-transform",
                "--out",
                str(model_weights),
                "--holdout-out",
                str(model_holdout),
            ]
            train_elapsed = run_logged(train_cmd, cwd=parapet_root, log_path=logs_dir / "train.log")

            weights_target.write_text(model_weights.read_text(encoding="utf-8"), encoding="utf-8")

            build_cmd = [
                "cargo",
                "build",
                "--features",
                "eval",
                "--release",
                "--bin",
                "parapet-eval",
            ]
            build_elapsed = run_logged(build_cmd, cwd=rust_crate_dir, log_path=logs_dir / "build.log")

            cfg = yaml.safe_load(eval_config_base.read_text(encoding="utf-8"))
            if "layers" not in cfg or "L1" not in cfg["layers"]:
                raise ValueError(f"{eval_config_base}: missing layers.L1")
            cfg["layers"]["L1"]["threshold"] = float(args.threshold)
            eval_cfg_path = eval_dir / "eval_config_l1_threshold.yaml"
            eval_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            eval_metrics: dict[str, dict[str, Any]] = {}
            eval_elapsed: dict[str, float] = {}
            for target in targets:
                out_path = eval_dir / f"{target.name}.json"
                eval_cmd = [
                    str(eval_bin),
                    "--config",
                    str(eval_cfg_path),
                    "--dataset",
                    str(target.dataset_dir),
                    "--source",
                    target.source,
                    "--layer",
                    "l1",
                    "--json",
                    "--output",
                    str(out_path),
                ]
                elapsed = run_logged(
                    eval_cmd,
                    cwd=parapet_root,
                    log_path=logs_dir / f"eval_{target.name}.log",
                    ok_returncodes={0, 1},
                )
                eval_elapsed[target.name] = elapsed
                eval_metrics[target.name] = parse_eval_metrics(out_path)

            result = {
                "cell_id": cell_id,
                "completed_at": utc_now(),
                "ratio": {
                    "mirror_pct": ratio.mirror_pct,
                    "non_mirror_pct": ratio.non_mirror_pct,
                    "label": ratio.label,
                },
                "seed": seed,
                "threshold": float(args.threshold),
                "target_per_class": target_per_class,
                "composition": {
                    "attacks": atk_mix,
                    "benign": ben_mix,
                },
                "timing_sec": {
                    "train": round(train_elapsed, 3),
                    "build": round(build_elapsed, 3),
                    **{f"eval_{k}": round(v, 3) for k, v in eval_elapsed.items()},
                },
                "metrics": eval_metrics,
                "artifacts": {
                    "weights": str(model_weights),
                    "weights_sha256": sha256_file(model_weights),
                    "active_weights": str(weights_target),
                    "eval_config": str(eval_cfg_path),
                    "eval_outputs": {k: str(eval_dir / f"{k}.json") for k in eval_metrics},
                },
            }
            write_json(result_path, result)
            write_json(
                cell_dir / "status.json",
                {
                    "cell_id": cell_id,
                    "status": "completed",
                    "updated_at": utc_now(),
                },
            )
            write_aggregate_files(output_root)

            completed += 1
            print(f"completed cell: {cell_id}", file=sys.stderr)

        except Exception as exc:  # noqa: BLE001
            failed += 1
            write_json(
                cell_dir / "status.json",
                {
                    "cell_id": cell_id,
                    "status": "failed",
                    "updated_at": utc_now(),
                    "error": str(exc),
                },
            )
            print(f"failed cell: {cell_id} | {exc}", file=sys.stderr)
            if not args.continue_on_error:
                write_aggregate_files(output_root)
                raise

    write_aggregate_files(output_root)
    print(f"done | completed={completed} failed={failed} output={output_root}", file=sys.stderr)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
