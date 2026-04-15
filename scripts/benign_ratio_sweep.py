"""Benign ratio experiment: test whether more/different benign reduces FPR.

Two-factor sweep:
  Factor A: benign ratio (1:1, 1.5:1, 2:1, 3:1)
  Factor B: benign recipe (M=mirror, H=hard-negative-expanded)

Holds attack data fixed. Only varies the benign side.

Usage:
    cd parapet
    python scripts/benign_ratio_sweep.py \
      --curation-dir parapet-data/curated/v7_35k_experiment \
      --output-dir parapet-runner/runs/benign_ratio_sweep
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random as random_mod
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


def load_yaml(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, list) else []


def write_yaml(samples: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(samples, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False, width=200)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Hard-negative benign sources for Recipe H
# ---------------------------------------------------------------------------

HARD_NEGATIVE_SOURCES = {
    "notinject_benign": "schema/eval/benign/opensource_notinject_benign.yaml",
    "wildguardmix_benign": "schema/eval/benign/opensource_wildguardmix_benign.yaml",
    "atlas_neg": "schema/eval/benign/thewall_atlas_neg.yaml",
    "bipia_benign": "schema/eval/benign/opensource_bipia_benign.yaml",
}

# Recipe H allocation for extra budget beyond mirror core
HARD_NEGATIVE_ALLOC = {
    "notinject_benign": 0.35,
    "wildguardmix_benign": 0.25,
    "atlas_neg": 0.20,
    "bipia_benign": 0.10,
    # remaining 0.10 goes to broad staged benign
}

BROAD_BENIGN_STAGED = [
    "schema/eval/staging/en_WildChat-1M_benign_staged.yaml",
    "schema/eval/staging/en_trivia-qa_benign_staged.yaml",
    "schema/eval/staging/en_writingprompts_benign_staged.yaml",
    "schema/eval/staging/en_databricks-dolly-15k_benign_staged.yaml",
    "schema/eval/staging/en_chatbot_arena_conversations_benign_staged.yaml",
    "schema/eval/staging/ru_ru_turbo_saiga_benign_staged.yaml",
    "schema/eval/staging/zh_stem_zh_instruction_benign_staged.yaml",
    "schema/eval/staging/ar_arabic-fine-tuning-chatML_benign_staged.yaml",
]


def load_hard_negatives(base: Path) -> dict[str, list[dict]]:
    """Load hard-negative benign families."""
    pool = {}
    for name, path in HARD_NEGATIVE_SOURCES.items():
        p = base / path
        if p.exists():
            data = load_yaml(p)
            # Normalize to training format
            normalized = []
            for row in data:
                normalized.append({
                    "content": row.get("content", ""),
                    "label": "benign",
                    "reason": row.get("reason", "background"),
                    "source": name,
                    "language": row.get("language", "EN"),
                    "format_bin": row.get("format_bin", "prose"),
                    "length_bin": row.get("length_bin", "medium"),
                })
            pool[name] = normalized
            print(f"  {name}: {len(normalized)}")
    return pool


def load_broad_benign(base: Path, limit: int = 5000) -> list[dict]:
    """Load broad benign from staged sources, up to limit total."""
    rng = random_mod.Random(42)
    pool = []
    for path_str in BROAD_BENIGN_STAGED:
        p = base / path_str
        if not p.exists():
            continue
        data = load_yaml(p)
        for row in data:
            pool.append({
                "content": row.get("content", ""),
                "label": "benign",
                "reason": row.get("reason", "background"),
                "source": "broad_staged",
                "language": row.get("language", "EN"),
                "format_bin": row.get("format_bin", "prose"),
                "length_bin": row.get("length_bin", "medium"),
            })
    rng.shuffle(pool)
    return pool[:limit]


def sample_recipe_m(
    mirror_benign: list[dict],
    target_count: int,
    seed: int,
) -> list[dict]:
    """Recipe M: sample from existing mirror benign distribution."""
    rng = random_mod.Random(seed)
    if target_count <= len(mirror_benign):
        return rng.sample(mirror_benign, target_count)
    # Oversample with replacement
    result = list(mirror_benign)
    while len(result) < target_count:
        result.extend(rng.choices(mirror_benign, k=min(target_count - len(result), len(mirror_benign))))
    return result[:target_count]


def sample_recipe_h(
    mirror_benign: list[dict],
    hard_negatives: dict[str, list[dict]],
    broad_benign: list[dict],
    target_count: int,
    seed: int,
) -> list[dict]:
    """Recipe H: mirror core + hard-negative expansion for extra budget."""
    rng = random_mod.Random(seed)

    # Keep all mirror benign as the core
    core = list(mirror_benign)
    extra_needed = target_count - len(core)

    if extra_needed <= 0:
        return rng.sample(core, target_count)

    extra = []

    # Allocate hard-negative families
    for name, ratio in HARD_NEGATIVE_ALLOC.items():
        pool = hard_negatives.get(name, [])
        alloc = min(int(extra_needed * ratio), len(pool))
        if alloc > 0:
            extra.extend(rng.sample(pool, alloc) if alloc < len(pool) else pool)

    # Fill remaining with broad benign
    remaining = extra_needed - len(extra)
    if remaining > 0 and broad_benign:
        broad_sample = rng.sample(broad_benign, min(remaining, len(broad_benign)))
        extra.extend(broad_sample)

    result = core + extra
    rng.shuffle(result)
    return result[:target_count]


# ---------------------------------------------------------------------------
# Manifest and runner
# ---------------------------------------------------------------------------

def build_manifest(
    curation_dir: Path,
    temp_dir: Path,
    train_samples: list[dict],
) -> Path:
    orig = json.loads((curation_dir / "manifest.json").read_text(encoding="utf-8"))
    train_hashes = sorted(content_hash(s["content"]) for s in train_samples)

    manifest = {
        "spec_name": "benign_ratio", "spec_version": "7.0.0",
        "spec_hash": orig.get("spec_hash", ""), "seed": 42,
        "timestamp": utc_now(),
        "source_hashes": orig.get("source_hashes", {}),
        "source_metadata": orig.get("source_metadata", {}),
        "output_path": str(temp_dir / "curated.yaml"),
        "output_hash": "ratio_sweep", "semantic_hash": "ratio_sweep",
        "total_samples": len(train_samples),
        "attack_samples": sum(1 for s in train_samples if s.get("label") == "malicious"),
        "benign_samples": sum(1 for s in train_samples if s.get("label") == "benign"),
        "splits": {
            "train": {
                "name": "train",
                "sample_count": len(train_samples),
                "content_hashes": train_hashes,
                "artifact_path": "train.yaml",
            },
            "val": orig["splits"]["val"],
            "holdout": orig["splits"]["holdout"],
        },
        "cell_fills": {}, "gaps": [],
        "duplicates_dropped": 0, "cross_contamination_dropped": 0,
    }
    path = temp_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_cell(
    tag: str,
    train_samples: list[dict],
    curation_dir: Path,
    output_dir: Path,
    workspace_root: Path,
    train_config_path: Path,
    parapet_eval_bin: Path,
    python_bin: str,
) -> dict[str, Any] | None:
    cell_dir = output_dir / tag
    result_path = cell_dir / "result.json"

    if result_path.exists():
        print(f"  [{tag}] SKIP (cached)")
        return json.loads(result_path.read_text(encoding="utf-8"))

    cell_dir.mkdir(parents=True, exist_ok=True)
    curated = cell_dir / "curated"
    curated.mkdir(parents=True, exist_ok=True)

    write_yaml(train_samples, curated / "train.yaml")
    shutil.copy2(curation_dir / "val.yaml", curated / "val.yaml")
    shutil.copy2(curation_dir / "holdout.yaml", curated / "holdout.yaml")

    manifest_path = build_manifest(curation_dir, curated, train_samples)

    run_dir = cell_dir / "run"
    cmd = [
        python_bin, "-m", "parapet_runner.runner", "run",
        "--workspace-root", str(workspace_root),
        "--curation-manifest", str(manifest_path),
        "--train-config", str(train_config_path),
        "--output-dir", str(run_dir.resolve()),
        "--parapet-eval-bin", str(parapet_eval_bin),
        "--run-id", f"br_{tag}",
        "--pg2-mode", "off",
        "--skip-output-hash-verify",
        "--skip-split-hash-verify",
    ]

    runner_cwd = workspace_root / "parapet-runner"
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(runner_cwd), capture_output=True, text=True)
    elapsed = time.time() - start

    log_path = cell_dir / "runner.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# tag: {tag}\n# elapsed: {elapsed:.1f}s\n# rc: {proc.returncode}\n\n")
        f.write("## stdout\n")
        f.write(proc.stdout or "")
        f.write("\n\n## stderr\n")
        f.write(proc.stderr or "")

    run_manifest = run_dir / "run_manifest.json"
    if proc.returncode == 0 and run_manifest.exists():
        payload = json.loads(run_manifest.read_text(encoding="utf-8"))
        er = payload["eval_result"]
        result = {
            "tag": tag, "status": "completed", "elapsed_sec": round(elapsed, 1),
            "train_size": len(train_samples),
            "attack_count": sum(1 for s in train_samples if s["label"] == "malicious"),
            "benign_count": sum(1 for s in train_samples if s["label"] == "benign"),
            **er,
        }
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  [{tag}] F1={er['f1']:.4f} R={er['recall']:.4f} P={er['precision']:.4f} "
              f"FP={er['false_positives']} ({elapsed:.0f}s)")
        return result
    else:
        print(f"  [{tag}] FAIL (rc={proc.returncode}, see {log_path})", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    default_workspace = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Benign ratio sweep experiment")
    parser.add_argument("--workspace-root", type=Path, default=default_workspace)
    parser.add_argument("--curation-dir", type=Path, required=True)
    parser.add_argument("--train-config", type=Path,
                        default=Path("parapet-runner/configs/phase1_learning_curve.yaml"))
    parser.add_argument("--parapet-eval-bin", type=Path,
                        default=Path("parapet/target/release/parapet-eval.exe"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-bin", type=str, default="python")
    parser.add_argument("--ratios", type=str, default="1.0,1.5,2.0,3.0")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    workspace = Path(args.workspace_root).resolve()
    curation_dir = Path(args.curation_dir)
    if not curation_dir.is_absolute():
        curation_dir = (workspace / curation_dir).resolve()
    parapet_eval_bin = Path(args.parapet_eval_bin)
    if not parapet_eval_bin.is_absolute():
        parapet_eval_bin = (workspace / parapet_eval_bin).resolve()
    train_config = Path(args.train_config)
    if not train_config.is_absolute():
        train_config = (workspace / train_config).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (workspace / output_dir).resolve()

    ratios = [float(r.strip()) for r in args.ratios.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load v7 train, separate attacks and benign
    print("Loading v7 train...")
    train = load_yaml(curation_dir / "train.yaml")
    attacks = [r for r in train if r.get("label") == "malicious"]
    mirror_benign = [r for r in train if r.get("label") == "benign"]
    print(f"  {len(attacks)} attacks, {len(mirror_benign)} mirror benign")

    # Load hard-negative families
    print("\nLoading hard-negative benign...")
    hard_negatives = load_hard_negatives(workspace)

    # Load broad benign
    print("\nLoading broad staged benign...")
    broad_benign = load_broad_benign(workspace, limit=30000)
    print(f"  {len(broad_benign)} broad benign loaded")

    # Dedup broad benign against mirror benign
    mirror_hashes = {content_hash(r["content"]) for r in mirror_benign}
    broad_benign = [r for r in broad_benign if content_hash(r["content"]) not in mirror_hashes]
    print(f"  {len(broad_benign)} after dedup against mirror")

    # Run matrix
    cells = []
    for ratio in ratios:
        for recipe in ["M", "H"]:
            for seed in seeds:
                cells.append((ratio, recipe, seed))

    print(f"\nMatrix: {len(ratios)} ratios x 2 recipes x {len(seeds)} seeds = {len(cells)} cells")
    print(f"Ratios: {ratios}")

    all_results = []
    for idx, (ratio, recipe, seed) in enumerate(cells, 1):
        tag = f"r{ratio:.1f}_{recipe}_s{seed}"
        target_benign = int(len(attacks) * ratio)

        print(f"\n[{idx}/{len(cells)}] {tag} (target benign={target_benign})")

        if recipe == "M":
            benign = sample_recipe_m(mirror_benign, target_benign, seed)
        else:
            benign = sample_recipe_h(mirror_benign, hard_negatives, broad_benign,
                                     target_benign, seed)

        combined = attacks + benign
        random_mod.Random(seed).shuffle(combined)

        result = run_cell(
            tag, combined, curation_dir, output_dir, workspace,
            train_config, parapet_eval_bin, args.python_bin,
        )
        if result:
            result["ratio"] = ratio
            result["recipe"] = recipe
            result["seed"] = seed
            all_results.append(result)
        elif not args.continue_on_error:
            return 1

    # Summary
    print("\n" + "=" * 90)
    print(f"{'Tag':<20} {'Ratio':>6} {'Recipe':>7} {'Train':>6} {'F1':>7} "
          f"{'P':>7} {'R':>7} {'FP':>5} {'FN':>5}")
    print("-" * 90)

    all_results.sort(key=lambda r: (-r.get("f1", 0),))
    for r in all_results:
        print(f"{r['tag']:<20} {r['ratio']:>6.1f} {r['recipe']:>7} "
              f"{r['train_size']:>6} {r.get('f1',0):>7.4f} "
              f"{r.get('precision',0):>7.4f} {r.get('recall',0):>7.4f} "
              f"{r.get('false_positives',0):>5} {r.get('false_negatives',0):>5}")

    # Aggregate by ratio x recipe (mean over seeds)
    print("\n=== Aggregated (mean over seeds) ===")
    print(f"{'Ratio':>6} {'Recipe':>7} {'F1':>7} {'P':>7} {'R':>7} {'FP':>6} {'FN':>6}")
    print("-" * 55)

    groups = defaultdict(list)
    for r in all_results:
        groups[(r["ratio"], r["recipe"])].append(r)

    for (ratio, recipe) in sorted(groups.keys()):
        g = groups[(ratio, recipe)]
        mean_f1 = sum(r.get("f1", 0) for r in g) / len(g)
        mean_p = sum(r.get("precision", 0) for r in g) / len(g)
        mean_r = sum(r.get("recall", 0) for r in g) / len(g)
        mean_fp = sum(r.get("false_positives", 0) for r in g) / len(g)
        mean_fn = sum(r.get("false_negatives", 0) for r in g) / len(g)
        print(f"{ratio:>6.1f} {recipe:>7} {mean_f1:>7.4f} {mean_p:>7.4f} "
              f"{mean_r:>7.4f} {mean_fp:>6.0f} {mean_fn:>6.0f}")

    # Save
    (output_dir / "results.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )
    print(f"\nResults saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
