"""Step 2: Feature family ablation for L2b.

Trains the L2b model (RF and XGB) on multiple feature configurations and
evaluates end-to-end cascade performance on the Phase 4 challenge set.

Configurations:
  A: harness-only (~5 features)
  B: harness + structural (~20 features)
  C: harness + tfidf-2.5K (~2,505 features)
  D: harness + structural + tfidf-2.5K (~2,520 features)
  E: harness + structural + tfidf-10K (~10,020 features)

Usage:
    cd parapet
    python scripts/ablate_l2b_features.py \
      --residuals-dir parapet-runner/runs/l2b_residuals \
      --phase4-eval parapet-runner/runs/phase4_final/run/_eval_holdout_0p0/eval.json \
      --challenge-attack schema/eval/challenges/tough_attack_v2/tough_attack_v6_novel.yaml \
      --challenge-neutral schema/eval/challenges/tough_neutral_v2/tough_neutral_v6_novel.yaml \
      --output-dir parapet-runner/runs/l2b_feature_ablation
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


CONFIGS = [
    {"name": "A_harness_only", "features": "harness", "max_tfidf": 0},
    {"name": "B_harness_structural", "features": "harness+structural", "max_tfidf": 0},
    {"name": "C_harness_tfidf2500", "features": "harness+tfidf", "max_tfidf": 2500},
    {"name": "D_harness_structural_tfidf2500", "features": "harness+structural+tfidf", "max_tfidf": 2500},
    {"name": "E_harness_structural_tfidf10000", "features": "harness+structural+tfidf", "max_tfidf": 10000},
]

MODELS = ["rf", "xgb"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 2: L2b feature family ablation")
    parser.add_argument("--residuals-dir", type=Path, required=True)
    parser.add_argument("--phase4-eval", type=Path, required=True)
    parser.add_argument("--challenge-attack", type=Path, required=True)
    parser.add_argument("--challenge-neutral", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-bin", type=str, default="python")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    residuals_dir = Path(args.residuals_dir).resolve()
    phase4_eval = Path(args.phase4_eval).resolve()
    challenge_attack = Path(args.challenge_attack).resolve()
    challenge_neutral = Path(args.challenge_neutral).resolve()

    scripts_dir = Path(__file__).resolve().parent

    all_results: list[dict[str, Any]] = []

    total_cells = len(CONFIGS) * len(MODELS)
    cell_idx = 0

    for config in CONFIGS:
        for model_type in MODELS:
            cell_idx += 1
            cell_name = f"{config['name']}_{model_type}"
            cell_dir = output_dir / cell_name
            model_dir = cell_dir / "model"
            cascade_dir = cell_dir / "cascade"

            # Skip if already complete
            cascade_result = cascade_dir / "best_operating_point.json"
            if cascade_result.exists():
                print(f"[{cell_idx}/{total_cells}] SKIP {cell_name} (cached)")
                result = json.loads(cascade_result.read_text(encoding="utf-8"))
                result["config"] = config["name"]
                result["model_type"] = model_type
                all_results.append(result)
                continue

            print(f"\n[{cell_idx}/{total_cells}] {cell_name}")

            # Train
            train_cmd = [
                args.python_bin, str(scripts_dir / "train_l2b.py"),
                "--residuals-dir", str(residuals_dir),
                "--output-dir", str(model_dir),
                "--model", model_type,
                "--features", config["features"],
            ]
            if config["max_tfidf"] > 0:
                train_cmd.extend(["--max-tfidf-features", str(config["max_tfidf"])])

            print(f"  Training {model_type} with features={config['features']}...")
            proc = subprocess.run(
                train_cmd, capture_output=True, text=True,
            )
            if proc.returncode != 0:
                print(f"  TRAIN FAILED (rc={proc.returncode})", file=sys.stderr)
                print(proc.stderr[-500:] if proc.stderr else "", file=sys.stderr)
                continue

            # Extract val metrics from train manifest
            train_manifest_path = model_dir / "train_manifest.json"
            if train_manifest_path.exists():
                train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
                model_key = {"rf": "random_forest", "xgb": "xgboost", "lr": "logistic"}.get(model_type, model_type)
                val_result = train_manifest.get("results", {}).get(model_key, {})
                val_f1 = val_result.get("f1", 0)
                val_auc = val_result.get("roc_auc", 0)
                print(f"  Val: F1={val_f1:.4f} ROC-AUC={val_auc:.4f}")
            else:
                val_f1 = val_auc = 0

            # Cascade eval
            cascade_cmd = [
                args.python_bin, str(scripts_dir / "eval_cascade.py"),
                "--phase4-eval", str(phase4_eval),
                "--challenge-attack", str(challenge_attack),
                "--challenge-neutral", str(challenge_neutral),
                "--l2b-model-dir", str(model_dir),
                "--output-dir", str(cascade_dir),
            ]

            print(f"  Cascade eval...")
            proc = subprocess.run(
                cascade_cmd, capture_output=True, text=True,
            )
            if proc.returncode != 0:
                print(f"  CASCADE FAILED (rc={proc.returncode})", file=sys.stderr)
                print(proc.stderr[-500:] if proc.stderr else "", file=sys.stderr)
                continue

            # Read cascade result
            if cascade_result.exists():
                result = json.loads(cascade_result.read_text(encoding="utf-8"))
                result["config"] = config["name"]
                result["model_type"] = model_type
                result["val_f1"] = val_f1
                result["val_roc_auc"] = val_auc
                result["total_features"] = train_manifest.get("total_features", 0)
                all_results.append(result)
                print(f"  Cascade: F1={result['f1']:.4f} FPR={result['fpr']:.4f} "
                      f"Route={result['route_rate']*100:.1f}%")

                # Read ctf FPR from sweep results
                sweep_path = cascade_dir / "sweep_results.json"
                if sweep_path.exists():
                    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
                    # Find the matching operating point
                    for sr in sweep:
                        if sr["t_allow"] == result["t_allow"] and sr["t_block"] == result["t_block"]:
                            result["sweep_match"] = True
                            break

            else:
                print(f"  No cascade result")

    # Write summary
    print(f"\n{'='*100}")
    print(f"{'Config':<35} {'Model':<5} {'Feats':>6} {'ValF1':>6} {'CascF1':>7} "
          f"{'CascP':>7} {'CascR':>7} {'FPR':>6} {'Route%':>7}")
    print("-" * 100)

    all_results.sort(key=lambda r: -r.get("f1", 0))
    for r in all_results:
        print(f"{r.get('config','?'):<35} {r.get('model_type','?'):<5} "
              f"{r.get('total_features',0):>6} "
              f"{r.get('val_f1',0):>6.4f} "
              f"{r.get('f1',0):>7.4f} {r.get('precision',0):>7.4f} "
              f"{r.get('recall',0):>7.4f} {r.get('fpr',0):>6.4f} "
              f"{r.get('route_rate',0)*100:>6.1f}%")

    (output_dir / "summary.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )

    # Markdown summary
    lines = [
        "# Step 2: Feature Ablation Results",
        "",
        f"Updated: {utc_now()}",
        "",
        "| Config | Model | Features | Val F1 | Cascade F1 | Cascade P | Cascade R | FPR | Route% |",
        "|--------|-------|-------:|-------:|-----------:|----------:|----------:|----:|-------:|",
    ]
    for r in all_results:
        lines.append(
            f"| {r.get('config','?')} | {r.get('model_type','?')} | "
            f"{r.get('total_features',0)} | {r.get('val_f1',0):.4f} | "
            f"{r.get('f1',0):.4f} | {r.get('precision',0):.4f} | "
            f"{r.get('recall',0):.4f} | {r.get('fpr',0):.4f} | "
            f"{r.get('route_rate',0)*100:.1f}% |"
        )

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nResults saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
