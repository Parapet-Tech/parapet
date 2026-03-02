"""Sweep L1 threshold on a holdout split and emit CSV of precision/recall/F1/FP/FN."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def run_sweep(
    *,
    parapet_eval_bin: Path,
    base_config: Path,
    dataset_dir: Path,
    source: str,
    output_dir: Path,
    thresholds: list[float],
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_data = yaml.safe_load(base_config.read_text(encoding="utf-8"))

    results = []
    for t in thresholds:
        # Set threshold in config
        cfg = json.loads(json.dumps(config_data))  # deep copy
        if "layers" in cfg and "L1" in cfg["layers"]:
            cfg["layers"]["L1"]["threshold"] = float(t)
        else:
            raise ValueError(f"No L1 layer in config: {base_config}")

        tag = str(t).replace(".", "p").replace("-", "m")
        config_path = output_dir / f"config_t{tag}.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        output_json = output_dir / f"eval_t{tag}.json"
        cmd = [
            str(parapet_eval_bin),
            "--config", str(config_path),
            "--dataset", str(dataset_dir),
            "--source", source,
            "--layer", "l1",
            "--json",
            "--output", str(output_json),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if not output_json.exists():
            print(f"SKIP t={t}: no output (rc={result.returncode})", file=sys.stderr)
            continue

        payload = json.loads(output_json.read_text(encoding="utf-8"))
        layer = payload["layers"][0]
        row = {
            "threshold": t,
            "tp": layer["tp"],
            "fp": layer["fp"],
            "fn": layer["fn_count"],
            "tn": layer["tn"],
            "precision": round(layer["precision"], 4),
            "recall": round(layer["recall"], 4),
            "f1": round(layer["f1"], 4),
            "fpr": round(layer["fp"] / (layer["fp"] + layer["tn"]), 4) if (layer["fp"] + layer["tn"]) > 0 else 0,
        }
        results.append(row)
        print(f"t={t:+.2f}  P={row['precision']:.4f}  R={row['recall']:.4f}  F1={row['f1']:.4f}  FP={row['fp']}  FN={row['fn']}  FPR={row['fpr']:.4f}")

    # Write CSV
    csv_path = output_dir / "sweep_results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("threshold,tp,fp,fn,tn,precision,recall,f1,fpr\n")
        for r in results:
            f.write(f"{r['threshold']},{r['tp']},{r['fp']},{r['fn']},{r['tn']},{r['precision']},{r['recall']},{r['f1']},{r['fpr']}\n")
    print(f"\nWrote {csv_path}")
    return results


if __name__ == "__main__":
    workspace = Path(__file__).resolve().parent.parent.parent  # parapet/
    thresholds = [-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    eval_bin = workspace / "parapet" / "target" / "release" / "parapet-eval.exe"
    base_config = workspace / "schema" / "eval" / "eval_config_l1_only.yaml"
    base_output = workspace / "parapet-runner" / "runs" / "mirror_v2_tier1_full"

    sets = [
        ("holdout", base_output / "_eval_holdout_m0p5" / "dataset", "holdout"),
        ("tough_neutral", workspace / "schema" / "eval" / "challenges" / "tough_neutral_v1", "tough_neutral_mirror_v2_novel"),
        ("tough_attack", workspace / "schema" / "eval" / "challenges" / "tough_attack_v1", "tough_attack_mirror_v2_novel"),
    ]

    for name, dataset_dir, source in sets:
        print(f"\n{'='*80}")
        print(f"  {name.upper()}: {source}")
        print(f"{'='*80}")
        run_sweep(
            parapet_eval_bin=eval_bin,
            base_config=base_config,
            dataset_dir=dataset_dir,
            source=source,
            output_dir=base_output / f"_threshold_sweep_{name}",
            thresholds=thresholds,
        )
