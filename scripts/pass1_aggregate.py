"""Aggregate Pass 1 per-model results into a comparison leaderboard.

Reads `parapet-runner/runs/l2_pass1_ots/results/*.jsonl` and emits:
- leaderboard.json     — machine-readable summary per model
- leaderboard.md       — human-readable table + per-source breakdowns
- operating_points.json — chosen thresholds (F1-optimal and FPR@R=0.85/0.90)

Metrics:
- F1@0.5            — at the default decision threshold
- F1*               — optimal F1 over all thresholds
- FPR@R=0.85        — lowest FPR achievable while keeping recall >= 0.85
- FPR@R=0.90        — same at recall 0.90
- Per-source recall for attack sources, FPR for benign sources

Run:
    cd parapet
    python scripts/pass1_aggregate.py --dir parapet-runner/runs/l2_pass1_ots
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def load_results(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def confusion_at(threshold: float, rows: list[dict]) -> tuple[int, int, int, int]:
    tp = fp = fn = tn = 0
    for r in rows:
        pred_pos = r["score"] >= threshold
        is_pos = r["true_label"] == "malicious"
        if pred_pos and is_pos:
            tp += 1
        elif pred_pos and not is_pos:
            fp += 1
        elif not pred_pos and is_pos:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def prf_fpr(tp: int, fp: int, fn: int, tn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": p, "recall": r, "f1": f1, "fpr": fpr}


def sweep(rows: list[dict], steps: int = 201) -> list[tuple[float, dict]]:
    """Grid sweep thresholds from 0 to 1. Returns [(threshold, metrics), ...]."""
    out: list[tuple[float, dict]] = []
    for i in range(steps):
        t = i / (steps - 1)
        out.append((t, prf_fpr(*confusion_at(t, rows))))
    return out


def fpr_at_recall(sweep_results: list[tuple[float, dict]], target_recall: float) -> dict:
    """Lowest FPR at a threshold that achieves recall >= target_recall."""
    best = None
    for t, m in sweep_results:
        if m["recall"] >= target_recall:
            if best is None or m["fpr"] < best["fpr"]:
                best = {**m, "threshold": t}
    return best or {}


def optimal_f1(sweep_results: list[tuple[float, dict]]) -> dict:
    best_t, best_m = max(sweep_results, key=lambda tm: tm[1]["f1"])
    return {**best_m, "threshold": best_t}


def per_source_breakdown(rows: list[dict], threshold: float) -> dict[str, dict]:
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    out: dict[str, dict] = {}
    for src, srows in by_src.items():
        tp, fp, fn, tn = confusion_at(threshold, srows)
        # Derive the most useful per-source metric.
        mal = sum(1 for r in srows if r["true_label"] == "malicious")
        ben = sum(1 for r in srows if r["true_label"] == "benign")
        rec = tp / (tp + fn) if (tp + fn) else None
        fpr = fp / (fp + tn) if (fp + tn) else None
        out[src] = {"n": len(srows), "n_mal": mal, "n_ben": ben,
                    "recall": rec, "fpr": fpr}
    return out


def build_report(results_dir: Path, meta_dir: Path) -> dict:
    models: dict[str, dict] = {}
    for rpath in sorted(results_dir.glob("*.jsonl")):
        slug = rpath.stem.replace("__", "/")
        rows = load_results(rpath)
        if not rows:
            continue
        sw = sweep(rows)
        m = {
            "slug": slug,
            "n": len(rows),
            "f1_at_0_5": prf_fpr(*confusion_at(0.5, rows)),
            "f1_optimal": optimal_f1(sw),
            "fpr_at_r_0_85": fpr_at_recall(sw, 0.85),
            "fpr_at_r_0_90": fpr_at_recall(sw, 0.90),
        }
        # Per-source at optimal-F1 threshold.
        t = m["f1_optimal"]["threshold"]
        m["per_source_at_optimal"] = per_source_breakdown(rows, t)

        meta_path = meta_dir / (rpath.stem + ".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            m["latency_ms"] = meta.get("latency_ms")
            m["device"] = meta.get("device")
            m["id2label"] = meta.get("id2label")

        models[slug] = m

    return {"models": models}


def fmt_metric(m: dict, key: str, fmt: str = ".3f") -> str:
    v = m.get(key)
    if v is None:
        return "—"
    return format(v, fmt)


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# Pass 1 — Off-the-Shelf PI Classifier Leaderboard\n")
    lines.append("Challenge set: `tough_attack_v2` + `tough_neutral_v2` (unified).\n")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| Model | N | F1@0.5 | F1\\* | FPR@R=0.85 | FPR@R=0.90 | p50 ms | p95 ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for slug, m in sorted(report["models"].items(), key=lambda kv: -kv[1]["f1_optimal"].get("f1", 0.0)):
        lat = m.get("latency_ms") or {}
        lines.append(
            f"| `{slug}` | {m['n']} | "
            f"{fmt_metric(m['f1_at_0_5'], 'f1')} | "
            f"{fmt_metric(m['f1_optimal'], 'f1')} | "
            f"{fmt_metric(m.get('fpr_at_r_0_85', {}), 'fpr')} | "
            f"{fmt_metric(m.get('fpr_at_r_0_90', {}), 'fpr')} | "
            f"{fmt_metric(lat, 'p50', '.1f')} | "
            f"{fmt_metric(lat, 'p95', '.1f')} |"
        )
    lines.append("")
    lines.append("*F1\\* = best F1 over threshold sweep.*\n")

    lines.append("## Per-source (at each model's optimal-F1 threshold)")
    lines.append("")
    for slug, m in report["models"].items():
        lines.append(f"### `{slug}` — threshold={m['f1_optimal']['threshold']:.3f}")
        lines.append("")
        lines.append("| Source | N | Mal | Ben | Recall | FPR |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for src in sorted(m["per_source_at_optimal"]):
            s = m["per_source_at_optimal"][src]
            rec = f"{s['recall']:.3f}" if s["recall"] is not None else "—"
            fpr = f"{s['fpr']:.3f}" if s["fpr"] is not None else "—"
            lines.append(f"| {src} | {s['n']} | {s['n_mal']} | {s['n_ben']} | {rec} | {fpr} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="parapet-runner/runs/l2_pass1_ots")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent.parent  # parapet/
    out_dir = (base / args.dir).resolve()
    results_dir = out_dir / "results"
    meta_dir = out_dir / "meta"

    if not results_dir.exists():
        print(f"No results dir at {results_dir}")
        return 1

    report = build_report(results_dir, meta_dir)
    (out_dir / "leaderboard.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )
    (out_dir / "leaderboard.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {out_dir / 'leaderboard.md'}")
    print(f"Wrote {out_dir / 'leaderboard.json'}")

    # Brief console summary.
    print("\nTop by F1*:")
    ranked = sorted(report["models"].items(),
                    key=lambda kv: -kv[1]["f1_optimal"].get("f1", 0.0))
    for slug, m in ranked:
        lat = m.get("latency_ms") or {}
        print(f"  {slug:<55}  F1*={m['f1_optimal']['f1']:.3f}  "
              f"FPR@R.85={fmt_metric(m.get('fpr_at_r_0_85', {}), 'fpr')}  "
              f"p95={fmt_metric(lat, 'p95', '.1f')}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
