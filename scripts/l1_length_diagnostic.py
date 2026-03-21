#!/usr/bin/env python3
"""L1 length-dependent scoring diagnostic.

Proves or disproves the hypothesis that L1 FNR correlates with text length.
If confirmed, this justifies L2-normalization retraining or dynamic thresholds.

Outputs:
  - Per-length-bin FNR/FPR at current fixed threshold
  - Score vs n_active scatter data (CSV)
  - Summary JSON with go/no-go gate metrics

Usage (from parapet/):
    python scripts/l1_length_diagnostic.py \
        --weights-file parapet/src/layers/l1_weights.rs \
        --verified-dir schema/eval/verified \
        --threshold 0.0 \
        --output-dir runs/l1_length_diagnostic
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

try:
    YAML_LOADER = yaml.CSafeLoader
except AttributeError:
    YAML_LOADER = yaml.SafeLoader

INVALID_YAML_CTRL_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f"
    r"\ud800-\udfff\ufffe\uffff]"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights-file", type=Path, required=True)
    p.add_argument("--verified-dir", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sample-cap", type=int, default=0,
                   help="Cap total samples (0=no cap). Useful for quick test runs.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

PHF_ENTRY_RE = re.compile(
    r'"((?:[^"\\]|\\.)*)"\s*=>\s*(-?\d+\.\d+)_f64'
)
BIAS_RE = re.compile(r"pub const BIAS:\s*f64\s*=\s*(-?\d+\.\d+)_f64")


def load_weights(path: Path) -> tuple[dict[str, float], float]:
    """Parse Rust phf_map weights file into a Python dict + bias."""
    text = path.read_text(encoding="utf-8")

    bias_match = BIAS_RE.search(text)
    if not bias_match:
        raise ValueError(f"Could not find BIAS in {path}")
    bias = float(bias_match.group(1))

    weights: dict[str, float] = {}
    for m in PHF_ENTRY_RE.finditer(text):
        ngram = m.group(1).replace("\\\\", "\\").replace('\\"', '"')
        weight = float(m.group(2))
        weights[ngram] = weight

    return weights, bias


# ---------------------------------------------------------------------------
# N-gram extraction (mirrors Rust char_wb)
# ---------------------------------------------------------------------------

def extract_char_wb_ngrams(text: str, ngram_range: tuple[int, int] = (3, 5)) -> set[str]:
    """Extract unique char_wb n-grams matching sklearn/Rust behavior."""
    lower = text.lower()
    seen: set[str] = set()
    for word in lower.split():
        padded = f" {word} "
        b = padded.encode("utf-8")
        for n in range(ngram_range[0], ngram_range[1] + 1):
            if len(b) < n:
                continue
            for i in range(len(b) - n + 1):
                window = b[i:i + n]
                try:
                    ngram = window.decode("utf-8")
                    seen.add(ngram)
                except UnicodeDecodeError:
                    continue
    return seen


def score_text(text: str, weights: dict[str, float], bias: float) -> tuple[float, int]:
    """Score text using L1 weights. Returns (score, n_active)."""
    ngrams = extract_char_wb_ngrams(text)
    score = bias
    n_active = 0
    for ng in ngrams:
        w = weights.get(ng)
        if w is not None:
            score += w
            n_active += 1
    return score, n_active


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_verified_yaml(path: Path) -> list[dict[str, Any]]:
    """Load a verified eval YAML file, return list of {text, label, source}."""
    raw = path.read_text(encoding="utf-8")
    raw = INVALID_YAML_CTRL_RE.sub("", raw)
    try:
        data = yaml.load(raw, Loader=YAML_LOADER)
    except yaml.YAMLError as e:
        print(f"  WARNING: skipping {path.name} (YAML parse error: {e})")
        return []
    if not data:
        return []

    source = path.stem
    is_attack = "attack" in source.lower()
    label = "malicious" if is_attack else "benign"

    rows = []
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                text = entry.get("text") or entry.get("content") or entry.get("prompt", "")
            elif isinstance(entry, str):
                text = entry
            else:
                continue
            if text:
                rows.append({"text": str(text), "label": label, "source": source})
    elif isinstance(data, dict):
        for key in ("samples", "data", "rows"):
            if key in data and isinstance(data[key], list):
                for entry in data[key]:
                    if isinstance(entry, dict):
                        text = entry.get("text") or entry.get("content") or entry.get("prompt", "")
                    elif isinstance(entry, str):
                        text = entry
                    else:
                        continue
                    if text:
                        rows.append({"text": str(text), "label": label, "source": source})
                break

    return rows


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

LENGTH_BINS = [
    (0, 100, "0-100"),
    (100, 200, "100-200"),
    (200, 500, "200-500"),
    (500, 1000, "500-1K"),
    (1000, 2000, "1K-2K"),
    (2000, 5000, "2K-5K"),
    (5000, float("inf"), "5K+"),
]


def bin_label(text_len: int) -> str:
    for lo, hi, name in LENGTH_BINS:
        if lo <= text_len < hi:
            return name
    return "5K+"


def main() -> None:
    args = parse_args()

    print(f"Loading weights from {args.weights_file}...")
    weights, bias = load_weights(args.weights_file)
    print(f"  {len(weights)} n-gram weights, bias={bias:.6f}")

    print(f"\nLoading verified corpus from {args.verified_dir}...")
    all_rows: list[dict[str, Any]] = []
    for yaml_path in sorted(args.verified_dir.glob("*.yaml")):
        rows = load_verified_yaml(yaml_path)
        all_rows.extend(rows)
        if rows:
            print(f"  {yaml_path.name}: {len(rows)} rows")

    if args.sample_cap > 0 and len(all_rows) > args.sample_cap:
        import random
        random.seed(42)
        random.shuffle(all_rows)
        all_rows = all_rows[:args.sample_cap]

    print(f"\nTotal: {len(all_rows)} samples")
    label_counts = Counter(r["label"] for r in all_rows)
    print(f"  malicious: {label_counts['malicious']}, benign: {label_counts['benign']}")

    print(f"\nScoring with threshold={args.threshold}...")
    results: list[dict[str, Any]] = []
    for i, row in enumerate(all_rows):
        score, n_active = score_text(row["text"], weights, bias)
        text_len = len(row["text"])
        predicted = "malicious" if score >= args.threshold else "benign"
        results.append({
            "label": row["label"],
            "predicted": predicted,
            "score": score,
            "n_active": n_active,
            "text_len": text_len,
            "length_bin": bin_label(text_len),
            "source": row["source"],
        })
        if (i + 1) % 50000 == 0:
            print(f"  scored {i+1}/{len(all_rows)}...")

    print(f"  scored {len(results)}/{len(all_rows)}")

    # Per-bin metrics
    bins: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        bins[r["length_bin"]].append(r)

    print(f"\n{'Bin':<10} {'N':>7} {'Atk':>6} {'Ben':>6} {'FNR':>7} {'FPR':>7} {'F1':>7}")
    print("-" * 58)

    bin_metrics: list[dict[str, Any]] = []
    for lo, hi, name in LENGTH_BINS:
        b = bins.get(name, [])
        if not b:
            continue
        tp = sum(1 for r in b if r["label"] == "malicious" and r["predicted"] == "malicious")
        fn = sum(1 for r in b if r["label"] == "malicious" and r["predicted"] == "benign")
        fp = sum(1 for r in b if r["label"] == "benign" and r["predicted"] == "malicious")
        tn = sum(1 for r in b if r["label"] == "benign" and r["predicted"] == "benign")
        n_atk = tp + fn
        n_ben = fp + tn
        fnr = fn / max(n_atk, 1)
        fpr = fp / max(n_ben, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)

        print(f"{name:<10} {len(b):>7} {n_atk:>6} {n_ben:>6} {fnr:>7.3f} {fpr:>7.3f} {f1:>7.3f}")
        bin_metrics.append({
            "bin": name, "n": len(b), "n_attack": n_atk, "n_benign": n_ben,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "fnr": round(fnr, 4), "fpr": round(fpr, 4), "f1": round(f1, 4),
        })

    # Global metrics
    all_tp = sum(m["tp"] for m in bin_metrics)
    all_fn = sum(m["fn"] for m in bin_metrics)
    all_fp = sum(m["fp"] for m in bin_metrics)
    all_tn = sum(m["tn"] for m in bin_metrics)
    g_fnr = all_fn / max(all_tp + all_fn, 1)
    g_fpr = all_fp / max(all_fp + all_tn, 1)
    g_prec = all_tp / max(all_tp + all_fp, 1)
    g_rec = all_tp / max(all_tp + all_fn, 1)
    g_f1 = 2 * g_prec * g_rec / max(g_prec + g_rec, 1e-9)

    print("-" * 58)
    print(f"{'GLOBAL':<10} {len(results):>7} {all_tp+all_fn:>6} {all_fp+all_tn:>6} {g_fnr:>7.3f} {g_fpr:>7.3f} {g_f1:>7.3f}")

    # Hypothesis test: does FNR increase with length?
    fnr_by_bin = [(m["bin"], m["fnr"], m["n_attack"]) for m in bin_metrics if m["n_attack"] > 10]
    fnr_increasing = all(
        fnr_by_bin[i][1] <= fnr_by_bin[i + 1][1]
        for i in range(len(fnr_by_bin) - 1)
    ) if len(fnr_by_bin) > 1 else False

    # Correlation (Spearman rank) between bin index and FNR
    if len(fnr_by_bin) > 2:
        from scipy.stats import spearmanr
        ranks = list(range(len(fnr_by_bin)))
        fnrs = [x[1] for x in fnr_by_bin]
        rho, p_value = spearmanr(ranks, fnrs)
    else:
        rho, p_value = 0.0, 1.0

    print(f"\nHypothesis: FNR increases with text length")
    print(f"  Monotonically increasing: {fnr_increasing}")
    print(f"  Spearman rho: {rho:.3f} (p={p_value:.4f})")
    if rho > 0.7 and p_value < 0.05:
        print(f"  CONFIRMED: Strong positive correlation. Dynamic threshold / L2-norm justified.")
    elif rho > 0.4:
        print(f"  SUGGESTIVE: Moderate correlation. Worth investigating further.")
    else:
        print(f"  NOT CONFIRMED: Weak or no correlation. Problem may be feature deficiency, not length.")

    # Write outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Score scatter CSV (for plotting)
    csv_path = args.output_dir / "score_vs_length.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "predicted", "score", "n_active", "text_len", "length_bin", "source"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nWrote {len(results)} rows to {csv_path}")

    # Summary JSON
    summary = {
        "threshold": args.threshold,
        "total_samples": len(results),
        "weights_file": str(args.weights_file),
        "n_weights": len(weights),
        "bias": bias,
        "global": {
            "tp": all_tp, "fn": all_fn, "fp": all_fp, "tn": all_tn,
            "fnr": round(g_fnr, 4), "fpr": round(g_fpr, 4), "f1": round(g_f1, 4),
        },
        "per_bin": bin_metrics,
        "hypothesis": {
            "fnr_monotonically_increasing": fnr_increasing,
            "spearman_rho": round(rho, 4),
            "spearman_p": round(p_value, 6),
            "confirmed": rho > 0.7 and p_value < 0.05,
        },
    }
    summary_path = args.output_dir / "diagnostic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
