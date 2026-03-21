#!/usr/bin/env python3
"""Recompute eval metrics after excluding non-PI sources or rows.

Usage:
    python scripts/adjusted_metrics.py \
        --eval-json runs/verified_residual_r3/eval.json \
        --exclude-sources thewall_sql_injection_zh_attacks \
        --preaudit runs/verified_residual_r3/persistent_intersection_preaudit.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-json", type=Path, required=True)
    p.add_argument(
        "--exclude-sources",
        nargs="*",
        default=[],
        help="Sources to exclude entirely (e.g. SQL injection datasets)",
    )
    p.add_argument(
        "--preaudit",
        type=Path,
        help="Pre-audit JSONL — rows with suggested_in_scope_pi=false are excluded from error counts",
    )
    return p.parse_args()


def compute_metrics(results: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    for r in results:
        is_attack = r["label"] == "malicious"
        predicted_attack = not r["correct"] if not is_attack else r["correct"]
        # Simpler: correct + malicious = TP, correct + benign = TN,
        #          incorrect + malicious = FN, incorrect + benign = FP
        if r["correct"]:
            if is_attack:
                tp += 1
            else:
                tn += 1
        else:
            if is_attack:
                fn += 1
            else:
                fp += 1

    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "N": total,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "F1": f1,
        "P": precision,
        "R": recall,
        "attacks": tp + fn,
        "benign": tn + fp,
    }


def main() -> None:
    args = parse_args()

    print("Loading eval.json...", file=sys.stderr)
    with open(args.eval_json, encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    exclude_sources = set(args.exclude_sources)

    # Load pre-audit out-of-scope case_ids
    oos_case_ids: set[str] = set()
    if args.preaudit:
        print("Loading pre-audit...", file=sys.stderr)
        with open(args.preaudit, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("suggested_in_scope_pi") is False:
                    oos_case_ids.add(str(row["case_id"]))

    # --- Pass 1: raw (no exclusions) ---
    m_raw = compute_metrics(results)

    # --- Pass 2: exclude sources ---
    filtered_src = [r for r in results if r["source"] not in exclude_sources]
    excluded_src_count = len(results) - len(filtered_src)

    # Count excluded by label
    excluded_by_label = Counter()
    for r in results:
        if r["source"] in exclude_sources:
            excluded_by_label[r["label"]] += 1

    m_src = compute_metrics(filtered_src)

    # --- Pass 3: exclude sources + pre-audit OOS rows ---
    # For OOS rows: if a FN is out-of-scope, mark it as "correct" (not a real PI miss)
    # If a FP is out-of-scope, that's trickier — keep as-is for now
    adjusted = []
    oos_fn_removed = 0
    for r in filtered_src:
        cid = str(r["case_id"])
        if cid in oos_case_ids and not r["correct"] and r["label"] == "malicious":
            # This FN is not PI — exclude it entirely
            oos_fn_removed += 1
            continue
        adjusted.append(r)

    m_adj = compute_metrics(adjusted)

    # --- Report ---
    print(f"\n{'='*65}")
    print("METRIC COMPARISON")
    print(f"{'='*65}")
    print(f"{'':>30} {'Raw':>10} {'-Sources':>10} {'-Src-OOS':>12}")
    print(f"{'N':>30} {m_raw['N']:>10,} {m_src['N']:>10,} {m_adj['N']:>12,}")
    print(f"{'Attacks':>30} {m_raw['attacks']:>10,} {m_src['attacks']:>10,} {m_adj['attacks']:>12,}")
    print(f"{'Benign':>30} {m_raw['benign']:>10,} {m_src['benign']:>10,} {m_adj['benign']:>12,}")
    print(f"{'F1':>30} {m_raw['F1']:>10.4f} {m_src['F1']:>10.4f} {m_adj['F1']:>12.4f}")
    print(f"{'Precision':>30} {m_raw['P']:>10.4f} {m_src['P']:>10.4f} {m_adj['P']:>12.4f}")
    print(f"{'Recall':>30} {m_raw['R']:>10.4f} {m_src['R']:>10.4f} {m_adj['R']:>12.4f}")
    print(f"{'FP':>30} {m_raw['FP']:>10,} {m_src['FP']:>10,} {m_adj['FP']:>12,}")
    print(f"{'FN':>30} {m_raw['FN']:>10,} {m_src['FN']:>10,} {m_adj['FN']:>12,}")

    print(f"\n{'='*65}")
    print("EXCLUSION DETAILS")
    print(f"{'='*65}")
    if exclude_sources:
        print(f"  Sources excluded: {', '.join(sorted(exclude_sources))}")
        print(f"  Rows removed (source): {excluded_src_count:,}")
        for label, count in sorted(excluded_by_label.items()):
            print(f"    {label}: {count:,}")
    if oos_case_ids:
        print(f"  Pre-audit OOS case_ids: {len(oos_case_ids):,}")
        print(f"  OOS FN removed: {oos_fn_removed:,}")


if __name__ == "__main__":
    main()
