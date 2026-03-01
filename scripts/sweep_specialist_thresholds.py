"""Sweep per-specialist thresholds to find 99.5%+ recall operating points.

Reads the CSV from --l1-scores and for each specialist finds the threshold
that achieves target recall, reporting FP at that threshold.

Also sweeps min_agree (1..7) × per-specialist thresholds to find the
best consensus configuration.

Usage:
  python scripts/sweep_specialist_thresholds.py schema/eval/ensemble/l1_specialist_scores.csv
"""

import csv
import sys
from collections import defaultdict

def load_scores(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        specialists = [c for c in reader.fieldnames if c not in ("case_id", "label", "source")]
        for row in reader:
            entry = {
                "case_id": row["case_id"],
                "label": row["label"],
                "source": row["source"],
                "scores": {s: float(row[s]) for s in specialists},
            }
            rows.append(entry)
    return rows, specialists


def sweep_single(rows, specialists, target_recall=0.995):
    """For each specialist, find threshold that achieves target recall."""
    malicious = [r for r in rows if r["label"] == "malicious"]
    benign = [r for r in rows if r["label"] == "benign"]
    n_mal = len(malicious)
    n_ben = len(benign)

    print(f"\nDataset: {len(rows)} total ({n_mal} malicious, {n_ben} benign)")
    print(f"Target recall: {target_recall*100:.1f}%")
    print()

    print(f"{'Specialist':<25} {'Thresh@99.5R':>12} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    print("-" * 90)

    results = {}
    for spec in specialists:
        # Sort malicious by score descending
        mal_scores = sorted([r["scores"][spec] for r in malicious], reverse=True)
        ben_scores = [r["scores"][spec] for r in benign]

        # Find threshold where we catch target_recall of malicious
        target_tp = int(n_mal * target_recall)
        if target_tp >= len(mal_scores):
            target_tp = len(mal_scores) - 1

        # Threshold = score of the target_tp-th malicious sample (0-indexed)
        threshold = mal_scores[target_tp]

        # Count at this threshold
        tp = sum(1 for s in mal_scores if s >= threshold)
        fn = n_mal - tp
        fp = sum(1 for s in ben_scores if s >= threshold)
        tn = n_ben - fp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0

        print(f"{spec:<25} {threshold:>12.4f} {tp:>6} {fp:>6} {fn:>6} {prec:>7.3f} {recall:>7.3f} {f1:>7.3f}")
        results[spec] = {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn}

    return results


def sweep_consensus(rows, specialists, thresholds_by_spec, min_agree_values=None):
    """Sweep min_agree with given per-specialist thresholds."""
    if min_agree_values is None:
        min_agree_values = list(range(1, len(specialists) + 1))

    malicious = [r for r in rows if r["label"] == "malicious"]
    benign = [r for r in rows if r["label"] == "benign"]
    n_mal = len(malicious)
    n_ben = len(benign)

    print(f"\n\nConsensus sweep (per-specialist thresholds set for 99.5% individual recall)")
    print(f"{'min_agree':>10} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>7} {'Recall':>7} {'F1':>7} {'FPR':>7}")
    print("-" * 65)

    for min_agree in min_agree_values:
        tp = 0
        fp = 0
        for r in malicious:
            breach_count = sum(
                1 for s in specialists
                if r["scores"][s] >= thresholds_by_spec[s]
            )
            if breach_count >= min_agree:
                tp += 1
        fn = n_mal - tp
        for r in benign:
            breach_count = sum(
                1 for s in specialists
                if r["scores"][s] >= thresholds_by_spec[s]
            )
            if breach_count >= min_agree:
                fp += 1
        tn = n_ben - fp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
        fpr = fp / n_ben if n_ben > 0 else 0

        print(f"{min_agree:>10} {tp:>6} {fp:>6} {fn:>6} {prec:>7.3f} {recall:>7.3f} {f1:>7.3f} {fpr:>6.2f}%")


def sweep_fixed_thresholds(rows, specialists):
    """Sweep common thresholds across all specialists with various min_agree."""
    malicious = [r for r in rows if r["label"] == "malicious"]
    benign = [r for r in rows if r["label"] == "benign"]
    n_mal = len(malicious)
    n_ben = len(benign)

    thresholds = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    min_agrees = [1, 2, 3]

    print(f"\n\nGrid sweep: fixed threshold × min_agree")
    print(f"{'thresh':>7} {'min_agree':>10} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>7} {'Recall':>7} {'F1':>7} {'FPR':>7}")
    print("-" * 75)

    for thresh in thresholds:
        for min_agree in min_agrees:
            tp = 0
            fp = 0
            for r in malicious:
                breach_count = sum(1 for s in specialists if r["scores"][s] >= thresh)
                if breach_count >= min_agree:
                    tp += 1
            fn = n_mal - tp
            for r in benign:
                breach_count = sum(1 for s in specialists if r["scores"][s] >= thresh)
                if breach_count >= min_agree:
                    fp += 1

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
            fpr = fp / n_ben if n_ben > 0 else 0

            print(f"{thresh:>7.1f} {min_agree:>10} {tp:>6} {fp:>6} {fn:>6} {prec:>7.3f} {recall:>7.3f} {f1:>7.3f} {fpr:>6.2f}%")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "schema/eval/ensemble/l1_specialist_scores.csv"
    rows, specialists = load_scores(path)

    # 1. Per-specialist threshold sweep targeting 99.5% recall
    results = sweep_single(rows, specialists, target_recall=0.995)

    # 2. Consensus sweep using those thresholds
    thresholds = {s: results[s]["threshold"] for s in specialists}
    sweep_consensus(rows, specialists, thresholds)

    # 3. Grid sweep: fixed thresholds × min_agree
    sweep_fixed_thresholds(rows, specialists)


if __name__ == "__main__":
    main()
