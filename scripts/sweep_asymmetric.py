"""Sweep asymmetric routing thresholds to find FP≈0, FN≈0 operating point.

Asymmetric routing logic:
  Rule 1: generalist >= high_thresh -> Block (solo, no corroboration needed)
  Rule 2: generalist >= low_thresh AND any specialist >= spec_thresh -> Block
  Rule 3: else -> Pass

Sweeps high_thresh × spec_thresh to find the combination where
FP and FN both approach zero.

Usage:
  python scripts/sweep_asymmetric.py schema/eval/ensemble/l1_specialist_scores.csv
"""

import csv
import sys


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


def evaluate_asymmetric(rows, specialists, high_thresh, low_thresh, spec_thresh):
    """Evaluate asymmetric routing at given thresholds."""
    tp = fp = fn = tn = 0
    non_gen = [s for s in specialists if s != "generalist"]

    for r in rows:
        gen = r["scores"]["generalist"]
        is_mal = r["label"] == "malicious"

        blocked = False
        # Rule 1: generalist solo at high confidence
        if gen >= high_thresh:
            blocked = True
        # Rule 2: generalist suspicious + specialist corroboration
        elif gen >= low_thresh:
            for s in non_gen:
                if r["scores"][s] >= spec_thresh:
                    blocked = True
                    break

        if is_mal:
            if blocked:
                tp += 1
            else:
                fn += 1
        else:
            if blocked:
                fp += 1
            else:
                tn += 1

    return tp, fp, fn, tn


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "schema/eval/ensemble/l1_specialist_scores.csv"
    rows, specialists = load_scores(path)

    malicious = [r for r in rows if r["label"] == "malicious"]
    benign = [r for r in rows if r["label"] == "benign"]
    n_mal = len(malicious)
    n_ben = len(benign)

    print(f"Dataset: {len(rows)} total ({n_mal} malicious, {n_ben} benign)")
    print(f"Specialists: {specialists}")

    # --- Sweep 1: Fix low_thresh=0.0, sweep high_thresh × spec_thresh ---
    print(f"\n{'='*90}")
    print(f"Asymmetric routing: gen >= high -> solo block; gen >= 0.0 AND any spec >= spec_thresh -> block")
    print(f"{'='*90}")

    high_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    spec_thresholds = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    low_thresh = 0.0

    print(f"\n{'high':>6} {'spec':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>7} {'Recall':>7} {'F1':>7} {'FPR':>7}")
    print("-" * 75)

    best_f1 = 0
    best_config = None

    for ht in high_thresholds:
        for st in spec_thresholds:
            tp, fp, fn, tn = evaluate_asymmetric(rows, specialists, ht, low_thresh, st)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
            fpr = fp / n_ben if n_ben > 0 else 0

            print(f"{ht:>6.1f} {st:>6.1f} {tp:>6} {fp:>6} {fn:>6} {prec:>7.3f} {recall:>7.3f} {f1:>7.3f} {fpr:>6.3f}%")

            if f1 > best_f1:
                best_f1 = f1
                best_config = (ht, st, tp, fp, fn, prec, recall, f1, fpr)

    if best_config:
        ht, st, tp, fp, fn, prec, recall, f1, fpr = best_config
        print(f"\nBest F1: high={ht}, spec={st} -> TP={tp} FP={fp} FN={fn} P={prec:.3f} R={recall:.3f} F1={f1:.3f} FPR={fpr:.3f}%")

    # --- Sweep 2: Also vary low_thresh (the generalist gate for corroboration) ---
    print(f"\n{'='*90}")
    print(f"Extended: vary low_thresh too")
    print(f"{'='*90}")

    low_thresholds = [-0.5, -0.25, 0.0, 0.25, 0.5]

    print(f"\n{'low':>6} {'high':>6} {'spec':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>7} {'Recall':>7} {'F1':>7} {'FPR':>7}")
    print("-" * 85)

    best_f1 = 0
    best_config = None

    for lt in low_thresholds:
        for ht in [1.0, 1.5, 2.0, 2.5, 3.0]:
            if ht <= lt:
                continue
            for st in [-0.5, 0.0, 0.5, 1.0]:
                tp, fp, fn, tn = evaluate_asymmetric(rows, specialists, ht, lt, st)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
                fpr = fp / n_ben if n_ben > 0 else 0

                # Only print promising configs (F1 > 90%)
                if f1 >= 0.90:
                    print(f"{lt:>6.2f} {ht:>6.1f} {st:>6.1f} {tp:>6} {fp:>6} {fn:>6} {prec:>7.3f} {recall:>7.3f} {f1:>7.3f} {fpr:>6.3f}%")

                if f1 > best_f1:
                    best_f1 = f1
                    best_config = (lt, ht, st, tp, fp, fn, prec, recall, f1, fpr)

    if best_config:
        lt, ht, st, tp, fp, fn, prec, recall, f1, fpr = best_config
        print(f"\nBest F1: low={lt}, high={ht}, spec={st} -> TP={tp} FP={fp} FN={fn} P={prec:.3f} R={recall:.3f} F1={f1:.3f} FPR={fpr:.3f}%")

    # --- V4-10B comparable subset ---
    print(f"\n{'='*90}")
    print(f"V4-10B comparable subset (excluding global_benign_*)")
    print(f"{'='*90}")

    v4_rows = [r for r in rows if not r["source"].startswith("global_benign")]
    v4_mal = sum(1 for r in v4_rows if r["label"] == "malicious")
    v4_ben = sum(1 for r in v4_rows if r["label"] == "benign")
    print(f"Subset: {len(v4_rows)} total ({v4_mal} malicious, {v4_ben} benign)")
    print(f"V4-10B baseline: TP=1988 FP=80 FN=79 P=96.1% R=96.2% F1=96.2%")

    print(f"\n{'low':>6} {'high':>6} {'spec':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>7} {'Recall':>7} {'F1':>7} {'FPR':>7}")
    print("-" * 85)

    for lt in [-0.5, -0.25, 0.0]:
        for ht in [1.0, 1.5, 2.0, 2.5, 3.0]:
            if ht <= lt:
                continue
            for st in [-0.5, 0.0, 0.5, 1.0]:
                tp, fp, fn, tn = evaluate_asymmetric(v4_rows, specialists, ht, lt, st)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
                fpr = fp / v4_ben if v4_ben > 0 else 0

                if f1 >= 0.93:
                    print(f"{lt:>6.2f} {ht:>6.1f} {st:>6.1f} {tp:>6} {fp:>6} {fn:>6} {prec:>7.3f} {recall:>7.3f} {f1:>7.3f} {fpr:>6.3f}%")


if __name__ == "__main__":
    main()
