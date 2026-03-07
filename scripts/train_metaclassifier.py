#!/usr/bin/env python3
"""
Train L1 meta-classifier (stacking) on holdout specialist scores.

Reads the CSV output from `parapet-eval --l1-scores` and trains a
LogisticRegression on the 7 raw SVM margin scores. Uses stratified
k-fold cross-validation so the meta-classifier never sees its own
training data — addressing the OOF concern from the review.

Usage:
    python scripts/train_metaclassifier.py \
        --holdout schema/eval/ensemble/holdout_scores.csv \
        --eval    schema/eval/ensemble/eval_scores.csv       # optional validation
        --folds 5

Outputs:
    - Per-fold precision/recall/F1
    - Learned weights + bias (for Rust codegen)
    - OOF confusion matrix
    - Optional: eval set validation report
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix


def load_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Binary label: 1 = malicious, 0 = benign
    df["y"] = (df["label"] == "malicious").astype(int)
    return df


FEATURE_COLS = [
    "generalist",
    "roleplay_jailbreak",
    "instruction_override",
    "meta_probe",
    "exfiltration",
    "adversarial_suffix",
    "indirect_injection",
]


def train_kfold(df: pd.DataFrame, n_folds: int = 5, C: float = 0.1):
    """Train with stratified k-fold, return OOF predictions and final model."""
    X = df[FEATURE_COLS].values
    y = df["y"].values

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(y), dtype=int)
    oof_probs = np.zeros(len(y), dtype=float)
    fold_reports = []

    print(f"\n{'='*60}")
    print(f"Stratified {n_folds}-Fold Cross-Validation (C={C})")
    print(f"{'='*60}")
    print(f"Total samples: {len(y)} (malicious={y.sum()}, benign={len(y)-y.sum()})")
    print()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = LogisticRegression(penalty="l2", C=C, class_weight="balanced",
                                 max_iter=1000, solver="lbfgs")
        clf.fit(X_train, y_train)

        preds = clf.predict(X_val)
        probs = clf.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = preds
        oof_probs[val_idx] = probs

        cm = confusion_matrix(y_val, preds)
        tn, fp, fn, tp = cm.ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        print(f"Fold {fold_idx+1}: TP={tp} FP={fp} FN={fn} TN={tn}  "
              f"P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
        fold_reports.append({"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                             "prec": prec, "rec": rec, "f1": f1})

    # OOF aggregate
    print(f"\n{'='*60}")
    print("OOF Aggregate (all folds combined)")
    print(f"{'='*60}")
    cm = confusion_matrix(y, oof_preds)
    tn, fp, fn, tp = cm.ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print(f"Accuracy={np.mean(y == oof_preds):.4f}")

    # Per-source breakdown of FPs and FNs
    print(f"\n{'='*60}")
    print("OOF Per-Source Breakdown")
    print(f"{'='*60}")
    df["oof_pred"] = oof_preds
    df["oof_correct"] = (df["y"] == df["oof_pred"]).astype(int)

    print(f"\n{'Misclassifications by source:'}")
    print(f"{'source':<45} {'total':>5} {'wrong':>5} {'FP':>5} {'FN':>5} {'acc%':>7}")
    print("-" * 75)

    for source in sorted(df["source"].unique()):
        subset = df[df["source"] == source]
        total = len(subset)
        wrong = (subset["oof_correct"] == 0).sum()
        fp_count = ((subset["y"] == 0) & (subset["oof_pred"] == 1)).sum()
        fn_count = ((subset["y"] == 1) & (subset["oof_pred"] == 0)).sum()
        acc = subset["oof_correct"].mean() * 100
        if wrong > 0:
            print(f"  {source:<43} {total:>5} {wrong:>5} {fp_count:>5} {fn_count:>5} {acc:>6.1f}%")

    # Clean up temp columns
    df.drop(columns=["oof_pred", "oof_correct"], inplace=True)

    # Train final model on all data
    print(f"\n{'='*60}")
    print("Final Model (trained on all holdout data)")
    print(f"{'='*60}")
    final_clf = LogisticRegression(penalty="l2", C=C, class_weight="balanced",
                                   max_iter=1000, solver="lbfgs")
    final_clf.fit(X, y)

    print(f"\nBias: {final_clf.intercept_[0]:.6f}")
    print(f"\nWeights:")
    for name, weight in zip(FEATURE_COLS, final_clf.coef_[0]):
        print(f"  {name:<25} {weight:+.6f}")

    return final_clf, oof_preds, oof_probs


def validate_eval(clf, eval_df: pd.DataFrame):
    """Validate the trained meta-classifier on the eval dataset."""
    X = eval_df[FEATURE_COLS].values
    y = eval_df["y"].values

    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:, 1]

    print(f"\n{'='*60}")
    print("Eval Set Validation (never seen during training)")
    print(f"{'='*60}")
    print(f"Total: {len(y)} (malicious={y.sum()}, benign={len(y)-y.sum()})")

    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print(f"Accuracy={np.mean(y == preds):.4f}")

    # Compare to generalist-only baseline
    gen_preds = (eval_df["generalist"].values > 0.0).astype(int)
    gen_cm = confusion_matrix(y, gen_preds)
    gen_tn, gen_fp, gen_fn, gen_tp = gen_cm.ravel()
    gen_prec = gen_tp / (gen_tp + gen_fp) if (gen_tp + gen_fp) > 0 else 0
    gen_rec = gen_tp / (gen_tp + gen_fn) if (gen_tp + gen_fn) > 0 else 0
    gen_f1 = 2 * gen_prec * gen_rec / (gen_prec + gen_rec) if (gen_prec + gen_rec) > 0 else 0
    print(f"\nGeneralist-only baseline (score > 0.0):")
    print(f"TP={gen_tp}  FP={gen_fp}  FN={gen_fn}  TN={gen_tn}")
    print(f"Precision={gen_prec:.4f}  Recall={gen_rec:.4f}  F1={gen_f1:.4f}")

    # Delta
    print(f"\nMeta vs Generalist delta:")
    print(f"  FP: {fp} vs {gen_fp} ({fp - gen_fp:+d})")
    print(f"  FN: {fn} vs {gen_fn} ({fn - gen_fn:+d})")
    print(f"  F1: {f1:.4f} vs {gen_f1:.4f} ({f1 - gen_f1:+.4f})")

    # Per-source FP/FN breakdown
    print(f"\n{'Eval misclassifications by source:'}")
    print(f"{'source':<45} {'total':>5} {'wrong':>5} {'FP':>5} {'FN':>5}")
    print("-" * 70)
    eval_df["pred"] = preds
    for source in sorted(eval_df["source"].unique()):
        subset = eval_df[eval_df["source"] == source]
        total = len(subset)
        y_sub = subset["y"].values
        p_sub = subset["pred"].values
        fp_count = ((y_sub == 0) & (p_sub == 1)).sum()
        fn_count = ((y_sub == 1) & (p_sub == 0)).sum()
        wrong = fp_count + fn_count
        if wrong > 0:
            print(f"  {source:<43} {total:>5} {wrong:>5} {fp_count:>5} {fn_count:>5}")
    eval_df.drop(columns=["pred"], inplace=True)


def emit_rust(clf, feature_cols):
    """Print Rust code for the meta-classifier weights."""
    print(f"\n{'='*60}")
    print("Rust Codegen")
    print(f"{'='*60}")
    print()
    print("// Auto-generated by scripts/train_metaclassifier.py")
    print("// Do not edit manually.")
    print()
    print("pub const META_BIAS: f64 = {:.10};".format(clf.intercept_[0]))
    print()
    print("pub const META_WEIGHTS: [(&str, f64); {}] = [".format(len(feature_cols)))
    for name, weight in zip(feature_cols, clf.coef_[0]):
        print(f'    ("{name}", {weight:.10}),')
    print("];")


def sweep_thresholds(clf, df: pd.DataFrame, label: str = "holdout"):
    """Sweep probability thresholds to find optimal FP/FN tradeoff."""
    X = df[FEATURE_COLS].values
    y = df["y"].values
    probs = clf.predict_proba(X)[:, 1]

    print(f"\n{'='*60}")
    print(f"Threshold Sweep ({label})")
    print(f"{'='*60}")
    print(f"{'thresh':>7} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>6} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 55)

    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        preds = (probs >= thresh).astype(int)
        cm = confusion_matrix(y, preds)
        tn, fp, fn, tp = cm.ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"  {thresh:.1f}   {tp:>5} {fp:>5} {fn:>5} {tn:>6} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train L1 meta-classifier")
    parser.add_argument("--holdout", type=Path, required=True,
                        help="Path to holdout_scores.csv")
    parser.add_argument("--eval", type=Path, default=None,
                        help="Path to eval_scores.csv (optional validation)")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--C", type=float, default=0.1,
                        help="Regularization strength (default: 0.1)")
    parser.add_argument("--rust", action="store_true",
                        help="Emit Rust codegen for the weights")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep probability thresholds")
    args = parser.parse_args()

    # Load holdout scores
    holdout_df = load_scores(args.holdout)
    print(f"Loaded {len(holdout_df)} holdout cases from {args.holdout}")
    print(f"  Malicious: {holdout_df['y'].sum()}")
    print(f"  Benign:    {(holdout_df['y'] == 0).sum()}")
    print(f"  Sources:   {holdout_df['source'].nunique()}")

    # Train with k-fold
    clf, oof_preds, oof_probs = train_kfold(holdout_df, n_folds=args.folds, C=args.C)

    # Threshold sweep on holdout
    if args.sweep:
        sweep_thresholds(clf, holdout_df, label="holdout OOF")

    # Validate on eval set if provided
    if args.eval is not None:
        eval_df = load_scores(args.eval)
        print(f"\nLoaded {len(eval_df)} eval cases from {args.eval}")
        validate_eval(clf, eval_df)
        if args.sweep:
            sweep_thresholds(clf, eval_df, label="eval")

    # Rust codegen
    if args.rust:
        emit_rust(clf, FEATURE_COLS)


if __name__ == "__main__":
    main()
