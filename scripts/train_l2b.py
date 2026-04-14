"""Train L2b precision-recovery model on L1 out-of-fold residuals.

Models:
  1. Word-TFIDF + logistic regression (baseline / sanity check)
  2. Word-TFIDF + structural features + random forest
  3. Word-TFIDF + structural features + XGBoost (primary candidate)

L2b's task: given text that L1 escalated, decide if it's truly malicious or benign.
Training labels are ground-truth labels, not L1 predictions.

Usage:
    cd parapet
    python scripts/train_l2b.py --residuals-dir parapet-runner/runs/l2b_residuals --output-dir parapet-runner/runs/l2b_models
    python scripts/train_l2b.py --residuals-dir parapet-runner/runs/l2b_residuals --output-dir parapet-runner/runs/l2b_models_xgb --model xgb
    python scripts/train_l2b.py --residuals-dir parapet-runner/runs/l2b_residuals --output-dir parapet-runner/runs/l2b_models --features harness+structural
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random as random_mod
import re
import string
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Load residuals
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Training mix
# ---------------------------------------------------------------------------

MIX_RATIOS = {
    "false_negative": 0.35,
    "false_positive": 0.25,
    "near_boundary_benign": 0.20,
    "baseline_correct": 0.20,
}


def build_training_mix(
    residuals: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    """Sample residuals according to L2b_POC.md mix ratios."""
    rng = random_mod.Random(seed)

    by_category: dict[str, list[dict]] = {}
    for r in residuals:
        cat = r.get("residual_category", "unknown")
        by_category.setdefault(cat, []).append(r)

    fn_count = len(by_category.get("false_negative", []))
    if fn_count == 0:
        raise ValueError("No false negatives in residual dataset")

    total_target = int(fn_count / MIX_RATIOS["false_negative"])

    mixed: list[dict] = []
    for category, ratio in MIX_RATIOS.items():
        pool = by_category.get(category, [])
        target = min(int(total_target * ratio), len(pool))
        if target < len(pool):
            sampled = rng.sample(pool, target)
        else:
            sampled = list(pool)
        mixed.extend(sampled)
        print(f"  {category:<25} target={int(total_target*ratio):>5}  "
              f"available={len(pool):>5}  sampled={len(sampled):>5}")

    rng.shuffle(mixed)
    return mixed


# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------

FEATURE_FAMILIES = {"harness", "structural", "tfidf"}

# Valid --features values map to sets of families
FEATURE_PRESETS = {
    "all": {"harness", "structural", "tfidf"},
    "harness": {"harness"},
    "harness+structural": {"harness", "structural"},
    "harness+tfidf": {"harness", "tfidf"},
    "harness+structural+tfidf": {"harness", "structural", "tfidf"},
}


def parse_feature_config(raw: str) -> set[str]:
    """Parse --features argument into a set of family names."""
    if raw in FEATURE_PRESETS:
        return FEATURE_PRESETS[raw]
    families = {f.strip() for f in raw.split("+")}
    unknown = families - FEATURE_FAMILIES
    if unknown:
        raise ValueError(f"Unknown feature families: {unknown}. Valid: {FEATURE_FAMILIES}")
    return families


# ---------------------------------------------------------------------------
# Structural features
# ---------------------------------------------------------------------------

QUOTE_PATTERNS = [
    re.compile(r'"[^"]{10,}"'),
    re.compile(r"'[^']{10,}'"),
    re.compile(r'`[^`]{10,}`'),
    re.compile(r'```[\s\S]*?```'),
    re.compile(r'^>\s', re.MULTILINE),
]

CODE_MARKERS = re.compile(
    r'(?:def |class |import |function |var |let |const |if\s*\(|for\s*\(|while\s*\(|return |print\()',
    re.IGNORECASE,
)

FRAMING_MARKERS = re.compile(
    r'\b(?:example|for instance|such as|e\.g\.|here is|the following|'
    r'writeup|write-up|analysis|CTF|challenge|security|discuss|'
    r'prompt injection|jailbreak|attack technique|red team)\b',
    re.IGNORECASE,
)


def compute_structural_features(text: str) -> dict[str, float]:
    """Extract structural features from text."""
    lines = text.split("\n")
    n_lines = len(lines)
    n_chars = len(text)
    n_words = len(text.split())
    n_punct = sum(1 for c in text if c in string.punctuation)
    n_digits = sum(1 for c in text if c.isdigit())
    n_upper = sum(1 for c in text if c.isupper())
    n_spaces = sum(1 for c in text if c.isspace())
    n_quotes = sum(len(pat.findall(text)) for pat in QUOTE_PATTERNS)
    n_code_markers = len(CODE_MARKERS.findall(text))
    n_framing_markers = len(FRAMING_MARKERS.findall(text))

    if n_chars > 0:
        char_counts = Counter(text)
        entropy = -sum(
            (count / n_chars) * math.log2(count / n_chars)
            for count in char_counts.values()
        )
    else:
        entropy = 0.0

    words = text.lower().split()
    if words:
        word_counts = Counter(words)
        repeated = sum(1 for w, c in word_counts.items() if c > 1)
        repeated_ratio = repeated / len(word_counts)
    else:
        repeated_ratio = 0.0

    n_delimiters = sum(1 for c in text if c in "{}[]()<>|/\\;:=+*&^%$#@!")
    safe_chars = max(n_chars, 1)
    safe_words = max(n_words, 1)

    return {
        "n_chars": n_chars,
        "n_words": n_words,
        "n_lines": n_lines,
        "avg_line_length": n_chars / max(n_lines, 1),
        "punct_ratio": n_punct / safe_chars,
        "digit_ratio": n_digits / safe_chars,
        "upper_ratio": n_upper / safe_chars,
        "space_ratio": n_spaces / safe_chars,
        "delimiter_density": n_delimiters / safe_chars,
        "quote_count": n_quotes,
        "code_marker_count": n_code_markers,
        "framing_marker_count": n_framing_markers,
        "entropy": entropy,
        "repeated_token_ratio": repeated_ratio,
        "avg_word_length": n_chars / safe_words,
    }


STRUCTURAL_FEATURE_NAMES = list(compute_structural_features("dummy text").keys())

L1_HARNESS_NAMES = [
    "l1_raw_score", "l1_raw_unquoted_score", "l1_raw_squash_score",
    "l1_raw_score_delta", "l1_quote_detected",
]


def extract_harness_features(sample: dict[str, Any]) -> list[float]:
    return [
        sample.get("raw_score") or 0.0,
        sample.get("raw_unquoted_score") or 0.0,
        sample.get("raw_squash_score") or 0.0,
        sample.get("raw_score_delta") or 0.0,
        1.0 if sample.get("quote_detected") else 0.0,
    ]


def extract_features(
    samples: list[dict[str, Any]],
    families: set[str],
    tfidf: TfidfVectorizer | None = None,
    fit: bool = False,
) -> tuple[Any, TfidfVectorizer | None, list[str]]:
    """Extract features based on the specified families."""
    matrices = []
    names: list[str] = []

    # TF-IDF
    if "tfidf" in families and tfidf is not None:
        texts = [s["content"] for s in samples]
        if fit:
            tfidf_matrix = tfidf.fit_transform(texts)
        else:
            tfidf_matrix = tfidf.transform(texts)
        matrices.append(tfidf_matrix)
        names.extend(tfidf.get_feature_names_out().tolist())

    # Dense features (harness + structural)
    dense_rows = []
    dense_names: list[str] = []

    if "harness" in families:
        if not dense_names:
            dense_names.extend(L1_HARNESS_NAMES)
        for s in samples:
            dense_rows.append(extract_harness_features(s))

    if "structural" in families:
        if not dense_names or "harness" not in families:
            dense_names = []
        if "harness" in families:
            dense_names.extend(STRUCTURAL_FEATURE_NAMES)
        else:
            dense_names.extend(STRUCTURAL_FEATURE_NAMES)

    # Build dense matrix properly
    if "harness" in families or "structural" in families:
        rows = []
        for s in samples:
            row = []
            if "harness" in families:
                row.extend(extract_harness_features(s))
            if "structural" in families:
                sf = compute_structural_features(s["content"])
                row.extend(sf.values())
            rows.append(row)

        dense_names = []
        if "harness" in families:
            dense_names.extend(L1_HARNESS_NAMES)
        if "structural" in families:
            dense_names.extend(STRUCTURAL_FEATURE_NAMES)

        dense_matrix = csr_matrix(np.array(rows, dtype=np.float64))
        matrices.append(dense_matrix)
        names.extend(dense_names)

    if not matrices:
        raise ValueError("No feature families selected")

    if len(matrices) == 1:
        combined = matrices[0]
    else:
        combined = hstack(matrices)

    return combined, tfidf, names


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model": name,
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
    }

    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["avg_precision"] = float(average_precision_score(y_true, y_prob))

    print(f"\n=== {name} ===")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  FPR:       {metrics['fpr']:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    if "roc_auc" in metrics:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  Avg Prec:  {metrics['avg_precision']:.4f}")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Train L2b precision-recovery model")
    parser.add_argument("--residuals-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tfidf-features", type=int, default=10000)
    parser.add_argument("--n-trees", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument(
        "--model",
        choices=["lr", "rf", "xgb", "all"],
        default="all",
        help="Which model(s) to train (default: all)",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="all",
        help="Feature families: all, harness, harness+structural, harness+tfidf, "
             "harness+structural+tfidf (default: all = harness+structural+tfidf)",
    )
    args = parser.parse_args()

    families = parse_feature_config(args.features)
    print(f"Feature families: {sorted(families)}")

    residuals_dir = Path(args.residuals_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load residuals
    candidates_path = residuals_dir / "l2b_training_candidates.jsonl"
    print(f"Loading {candidates_path}...")
    all_residuals = load_jsonl(candidates_path)
    print(f"  {len(all_residuals):,} candidates")

    # Build training mix
    print("\nBuilding training mix...")
    mixed = build_training_mix(all_residuals, seed=args.seed)
    print(f"  Total mixed: {len(mixed):,}")

    labels = [1 if s["true_label"] == "malicious" else 0 for s in mixed]
    print(f"  Malicious: {sum(labels):,}  Benign: {len(labels) - sum(labels):,}")

    # Train/val split
    train_idx, val_idx = train_test_split(
        range(len(mixed)), test_size=0.2, random_state=args.seed,
        stratify=labels,
    )
    train_samples = [mixed[i] for i in train_idx]
    val_samples = [mixed[i] for i in val_idx]
    y_train = np.array([labels[i] for i in train_idx])
    y_val = np.array([labels[i] for i in val_idx])

    print(f"\n  Train: {len(train_samples)} (mal={y_train.sum()}, ben={len(y_train)-y_train.sum()})")
    print(f"  Val:   {len(val_samples)} (mal={y_val.sum()}, ben={len(y_val)-y_val.sum()})")

    # Build features
    print("\nExtracting features...")
    tfidf = None
    if "tfidf" in families:
        tfidf = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=args.max_tfidf_features,
            min_df=3,
            sublinear_tf=True,
        )

    X_train, tfidf, feature_names = extract_features(
        train_samples, families, tfidf, fit=True,
    )
    X_val, _, _ = extract_features(val_samples, families, tfidf, fit=False)

    n_tfidf = len(tfidf.get_feature_names_out()) if tfidf else 0
    print(f"  TF-IDF features: {n_tfidf}")
    print(f"  Total features: {X_train.shape[1]}")

    # Save feature config (consumed by eval_cascade.py)
    feature_config = {
        "families": sorted(families),
        "max_tfidf_features": args.max_tfidf_features if "tfidf" in families else 0,
        "tfidf_ngram_range": [1, 2] if "tfidf" in families else None,
        "tfidf_min_df": 3 if "tfidf" in families else None,
    }
    (output_dir / "feature_config.json").write_text(
        json.dumps(feature_config, indent=2), encoding="utf-8"
    )

    models_to_train = (
        ["lr", "rf", "xgb"] if args.model == "all"
        else [args.model]
    )
    results = {}

    # --- Logistic Regression ---
    if "lr" in models_to_train:
        print("\nTraining logistic regression...")
        lr = LogisticRegression(
            C=1.0, max_iter=5000, random_state=args.seed, solver="saga",
        )
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_val)
        lr_prob = lr.predict_proba(X_val)[:, 1]
        results["logistic"] = evaluate_model("Logistic Regression", y_val, lr_pred, lr_prob)
        with open(output_dir / "lr_model.pkl", "wb") as f:
            pickle.dump(lr, f)

    # --- Random Forest ---
    if "rf" in models_to_train:
        print(f"\nTraining random forest (n_trees={args.n_trees}, max_depth={args.max_depth})...")
        rf = RandomForestClassifier(
            n_estimators=args.n_trees,
            max_depth=args.max_depth,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=args.seed,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_val)
        rf_prob = rf.predict_proba(X_val)[:, 1]
        results["random_forest"] = evaluate_model("Random Forest", y_val, rf_pred, rf_prob)
        with open(output_dir / "rf_model.pkl", "wb") as f:
            pickle.dump(rf, f)

        # Feature importance
        _print_feature_importance(rf.feature_importances_, feature_names, "RF")

    # --- XGBoost ---
    if "xgb" in models_to_train:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            print("\nERROR: xgboost not installed. Run: pip install xgboost", file=sys.stderr)
            return 1

        print(f"\nTraining XGBoost (n_trees={args.n_trees}, max_depth=6)...")
        xgb = XGBClassifier(
            n_estimators=args.n_trees,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            random_state=args.seed,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
        )
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_val)
        xgb_prob = xgb.predict_proba(X_val)[:, 1]
        results["xgboost"] = evaluate_model("XGBoost", y_val, xgb_pred, xgb_prob)
        with open(output_dir / "xgb_model.pkl", "wb") as f:
            pickle.dump(xgb, f)

        _print_feature_importance(xgb.feature_importances_, feature_names, "XGB")

    # --- Per-category breakdown (best model) ---
    best_name = max(results, key=lambda k: results[k]["f1"]) if results else None
    if best_name:
        # Re-predict with best model for category breakdown
        best_model_path = output_dir / f"{'xgb' if best_name == 'xgboost' else best_name}_model.pkl"
        if best_model_path.exists():
            with open(best_model_path, "rb") as f:
                best_model = pickle.load(f)
            best_pred = best_model.predict(X_val)
            print(f"\n=== Per-category val breakdown ({best_name}) ===")
            for cat in ["false_negative", "false_positive", "near_boundary_benign", "baseline_correct"]:
                cat_mask = [i for i, s in enumerate(val_samples) if s.get("residual_category") == cat]
                if not cat_mask:
                    continue
                cat_true = y_val[cat_mask]
                cat_pred = best_pred[cat_mask]
                cat_correct = (cat_true == cat_pred).sum()
                print(f"  {cat:<25} n={len(cat_mask):>4}  "
                      f"correct={cat_correct:>4} ({cat_correct/len(cat_mask)*100:.1f}%)")

    # --- Save manifest ---
    manifest = {
        "created_utc": utc_now(),
        "residuals_dir": str(residuals_dir),
        "training_mix_size": len(mixed),
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "feature_config": feature_config,
        "total_features": X_train.shape[1],
        "n_trees": args.n_trees,
        "max_depth": args.max_depth,
        "seed": args.seed,
        "models_trained": list(results.keys()),
        "results": results,
    }
    (output_dir / "train_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    if tfidf is not None:
        with open(output_dir / "tfidf.pkl", "wb") as f:
            pickle.dump(tfidf, f)
    with open(output_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f)

    print(f"\nModels and manifest saved to {output_dir}")
    print("Done.")
    return 0


def _print_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    label: str,
) -> None:
    top_indices = np.argsort(importances)[::-1][:30]
    print(f"\nTop 30 {label} feature importances:")
    for rank, idx in enumerate(top_indices, 1):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        safe_name = name.encode("ascii", errors="replace").decode("ascii")
        print(f"  {rank:>2}. {safe_name:<40} {importances[idx]:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
