"""Train the L2 semantic classifier head (LR over MiniLM embeddings + harness signals).

Uses L1 out-of-fold residuals as training data. Generates:
  - LR weights + bias
  - Scaler mean + scale for harness signals
  - Codegen-ready l2_weights.rs

Usage:
    cd parapet
    python scripts/train_l2_semantic.py \
      --residuals-dir parapet-runner/runs/l2b_residuals \
      --output-dir parapet-runner/runs/l2_semantic
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


MIX_RATIOS = {
    "false_negative": 0.35,
    "false_positive": 0.25,
    "near_boundary_benign": 0.20,
    "baseline_correct": 0.20,
}


def build_training_mix(residuals, seed):
    import random
    rng = random.Random(seed)
    by_cat = {}
    for r in residuals:
        cat = r.get("residual_category", "unknown")
        by_cat.setdefault(cat, []).append(r)

    fn_count = len(by_cat.get("false_negative", []))
    if fn_count == 0:
        raise ValueError("No false negatives")
    total_target = int(fn_count / MIX_RATIOS["false_negative"])

    mixed = []
    for category, ratio in MIX_RATIOS.items():
        pool = by_cat.get(category, [])
        target = min(int(total_target * ratio), len(pool))
        sampled = rng.sample(pool, target) if target < len(pool) else list(pool)
        mixed.extend(sampled)
        print(f"  {category:<25} target={int(total_target*ratio):>5}  "
              f"available={len(pool):>5}  sampled={len(sampled):>5}")
    rng.shuffle(mixed)
    return mixed


HARNESS_CLAMP = 10.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train L2 semantic classifier head")
    parser.add_argument("--residuals-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    residuals_dir = Path(args.residuals_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load residuals
    candidates = load_jsonl(residuals_dir / "l2b_training_candidates.jsonl")
    print(f"Loaded {len(candidates):,} residual candidates")

    # Build training mix
    print("\nBuilding training mix...")
    mixed = build_training_mix(candidates, seed=args.seed)
    print(f"  Total: {len(mixed):,}")

    labels = np.array([1 if s["true_label"] == "malicious" else 0 for s in mixed])
    print(f"  Malicious: {labels.sum():,}  Benign: {(1-labels).sum():,}")

    # Train/val split
    train_idx, val_idx = train_test_split(
        range(len(mixed)), test_size=0.2, random_state=args.seed, stratify=labels,
    )
    train_samples = [mixed[i] for i in train_idx]
    val_samples = [mixed[i] for i in val_idx]
    y_train = labels[train_idx]
    y_val = labels[val_idx]

    print(f"\n  Train: {len(train_samples)} (mal={y_train.sum()}, ben={len(y_train)-y_train.sum()})")
    print(f"  Val:   {len(val_samples)} (mal={y_val.sum()}, ben={len(y_val)-y_val.sum()})")

    # Generate embeddings
    print("\nGenerating MiniLM embeddings...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    train_texts = [s["content"] for s in train_samples]
    val_texts = [s["content"] for s in val_samples]

    train_emb = model.encode(train_texts, normalize_embeddings=True, show_progress_bar=True)
    val_emb = model.encode(val_texts, normalize_embeddings=True, show_progress_bar=True)
    print(f"  Train embeddings: {train_emb.shape}")
    print(f"  Val embeddings:   {val_emb.shape}")

    # Extract harness signals
    def harness_features(samples):
        return np.array([
            [
                s.get("raw_score") or 0.0,
                s.get("raw_unquoted_score") or 0.0,
                s.get("raw_squash_score") or 0.0,
                s.get("raw_score_delta") or 0.0,
                1.0 if s.get("quote_detected") else 0.0,
            ]
            for s in samples
        ], dtype=np.float32)

    train_harness = harness_features(train_samples)
    val_harness = harness_features(val_samples)

    # Scale harness signals
    scaler = StandardScaler()
    train_harness_scaled = scaler.fit_transform(train_harness)
    val_harness_scaled = scaler.transform(val_harness)

    # Clamp
    train_harness_scaled = np.clip(train_harness_scaled, -HARNESS_CLAMP, HARNESS_CLAMP)
    val_harness_scaled = np.clip(val_harness_scaled, -HARNESS_CLAMP, HARNESS_CLAMP)

    # Concatenate
    X_train = np.hstack([train_emb, train_harness_scaled])
    X_val = np.hstack([val_emb, val_harness_scaled])
    print(f"  Feature vector: {X_train.shape[1]} dims (384 emb + 5 harness)")

    # Train LR
    print("\nTraining logistic regression...")
    lr = LogisticRegression(C=1.0, max_iter=5000, random_state=args.seed, solver="lbfgs")
    lr.fit(X_train, y_train)

    # Evaluate
    y_pred = lr.predict(X_val)
    y_prob = lr.predict_proba(X_val)[:, 1]
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()

    f1 = f1_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    roc = roc_auc_score(y_val, y_prob)
    ap = average_precision_score(y_val, y_prob)

    print(f"\n=== L2 Semantic Classifier (MiniLM + LR) ===")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  FPR:       {fpr:.4f}")
    print(f"  ROC-AUC:   {roc:.4f}")
    print(f"  Avg Prec:  {ap:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    # Per-category breakdown
    print(f"\n=== Per-category val breakdown ===")
    for cat in ["false_negative", "false_positive", "near_boundary_benign", "baseline_correct"]:
        cat_mask = [i for i, s in enumerate(val_samples) if s.get("residual_category") == cat]
        if not cat_mask:
            continue
        cat_true = y_val[cat_mask]
        cat_pred = y_pred[cat_mask]
        correct = (cat_true == cat_pred).sum()
        print(f"  {cat:<25} n={len(cat_mask):>4}  correct={correct:>4} ({correct/len(cat_mask)*100:.1f}%)")

    # Export weights
    weights = lr.coef_[0].astype(np.float32)
    bias = float(lr.intercept_[0])
    harness_mean = scaler.mean_.astype(np.float32)
    harness_scale = scaler.scale_.astype(np.float32)

    weights_json = {
        "weights": weights.tolist(),
        "bias": bias,
        "harness_mean": harness_mean.tolist(),
        "harness_scale": harness_scale.tolist(),
    }
    (output_dir / "l2_weights.json").write_text(
        json.dumps(weights_json), encoding="utf-8"
    )

    # Codegen Rust constants
    def fmt_array(arr, name, typ, n):
        vals = ", ".join(f"{v:.8e}" for v in arr)
        return f"pub const {name}: [{typ}; {n}] = [{vals}];"

    rust_code = f"""\
// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// L2 semantic classifier weights (MiniLM + LR).
//
// Generated by scripts/train_l2_semantic.py on {utc_now()}.
// Do not edit manually.
//
// Training: {len(train_samples)} samples, {X_train.shape[1]} features
// Val: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}, FPR={fpr:.4f}, ROC-AUC={roc:.4f}

/// Embedding dimension from all-MiniLM-L6-v2.
pub const EMBEDDING_DIM: usize = {train_emb.shape[1]};

/// Number of L1 harness signal features.
pub const HARNESS_DIM: usize = {train_harness.shape[1]};

/// Total feature dimension: embedding + harness signals.
pub const TOTAL_DIM: usize = EMBEDDING_DIM + HARNESS_DIM;

/// LR weight vector. Length = TOTAL_DIM = {X_train.shape[1]}.
{fmt_array(weights, "L2_WEIGHTS", "f32", len(weights))}

/// LR bias term.
pub const L2_BIAS: f32 = {bias:.8e};

/// StandardScaler mean for the 5 harness signals.
/// Order: raw_score, raw_unquoted_score, raw_squash_score, raw_score_delta, quote_detected.
{fmt_array(harness_mean, "L2_HARNESS_MEAN", "f32", len(harness_mean))}

/// StandardScaler scale (std dev) for the 5 harness signals.
{fmt_array(harness_scale, "L2_HARNESS_SCALE", "f32", len(harness_scale))}
"""
    rust_path = output_dir / "l2_weights.rs"
    rust_path.write_text(rust_code, encoding="utf-8")
    print(f"\n  Weights JSON: {output_dir / 'l2_weights.json'}")
    print(f"  Rust codegen: {rust_path}")

    # Save manifest
    manifest = {
        "created_utc": utc_now(),
        "training_mix_size": len(mixed),
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "embedding_dim": int(train_emb.shape[1]),
        "total_features": int(X_train.shape[1]),
        "results": {
            "f1": f1, "precision": precision, "recall": recall,
            "fpr": fpr, "roc_auc": roc, "avg_precision": ap,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        },
    }
    (output_dir / "train_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    with open(output_dir / "lr_model.pkl", "wb") as f:
        pickle.dump(lr, f)
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
