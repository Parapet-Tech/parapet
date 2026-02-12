# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0

"""
Train the L1 lightweight classifier and codegen Rust phf_map weights.

Pipeline:
  1. Load all labeled YAML datasets from schema/eval/opensource_*.yaml
  2. Train CountVectorizer(analyzer='char_wb', ngram_range=(3,5), binary=True) + LinearSVC
  3. Export raw SVM coefficients (binary features = no normalization needed)
  4. Prune near-zero weights (|w| < threshold)
  5. Codegen parapet/src/layers/l1_weights.rs as phf_map!

Datasets (all from schema/eval/):
  Attacks: deepset, jailbreak_cls, hackaprompt, wildjailbreak, mosscap, gandalf, giskard, jbb
  Benign:  deepset, jailbreak_cls, wildjailbreak, jbb, hc3, prompts_chat, wildchat

Usage:
  python scripts/train_l1.py
  python scripts/train_l1.py --data-dir schema/eval --prune-threshold 0.001
"""

import argparse
import glob
import random
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score


def load_data(data_dir, max_per_file=0, seed=42):
    """Load all labeled YAML datasets from data_dir/opensource_*.yaml.

    Each YAML file is a list of entries with at least:
      - content: str  (the text)
      - label: "malicious" | "benign"

    If max_per_file > 0, randomly sample at most that many entries per file
    so no single dataset dominates training.

    Returns (texts, labels) where labels are 1=attack, 0=benign.
    """
    pattern = str(Path(data_dir) / "opensource_*.yaml")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"ERROR: No opensource_*.yaml files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(seed)
    texts = []
    labels = []
    stats = {"attack_files": 0, "benign_files": 0, "attacks": 0, "benign": 0, "skipped": 0}

    print(f"Loading datasets from {data_dir}/opensource_*.yaml...", file=sys.stderr)
    if max_per_file > 0:
        print(f"  (capping at {max_per_file} samples per file)", file=sys.stderr)

    for filepath in files:
        fname = Path(filepath).name
        with open(filepath, "r", encoding="utf-8") as f:
            entries = yaml.safe_load(f)

        if not entries:
            print(f"  {fname}: empty, skipping", file=sys.stderr)
            continue

        # Filter to valid entries first
        valid = []
        for entry in entries:
            content = entry.get("content", "")
            label = entry.get("label", "")
            if not content or not content.strip():
                stats["skipped"] += 1
                continue
            if label not in ("malicious", "benign"):
                stats["skipped"] += 1
                continue
            valid.append(entry)

        # Cap per file if requested
        raw_count = len(valid)
        if max_per_file > 0 and len(valid) > max_per_file:
            valid = rng.sample(valid, max_per_file)

        file_attacks = 0
        file_benign = 0
        for entry in valid:
            content = entry["content"]
            label = entry["label"]
            if label == "malicious":
                texts.append(content)
                labels.append(1)
                file_attacks += 1
            else:
                texts.append(content)
                labels.append(0)
                file_benign += 1

        if file_attacks > 0:
            stats["attack_files"] += 1
            stats["attacks"] += file_attacks
        if file_benign > 0:
            stats["benign_files"] += 1
            stats["benign"] += file_benign

        used = file_attacks + file_benign
        label_str = f"{file_attacks}atk" if file_attacks else f"{file_benign}ben"
        cap_str = f" (capped from {raw_count})" if raw_count > used else ""
        print(f"  {fname}: {used} ({label_str}){cap_str}", file=sys.stderr)

    print(f"\nTotal: {len(texts)} samples "
          f"({stats['attacks']} attacks from {stats['attack_files']} files, "
          f"{stats['benign']} benign from {stats['benign_files']} files, "
          f"{stats['skipped']} skipped)", file=sys.stderr)

    return texts, labels


def train(texts, labels):
    """Train binary char n-gram + LinearSVC and return vectorizer + model.

    Uses CountVectorizer(binary=True) instead of TfidfVectorizer so the SVM
    coefficients directly correspond to binary-presence scoring at runtime.
    No TF-IDF normalization means the exported weights work as-is with
    score = bias + sum(coef[i] for each unique ngram i present in text).
    """
    from sklearn.feature_extraction.text import CountVectorizer

    print("Training binary char_wb + LinearSVC...", file=sys.stderr)

    vectorizer = CountVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=20_000,
        binary=True,
    )
    X = vectorizer.fit_transform(texts)

    model = LinearSVC(C=0.1, max_iter=100_000, dual="auto", class_weight="balanced")

    # Cross-validate first
    scores = cross_val_score(model, X, labels, cv=5, scoring="f1")
    print(f"  5-fold CV F1: {scores.mean():.4f} (+/- {scores.std():.4f})", file=sys.stderr)

    # Fit on full data
    model.fit(X, labels)
    print(f"  Feature count: {X.shape[1]}", file=sys.stderr)

    return vectorizer, model


def extract_weights(vectorizer, model, prune_threshold, min_ngram_chars=3):
    """
    Export SVM coefficients as runtime weights and prune.

    With binary features, the decision function is exactly:
      f(x) = coef @ binary_features(x) + intercept
           = bias + sum(coef[i] for each unique ngram i present in text)

    This matches our Rust runtime scoring directly — no normalization needed.

    The char_wb analyzer pads words with spaces at boundaries, producing
    n-grams like " ign", "igno", "gnor", "ore ", " ignore ", etc.
    We keep the space padding in the exported n-grams so runtime matching
    uses the same convention: prepend/append spaces at word boundaries.
    """
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]  # shape: (n_features,)
    bias = float(model.intercept_[0])

    print(f"  Bias (intercept): {bias:.6f}", file=sys.stderr)
    print(f"  Coef range: [{coefs.min():.6f}, {coefs.max():.6f}]", file=sys.stderr)

    # Build (ngram, weight) pairs, pruning near-zero and too-short
    weights = {}
    skipped_short = 0
    skipped_small = 0
    for name, coef in zip(feature_names, coefs):
        if abs(coef) < prune_threshold:
            skipped_small += 1
            continue
        # char_wb n-grams include space padding; keep it for matching
        # but skip if the stripped content is too short (single chars)
        stripped = name.strip()
        if len(stripped) < min_ngram_chars:
            skipped_short += 1
            continue
        # Keep original name with space padding for char_wb matching
        weights[name] = float(coef)

    print(f"  Skipped {skipped_small} near-zero, {skipped_short} too-short", file=sys.stderr)
    print(f"  Kept {len(weights)} features", file=sys.stderr)

    # Report top positive and negative
    sorted_w = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    print("\n  Top 20 injection indicators:", file=sys.stderr)
    for name, w in sorted_w[:20]:
        print(f"    {w:+.4f}  {name!r}", file=sys.stderr)
    print("\n  Top 20 benign indicators:", file=sys.stderr)
    for name, w in sorted_w[-20:]:
        print(f"    {w:+.4f}  {name!r}", file=sys.stderr)

    return bias, weights


def codegen_phf(bias, weights, out_path, n_attacks=0, n_benign=0):
    """Generate l1_weights.rs with a phf_map! macro."""
    print(f"\nWriting {out_path}...", file=sys.stderr)

    # Sort by weight descending for readability
    sorted_weights = sorted(weights.items(), key=lambda kv: -kv[1])

    # Escape strings for Rust
    def rust_str(s):
        return s.replace("\\", "\\\\").replace('"', '\\"')

    lines = []
    for ngram, w in sorted_weights:
        lines.append(f'    "{rust_str(ngram)}" => {w:.8f}_f64,')

    entries = "\n".join(lines)
    total = n_attacks + n_benign

    code = (
        f"// Copyright 2026 The Parapet Project\n"
        f"// SPDX-License-Identifier: Apache-2.0\n"
        f"\n"
        f"// AUTO-GENERATED by scripts/train_l1.py — do not edit manually.\n"
        f"//\n"
        f"// Training data: {total} samples ({n_attacks} attacks, {n_benign} benign)\n"
        f"// Sources: schema/eval/opensource_*.yaml\n"
        f"// Model: CountVectorizer(analyzer='char_wb', ngram_range=(3,5), binary=True) + LinearSVC\n"
        f"// Features: {len(weights)} character n-grams after pruning\n"
        f"\n"
        f"use phf::phf_map;\n"
        f"\n"
        f"/// SVM intercept (bias term).\n"
        f"pub const BIAS: f64 = {bias:.8f}_f64;\n"
        f"\n"
        f"/// Character n-gram weights from trained SVM.\n"
        f"/// Positive = injection indicator, negative = benign indicator.\n"
        f"pub static WEIGHTS: phf::Map<&'static str, f64> = phf_map! {{\n"
        f"{entries}\n"
        f"}};\n"
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(code, encoding="utf-8")
    print(f"  Wrote {len(weights)} entries + bias to {out_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Train L1 classifier and codegen Rust weights")
    parser.add_argument("--data-dir", type=str, default="schema/eval",
                        help="Directory containing opensource_*.yaml datasets (default: schema/eval)")
    parser.add_argument("--max-per-file", type=int, default=1000,
                        help="Max samples per dataset file to prevent dominance (default: 1000, 0=unlimited)")
    parser.add_argument("--prune-threshold", type=float, default=0.05,
                        help="Prune weights with |w| below this (default: 0.05)")
    parser.add_argument("--out", type=str,
                        default="parapet/src/layers/l1_weights.rs",
                        help="Output path for generated Rust file")
    args = parser.parse_args()

    texts, labels = load_data(args.data_dir, max_per_file=args.max_per_file)
    n_attacks = sum(labels)
    n_benign = len(labels) - n_attacks

    vectorizer, model = train(texts, labels)
    bias, weights = extract_weights(vectorizer, model, args.prune_threshold)
    codegen_phf(bias, weights, args.out, n_attacks=n_attacks, n_benign=n_benign)

    print("\nDone. Next steps:", file=sys.stderr)
    print("  1. cargo build  (verify phf_map compiles)", file=sys.stderr)
    print("  2. cargo test   (verify unit tests pass)", file=sys.stderr)
    print("  3. cargo run --bin parapet-eval -- \\", file=sys.stderr)
    print("       --config schema/eval/eval_config_l1_only.yaml \\", file=sys.stderr)
    print("       --dataset schema/eval/ --layer l1", file=sys.stderr)


if __name__ == "__main__":
    main()
