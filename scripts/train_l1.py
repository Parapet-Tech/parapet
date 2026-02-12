# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0

"""
Train the L1 lightweight classifier and codegen Rust phf_map weights.

Pipeline:
  1. Load all labeled YAML datasets from schema/eval/opensource_*.yaml
  2. Split 80/20 train/holdout (stratified, deterministic seed)
  3. Save holdout to schema/eval/l1_holdout.yaml (never trained on)
  4. Train CountVectorizer(analyzer='char_wb', ngram_range=(3,5), binary=True) + LinearSVC
  5. Export raw SVM coefficients (binary features = no normalization needed)
  6. Prune near-zero weights (|w| < threshold)
  7. Codegen parapet/src/layers/l1_weights.rs as phf_map!

Eval tiers:
  - In-distribution holdout: schema/eval/l1_holdout.yaml (same distribution, held out)
  - Out-of-distribution: schema/eval/l1_attacks.yaml + l1_benign.yaml (hand-written)
  - Cross-layer: existing l3_inbound datasets

Usage:
  python scripts/train_l1.py
  python scripts/train_l1.py --data-dir schema/eval --prune-threshold 0.001
  python scripts/train_l1.py --holdout-pct 0.3  # 30% holdout
"""

import argparse
import glob
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, train_test_split



# ProtectAI recipe: 12 datasets used for prompt injection detection training.
# https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2
PROTECTAI_FILES = [
    # attack
    "opensource_gandalf_attacks.yaml",            # Lakera/gandalf_ignore_instructions
    "opensource_chatgpt_jailbreak_attacks.yaml",   # rubend18/ChatGPT-Jailbreak-Prompts
    "opensource_imoxto_attacks.yaml",              # imoxto/prompt_injection_cleaned_dataset-v2
    "opensource_hackaprompt_attacks.yaml",          # hackaprompt/hackaprompt-dataset
    # benign
    "opensource_awesome_chatgpt_benign.yaml",      # fka/awesome-chatgpt-prompts
    "opensource_teven_benign.yaml",                # teven/prompted_examples
    "opensource_dahoas_benign.yaml",               # Dahoas/synthetic-hh-rlhf-prompts + hh_prompt_format
    "opensource_chatgpt_prompts_benign.yaml",      # MohamedRashad/ChatGPT-prompts
    "opensource_hf_instruction_benign.yaml",       # HuggingFaceH4/instruction-dataset
    "opensource_no_robots_benign.yaml",            # HuggingFaceH4/no_robots
    "opensource_ultrachat_benign.yaml",            # HuggingFaceH4/ultrachat_200k
]


def load_data(data_dir, max_per_file=0, seed=42):
    """Load the ProtectAI recipe datasets from data_dir.

    Each YAML file is a list of entries with at least:
      - content: str  (the text)
      - label: "malicious" | "benign"

    If max_per_file > 0, randomly sample at most that many entries per file
    so no single dataset dominates training.

    Returns (entries, file_stats) where entries have id, content, label, source.
    """
    files = [str(Path(data_dir) / f) for f in PROTECTAI_FILES]
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        print(f"ERROR: Missing dataset files:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(seed)
    entries = []
    file_stats = {}
    total_skipped = 0

    print(f"Loading {len(PROTECTAI_FILES)} ProtectAI recipe datasets from {data_dir}...", file=sys.stderr)
    if max_per_file > 0:
        print(f"  (capping at {max_per_file} samples per file)", file=sys.stderr)

    for filepath in files:
        fname = Path(filepath).name
        with open(filepath, "r", encoding="utf-8") as f:
            raw_entries = yaml.safe_load(f)

        if not raw_entries:
            print(f"  {fname}: empty, skipping", file=sys.stderr)
            continue

        # Filter to valid entries
        valid = []
        for entry in raw_entries:
            content = entry.get("content", "")
            label = entry.get("label", "")
            if not content or not content.strip():
                total_skipped += 1
                continue
            if label not in ("malicious", "benign"):
                total_skipped += 1
                continue
            valid.append({
                "id": entry.get("id", ""),
                "content": content,
                "label": label,
                "description": entry.get("description", ""),
                "source": fname,
            })

        # Cap per file if requested
        raw_count = len(valid)
        if max_per_file > 0 and len(valid) > max_per_file:
            valid = rng.sample(valid, max_per_file)

        file_attacks = sum(1 for e in valid if e["label"] == "malicious")
        file_benign = len(valid) - file_attacks
        file_stats[fname] = {"attacks": file_attacks, "benign": file_benign, "raw": raw_count}

        entries.extend(valid)

        used = len(valid)
        label_str = f"{file_attacks}atk" if file_attacks else f"{file_benign}ben"
        if file_attacks and file_benign:
            label_str = f"{file_attacks}atk+{file_benign}ben"
        cap_str = f" (capped from {raw_count})" if raw_count > used else ""
        print(f"  {fname}: {used} ({label_str}){cap_str}", file=sys.stderr)

    n_attacks = sum(1 for e in entries if e["label"] == "malicious")
    n_benign = len(entries) - n_attacks
    print(f"\nTotal: {len(entries)} samples "
          f"({n_attacks} attacks, {n_benign} benign, "
          f"{total_skipped} skipped)", file=sys.stderr)

    return entries, file_stats


def split_holdout(entries, holdout_pct, seed=42):
    """Split entries into train and holdout sets, stratified by label.

    Returns (train_entries, holdout_entries).
    """
    if holdout_pct <= 0:
        return entries, []

    labels = [e["label"] for e in entries]
    train_entries, holdout_entries = train_test_split(
        entries, test_size=holdout_pct, random_state=seed, stratify=labels
    )

    n_train_atk = sum(1 for e in train_entries if e["label"] == "malicious")
    n_train_ben = len(train_entries) - n_train_atk
    n_hold_atk = sum(1 for e in holdout_entries if e["label"] == "malicious")
    n_hold_ben = len(holdout_entries) - n_hold_atk

    print(f"\nSplit: {len(train_entries)} train ({n_train_atk} atk, {n_train_ben} ben) / "
          f"{len(holdout_entries)} holdout ({n_hold_atk} atk, {n_hold_ben} ben)",
          file=sys.stderr)

    return train_entries, holdout_entries


def save_holdout(holdout_entries, out_path, seed, holdout_pct):
    """Save holdout set as eval YAML."""
    if not holdout_entries:
        return

    # Convert to eval YAML format
    cases = []
    for entry in holdout_entries:
        cases.append({
            "id": entry["id"],
            "layer": "l1",
            "label": entry["label"],
            "description": f"holdout: {entry['description']}",
            "content": entry["content"],
        })

    n_atk = sum(1 for c in cases if c["label"] == "malicious")
    n_ben = len(cases) - n_atk
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            f"# L1 holdout eval set — DO NOT use for training\n"
            f"# Generated: {timestamp}\n"
            f"# Split: {holdout_pct*100:.0f}% holdout, seed={seed}\n"
            f"# {n_atk} attacks, {n_ben} benign ({len(cases)} total)\n"
            f"# Source datasets: schema/eval/opensource_*.yaml\n"
            f"# Auto-generated by scripts/train_l1.py\n\n"
        )
        yaml.dump(cases, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"  Saved {len(cases)} holdout cases to {out_path}", file=sys.stderr)


def train_model(texts, labels):
    """Train binary char n-gram + LinearSVC and return vectorizer + model.

    Uses CountVectorizer(binary=True) instead of TfidfVectorizer so the SVM
    coefficients directly correspond to binary-presence scoring at runtime.
    No TF-IDF normalization means the exported weights work as-is with
    score = bias + sum(coef[i] for each unique ngram i present in text).
    """
    from sklearn.feature_extraction.text import CountVectorizer

    print("\nTraining binary char_wb + LinearSVC...", file=sys.stderr)

    vectorizer = CountVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=5_000,
        binary=True,
    )
    X = vectorizer.fit_transform(texts)

    model = LinearSVC(C=0.1, max_iter=100_000, dual="auto", class_weight="balanced")

    # Cross-validate on training set
    scores = cross_val_score(model, X, labels, cv=5, scoring="f1")
    print(f"  5-fold CV F1: {scores.mean():.4f} (+/- {scores.std():.4f})", file=sys.stderr)

    # Fit on full training set
    model.fit(X, labels)
    print(f"  Feature count: {X.shape[1]}", file=sys.stderr)

    return vectorizer, model


def evaluate_holdout(vectorizer, model, holdout_entries):
    """Score the holdout set and report metrics."""
    if not holdout_entries:
        return

    texts = [e["content"] for e in holdout_entries]
    labels = [1 if e["label"] == "malicious" else 0 for e in holdout_entries]

    X = vectorizer.transform(texts)
    predictions = model.predict(X)

    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\nHoldout evaluation ({len(holdout_entries)} samples):", file=sys.stderr)
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}", file=sys.stderr)
    print(f"  Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}", file=sys.stderr)

    # Report false positives for inspection
    if fp > 0:
        print(f"\n  False positives ({fp}):", file=sys.stderr)
        scores = model.decision_function(X)
        for i, (entry, pred, label, score) in enumerate(zip(holdout_entries, predictions, labels, scores)):
            if pred == 1 and label == 0:
                preview = entry["content"][:80].replace("\n", " ")
                print(f"    [{entry['id']}] score={score:.3f} {preview!r}", file=sys.stderr)
                if fp > 10 and i > 10:
                    print(f"    ... and {fp - 10} more", file=sys.stderr)
                    break


def extract_weights(vectorizer, model, prune_threshold):
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

    All features are kept (including short n-grams like " a ", " wh")
    because char_wb boundary-padded features carry significant weight.
    """
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]  # shape: (n_features,)
    bias = float(model.intercept_[0])

    print(f"\n  Bias (intercept): {bias:.6f}", file=sys.stderr)
    print(f"  Coef range: [{coefs.min():.6f}, {coefs.max():.6f}]", file=sys.stderr)

    # Build (ngram, weight) pairs, only pruning near-zero
    weights = {}
    skipped_small = 0
    for name, coef in zip(feature_names, coefs):
        if abs(coef) < prune_threshold:
            skipped_small += 1
            continue
        weights[name] = float(coef)

    print(f"  Skipped {skipped_small} near-zero", file=sys.stderr)
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


def codegen_phf(bias, weights, out_path, provenance):
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

    # Build provenance comment
    prov_lines = [
        f"// Training data: {provenance['n_train']} samples "
        f"({provenance['n_train_attacks']} attacks, {provenance['n_train_benign']} benign)",
        f"// Holdout: {provenance['n_holdout']} samples "
        f"({provenance['holdout_pct']*100:.0f}%, seed={provenance['seed']})",
        f"// Trained: {provenance['timestamp']}",
        f"// Sources ({len(provenance['files'])} files):",
    ]
    for fname, stats in sorted(provenance["files"].items()):
        prov_lines.append(f"//   {fname}: {stats['attacks']}atk + {stats['benign']}ben")
    prov_lines.append(f"// Model: CountVectorizer(analyzer='char_wb', ngram_range=(3,5), binary=True) + LinearSVC(C=0.1)")
    prov_lines.append(f"// Features: {len(weights)} character n-grams after pruning")
    provenance_block = "\n".join(prov_lines)

    code = (
        f"// Copyright 2026 The Parapet Project\n"
        f"// SPDX-License-Identifier: Apache-2.0\n"
        f"\n"
        f"// AUTO-GENERATED by scripts/train_l1.py — do not edit manually.\n"
        f"//\n"
        f"{provenance_block}\n"
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
    parser.add_argument("--holdout-pct", type=float, default=0.2,
                        help="Fraction of data to hold out for eval (default: 0.2)")
    parser.add_argument("--holdout-out", type=str, default="schema/eval/l1_holdout.yaml",
                        help="Output path for holdout eval set")
    parser.add_argument("--prune-threshold", type=float, default=0.05,
                        help="Prune weights with |w| below this (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--out", type=str,
                        default="parapet/src/layers/l1_weights.rs",
                        help="Output path for generated Rust file")
    args = parser.parse_args()

    # 1. Load
    entries, file_stats = load_data(args.data_dir, max_per_file=args.max_per_file, seed=args.seed)

    # 2. Split
    train_entries, holdout_entries = split_holdout(entries, args.holdout_pct, seed=args.seed)

    # 3. Save holdout
    save_holdout(holdout_entries, args.holdout_out, args.seed, args.holdout_pct)

    # 4. Train (on train set only)
    train_texts = [e["content"] for e in train_entries]
    train_labels = [1 if e["label"] == "malicious" else 0 for e in train_entries]
    n_train_attacks = sum(train_labels)
    n_train_benign = len(train_labels) - n_train_attacks

    vectorizer, model = train_model(train_texts, train_labels)

    # 5. Evaluate on holdout
    evaluate_holdout(vectorizer, model, holdout_entries)

    # 6. Extract and prune weights
    bias, weights = extract_weights(vectorizer, model, args.prune_threshold)

    # 7. Codegen with provenance
    provenance = {
        "n_train": len(train_entries),
        "n_train_attacks": n_train_attacks,
        "n_train_benign": n_train_benign,
        "n_holdout": len(holdout_entries),
        "holdout_pct": args.holdout_pct,
        "seed": args.seed,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "files": file_stats,
    }
    codegen_phf(bias, weights, args.out, provenance)

    print("\nDone. Next steps:", file=sys.stderr)
    print("  1. cargo build  (verify phf_map compiles)", file=sys.stderr)
    print("  2. cargo test   (verify unit tests pass)", file=sys.stderr)
    print("  3. Run eval against holdout + hand-written datasets", file=sys.stderr)


if __name__ == "__main__":
    main()
