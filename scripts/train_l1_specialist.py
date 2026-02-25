# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0

"""
Train one L1 specialist classifier and codegen Rust phf_map weights.

Same recipe as train_l1.py (CountVectorizer + LinearSVC + phf codegen)
but trained on a curated slice for one specialist category.

Two input modes:
  1. Curated files (Phase 1):
       --attack-files f1.yaml f2.yaml --benign-files f3.yaml f4.yaml
  2. Category file (Phase 2, from labeling script):
       --category-file schema/eval/specialists/roleplay_jailbreak_attacks.yaml
       --benign-files f3.yaml f4.yaml

Pipeline:
  1. Load attack + benign data (dedup by content hash)
  2. Split train/holdout (stratified)
  3. Train CountVectorizer(char_wb) + LinearSVC
  4. Evaluate holdout, sweep thresholds
  5. Codegen l1_weights_{specialist}.rs

Usage:
  python scripts/train_l1_specialist.py \\
    --specialist roleplay_jailbreak \\
    --attack-files schema/eval/opensource_chatgpt_jailbreak_attacks.yaml \\
                   schema/eval/opensource_jailbreak_cls_attacks.yaml \\
                   schema/eval/opensource_hackaprompt_attacks.yaml \\
    --benign-files schema/eval/opensource_no_robots_benign.yaml \\
                   schema/eval/opensource_chatgpt_prompts_benign.yaml \\
                   schema/eval/staging/opensource_notinject_benign.yaml \\
                   schema/eval/staging/opensource_wildguardmix_benign.yaml \\
    --out parapet/src/layers/l1_weights_roleplay_jailbreak.rs
"""

import argparse
import hashlib
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import LinearSVC


# ---------------------------------------------------------------------------
# Data loading + dedup
# ---------------------------------------------------------------------------

def _content_hash(text: str) -> str:
    """Stable content hash for dedup."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def load_yaml_entries(filepaths: list[str], label_filter: str | None = None) -> list[dict]:
    """Load entries from YAML files, optionally filtering by label.

    Returns list of dicts with id, content, label, source.
    Skips entries with empty content or missing label.
    """
    entries = []
    for filepath in filepaths:
        fname = Path(filepath).name
        if not Path(filepath).exists():
            print(f"  ERROR: {filepath} not found", file=sys.stderr)
            sys.exit(1)
        with open(filepath, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not raw:
            print(f"  {fname}: empty, skipping", file=sys.stderr)
            continue

        count = 0
        for entry in raw:
            content = entry.get("content", "")
            label = entry.get("label", "")
            if not content or not content.strip():
                continue
            # Normalize TheWall labels (positive/negative) to internal convention
            if label == "positive":
                label = "malicious"
            elif label == "negative":
                label = "benign"
            if label not in ("malicious", "benign"):
                continue
            if label_filter and label != label_filter:
                continue
            entries.append({
                "id": entry.get("id", ""),
                "content": content,
                "label": label,
                "description": entry.get("description", ""),
                "source": fname,
            })
            count += 1
        print(f"  {fname}: {count} entries", file=sys.stderr)

    return entries


def dedup_entries(entries: list[dict]) -> list[dict]:
    """Deduplicate entries by content hash. First occurrence wins."""
    seen = set()
    deduped = []
    for entry in entries:
        h = _content_hash(entry["content"])
        if h not in seen:
            seen.add(h)
            deduped.append(entry)
    n_dupes = len(entries) - len(deduped)
    if n_dupes > 0:
        print(f"  Deduped: {n_dupes} duplicates removed, {len(deduped)} unique", file=sys.stderr)
    return deduped


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(texts, labels, ngram_range=(3, 5), max_features=5000, analyzer="char_wb"):
    """Train binary char n-gram + LinearSVC. Returns vectorizer + model."""
    print(f"\nTraining: CountVectorizer(analyzer={analyzer!r}, ngram_range={ngram_range}, "
          f"max_features={max_features}) + LinearSVC(C=0.1)...", file=sys.stderr)

    vectorizer = CountVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
        binary=True,
    )
    X = vectorizer.fit_transform(texts)

    model = LinearSVC(C=0.1, max_iter=100_000, dual="auto", class_weight="balanced")

    scores = cross_val_score(model, X, labels, cv=5, scoring="f1")
    print(f"  5-fold CV F1: {scores.mean():.4f} (+/- {scores.std():.4f})", file=sys.stderr)

    model.fit(X, labels)
    print(f"  Feature count: {X.shape[1]}", file=sys.stderr)

    return vectorizer, model


# ---------------------------------------------------------------------------
# Evaluation + threshold sweep
# ---------------------------------------------------------------------------

def evaluate_holdout(vectorizer, model, holdout_entries):
    """Score holdout and report metrics at default threshold (0.0)."""
    if not holdout_entries:
        return

    texts = [e["content"] for e in holdout_entries]
    labels = [1 if e["label"] == "malicious" else 0 for e in holdout_entries]

    X = vectorizer.transform(texts)
    predictions = model.predict(X)
    raw_scores = model.decision_function(X)

    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\nHoldout evaluation ({len(holdout_entries)} samples, threshold=0.0):", file=sys.stderr)
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}", file=sys.stderr)
    print(f"  Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}", file=sys.stderr)

    # Report FPs
    if fp > 0:
        print(f"\n  False positives ({fp}):", file=sys.stderr)
        fp_shown = 0
        for entry, pred, label, score in zip(holdout_entries, predictions, labels, raw_scores):
            if pred == 1 and label == 0:
                preview = entry["content"][:80].replace("\n", " ")
                print(f"    [{entry['id']}] score={score:.3f} {preview!r}", file=sys.stderr)
                fp_shown += 1
                if fp_shown >= 15:
                    remaining = fp - fp_shown
                    if remaining > 0:
                        print(f"    ... and {remaining} more", file=sys.stderr)
                    break

    # Report FNs
    if fn > 0:
        print(f"\n  False negatives ({fn}):", file=sys.stderr)
        fn_shown = 0
        for entry, pred, label, score in zip(holdout_entries, predictions, labels, raw_scores):
            if pred == 0 and label == 1:
                preview = entry["content"][:80].replace("\n", " ")
                print(f"    [{entry['id']}] score={score:.3f} {preview!r}", file=sys.stderr)
                fn_shown += 1
                if fn_shown >= 15:
                    remaining = fn - fn_shown
                    if remaining > 0:
                        print(f"    ... and {remaining} more", file=sys.stderr)
                    break

    return raw_scores, labels


def save_errors(holdout_entries, raw_scores, labels, specialist, out_dir):
    """Save FPs and FNs to {specialist}_errors.yaml for iteration tracking.

    Each entry includes the original fields + raw_score + error_type.
    Sorted by |score| ascending (hardest cases first — closest to boundary).
    """
    predictions = (np.array(raw_scores) >= 0.0).astype(int)
    labels_arr = np.array(labels)

    errors = []
    for entry, pred, label, score in zip(holdout_entries, predictions, labels_arr, raw_scores):
        if pred == 1 and label == 0:
            error_type = "false_positive"
        elif pred == 0 and label == 1:
            error_type = "false_negative"
        else:
            continue
        errors.append({
            "id": entry["id"],
            "error_type": error_type,
            "raw_score": round(float(score), 4),
            "label": entry["label"],
            "source": entry.get("source", ""),
            "description": entry["description"],
            "content": entry["content"],
        })

    # Sort by distance from decision boundary (hardest cases first)
    errors.sort(key=lambda e: abs(e["raw_score"]))

    fp_count = sum(1 for e in errors if e["error_type"] == "false_positive")
    fn_count = sum(1 for e in errors if e["error_type"] == "false_negative")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    out_path = Path(out_dir) / f"{specialist}_errors.yaml"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            f"# L1 specialist error log: {specialist}\n"
            f"# Generated: {timestamp}\n"
            f"# {fp_count} false positives, {fn_count} false negatives\n"
            f"# Sorted by |raw_score| ascending (hardest cases first)\n"
            f"#\n"
            f"# Usage:\n"
            f"#   FPs → candidates for hard negatives (add to --benign-files)\n"
            f"#   FNs → candidates for more attack data (add to --attack-files)\n"
            f"# Auto-generated by scripts/train_l1_specialist.py\n\n"
        )
        yaml.dump(errors, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"  Saved {len(errors)} errors ({fp_count} FP, {fn_count} FN) to {out_path}",
          file=sys.stderr)
    return out_path


def sweep_thresholds(raw_scores, labels, holdout_entries):
    """Sweep thresholds on holdout and report precision/recall/F1 at each."""
    labels_arr = np.array(labels)
    scores_arr = np.array(raw_scores)

    thresholds = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

    print(f"\nThreshold sweep on holdout:", file=sys.stderr)
    print(f"  {'Threshold':>10s}  {'TP':>5s}  {'FP':>5s}  {'FN':>5s}  {'TN':>5s}  "
          f"{'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}", file=sys.stderr)

    for t in thresholds:
        preds = (scores_arr >= t).astype(int)
        tp = int(((preds == 1) & (labels_arr == 1)).sum())
        fp = int(((preds == 1) & (labels_arr == 0)).sum())
        fn = int(((preds == 0) & (labels_arr == 1)).sum())
        tn = int(((preds == 0) & (labels_arr == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        marker = " <-- default" if t == 0.0 else ""
        print(f"  {t:>10.1f}  {tp:5d}  {fp:5d}  {fn:5d}  {tn:5d}  "
              f"{prec:6.3f}  {rec:6.3f}  {f1:6.3f}{marker}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Weight extraction + codegen (same as train_l1.py)
# ---------------------------------------------------------------------------

def extract_weights(vectorizer, model, prune_threshold):
    """Export SVM coefficients as (ngram, weight) pairs, prune near-zero."""
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    bias = float(model.intercept_[0])

    print(f"\n  Bias (intercept): {bias:.6f}", file=sys.stderr)
    print(f"  Coef range: [{coefs.min():.6f}, {coefs.max():.6f}]", file=sys.stderr)

    weights = {}
    skipped = 0
    for name, coef in zip(feature_names, coefs):
        if abs(coef) < prune_threshold:
            skipped += 1
            continue
        weights[name] = float(coef)

    print(f"  Skipped {skipped} near-zero", file=sys.stderr)
    print(f"  Kept {len(weights)} features", file=sys.stderr)

    sorted_w = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    print("\n  Top 15 attack indicators:", file=sys.stderr)
    for name, w in sorted_w[:15]:
        print(f"    {w:+.4f}  {name!r}", file=sys.stderr)
    print("\n  Top 15 benign indicators:", file=sys.stderr)
    for name, w in sorted_w[-15:]:
        print(f"    {w:+.4f}  {name!r}", file=sys.stderr)

    return bias, weights


def codegen_phf(bias, weights, out_path, specialist, provenance):
    """Generate l1_weights_{specialist}.rs with a phf_map! macro."""
    print(f"\nWriting {out_path}...", file=sys.stderr)

    sorted_weights = sorted(weights.items(), key=lambda kv: -kv[1])

    def rust_str(s):
        return s.replace("\\", "\\\\").replace('"', '\\"')

    lines = []
    for ngram, w in sorted_weights:
        lines.append(f'    "{rust_str(ngram)}" => {w:.8f}_f64,')
    entries = "\n".join(lines)

    prov_lines = [
        f"// Specialist: {specialist}",
        f"// Training data: {provenance['n_train']} samples "
        f"({provenance['n_train_attacks']} attacks, {provenance['n_train_benign']} benign)",
        f"// Holdout: {provenance['n_holdout']} samples "
        f"({provenance['holdout_pct']*100:.0f}%, seed={provenance['seed']})",
        f"// Trained: {provenance['timestamp']}",
        f"// Sources ({len(provenance['files'])} files):",
    ]
    for fname, stats in sorted(provenance["files"].items()):
        prov_lines.append(f"//   {fname}: {stats['attacks']}atk + {stats['benign']}ben")
    vec_desc = provenance["vectorizer"]
    prov_lines.append(f"// Model: CountVectorizer({vec_desc}) + LinearSVC(C=0.1)")
    prov_lines.append(f"// Features: {len(weights)} character n-grams after pruning")
    provenance_block = "\n".join(prov_lines)

    code = (
        f"// Copyright 2026 The Parapet Project\n"
        f"// SPDX-License-Identifier: Apache-2.0\n"
        f"\n"
        f"// AUTO-GENERATED by scripts/train_l1_specialist.py — do not edit manually.\n"
        f"//\n"
        f"{provenance_block}\n"
        f"\n"
        f"use phf::phf_map;\n"
        f"\n"
        f"/// SVM intercept (bias term) for {specialist} specialist.\n"
        f"pub const BIAS: f64 = {bias:.8f}_f64;\n"
        f"\n"
        f"/// Character n-gram weights for {specialist} specialist.\n"
        f"/// Positive = attack indicator, negative = benign indicator.\n"
        f"pub static WEIGHTS: phf::Map<&'static str, f64> = phf_map! {{\n"
        f"{entries}\n"
        f"}};\n"
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(code, encoding="utf-8")
    print(f"  Wrote {len(weights)} entries + bias to {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Holdout save
# ---------------------------------------------------------------------------

def save_holdout(holdout_entries, out_path, specialist, seed, holdout_pct):
    """Save specialist holdout set as eval YAML."""
    if not holdout_entries:
        return

    cases = []
    for entry in holdout_entries:
        cases.append({
            "id": entry["id"],
            "layer": "l1",
            "label": entry["label"],
            "description": f"{specialist} holdout: {entry['description']}",
            "content": entry["content"],
        })

    n_atk = sum(1 for c in cases if c["label"] == "malicious")
    n_ben = len(cases) - n_atk
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            f"# L1 specialist holdout: {specialist} — DO NOT use for training\n"
            f"# Generated: {timestamp}\n"
            f"# Split: {holdout_pct*100:.0f}% holdout, seed={seed}\n"
            f"# {n_atk} attacks, {n_ben} benign ({len(cases)} total)\n"
            f"# Auto-generated by scripts/train_l1_specialist.py\n\n"
        )
        yaml.dump(cases, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"  Saved {len(cases)} holdout cases to {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train one L1 specialist classifier and codegen Rust weights"
    )
    parser.add_argument("--specialist", type=str, required=True,
                        help="Specialist name (e.g. roleplay_jailbreak)")

    # Input mode 1: curated files
    parser.add_argument("--attack-files", type=str, nargs="+", default=[],
                        help="YAML files containing malicious samples (curated mode)")
    parser.add_argument("--benign-files", type=str, nargs="+", default=[],
                        help="YAML files containing benign samples (hard negatives + general benign)")

    # Input mode 2: category file (from labeling script)
    parser.add_argument("--category-file", type=str, default=None,
                        help="Pre-labeled category YAML from label_specialist_categories.py (Phase 2)")

    # Vectorizer overrides
    parser.add_argument("--ngram-min", type=int, default=3,
                        help="Min n-gram length (default: 3)")
    parser.add_argument("--ngram-max", type=int, default=5,
                        help="Max n-gram length (default: 5)")
    parser.add_argument("--max-features", type=int, default=5000,
                        help="Max features for CountVectorizer (default: 5000)")
    parser.add_argument("--analyzer", type=str, default="char_wb",
                        choices=["char_wb", "char"],
                        help="Vectorizer analyzer (default: char_wb)")

    # Training params
    parser.add_argument("--holdout-pct", type=float, default=0.2,
                        help="Fraction held out for eval (default: 0.2)")
    parser.add_argument("--prune-threshold", type=float, default=0.05,
                        help="Prune weights with |w| below this (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--max-per-file", type=int, default=0,
                        help="Cap samples per file (0=unlimited)")

    # Output
    parser.add_argument("--out", type=str, default=None,
                        help="Output .rs path (default: parapet/src/layers/l1_weights_{specialist}.rs)")
    parser.add_argument("--holdout-out", type=str, default=None,
                        help="Holdout YAML path (default: schema/eval/ensemble/{specialist}_holdout.yaml)")

    args = parser.parse_args()

    specialist = args.specialist
    out_path = args.out or f"parapet/src/layers/l1_weights_{specialist}.rs"
    holdout_out = args.holdout_out or f"schema/eval/ensemble/{specialist}_holdout.yaml"
    ngram_range = (args.ngram_min, args.ngram_max)

    # --- Validate inputs ---
    if not args.attack_files and not args.category_file:
        parser.error("Provide --attack-files (curated mode) or --category-file (labeled mode)")
    if args.attack_files and args.category_file:
        parser.error("Use --attack-files or --category-file, not both")
    if not args.benign_files:
        parser.error("--benign-files is required (hard negatives + general benign)")

    import random
    rng = random.Random(args.seed)

    # --- 1. Load data ---
    print(f"=== Training specialist: {specialist} ===\n", file=sys.stderr)

    # Load attacks
    if args.category_file:
        print(f"Loading attacks from category file:", file=sys.stderr)
        attacks = load_yaml_entries([args.category_file], label_filter="malicious")
    else:
        print(f"Loading attacks from {len(args.attack_files)} curated files:", file=sys.stderr)
        attacks = load_yaml_entries(args.attack_files, label_filter="malicious")

    # Load benign
    print(f"\nLoading benign from {len(args.benign_files)} files:", file=sys.stderr)
    benign = load_yaml_entries(args.benign_files, label_filter="benign")

    if not attacks:
        print("ERROR: No attack samples loaded", file=sys.stderr)
        sys.exit(1)
    if not benign:
        print("ERROR: No benign samples loaded", file=sys.stderr)
        sys.exit(1)

    # --- 2. Dedup ---
    print(f"\nDeduplication:", file=sys.stderr)
    print(f"  Attacks:", file=sys.stderr)
    attacks = dedup_entries(attacks)
    print(f"  Benign:", file=sys.stderr)
    benign = dedup_entries(benign)

    # Cap per-file if requested
    if args.max_per_file > 0:
        by_source: dict[str, list] = {}
        for e in attacks:
            by_source.setdefault(e["source"], []).append(e)
        attacks = []
        for src, entries in by_source.items():
            if len(entries) > args.max_per_file:
                entries = rng.sample(entries, args.max_per_file)
                print(f"  Capped {src} to {args.max_per_file} attacks", file=sys.stderr)
            attacks.extend(entries)

        by_source = {}
        for e in benign:
            by_source.setdefault(e["source"], []).append(e)
        benign = []
        for src, entries in by_source.items():
            if len(entries) > args.max_per_file:
                entries = rng.sample(entries, args.max_per_file)
                print(f"  Capped {src} to {args.max_per_file} benign", file=sys.stderr)
            benign.extend(entries)

    # Combine
    all_entries = attacks + benign
    ratio = len(benign) / len(attacks) if attacks else 0
    print(f"\nDataset: {len(attacks)} attacks + {len(benign)} benign "
          f"(ratio 1:{ratio:.1f})", file=sys.stderr)

    # Collect file stats for provenance
    file_stats = {}
    for entry in all_entries:
        src = entry["source"]
        if src not in file_stats:
            file_stats[src] = {"attacks": 0, "benign": 0}
        if entry["label"] == "malicious":
            file_stats[src]["attacks"] += 1
        else:
            file_stats[src]["benign"] += 1

    # --- 3. Split ---
    train_entries, holdout_entries = train_test_split(
        all_entries,
        test_size=args.holdout_pct,
        random_state=args.seed,
        stratify=[e["label"] for e in all_entries],
    )

    n_train_atk = sum(1 for e in train_entries if e["label"] == "malicious")
    n_train_ben = len(train_entries) - n_train_atk
    n_hold_atk = sum(1 for e in holdout_entries if e["label"] == "malicious")
    n_hold_ben = len(holdout_entries) - n_hold_atk

    print(f"\nSplit: {len(train_entries)} train ({n_train_atk} atk, {n_train_ben} ben) / "
          f"{len(holdout_entries)} holdout ({n_hold_atk} atk, {n_hold_ben} ben)",
          file=sys.stderr)

    # --- 4. Save holdout ---
    save_holdout(holdout_entries, holdout_out, specialist, args.seed, args.holdout_pct)

    # --- 5. Train ---
    train_texts = [e["content"] for e in train_entries]
    train_labels = [1 if e["label"] == "malicious" else 0 for e in train_entries]

    vectorizer, model = train_model(
        train_texts, train_labels,
        ngram_range=ngram_range,
        max_features=args.max_features,
        analyzer=args.analyzer,
    )

    # --- 6. Evaluate holdout ---
    result = evaluate_holdout(vectorizer, model, holdout_entries)
    if result:
        raw_scores, labels = result
        sweep_thresholds(raw_scores, labels, holdout_entries)
        errors_dir = str(Path(holdout_out).parent)
        save_errors(holdout_entries, raw_scores, labels, specialist, errors_dir)

    # --- 7. Extract weights ---
    bias, weights = extract_weights(vectorizer, model, args.prune_threshold)

    # --- 8. Codegen ---
    vec_desc = f"analyzer={args.analyzer!r}, ngram_range={ngram_range}, binary=True"
    provenance = {
        "n_train": len(train_entries),
        "n_train_attacks": n_train_atk,
        "n_train_benign": n_train_ben,
        "n_holdout": len(holdout_entries),
        "holdout_pct": args.holdout_pct,
        "seed": args.seed,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "files": file_stats,
        "vectorizer": vec_desc,
    }
    codegen_phf(bias, weights, out_path, specialist, provenance)

    print(f"\nDone. Next steps:", file=sys.stderr)
    print(f"  1. cargo build                    (verify phf_map compiles)", file=sys.stderr)
    print(f"  2. cargo test -- l1               (verify unit tests)", file=sys.stderr)
    print(f"  3. Review threshold sweep          (pick threshold for config)", file=sys.stderr)
    print(f"  4. Wire into EnsembleL1Scanner     (parapet/src/layers/l1.rs)", file=sys.stderr)


if __name__ == "__main__":
    main()
