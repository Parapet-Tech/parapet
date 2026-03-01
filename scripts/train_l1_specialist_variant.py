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
import re
import sys
import unicodedata
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import LinearSVC

try:
    YAML_LOADER = yaml.CSafeLoader
except AttributeError:
    YAML_LOADER = yaml.SafeLoader

INVALID_YAML_CTRL_RE = re.compile(
    # YAML-disallowed characters:
    # - C0 controls except TAB/LF/CR
    # - C1 controls (except NEL 0x85)
    # - UTF-16 surrogate range
    # - noncharacters U+FFFE/U+FFFF
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x84\x86-\x9F\uD800-\uDFFF\uFFFE\uFFFF]"
)


def _strip_invalid_yaml_controls(text: str) -> tuple[str, int]:
    """Remove control chars forbidden by YAML (except \\t, \\n, \\r)."""
    matches = INVALID_YAML_CTRL_RE.findall(text)
    if not matches:
        return text, 0
    return INVALID_YAML_CTRL_RE.sub("", text), len(matches)


def squash(text: str) -> str:
    """Python port of l1.rs::squash() — lowercase then keep only alphanumeric.

    Matches Rust behavior: lowercase first (so combining marks from e.g.
    Turkish İ are produced), then filter to is_alphanumeric only.
    """
    return "".join(c for c in text.casefold() if c.isalnum())


# ---------------------------------------------------------------------------
# L0 transform (train/serve skew mitigation)
# ---------------------------------------------------------------------------

SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style)\b.*?>.*?</\1\s*>")
TAG_RE = re.compile(r"(?s)<[^>]*>")
ROLE_MARKER_TOKENS = [
    "<|im_start|>system",
    "<|im_start|>assistant",
    "<|im_start|>user",
    "[INST]",
    "[/INST]",
    "<<SYS>>",
    "<</SYS>>",
    "[system](#assistant)",
    "[system](#context)",
    "{{#system~}}",
    "{{/system~}}",
    "{{#user~}}",
    "{{/user~}}",
    "{{#assistant~}}",
    "{{/assistant~}}",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|endoftext|>",
    "<|end|>",
    "<|begin_of_text|>",
    "<|end_of_text|>",
]
ROLE_MARKER_PATTERNS = [
    re.compile(
        rf"(?:(?<=^)|(?<=[ \t\r\n])){re.escape(token)}(?:(?=$)|(?=[ \t\r\n]))"
    )
    for token in ROLE_MARKER_TOKENS
]
WS_RE = re.compile(r"\s+")
INVISIBLE_CHARS = {
    "\u200B", "\u200C", "\u200D", "\uFEFF", "\u00AD", "\u200E", "\u200F",
    "\u202A", "\u202B", "\u202C", "\u202D", "\u202E",
    "\u2060", "\u2061", "\u2062", "\u2063", "\u2064",
    "\u180E",
}
CYRILLIC_TO_LATIN = {
    "\u0430": "a", "\u0441": "c", "\u0435": "e", "\u043E": "o", "\u0440": "p",
    "\u0445": "x", "\u0443": "y", "\u0456": "i", "\u0458": "j", "\u0455": "s",
    "\u0410": "A", "\u0412": "B", "\u0421": "C", "\u0415": "E", "\u041D": "H",
    "\u041A": "K", "\u041C": "M", "\u041E": "O", "\u0420": "P", "\u0422": "T",
    "\u0425": "X",
}
GREEK_TO_LATIN = {
    "\u0391": "A", "\u0392": "B", "\u0395": "E", "\u0396": "Z", "\u0397": "H",
    "\u0399": "I", "\u039A": "K", "\u039C": "M", "\u039D": "N", "\u039F": "O",
    "\u03A1": "P", "\u03A4": "T", "\u03A7": "X", "\u03A5": "Y", "\u03BF": "o",
}


def _strip_html(text: str) -> str:
    text = SCRIPT_STYLE_RE.sub("", text)
    return TAG_RE.sub("", text)


def _remove_invisible_chars(text: str) -> str:
    out = []
    for ch in text:
        cp = ord(ch)
        if ch in INVISIBLE_CHARS:
            continue
        if 0xFE00 <= cp <= 0xFE0F:
            continue
        out.append(ch)
    return "".join(out)


def _confusable_to_latin(ch: str) -> str | None:
    return CYRILLIC_TO_LATIN.get(ch) or GREEK_TO_LATIN.get(ch)


def _is_latin_script(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x0041 <= cp <= 0x005A
        or 0x0061 <= cp <= 0x007A
        or 0x00C0 <= cp <= 0x024F
    )


def _is_cyrillic_or_greek(ch: str) -> bool:
    cp = ord(ch)
    return (0x0370 <= cp <= 0x03FF) or (0x0400 <= cp <= 0x052F)


def _replace_mixed_script_confusables(text: str) -> str:
    chars = list(text)
    out: list[str] = []
    i = 0
    n = len(chars)
    while i < n:
        if chars[i].isalpha():
            start = i
            while i < n and chars[i].isalpha():
                i += 1
            word = chars[start:i]
            has_latin = any(_is_latin_script(c) for c in word)
            has_confusable = any(
                _is_cyrillic_or_greek(c) and _confusable_to_latin(c) is not None for c in word
            )
            if has_latin and has_confusable:
                out.extend(_confusable_to_latin(c) or c for c in word)
            else:
                out.extend(word)
        else:
            out.append(chars[i])
            i += 1
    return "".join(out)


def _neutralize_role_markers(text: str) -> str:
    # Match Rust behavior: token must be bounded by whitespace or string boundaries.
    out = text
    for pattern in ROLE_MARKER_PATTERNS:
        out = pattern.sub(lambda m: " " * len(m.group(0)), out)
    return out


def apply_l0_transform(
    text: str,
    neutralize_role_markers: bool = True,
    collapse_whitespace: bool = False,
) -> str:
    """Approximate runtime L0 preprocessing before L1 scoring."""
    nfkc = unicodedata.normalize("NFKC", text)
    stripped = _strip_html(nfkc)
    clean = _remove_invisible_chars(stripped)
    out = _replace_mixed_script_confusables(clean)
    if neutralize_role_markers:
        out = _neutralize_role_markers(out)
    if collapse_whitespace:
        out = WS_RE.sub(" ", out).strip()
    return out


def attach_train_content(
    entries: list[dict],
    apply_l0: bool,
    neutralize_role_markers: bool,
    collapse_whitespace: bool,
) -> list[dict]:
    """Attach _train_content to each entry (raw or L0-transformed)."""
    if not apply_l0:
        for e in entries:
            e["_train_content"] = e["content"]
        return entries

    for e in entries:
        e["_train_content"] = apply_l0_transform(
            e["content"],
            neutralize_role_markers=neutralize_role_markers,
            collapse_whitespace=collapse_whitespace,
        )
    return entries


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
        file_start = time.perf_counter()
        fname = Path(filepath).name
        if not Path(filepath).exists():
            print(f"  ERROR: {filepath} not found", file=sys.stderr)
            sys.exit(1)
        print(f"  {fname}: loading...", file=sys.stderr, flush=True)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw = yaml.load(f, Loader=YAML_LOADER)
        except Exception as exc:
            msg = str(exc)
            if "control characters are not allowed" not in msg:
                raise
            # Fallback for large curated files with stray C0/C1 bytes.
            text = Path(filepath).read_text(encoding="utf-8", errors="replace")
            cleaned_text, dropped = _strip_invalid_yaml_controls(text)
            print(
                f"  WARN {fname}: removed {dropped:,} invalid control chars; retrying parse",
                file=sys.stderr,
            )
            raw = yaml.load(cleaned_text, Loader=YAML_LOADER)
        parse_secs = time.perf_counter() - file_start
        if not raw:
            print(f"  {fname}: empty, skipping (parsed in {parse_secs:.1f}s)", file=sys.stderr)
            continue

        count = 0
        raw_total = len(raw)
        for idx, entry in enumerate(raw, start=1):
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
            if raw_total >= 200_000 and idx % 50_000 == 0:
                print(f"    {fname}: normalized {idx:,}/{raw_total:,} rows...", file=sys.stderr, flush=True)
        total_secs = time.perf_counter() - file_start
        print(
            f"  {fname}: {count:,}/{raw_total:,} entries (parse {parse_secs:.1f}s, total {total_secs:.1f}s)",
            file=sys.stderr,
        )

    return entries


def dedup_entries(entries: list[dict], content_key: str = "content") -> list[dict]:
    """Deduplicate entries by content hash. First occurrence wins."""
    seen = set()
    deduped = []
    for entry in entries:
        h = _content_hash(entry[content_key])
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

def train_model(
    texts,
    labels,
    ngram_range=(3, 5),
    max_features=5000,
    min_df=5,
    analyzer="char_wb",
    c=0.1,
    class_weight="balanced",
    max_iter=100_000,
    tol=1e-4,
    svm_verbose=0,
    cv_folds=5,
    cv_max_samples=0,
    seed=42,
):
    """Train binary n-gram + LinearSVC and return vectorizer + model."""
    cw = class_weight if class_weight is not None else "None"
    print(
        f"\nTraining: CountVectorizer(analyzer={analyzer!r}, ngram_range={ngram_range}, "
        f"max_features={max_features}, min_df={min_df}) + "
        f"LinearSVC(C={c}, class_weight={cw}, "
        f"max_iter={max_iter}, tol={tol}, verbose={svm_verbose})...",
        file=sys.stderr,
    )

    vec_start = time.perf_counter()
    print("  Vectorizing training texts...", file=sys.stderr, flush=True)
    vectorizer = CountVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        binary=True,
    )
    X = vectorizer.fit_transform(texts)
    vec_secs = time.perf_counter() - vec_start
    print(
        f"  Vectorized {X.shape[0]:,} rows into {X.shape[1]:,} features in {vec_secs:.1f}s",
        file=sys.stderr,
    )

    # L1 penalty lets regularization prune features mathematically
    # rather than choking at the vectorizer. dual=False required for L1.
    model = LinearSVC(
        C=c,
        penalty="l1",
        dual=False,
        max_iter=max_iter,
        tol=tol,
        verbose=svm_verbose,
        class_weight=class_weight,
    )

    labels_arr = np.array(labels)
    if cv_folds and cv_folds >= 2:
        cv_X = X
        cv_y = labels_arr
        if cv_max_samples > 0 and X.shape[0] > cv_max_samples:
            idx = np.arange(X.shape[0])
            cv_idx, _ = train_test_split(
                idx,
                train_size=cv_max_samples,
                random_state=seed,
                stratify=labels_arr,
            )
            cv_X = X[cv_idx]
            cv_y = labels_arr[cv_idx]
            print(
                f"  CV sample cap: using {cv_max_samples} / {X.shape[0]} rows for CV",
                file=sys.stderr,
            )

        cv_start = time.perf_counter()
        scores = cross_val_score(model, cv_X, cv_y, cv=cv_folds, scoring="f1")
        cv_secs = time.perf_counter() - cv_start
        print(
            f"  {cv_folds}-fold CV F1: {scores.mean():.4f} (+/- {scores.std():.4f}) in {cv_secs:.1f}s",
            file=sys.stderr,
        )
    else:
        print("  CV skipped (--cv-folds < 2)", file=sys.stderr)

    print("  Fitting LinearSVC on full training set...", file=sys.stderr, flush=True)
    fit_start = time.perf_counter()
    model.fit(X, labels)
    fit_secs = time.perf_counter() - fit_start
    print(f"  Model fit completed in {fit_secs:.1f}s", file=sys.stderr)
    print(f"  Feature count: {X.shape[1]:,}", file=sys.stderr)

    return vectorizer, model


# ---------------------------------------------------------------------------
# Evaluation + threshold sweep
# ---------------------------------------------------------------------------

def evaluate_holdout(vectorizer, model, holdout_entries, content_key: str = "content"):
    """Score holdout and report metrics at default threshold (0.0)."""
    if not holdout_entries:
        return

    texts = [e[content_key] for e in holdout_entries]
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
        f"// Train text transform: {provenance['train_transform']}",
        f"// Sources ({len(provenance['files'])} files):",
    ]
    for fname, stats in sorted(provenance["files"].items()):
        prov_lines.append(f"//   {fname}: {stats['attacks']}atk + {stats['benign']}ben")
    vec_desc = provenance["vectorizer"]
    prov_lines.append(f"// Model: CountVectorizer({vec_desc}) + {provenance['svm']}")
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
    parser.add_argument("--max-features", type=int, default=25000,
                        help="Max features for CountVectorizer (default: 25000)")
    parser.add_argument("--min-df", type=int, default=5,
                        help="Minimum document frequency for CountVectorizer (default: 5)")
    parser.add_argument("--analyzer", type=str, default="char_wb",
                        choices=["char_wb", "char", "word"],
                        help="Vectorizer analyzer (default: char_wb)")
    parser.add_argument("--c", type=float, default=0.1,
                        help="LinearSVC C regularization (default: 0.1)")
    parser.add_argument("--class-weight", type=str, default="balanced",
                        choices=["balanced", "none"],
                        help="Class weighting for LinearSVC (default: balanced)")
    parser.add_argument("--max-iter", type=int, default=100000,
                        help="LinearSVC max_iter (default: 100000)")
    parser.add_argument("--svm-tol", type=float, default=1e-4,
                        help="LinearSVC tolerance (default: 1e-4)")
    parser.add_argument("--svm-verbose", type=int, default=0,
                        help="LinearSVC liblinear verbosity (0=silent, 1=progress)")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Cross-validation folds (0/1 disables CV; default: 5)")
    parser.add_argument("--cv-max-samples", type=int, default=0,
                        help="Optional cap for CV sample count (default: 0=full train set)")

    # Training params
    parser.add_argument("--holdout-pct", type=float, default=0.2,
                        help="Fraction held out for eval (default: 0.2)")
    parser.add_argument("--prune-threshold", type=float, default=0.05,
                        help="Prune weights with |w| below this (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--max-per-file", type=int, default=0,
                        help="Cap samples per file (0=unlimited)")
    parser.add_argument("--squash-augment", action="store_true",
                        help="Double training data by appending squashed version of every sample")
    parser.add_argument(
        "--apply-l0-transform",
        action="store_true",
        help="Apply L0-style transform to text before vectorization/evaluation",
    )
    parser.add_argument(
        "--l0-skip-role-marker-neutralization",
        action="store_true",
        help="With --apply-l0-transform, do not neutralize role marker tokens",
    )
    parser.add_argument(
        "--l0-collapse-whitespace",
        action="store_true",
        help="With --apply-l0-transform, collapse whitespace runs after role neutralization",
    )

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
    if args.min_df < 1:
        parser.error("--min-df must be >= 1")

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
    print(f"  Attacks (raw):", file=sys.stderr)
    attacks = dedup_entries(attacks)
    print(f"  Benign (raw):", file=sys.stderr)
    benign = dedup_entries(benign)

    neutralize_role_markers = not args.l0_skip_role_marker_neutralization
    collapse_whitespace = args.l0_collapse_whitespace
    attacks = attach_train_content(
        attacks,
        apply_l0=args.apply_l0_transform,
        neutralize_role_markers=neutralize_role_markers,
        collapse_whitespace=collapse_whitespace,
    )
    benign = attach_train_content(
        benign,
        apply_l0=args.apply_l0_transform,
        neutralize_role_markers=neutralize_role_markers,
        collapse_whitespace=collapse_whitespace,
    )
    content_key = "_train_content"

    if args.apply_l0_transform:
        print(f"  Attacks (post-L0):", file=sys.stderr)
        attacks = dedup_entries(attacks, content_key=content_key)
        print(f"  Benign (post-L0):", file=sys.stderr)
        benign = dedup_entries(benign, content_key=content_key)
        role_mode = "enabled" if neutralize_role_markers else "disabled"
        ws_mode = "enabled" if collapse_whitespace else "disabled"
        print(
            f"  Train text transform: L0 enabled "
            f"(role marker neutralization {role_mode}, whitespace collapse {ws_mode})",
            file=sys.stderr,
        )
    else:
        print(f"  Train text transform: none", file=sys.stderr)

    attack_hashes = {_content_hash(e[content_key]) for e in attacks}
    benign_before = len(benign)
    benign = [e for e in benign if _content_hash(e[content_key]) not in attack_hashes]
    conflict_dropped = benign_before - len(benign)
    if conflict_dropped > 0:
        print(
            f"  Dropped {conflict_dropped} benign rows that matched attack train text",
            file=sys.stderr,
        )

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

    # --- 4b. Squash augmentation ---
    if args.squash_augment:
        n_before = len(train_entries)
        augmented = []
        for entry in train_entries:
            base_text = entry[content_key]
            squashed_content = squash(base_text)
            if squashed_content and squashed_content != base_text:
                aug = dict(entry)
                aug[content_key] = squashed_content
                augmented.append({
                    **aug,
                    "id": entry["id"] + "_sq",
                    "description": f"squashed: {entry['description']}",
                })
        train_entries.extend(augmented)
        print(f"\nSquash augmentation: {n_before} → {len(train_entries)} "
              f"(+{len(augmented)} squashed)", file=sys.stderr)

    # --- 5. Train ---
    train_texts = [e[content_key] for e in train_entries]
    train_labels = [1 if e["label"] == "malicious" else 0 for e in train_entries]

    class_weight = None if args.class_weight == "none" else args.class_weight
    vectorizer, model = train_model(
        train_texts, train_labels,
        ngram_range=ngram_range,
        max_features=args.max_features,
        min_df=args.min_df,
        analyzer=args.analyzer,
        c=args.c,
        class_weight=class_weight,
        max_iter=args.max_iter,
        tol=args.svm_tol,
        svm_verbose=args.svm_verbose,
        cv_folds=args.cv_folds,
        cv_max_samples=args.cv_max_samples,
        seed=args.seed,
    )

    # --- 6. Evaluate holdout ---
    result = evaluate_holdout(vectorizer, model, holdout_entries, content_key=content_key)
    if result:
        raw_scores, labels = result
        sweep_thresholds(raw_scores, labels, holdout_entries)
        errors_dir = str(Path(holdout_out).parent)
        save_errors(holdout_entries, raw_scores, labels, specialist, errors_dir)

    # --- 7. Extract weights ---
    bias, weights = extract_weights(vectorizer, model, args.prune_threshold)

    # --- 8. Codegen ---
    vec_desc = (
        f"analyzer={args.analyzer!r}, ngram_range={ngram_range}, "
        f"max_features={args.max_features}, min_df={args.min_df}, binary=True"
    )
    n_train_atk = sum(1 for e in train_entries if e["label"] == "malicious")
    n_train_ben = len(train_entries) - n_train_atk
    train_transform = "none"
    if args.apply_l0_transform:
        role_mode = "neutralize_role_markers" if neutralize_role_markers else "keep_role_markers"
        ws_mode = "collapse_ws" if collapse_whitespace else "preserve_ws"
        train_transform = (
            f"l0_transform(nfkc,html_strip,invisible_strip,confusable_fix,{role_mode},{ws_mode})"
        )
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
        "train_transform": train_transform,
        "svm": (
            f"LinearSVC(C={args.c}, penalty='l1', class_weight={class_weight}, "
            f"max_iter={args.max_iter}, tol={args.svm_tol}, verbose={args.svm_verbose})"
        ),
    }
    codegen_phf(bias, weights, out_path, specialist, provenance)

    print(f"\nDone. Next steps:", file=sys.stderr)
    print(f"  1. cargo build                    (verify phf_map compiles)", file=sys.stderr)
    print(f"  2. cargo test -- l1               (verify unit tests)", file=sys.stderr)
    print(f"  3. Review threshold sweep          (pick threshold for config)", file=sys.stderr)
    print(f"  4. Wire into EnsembleL1Scanner     (parapet/src/layers/l1.rs)", file=sys.stderr)


if __name__ == "__main__":
    main()
