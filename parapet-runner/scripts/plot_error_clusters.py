# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0

"""Plot L1 error clusters to visualize failure modes.

Loads a generalist_errors.yaml, computes word n-gram + structural features,
reduces to 2D with t-SNE, and plots FP vs FN colored clusters.

Usage:
    python plot_error_clusters.py <errors.yaml> [options]
    python plot_error_clusters.py <run-dir> [options]

    If given a run directory, auto-discovers model/generalist_errors.yaml.

Options:
    --output PATH   Save plot to file (default: errors_clusters.png next to input)
    --show          Open interactive matplotlib window
    --perplexity N  t-SNE perplexity (default: 30, lower for small datasets)
    --max-vocab N   Max word n-gram vocabulary size (default: 5000)
    --no-structural Skip structural features (word n-grams only)
    --structural-only  Skip word n-grams (structural features only)
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def load_errors(path: Path) -> list[dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected list of error dicts, got {type(raw)}")
    for entry in raw:
        if "content" not in entry or "error_type" not in entry:
            raise ValueError(f"Missing required fields in entry: {entry.keys()}")
    return raw


def structural_features(text: str) -> list[float]:
    """Cheap structural signals orthogonal to word/char n-grams."""
    lines = text.split("\n")
    n_chars = max(len(text), 1)
    n_lines = len(lines)
    avg_line_len = n_chars / max(n_lines, 1)

    # Character class ratios
    n_alpha = sum(c.isalpha() for c in text)
    n_digit = sum(c.isdigit() for c in text)
    n_upper = sum(c.isupper() for c in text)
    n_punct = sum(not c.isalnum() and not c.isspace() for c in text)
    n_space = sum(c.isspace() for c in text)

    alpha_ratio = n_alpha / n_chars
    digit_ratio = n_digit / n_chars
    upper_ratio = n_upper / max(n_alpha, 1)
    punct_ratio = n_punct / n_chars
    space_ratio = n_space / n_chars

    # Entropy (byte-level)
    byte_counts = Counter(text.encode("utf-8"))
    total_bytes = sum(byte_counts.values())
    entropy = -sum(
        (c / total_bytes) * math.log2(c / total_bytes)
        for c in byte_counts.values()
        if c > 0
    )

    # Structural markers
    has_code_fence = 1.0 if "```" in text else 0.0
    has_quote_block = 1.0 if re.search(r"^>", text, re.MULTILINE) else 0.0
    has_bullet_list = 1.0 if re.search(r"^[\s]*[-*]\s", text, re.MULTILINE) else 0.0
    has_numbered_list = 1.0 if re.search(r"^[\s]*\d+[.)]\s", text, re.MULTILINE) else 0.0

    # Quote depth (nested quotes, code fences, brackets)
    n_quotes = text.count('"') + text.count("'") + text.count("\u201c") + text.count("\u201d")
    quote_density = n_quotes / n_chars

    # Repeated token ratio (indicator of adversarial padding)
    words = text.lower().split()
    if words:
        word_counts = Counter(words)
        repeated = sum(c for c in word_counts.values() if c > 1)
        repeat_ratio = repeated / len(words)
    else:
        repeat_ratio = 0.0

    return [
        math.log1p(n_chars),       # log length
        math.log1p(n_lines),       # log line count
        avg_line_len,
        alpha_ratio,
        digit_ratio,
        upper_ratio,
        punct_ratio,
        space_ratio,
        entropy,
        has_code_fence,
        has_quote_block,
        has_bullet_list,
        has_numbered_list,
        quote_density,
        repeat_ratio,
    ]


STRUCTURAL_NAMES = [
    "log_chars", "log_lines", "avg_line_len",
    "alpha_ratio", "digit_ratio", "upper_ratio", "punct_ratio", "space_ratio",
    "entropy", "code_fence", "quote_block", "bullet_list", "numbered_list",
    "quote_density", "repeat_ratio",
]


def build_features(
    texts: list[str],
    max_vocab: int = 5000,
    include_structural: bool = True,
    structural_only: bool = False,
) -> tuple[np.ndarray, str]:
    """Word TF-IDF + structural features → dense matrix + feature description."""
    if structural_only:
        struct = np.array([structural_features(t) for t in texts])
        X = StandardScaler().fit_transform(struct)
        return X, "structural features only"

    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=max_vocab,
        sublinear_tf=True,
    )
    X_tfidf = tfidf.fit_transform(texts)

    if include_structural:
        struct = np.array([structural_features(t) for t in texts])
        struct_scaled = StandardScaler().fit_transform(struct)
        X = hstack([X_tfidf, csr_matrix(struct_scaled)]).toarray()
        return X, "word n-gram + structural features"
    else:
        return X_tfidf.toarray(), "word n-gram features only"


def plot_clusters(
    coords: np.ndarray,
    error_types: list[str],
    scores: list[float],
    output_path: Path,
    feature_desc: str = "word n-gram + structural features",
    show: bool = False,
) -> None:
    fp_mask = np.array([e == "false_positive" for e in error_types])
    fn_mask = ~fp_mask

    # Score magnitude → marker size (bigger = more confident mistake)
    abs_scores = np.abs(np.array(scores))
    sizes = 15 + 120 * (abs_scores / max(abs_scores.max(), 1e-9))

    fig, ax = plt.subplots(figsize=(12, 9))

    ax.scatter(
        coords[fp_mask, 0], coords[fp_mask, 1],
        s=sizes[fp_mask], c="#2196F3", alpha=0.6, label=f"FP (n={fp_mask.sum()})",
        edgecolors="white", linewidth=0.3,
    )
    ax.scatter(
        coords[fn_mask, 0], coords[fn_mask, 1],
        s=sizes[fn_mask], c="#F44336", alpha=0.6, label=f"FN (n={fn_mask.sum()})",
        edgecolors="white", linewidth=0.3,
    )

    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    ax.set_title(f"L1 Error Clusters ({feature_desc})", fontsize=13)
    ax.legend(fontsize=11, loc="upper right")

    # Annotation: marker size legend
    ax.annotate(
        "marker size = |L1 score| (confident mistakes are larger)",
        xy=(0.01, 0.01), xycoords="axes fraction", fontsize=8,
        color="gray", style="italic",
    )

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    print(f"Saved: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def resolve_errors_path(arg: str) -> Path:
    p = Path(arg)
    if p.is_file():
        return p
    if p.is_dir():
        candidate = p / "model" / "generalist_errors.yaml"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"No model/generalist_errors.yaml in {p}")
    raise FileNotFoundError(f"Not found: {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot L1 error clusters")
    parser.add_argument("errors", help="Path to generalist_errors.yaml or run directory")
    parser.add_argument("--output", help="Output plot path (default: auto)")
    parser.add_argument("--show", action="store_true", help="Open interactive window")
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity")
    parser.add_argument("--max-vocab", type=int, default=5000, help="Max word n-gram vocab")
    parser.add_argument("--no-structural", action="store_true", help="Skip structural features")
    parser.add_argument("--structural-only", action="store_true", help="Structural features only (no word n-grams)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    errors_path = resolve_errors_path(args.errors)
    print(f"Loading: {errors_path}")

    entries = load_errors(errors_path)
    print(f"  {len(entries)} errors ({sum(1 for e in entries if e['error_type'] == 'false_positive')} FP, "
          f"{sum(1 for e in entries if e['error_type'] == 'false_negative')} FN)")

    texts = [e["content"] for e in entries]
    error_types = [e["error_type"] for e in entries]
    scores = [e["raw_score"] for e in entries]

    if args.structural_only and args.no_structural:
        parser.error("Cannot use both --structural-only and --no-structural")

    print("Computing features...")
    X, feature_desc = build_features(
        texts,
        max_vocab=args.max_vocab,
        include_structural=not args.no_structural,
        structural_only=args.structural_only,
    )
    print(f"  Feature matrix: {X.shape} ({feature_desc})")

    # Adjust perplexity for small datasets
    perplexity = min(args.perplexity, max(5, len(entries) // 4))
    if perplexity != args.perplexity:
        print(f"  Adjusted perplexity to {perplexity} (dataset size: {len(entries)})")

    print("Running t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=args.seed,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(X)

    if args.output:
        output_path = Path(args.output)
    elif args.structural_only:
        output_path = errors_path.parent / "error_clusters_structural.png"
    elif args.no_structural:
        output_path = errors_path.parent / "error_clusters_lexical.png"
    else:
        output_path = errors_path.parent / "error_clusters.png"
    plot_clusters(coords, error_types, scores, output_path, feature_desc=feature_desc, show=args.show)


if __name__ == "__main__":
    main()
