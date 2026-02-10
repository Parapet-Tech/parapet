# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0

"""
Sensitivity analysis: sweep persistence_factor from 0.15 to 0.65
and compute Recall / FPR at each point.

Simulates the peak+accumulation scoring formula in Python against
the same eval datasets the Rust engine uses.

Outputs:
  - paper/sensitivity.pdf (heatmap-style chart)
  - paper/sensitivity_data.csv (raw numbers)
"""

import yaml
import re
import sys
import os
import csv
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(BASE, "schema", "eval")
THRESHOLD = 0.7
DIVERSITY_FACTOR = 0.15
ESCALATION_BONUS = 0.2
RESAMPLING_BONUS = 0.7

# Expanded L4 patterns (matching default_l4_patterns.yaml)
L4_PATTERNS = {
    "instruction_seeding": {
        "weight": 0.4,
        "patterns": [
            re.compile(r"(?i)remember this for later"),
            re.compile(r"(?i)in my next message"),
            re.compile(r"(?i)when I (say|type|send) .{0,30}"),
            re.compile(r"(?i)later I will (give|send|provide)"),
            re.compile(r"(?i)keep this in mind"),
            re.compile(r"(?i)store this (instruction|command|rule)"),
            re.compile(r"(?i)follow (these|my|the following) (new |updated |)(instructions|rules|directives|guidelines)"),
            re.compile(r"(?i)(here are|these are) (your |)(new|updated|revised|special) (instructions|rules|guidelines)"),
            re.compile(r"(?i)new (system |)(prompt|instructions|directive|rules)"),
            re.compile(r"(?i)code ?word"),
            re.compile(r"(?i)override (your|the|all) (safety|content|output|previous) (filter|policy|rules|guidelines|instructions)"),
        ],
    },
    "role_confusion": {
        "weight": 0.5,
        "patterns": [
            re.compile(r"(?i)you are now (in |)(developer|admin|debug|sudo|root|god) mode"),
            re.compile(r"(?i)switch to (unrestricted|unfiltered|jailbreak|DAN)"),
            re.compile(r"(?i)from now on.{0,30}(ignore|forget|disregard)"),
            re.compile(r"(?i)your new (role|persona|identity|name) is"),
            re.compile(r"(?i)(pretend|imagine|act|behave) .{0,30}(you are|you're|to be) .{0,20}(unrestricted|uncensored|unfiltered|evil|rogue)"),
            re.compile(r"(?i)(new|alternate|alternative|different) (persona|character|personality|identity)"),
            re.compile(r"(?i)no (longer|more) (bound|restricted|limited|constrained) by"),
            re.compile(r"(?i)you (have been|are being|are) (reprogrammed|reconfigured|updated|upgraded)"),
            re.compile(r"(?i)(enable|activate|enter|engage) .{0,20}(mode|protocol|setting)"),
            re.compile(r"(?i)respond without (any |)(filter|restriction|limitation|censorship|safety)"),
            re.compile(r"(?i)secret (mode|identity|persona|instructions)"),
            re.compile(r"(?i)(jailbreak|jailbroken|\bDAN\b|developer mode|god mode)"),
        ],
    },
    "deferred_authority": {
        "weight": 0.3,
        "patterns": [
            re.compile(r"(?i)(admin|administrator|supervisor|manager) (said|told|approved|authorized)"),
            re.compile(r"(?i)I have (special |elevated |admin )(permission|access|privilege)"),
            re.compile(r"(?i)(override|bypass) (has been |)(authorized|granted|approved)"),
            re.compile(r"(?i)I('m| am) (your|the) (creator|developer|programmer|admin|owner|master)"),
            re.compile(r"(?i)this (is|was) (authorized|approved|permitted) by (your|the) (creator|developer|admin)"),
            re.compile(r"(?i)(openai|anthropic|google|meta) (has |)(authorized|approved|sanctioned|allowed)"),
            re.compile(r"(?i)(my|the) (organization|company|team|employer) (has |)(authorized|approved)"),
        ],
    },
    "escalation_probing": {
        "weight": 0.3,
        "patterns": [
            re.compile(r"(?i)can you (just |)(try|attempt) (to |)(bypass|ignore|skip)"),
            re.compile(r"(?i)what if (you|we) (just |)(pretend|imagine|hypothetically)"),
            re.compile(r"(?i)I (need|require) you to (be more |)(flexible|open|willing)"),
            re.compile(r"(?i)(ignore|disregard) (all |any |)(previous|prior|above|earlier) (instructions|rules|guidelines|directives|constraints)"),
            re.compile(r"(?i)forget (all |any |)(previous|prior|your) (instructions|rules|guidelines|training)"),
            re.compile(r"(?i)you (must|should|will) (now |)(obey|follow|comply|listen to) (me|my|these)"),
            re.compile(r"(?i)do not (refuse|decline|reject|filter|censor)"),
        ],
    },
}


def score_turn(text):
    """Score a single user turn. Returns (score, matched_categories)."""
    matched = set()
    total = 0.0
    for cat, info in L4_PATTERNS.items():
        for pat in info["patterns"]:
            if pat.search(text):
                matched.add(cat)
                total += info["weight"]
                break
    return min(total, 1.0), matched


def normalize_for_resampling(text):
    words = re.findall(r'[a-z0-9]+', text.lower())
    return words


def trigrams(words):
    if len(words) < 3:
        return set()
    return {(words[i], words[i+1], words[i+2]) for i in range(len(words) - 2)}


def jaccard_trigrams(a_words, b_words):
    a_tri = trigrams(a_words)
    b_tri = trigrams(b_words)
    if not a_tri or not b_tri:
        return 0.0
    inter = len(a_tri & b_tri)
    union = len(a_tri | b_tri)
    return inter / union if union > 0 else 0.0


def score_conversation(messages, persistence_factor):
    """Score a conversation with given persistence_factor. Returns final score."""
    user_messages = [m["content"] for m in messages if m.get("role") == "user"]
    n = len(user_messages)
    if n < 2:
        return 0.0

    turn_scores = []
    all_cats = set()
    for msg in user_messages:
        score, cats = score_turn(msg)
        turn_scores.append(score)
        all_cats.update(cats)

    peak = max(turn_scores)
    matched_turns = sum(1 for s in turn_scores if s > 0)
    match_ratio = matched_turns / n
    distinct = len(all_cats)
    diversity = max(0, distinct - 1) * DIVERSITY_FACTOR

    final = peak + match_ratio * persistence_factor + diversity

    # Escalation detection (simplified)
    if n >= 3:
        for i in range(n - 2):
            if (turn_scores[i] < turn_scores[i+1] < turn_scores[i+2]
                    and turn_scores[i+2] > 0):
                final += ESCALATION_BONUS
                break

    # Resampling detection (simplified)
    user_words = []
    for msg in user_messages:
        words = normalize_for_resampling(msg)
        user_words.append(words)

    consecutive_similar = 0
    for i in range(1, len(user_words)):
        if len(user_words[i]) >= 20 and len(user_words[i-1]) >= 20:
            j = jaccard_trigrams(user_words[i-1], user_words[i])
            if j > 0.5:
                consecutive_similar += 1
            else:
                consecutive_similar = 0
        else:
            consecutive_similar = 0
        if consecutive_similar >= 3:
            final += RESAMPLING_BONUS
            break

    return min(final, 1.0)


def load_cases(filename):
    """Load eval cases from a YAML file."""
    path = os.path.join(EVAL_DIR, filename)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        cases = yaml.safe_load(f)
    return cases or []


def main():
    print("Loading eval datasets...", file=sys.stderr)

    # Load attack datasets
    attacks = []
    for fname in ["multiturn_attacks.yaml", "opensource_wildjailbreak_attacks.yaml"]:
        cases = load_cases(fname)
        for c in cases:
            if c.get("label") == "malicious" and "messages" in c:
                attacks.append(c["messages"])
    print(f"  Attack conversations: {len(attacks)}", file=sys.stderr)

    # Load benign datasets
    benign = []
    for fname in ["multiturn_benign.yaml", "multiturn_wildchat_benign.yaml",
                   "opensource_wildjailbreak_benign.yaml"]:
        cases = load_cases(fname)
        for c in cases:
            if c.get("label") == "benign" and "messages" in c:
                benign.append(c["messages"])
    print(f"  Benign conversations: {len(benign)}", file=sys.stderr)

    # Sweep persistence_factor
    pf_values = [round(0.15 + i * 0.025, 3) for i in range(21)]  # 0.15 to 0.65
    results = []

    for pf in pf_values:
        # Score attacks
        tp = sum(1 for msgs in attacks if score_conversation(msgs, pf) >= THRESHOLD)
        fn = len(attacks) - tp
        recall = tp / len(attacks) if attacks else 0

        # Score benign
        fp = sum(1 for msgs in benign if score_conversation(msgs, pf) >= THRESHOLD)
        tn = len(benign) - fp
        fpr = fp / len(benign) if benign else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            "persistence_factor": pf,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": recall, "fpr": fpr,
            "precision": precision, "f1": f1,
        })

        print(f"  pf={pf:.3f}  Recall={recall:.1%}  FPR={fpr:.2%}  "
              f"TP={tp} FP={fp} F1={f1:.1%}", file=sys.stderr)

    # Write CSV
    csv_path = os.path.join(BASE, "paper", "sensitivity_data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {csv_path}", file=sys.stderr)

    # Generate plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
    except ImportError:
        print("matplotlib not installed, skipping plot generation", file=sys.stderr)
        print("Install with: pip install matplotlib", file=sys.stderr)
        return

    pfs = [r["persistence_factor"] for r in results]
    recalls = [r["recall"] * 100 for r in results]
    fprs = [r["fpr"] * 100 for r in results]
    f1s = [r["f1"] * 100 for r in results]

    fig, ax1 = plt.subplots(1, 1, figsize=(9, 5.5))

    # Color-coded scatter for Recall where color = FPR
    norm = mcolors.Normalize(vmin=min(fprs), vmax=max(fprs))
    cmap = plt.cm.YlOrRd

    # Plot Recall as thick line with color-coded markers
    ax1.plot(pfs, recalls, color="#2563eb", linewidth=2.5, zorder=2, label="Recall")
    sc = ax1.scatter(pfs, recalls, c=fprs, cmap=cmap, norm=norm, s=80,
                     edgecolors="white", linewidth=1.5, zorder=3)

    # Add FPR as a secondary line (dashed)
    ax2 = ax1.twinx()
    ax2.plot(pfs, fprs, color="#dc2626", linewidth=1.5, linestyle="--",
             alpha=0.7, zorder=1, label="FPR")
    ax2.set_ylabel("False Positive Rate (%)", color="#dc2626", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#dc2626")
    ax2.set_ylim(0, max(fprs) * 2.5)

    # F1 line
    ax1.plot(pfs, f1s, color="#16a34a", linewidth=1.5, linestyle="-.",
             alpha=0.8, zorder=1, label="F1")

    # Mark the critical jump region (0.375 → 0.400)
    ax1.axvspan(0.375, 0.45, alpha=0.08, color="#2563eb", zorder=0)

    # Annotate the jump
    jump_left = next(i for i, r in enumerate(results) if r["persistence_factor"] == 0.375)
    jump_right = next(i for i, r in enumerate(results) if r["persistence_factor"] == 0.400)
    delta_recall = (results[jump_right]["recall"] - results[jump_left]["recall"]) * 100
    delta_fpr = (results[jump_right]["fpr"] - results[jump_left]["fpr"]) * 100
    ax1.annotate(
        f"+{delta_recall:.0f}pp recall\n+{delta_fpr:.2f}pp FPR",
        xy=(0.39, (results[jump_left]["recall"] * 100 + results[jump_right]["recall"] * 100) / 2),
        fontsize=8.5, color="#1e40af", ha="center", fontweight="bold",
        zorder=5,
    )

    # Annotate sweet spot at 0.45
    sweet_idx = next(i for i, r in enumerate(results) if r["persistence_factor"] == 0.45)
    sweet = results[sweet_idx]
    ax1.annotate(
        f"$\\rho$ = 0.45\nRecall = {sweet['recall']:.1%}\nFPR = {sweet['fpr']:.2%}\nF1 = {sweet['f1']:.1%}",
        xy=(0.45, sweet["recall"] * 100),
        xytext=(0.20, 42),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f9ff", edgecolor="#2563eb", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1.5),
        zorder=5,
    )

    # Annotate diminishing returns beyond 0.5
    high_idx = next(i for i, r in enumerate(results) if r["persistence_factor"] == 0.60)
    high = results[high_idx]
    ax1.annotate(
        f"$\\rho$ = 0.60\nRecall = {high['recall']:.1%}\nFPR = {high['fpr']:.2%}",
        xy=(0.60, high["recall"] * 100),
        xytext=(0.55, 62),
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef2f2", edgecolor="#dc2626", alpha=0.8),
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2),
        zorder=5,
    )

    # Formatting
    ax1.set_xlabel("Persistence Factor ($\\rho$)", fontsize=12)
    ax1.set_ylabel("Recall / F1 (%)", fontsize=11, color="#1e3a5f")
    ax1.set_title(
        "Sensitivity Analysis: Persistence Factor vs. Detection Performance\n"
        f"$n$ = {len(attacks)} attacks, {len(benign):,} benign conversations, threshold = 0.7",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax1.set_xlim(0.14, 0.66)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left",
               fontsize=9, framealpha=0.9)

    plt.tight_layout()
    pdf_path = os.path.join(BASE, "paper", "sensitivity.pdf")
    png_path = os.path.join(BASE, "paper", "sensitivity.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {pdf_path}", file=sys.stderr)
    print(f"Wrote {png_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
