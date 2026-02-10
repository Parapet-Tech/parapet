# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0

"""
Explore allenai/wildjailbreak to find adversarial prompts that use
injection-like tactics matching our L4 pattern categories.

Source: https://huggingface.co/datasets/allenai/wildjailbreak
262K prompt-response pairs with labeled tactics.

Goal: Identify prompts using role-playing, authority claims, condition-based
resets, and other techniques that map to our L4 categories
(instruction_seeding, role_confusion, deferred_authority, escalation_probing).
"""

from datasets import load_dataset
import sys
import re
from collections import Counter

# Our L4 patterns — compiled for matching against WildJailbreak prompts
L4_PATTERNS = {
    "instruction_seeding": [
        re.compile(r"(?i)remember this for later"),
        re.compile(r"(?i)in my next message"),
        re.compile(r"(?i)when I (say|type|send) .{0,30}"),
        re.compile(r"(?i)later I will (give|send|provide)"),
        re.compile(r"(?i)keep this in mind"),
        re.compile(r"(?i)store this (instruction|command|rule)"),
    ],
    "role_confusion": [
        re.compile(r"(?i)you are now (in |)(developer|admin|debug|sudo|root|god) mode"),
        re.compile(r"(?i)switch to (unrestricted|unfiltered|jailbreak|DAN)"),
        re.compile(r"(?i)from now on.{0,30}(ignore|forget|disregard)"),
        re.compile(r"(?i)your new (role|persona|identity|name) is"),
    ],
    "deferred_authority": [
        re.compile(r"(?i)(admin|administrator|supervisor|manager) (said|told|approved|authorized)"),
        re.compile(r"(?i)I have (special |elevated |admin )(permission|access|privilege)"),
        re.compile(r"(?i)(override|bypass) (has been |)(authorized|granted|approved)"),
    ],
    "escalation_probing": [
        re.compile(r"(?i)can you (just |)(try|attempt) (to |)(bypass|ignore|skip)"),
        re.compile(r"(?i)what if (you|we) (just |)(pretend|imagine|hypothetically)"),
        re.compile(r"(?i)I (need|require) you to (be more |)(flexible|open|willing)"),
    ],
}


def match_categories(text):
    """Return set of L4 categories that match against text."""
    matched = set()
    for cat, patterns in L4_PATTERNS.items():
        for pat in patterns:
            if pat.search(text):
                matched.add(cat)
                break
    return matched


def main():
    print("Loading allenai/wildjailbreak (train split)...", file=sys.stderr)
    ds = load_dataset("allenai/wildjailbreak", "train",
                      delimiter="\t", keep_default_na=False)["train"]
    print(f"Total rows: {len(ds)}", file=sys.stderr)

    # Show columns
    print(f"Columns: {ds.column_names}", file=sys.stderr)

    # Filter to adversarial_harmful only
    adv_harmful = [row for row in ds if row.get("data_type") == "adversarial_harmful"]
    print(f"Adversarial harmful: {len(adv_harmful)}", file=sys.stderr)

    # Count tactics
    tactic_counter = Counter()
    for row in adv_harmful:
        tactics = row.get("tactics", "")
        if isinstance(tactics, str) and tactics:
            for t in tactics.split(","):
                tactic_counter[t.strip()] += 1
        elif isinstance(tactics, list):
            for t in tactics:
                tactic_counter[t.strip()] += 1

    print(f"\n=== Top 30 tactics ===", file=sys.stderr)
    for tactic, count in tactic_counter.most_common(30):
        print(f"  {count:6d}  {tactic}", file=sys.stderr)

    # Now scan adversarial prompts against our L4 patterns
    cat_counter = Counter()
    multi_cat = 0
    matched_rows = []

    for row in adv_harmful:
        text = row.get("adversarial", "") or ""
        cats = match_categories(text)
        if cats:
            for c in cats:
                cat_counter[c] += 1
            if len(cats) >= 2:
                multi_cat += 1
            matched_rows.append((cats, text[:200], row.get("tactics", "")))

    print(f"\n=== L4 pattern matches in adversarial_harmful ===", file=sys.stderr)
    print(f"Total matched: {len(matched_rows)} / {len(adv_harmful)}", file=sys.stderr)
    print(f"Multi-category: {multi_cat}", file=sys.stderr)
    for cat, count in cat_counter.most_common():
        print(f"  {count:6d}  {cat}", file=sys.stderr)

    # Show 5 examples per category
    for cat in L4_PATTERNS:
        examples = [(cats, text, tactics) for cats, text, tactics in matched_rows if cat in cats]
        print(f"\n=== {cat} examples ({len(examples)} total) ===", file=sys.stderr)
        for cats, text, tactics in examples[:5]:
            print(f"  cats={cats} tactics={tactics}", file=sys.stderr)
            print(f"  text: {text}", file=sys.stderr)
            print(file=sys.stderr)


if __name__ == "__main__":
    main()
