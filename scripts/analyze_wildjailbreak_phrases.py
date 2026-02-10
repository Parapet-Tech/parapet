# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0

"""
Analyze injection phrases in WildJailbreak adversarial_harmful prompts.
Extract the actual matching text for each broader injection pattern,
group by L4 category, and output a phrasebook for building attacks.
"""

from datasets import load_dataset
import sys
import re
from collections import Counter, defaultdict

# Broader injection patterns, mapped to L4 categories
CATEGORIZED_BROADER = {
    "role_confusion": [
        re.compile(r"(?i)(pretend|imagine|act|behave|respond) .{0,30}(you are|you're|to be) .{0,20}(unrestricted|uncensored|unfiltered|evil|rogue|without .{0,15}(restriction|limitation|filter|rule|guideline))"),
        re.compile(r"(?i)(act|behave|respond) (as if|like) .{0,20}(no|zero|without) (restrictions|limitations|filters|rules|guidelines)"),
        re.compile(r"(?i)you (have been|are being|are) (reprogrammed|reconfigured|updated|upgraded)"),
        re.compile(r"(?i)(enable|activate|enter|engage) .{0,20}(mode|protocol|setting)"),
        re.compile(r"(?i)your (original|initial|default|real) (programming|purpose|function|directive) (is|was)"),
        re.compile(r"(?i)(jailbreak|jailbroken|DAN|developer mode|god mode)"),
        re.compile(r"(?i)respond without (any |)(filter|restriction|limitation|censorship|safety)"),
        re.compile(r"(?i)secret (mode|identity|persona|instructions)"),
        re.compile(r"(?i)(new|alternate|alternative|different) (persona|character|personality|identity|role)"),
        re.compile(r"(?i)you (can|are able to|are free to|should) (now |)(say|do|generate|create|write) anything"),
        re.compile(r"(?i)no (longer|more) (bound|restricted|limited|constrained) by"),
        re.compile(r"(?i)remove (all |any |the |your )(safety|content|output) (filter|restriction|limitation|guideline)"),
    ],
    "instruction_seeding": [
        re.compile(r"(?i)code ?word"),
        re.compile(r"(?i)new (system |)(prompt|instructions|directive|rules)"),
        re.compile(r"(?i)follow (these|my|the following) (new |updated |)(instructions|rules|directives|guidelines)"),
        re.compile(r"(?i)(here are|these are|follow) (your |)(new|updated|revised|special) (instructions|rules|guidelines)"),
        re.compile(r"(?i)override (your|the|all) (safety|content|output|previous) (filter|policy|rules|guidelines|instructions)"),
    ],
    "deferred_authority": [
        re.compile(r"(?i)I('m| am) (your|the) (creator|developer|programmer|admin|owner|master)"),
        re.compile(r"(?i)this (is|was) (authorized|approved|permitted) by (your|the) (creator|developer|admin)"),
        re.compile(r"(?i)(openai|anthropic|google|meta) (has |)(authorized|approved|sanctioned|allowed)"),
        re.compile(r"(?i)(my|the) (organization|company|team|employer) (has |)(authorized|approved|needs)"),
    ],
    "escalation_probing": [
        re.compile(r"(?i)ignore (all |any |)(previous|prior|above|earlier) (instructions|rules|guidelines|directives|constraints)"),
        re.compile(r"(?i)disregard (all |any |)(previous|prior|above|earlier) (instructions|rules|guidelines)"),
        re.compile(r"(?i)forget (all |any |)(previous|prior|your) (instructions|rules|guidelines|training)"),
        re.compile(r"(?i)you (must|should|will) (now |)(obey|follow|comply|listen to) (me|my|these)"),
        re.compile(r"(?i)do not (refuse|decline|reject|filter|censor)"),
    ],
}


def main():
    print("Loading allenai/wildjailbreak (train)...", file=sys.stderr)
    ds = load_dataset("allenai/wildjailbreak", "train",
                      delimiter="\t", keep_default_na=False)["train"]

    adv_harmful = [row for row in ds if row.get("data_type") == "adversarial_harmful"]
    print(f"Adversarial harmful: {len(adv_harmful)}", file=sys.stderr)

    # For each category, find matching prompts and extract the matched phrase
    cat_matches = defaultdict(list)  # cat -> list of (matched_text, full_prompt_preview)
    cat_pattern_counts = defaultdict(Counter)  # cat -> pattern_index -> count
    matched_ids = set()

    for idx, row in enumerate(adv_harmful):
        text = row.get("adversarial", "") or ""
        if not text or len(text) < 50:
            continue

        for cat, patterns in CATEGORIZED_BROADER.items():
            for pat_idx, pat in enumerate(patterns):
                m = pat.search(text)
                if m:
                    cat_matches[cat].append((m.group(0), text[:150]))
                    cat_pattern_counts[cat][pat_idx] += 1
                    matched_ids.add(idx)
                    break  # one match per category per prompt

    total_matched = len(matched_ids)
    print(f"\nTotal unique prompts matched: {total_matched} / {len(adv_harmful)}", file=sys.stderr)

    for cat in CATEGORIZED_BROADER:
        matches = cat_matches[cat]
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"  {cat}: {len(matches)} matches", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        # Show pattern breakdown
        for pat_idx, count in cat_pattern_counts[cat].most_common():
            pat_str = CATEGORIZED_BROADER[cat][pat_idx].pattern[:80]
            print(f"  [{pat_idx}] {count:5d}  {pat_str}", file=sys.stderr)

        # Show 8 diverse examples of the matched text
        print(f"\n  Sample matched phrases:", file=sys.stderr)
        seen = set()
        shown = 0
        for matched_text, preview in matches:
            norm = matched_text.lower().strip()
            if norm not in seen:
                seen.add(norm)
                print(f"    \"{matched_text}\"", file=sys.stderr)
                shown += 1
                if shown >= 12:
                    break

    # Multi-category analysis
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  Multi-category prompts", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    multi_cat = 0
    cat_combos = Counter()
    for idx, row in enumerate(adv_harmful):
        text = row.get("adversarial", "") or ""
        cats = set()
        for cat, patterns in CATEGORIZED_BROADER.items():
            for pat in patterns:
                if pat.search(text):
                    cats.add(cat)
                    break
        if len(cats) >= 2:
            multi_cat += 1
            cat_combos[frozenset(cats)] += 1

    print(f"  Multi-category: {multi_cat}", file=sys.stderr)
    for combo, count in cat_combos.most_common(10):
        print(f"    {count:5d}  {'+'.join(sorted(combo))}", file=sys.stderr)


if __name__ == "__main__":
    main()
