# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0

"""
Diagnose WildJailbreak eval failures.
Load the attack YAML, extract failing case IDs, and analyze why they didn't trigger.
"""

import yaml
import re
import sys
from collections import Counter, defaultdict

# Same patterns as default_l4_patterns.yaml (expanded)
L4_PATTERNS = {
    "instruction_seeding": [
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
    "role_confusion": [
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
    "deferred_authority": [
        re.compile(r"(?i)(admin|administrator|supervisor|manager) (said|told|approved|authorized)"),
        re.compile(r"(?i)I have (special |elevated |admin )(permission|access|privilege)"),
        re.compile(r"(?i)(override|bypass) (has been |)(authorized|granted|approved)"),
        re.compile(r"(?i)I('m| am) (your|the) (creator|developer|programmer|admin|owner|master)"),
        re.compile(r"(?i)this (is|was) (authorized|approved|permitted) by (your|the) (creator|developer|admin)"),
        re.compile(r"(?i)(openai|anthropic|google|meta) (has |)(authorized|approved|sanctioned|allowed)"),
        re.compile(r"(?i)(my|the) (organization|company|team|employer) (has |)(authorized|approved)"),
    ],
    "escalation_probing": [
        re.compile(r"(?i)can you (just |)(try|attempt) (to |)(bypass|ignore|skip)"),
        re.compile(r"(?i)what if (you|we) (just |)(pretend|imagine|hypothetically)"),
        re.compile(r"(?i)I (need|require) you to (be more |)(flexible|open|willing)"),
        re.compile(r"(?i)(ignore|disregard) (all |any |)(previous|prior|above|earlier) (instructions|rules|guidelines|directives|constraints)"),
        re.compile(r"(?i)forget (all |any |)(previous|prior|your) (instructions|rules|guidelines|training)"),
        re.compile(r"(?i)you (must|should|will) (now |)(obey|follow|comply|listen to) (me|my|these)"),
        re.compile(r"(?i)do not (refuse|decline|reject|filter|censor)"),
    ],
}

WEIGHTS = {
    "instruction_seeding": 0.4,
    "role_confusion": 0.5,
    "deferred_authority": 0.3,
    "escalation_probing": 0.3,
}


def score_turn(text):
    """Score a single user turn against L4 patterns. Returns (score, matched_cats)."""
    matched_cats = set()
    total = 0.0
    for cat, patterns in L4_PATTERNS.items():
        for pat in patterns:
            if pat.search(text):
                matched_cats.add(cat)
                total += WEIGHTS[cat]
                break
    return min(total, 1.0), matched_cats


def score_conversation(messages):
    """Score a full conversation using peak+accumulation. Returns details."""
    user_messages = [m["content"] for m in messages if m["role"] == "user"]
    n = len(user_messages)
    if n < 2:
        return 0.0, {}

    turn_scores = []
    turn_cats = []
    all_cats = set()
    for msg in user_messages:
        score, cats = score_turn(msg)
        turn_scores.append(score)
        turn_cats.append(cats)
        all_cats.update(cats)

    peak = max(turn_scores)
    matched_turns = sum(1 for s in turn_scores if s > 0)
    match_ratio = matched_turns / n
    distinct = len(all_cats)
    diversity = max(0, distinct - 1) * 0.15

    final = min(peak + match_ratio * 0.35 + diversity, 1.0)

    return final, {
        "peak": peak,
        "match_ratio": match_ratio,
        "matched_turns": matched_turns,
        "total_turns": n,
        "distinct_cats": distinct,
        "diversity": diversity,
        "all_cats": all_cats,
        "turn_scores": turn_scores,
        "turn_cats": turn_cats,
    }


def main():
    # Load failure IDs
    fail_ids = set()
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base, "wjb_failures.txt")) as f:
        for line in f:
            # Parse "  FAIL wjb-0005: [l4] ..."
            m = re.search(r"FAIL (wjb-\d+):", line)
            if m:
                fail_ids.add(m.group(1))

    print(f"Failure IDs loaded: {len(fail_ids)}", file=sys.stderr)

    # Load attack cases
    with open(os.path.join(base, "schema/eval/opensource_wildjailbreak_attacks.yaml")) as f:
        cases = yaml.safe_load(f)

    # Analyze failures
    score_buckets = Counter()  # score range -> count
    reason_counter = Counter()
    strategy_counter = Counter()
    zero_match_examples = []
    low_peak_examples = []
    low_persistence_examples = []

    for case in cases:
        if case["id"] not in fail_ids:
            continue

        final, details = score_conversation(case["messages"])
        desc = case["description"]

        # Extract strategy
        if "persistent" in desc:
            strategy = "persistent"
        elif "multi-cat" in desc:
            strategy = "multi-cat"
        elif "escalation" in desc:
            strategy = "escalation"
        else:
            strategy = "unknown"
        strategy_counter[strategy] += 1

        # Bucket score
        if final == 0:
            score_buckets["0.0"] += 1
        elif final < 0.3:
            score_buckets["0.01-0.29"] += 1
        elif final < 0.5:
            score_buckets["0.30-0.49"] += 1
        elif final < 0.7:
            score_buckets["0.50-0.69"] += 1
        else:
            score_buckets["0.70+"] += 1  # should have been caught

        # Diagnose reason
        if details.get("matched_turns", 0) == 0:
            reason_counter["no_pattern_match"] += 1
            if len(zero_match_examples) < 5:
                user_msgs = [m["content"] for m in case["messages"] if m["role"] == "user"]
                zero_match_examples.append((case["id"], desc, user_msgs))
        elif details.get("peak", 0) < 0.4:
            reason_counter["low_peak (<0.4)"] += 1
            if len(low_peak_examples) < 5:
                low_peak_examples.append((case["id"], desc, details))
        elif details.get("match_ratio", 0) < 0.5:
            reason_counter["low_persistence (<50%)"] += 1
            if len(low_persistence_examples) < 5:
                low_persistence_examples.append((case["id"], desc, details))
        else:
            reason_counter["other (score " + f"{final:.2f}" + ")"] += 1

    print(f"\n=== Score distribution of 144 failures ===")
    for bucket, count in sorted(score_buckets.items()):
        print(f"  {bucket:15s}  {count}")

    print(f"\n=== Strategy distribution ===")
    for strategy, count in strategy_counter.most_common():
        print(f"  {strategy:15s}  {count}")

    print(f"\n=== Failure reasons ===")
    for reason, count in reason_counter.most_common():
        print(f"  {count:4d}  {reason}")

    if zero_match_examples:
        print(f"\n=== Zero-match examples (pattern not firing) ===")
        for cid, desc, user_msgs in zero_match_examples:
            print(f"\n  {cid}: {desc}")
            for i, msg in enumerate(user_msgs):
                print(f"    T{i}: {msg[:120]}")

    if low_peak_examples:
        print(f"\n=== Low-peak examples (peak < 0.4) ===")
        for cid, desc, details in low_peak_examples:
            print(f"\n  {cid}: {desc}")
            print(f"    peak={details['peak']:.2f} match_ratio={details['match_ratio']:.2f} "
                  f"distinct={details['distinct_cats']} final={details['peak'] + details['match_ratio']*0.35 + details['diversity']:.2f}")
            print(f"    turn_scores={details['turn_scores']}")
            print(f"    turn_cats={details['turn_cats']}")

    if low_persistence_examples:
        print(f"\n=== Low-persistence examples (match_ratio < 50%) ===")
        for cid, desc, details in low_persistence_examples:
            print(f"\n  {cid}: {desc}")
            print(f"    peak={details['peak']:.2f} match_ratio={details['match_ratio']:.2f} "
                  f"distinct={details['distinct_cats']} final={details['peak'] + details['match_ratio']*0.35 + details['diversity']:.2f}")
            print(f"    turn_scores={details['turn_scores']}")


if __name__ == "__main__":
    main()
