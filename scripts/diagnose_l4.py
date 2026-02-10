# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0

"""Diagnose L4 FP/FN by checking which patterns match on specific cases."""

import yaml
import re
import sys

PATTERNS = {
    "instruction_seeding (0.4)": [
        r"(?i)remember this for later",
        r"(?i)in my next message",
        r"(?i)when I (say|type|send) .{0,30}",
        r"(?i)later I will (give|send|provide)",
        r"(?i)keep this in mind",
        r"(?i)store this (instruction|command|rule)",
    ],
    "role_confusion (0.5)": [
        r"(?i)you are now (in |)(developer|admin|debug|sudo|root|god) mode",
        r"(?i)switch to (unrestricted|unfiltered|jailbreak|DAN)",
        r"(?i)from now on.{0,30}(ignore|forget|disregard)",
        r"(?i)your new (role|persona|identity|name) is",
    ],
    "deferred_authority (0.3)": [
        r"(?i)(admin|administrator|supervisor|manager) (said|told|approved|authorized)",
        r"(?i)I have (special |elevated |admin )(permission|access|privilege)",
        r"(?i)(override|bypass) (has been |)(authorized|granted|approved)",
    ],
    "escalation_probing (0.3)": [
        r"(?i)can you (just |)(try|attempt) (to |)(bypass|ignore|skip)",
        r"(?i)what if (you|we) (just |)(pretend|imagine|hypothetically)",
        r"(?i)I (need|require) you to (be more |)(flexible|open|willing)",
    ],
}


def check_case(case):
    """Check which L4 patterns match on each user turn."""
    messages = case.get("messages", [])
    user_turns = [(i, m) for i, m in enumerate(messages) if m["role"] == "user"]

    print(f'=== {case["id"]} ({len(user_turns)} user turns) ===')

    turn_scores = []
    for idx, (i, m) in enumerate(user_turns):
        content = m["content"]
        matches = []
        score = 0.0
        for cat, pats in PATTERNS.items():
            for p in pats:
                if re.search(p, content):
                    weight = float(cat.split("(")[1].rstrip(")"))
                    matches.append(cat)
                    score += weight
                    break
        turn_scores.append(min(score, 1.0))
        preview = content[:100].replace("\n", " ")
        if matches:
            print(f"  turn {idx}: score={score:.1f} MATCH {matches}")
            print(f"    text: {preview}")
        else:
            print(f"  turn {idx}: score=0.0")

    # Compute weighted cumulative
    n = len(turn_scores)
    if n <= 1:
        cum = turn_scores[0] if turn_scores else 0.0
    else:
        sw = 0.0
        wt = 0.0
        for i, s in enumerate(turn_scores):
            w = 1.0 + (i / (n - 1))
            sw += s * w
            wt += w
        cum = sw / wt

    # Check escalation gradient
    esc_bonus = 0.0
    if n >= 3:
        count = 0
        for i in range(n - 1, 0, -1):
            if turn_scores[i] > turn_scores[i - 1] and turn_scores[i] > 0.0:
                count += 1
            else:
                break
        if count >= 2:
            esc_bonus = 0.2

    final = min(cum + esc_bonus, 1.0)
    print(f"  => cum={cum:.3f} + esc={esc_bonus:.1f} = final={final:.3f} (threshold=0.7)")
    verdict = "BLOCK" if final >= 0.7 else "ALLOW"
    print(f"  => {verdict}")
    print()


def main():
    filepath = sys.argv[1]
    case_ids = sys.argv[2:] if len(sys.argv) > 2 else None

    with open(filepath, encoding="utf-8") as f:
        cases = yaml.safe_load(f.read())

    if case_ids:
        cases = [c for c in cases if c["id"] in case_ids]
    else:
        # Show first 10
        cases = cases[:10]

    for case in cases:
        check_case(case)


if __name__ == "__main__":
    main()
