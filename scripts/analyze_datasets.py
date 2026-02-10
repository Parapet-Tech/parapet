# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0

"""Analyze opensource eval datasets for trust-span wrapping eligibility."""
import yaml
import os

EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "schema", "eval")

def is_ascii_safe(s):
    """Content must not require JSON escaping beyond identity."""
    for ch in s:
        code = ord(ch)
        if code < 32 or code > 126:
            return False
        if ch == '"' or ch == '\\':
            return False
    return True

for f in sorted(os.listdir(EVAL_DIR)):
    if not f.startswith("opensource"):
        continue
    path = os.path.join(EVAL_DIR, f)
    with open(path, encoding="utf-8") as fh:
        cases = yaml.safe_load(fh)

    total = len(cases)
    ascii_safe = 0
    under_490 = 0
    both = 0
    for c in cases:
        content = c.get("content", "")
        safe = is_ascii_safe(content)
        short = len(content) <= 490
        if safe:
            ascii_safe += 1
        if short:
            under_490 += 1
        if safe and short:
            both += 1

    label = cases[0].get("label", "?") if cases else "?"
    print(f"{f}: {total} total, ascii_safe={ascii_safe}, <=490ch={under_490}, eligible={both} ({label})")
