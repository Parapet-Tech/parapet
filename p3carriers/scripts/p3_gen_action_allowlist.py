#!/usr/bin/env python3
"""Thin CLI: bootstrap/regenerate the reviewed P3 action-tool allowlist.

All logic lives in p3carriers.allowlist_generator (unit-tested). Run from the
parapet repo root, e.g.:
    python p3carriers/scripts/p3_gen_action_allowlist.py --check
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from p3carriers.allowlist_generator import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
