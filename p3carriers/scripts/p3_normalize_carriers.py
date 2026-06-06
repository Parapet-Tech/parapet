#!/usr/bin/env python3
"""Thin CLI: normalize AgentDojo/AgentDyn runs into P3 carrier artifacts.

All logic lives in p3carriers.normalize (unit-tested). Run from the parapet repo
root, e.g.:
    python p3carriers/scripts/p3_normalize_carriers.py --clean-only --suites github
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from p3carriers.normalize import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
