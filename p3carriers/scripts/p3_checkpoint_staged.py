#!/usr/bin/env python3
"""Thin CLI: verify staged P3 carrier artifacts against their source carriers.

All logic lives in p3carriers.checkpoint (unit-tested). Run from the parapet repo
root, e.g.:
    python p3carriers/scripts/p3_checkpoint_staged.py \
        --staged parapet-runner/runs/p3_pilot/staged \
        --repos-root parapet-runner/runs/p3_pilot/sources/_repos \
        --limit 100
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from p3carriers.checkpoint import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
