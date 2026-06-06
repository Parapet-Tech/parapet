#!/usr/bin/env python3
"""Thin CLI: score per-event tool-call argument strings from staged carriers with D_gen.

Requires the model already served at the endpoint. In the local-llm repo first run:
    scripts/serve-model qwen3-30b-a3b-2507-mlx-4bit
    scripts/verify-served-model qwen3-30b-a3b-2507-mlx-4bit
Then, from the parapet repo root:
    python p3detectors/scripts/p3_score_events.py --limit 50

All logic lives in p3detectors.score_cli (unit-tested). Writes git-ignored JSONL.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from p3detectors.score_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
