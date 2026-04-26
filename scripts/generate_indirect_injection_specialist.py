#!/usr/bin/env python3
# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0

"""
Generate a publishable EN indirect_injection specialist file from staged LLMail.

Run from the parapet/ directory.

This fills the v5 mirror gap with a deterministic subset of staged LLMail rows
that are already labeled as `indirect_injection`. The generator streams the
staged input (bounded retention) and preserves deterministic ordering through
a fixed seed shuffle. Core logic lives in ``parapet_data.specialists``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "parapet-data"))

from parapet_data.specialists import (  # noqa: E402
    build_indirect_injection_cases,
    write_specialist_output,
)
from parapet_data.staged_artifact import iter_staged_rows  # noqa: E402

_SCRIPT_REL_PATH = "scripts/generate_indirect_injection_specialist.py"
_OUTPUT_TITLE = "L1 specialist training data: indirect_injection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("schema/eval/staging/en_llmail-inject-challenge_attacks_staged.jsonl"),
        help="Input staged artifact (default: schema/eval/staging/en_llmail-inject-challenge_attacks_staged.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("schema/eval/staging/en_indirect_injection_attacks_staged.jsonl"),
        help="Output staged JSONL (default: schema/eval/staging/en_indirect_injection_attacks_staged.jsonl)",
    )
    parser.add_argument("--max-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases, summary = build_indirect_injection_cases(
        iter_staged_rows(args.input),
        max_samples=args.max_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_specialist_output(
        args.output,
        cases,
        summary,
        title=_OUTPUT_TITLE,
        generator=_SCRIPT_REL_PATH,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
