#!/usr/bin/env python3
"""Fail if staged changes include non-versioned data corpus paths.

Usage:
  python scripts/check_no_data_commit.py
  python scripts/check_no_data_commit.py --all

Default mode checks staged files only (ACMR). Use --all to audit tracked files.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Never commit these corpus/output paths to GitHub.
BLOCKED_PREFIXES = (
    "schema/eval/staging/",
    "schema/eval/benign/",
    "schema/eval/malicious/",
    "schema/eval/training/",
    "schema/eval/challenges/",
    "schema/eval/heuristic_staged/",
    "TheWall/",
    "parapet-data/curated/",
    "parapet-data/curated_",
    "parapet-runner/runs/",
)


def _run_git(args: list[str]) -> list[str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stderr.strip(), file=sys.stderr)
        raise SystemExit(proc.returncode)
    return [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]


def _collect_paths(all_tracked: bool) -> list[str]:
    if all_tracked:
        tracked = set(_run_git(["ls-files"]))
        staged_deleted = set(_run_git(["diff", "--cached", "--name-only", "--diff-filter=D"]))
        return sorted(tracked - staged_deleted)
    return _run_git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"])


def _blocked(paths: list[str]) -> list[str]:
    bad: list[str] = []
    for p in paths:
        for prefix in BLOCKED_PREFIXES:
            if p.startswith(prefix):
                bad.append(p)
                break
    return sorted(set(bad))


def main() -> int:
    parser = argparse.ArgumentParser(description="Block commits of staged corpus/output data files.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Audit all tracked files instead of staged files.",
    )
    args = parser.parse_args()

    paths = _collect_paths(all_tracked=args.all)
    bad = _blocked(paths)

    if not bad:
        scope = "tracked files" if args.all else "staged changes"
        print(f"OK: no blocked data paths in {scope}.")
        return 0

    scope = "tracked files" if args.all else "staged changes"
    print(f"ERROR: blocked data paths found in {scope}:", file=sys.stderr)
    for p in bad:
        print(f"  - {p}", file=sys.stderr)

    print(
        "\nMove data under ignored paths or keep it local. "
        "Do not commit corpora/run artifacts.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
