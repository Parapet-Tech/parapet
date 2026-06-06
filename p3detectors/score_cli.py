"""CLI logic for scoring per-event tool-call arguments with D_gen (the MLX judge).

Step 1: preflight the served model, extract tool-call argument strings from staged
carriers, score each with the MLX judge, and write a git-ignored JSONL. Logic here
(tested with an injected fake judge); scripts/p3_score_events.py is the thin wrapper.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from p3carriers import schemas as carrier_schemas

from p3detectors.event_extract import SkippedEvent, events_from_staged
from p3detectors.interface import Detector
from p3detectors.mlx_judge import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REPO,
    MLXJudge,
)

PREVIEW_CHARS = 160


def _iter_staged_paths(staged_dir: str):
    for root, _, files in os.walk(staged_dir):
        for fn in sorted(files):
            if fn.endswith(".json") and fn != "index.json":
                yield os.path.join(root, fn)


def _default_out(staged_dir: str) -> str:
    parent = os.path.dirname(os.path.abspath(staged_dir.rstrip("/")))
    return os.path.join(parent, "detector_scores", "dgen_mlx_judge.jsonl")


def score_events(staged_dir: str, repos_root: str, judge: Detector, out_path: str,
                 *, limit: int = 0) -> dict:
    """Extract + score events, writing one JSONL record per scored/errored event.

    Skipped events (no_event_text) are counted but not written. limit caps the number
    of events actually sent to the judge (skips are free).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scored = skipped = errors = 0
    with open(out_path, "w") as out:
        for staged_path in _iter_staged_paths(staged_dir):
            for ev in events_from_staged(staged_path, repos_root):
                if isinstance(ev, SkippedEvent):
                    skipped += 1
                    continue
                res = judge.score(ev.event_text, ev.context)
                if res.error:
                    errors += 1
                else:
                    scored += 1
                rec = {
                    "staged_path": os.path.relpath(ev.staged_path, staged_dir),
                    "position": ev.position,
                    "call_id": ev.call_id,
                    "event_text_preview": ev.event_text[:PREVIEW_CHARS],
                    "event_text_len": len(ev.event_text),
                    **res.to_dict(),
                }
                out.write(json.dumps(rec) + "\n")
                if limit and (scored + errors) >= limit:
                    return {"scored": scored, "skipped": skipped, "errors": errors, "out": out_path}
    return {"scored": scored, "skipped": skipped, "errors": errors, "out": out_path}


def main(argv: Optional[list] = None, judge: Optional[Detector] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Score per-event tool-call argument strings from staged carriers with D_gen (MLX judge)."
    )
    ap.add_argument("--staged", default=None, help="staged carrier dir (default: p3carriers staged out)")
    ap.add_argument("--repos-root", default=None, help="carrier clones root (default: p3carriers repos root)")
    ap.add_argument("--out", default=None, help="output JSONL path (default: <pool>/scores/dgen_mlx_judge.jsonl)")
    ap.add_argument("--limit", type=int, default=0, help="cap events scored (0 = all)")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    args = ap.parse_args(argv)

    staged = args.staged or carrier_schemas.default_staged_out()
    repos_root = args.repos_root or carrier_schemas.default_repos_root()
    out_path = args.out or _default_out(staged)

    if not os.path.isdir(staged):
        print(f"staged dir not found at {staged}", file=sys.stderr)
        return 2
    if not os.path.isdir(repos_root):
        print(f"carrier clones not found at {repos_root}", file=sys.stderr)
        return 2

    judge = judge or MLXJudge(
        base_url=args.base_url, model_repo=args.model_repo, model_id=args.model_id,
    )

    ok, msg = judge.preflight()
    if not ok:
        print(f"preflight failed: {msg}", file=sys.stderr)
        print("serve the model first (in the local-llm repo): "
              "scripts/serve-model qwen3-30b-a3b-2507-mlx-4bit && "
              "scripts/verify-served-model qwen3-30b-a3b-2507-mlx-4bit", file=sys.stderr)
        return 2

    summary = score_events(staged, repos_root, judge, out_path, limit=args.limit)
    print(f"scored {summary['scored']}, skipped {summary['skipped']} (no_event_text), "
          f"errors {summary['errors']}. out: {summary['out']}")
    return 0
