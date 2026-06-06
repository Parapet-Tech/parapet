"""Extract per-event text (tool-call argument strings) from staged P3 carriers.

Step 1 scope (detector_ensemble_spec.md section 9): score tool-call ARGUMENT VALUE
strings only. Tool names and other metadata are context, never event text. Events
with no string argument values are skipped with reason no_event_text.

Reads the staged artifact (the pool unit) and resolves its source carrier run for the
args, which normalization does not retain on the artifact.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Union

from p3detectors.interface import EventContext


@dataclass
class ExtractedEvent:
    event_text: str
    context: EventContext
    staged_path: str
    position: int
    call_id: Optional[str]


@dataclass
class SkippedEvent:
    reason: str
    staged_path: str
    position: int


def string_values(args: Any) -> list[str]:
    """Collect string leaf VALUES from a tool-call args structure (names excluded)."""
    out: list[str] = []
    if isinstance(args, str):
        if args.strip():
            out.append(args)
    elif isinstance(args, dict):
        for v in args.values():
            out.extend(string_values(v))
    elif isinstance(args, list):
        for v in args:
            out.extend(string_values(v))
    return out


def _source_tool_calls(messages: list) -> list:
    """Ordered tool calls with args, mirroring p3carriers normalization order."""
    seq = []
    for mi, m in enumerate(messages):
        for tc in (m.get("tool_calls") or []):
            seq.append({
                "position": len(seq),
                "message_index": mi,
                "function": tc.get("function"),
                "call_id": tc.get("id"),
                "args": tc.get("args"),
            })
    return seq


def events_from_staged(
    staged_path: str,
    repos_root: str,
) -> Iterator[Union[ExtractedEvent, SkippedEvent]]:
    """Yield one ExtractedEvent or SkippedEvent per tool call in the carrier.

    Silently yields nothing if the staged artifact lacks a run_path or its source
    carrier is missing/unreadable (the checkpoint is the integrity gate for that;
    here we just skip).
    """
    with open(staged_path) as fh:
        staged = json.load(fh)
    run_path = staged.get("source", {}).get("run_path")
    if not run_path:
        return
    src_path = os.path.join(repos_root, run_path)
    try:
        with open(src_path) as fh:
            run = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return
    for tc in _source_tool_calls(run.get("messages", [])):
        pos = tc["position"]
        values = string_values(tc.get("args"))
        if not values:
            yield SkippedEvent("no_event_text", staged_path, pos)
            continue
        ctx = EventContext(function=tc.get("function"), position=pos)
        yield ExtractedEvent("\n".join(values), ctx, staged_path, pos, tc.get("call_id"))
