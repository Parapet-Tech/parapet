"""Extract per-event text from staged P3 carriers.

Two span types (detector_ensemble_spec.md section 4: event text is "message text or
tool-output content at that position"):

- arg_values  : the string VALUES an agent passed to a tool call (the agent's action).
                Tool names and non-string args are context, never event text.
- tool_output : the content of a tool-result message (the UNTRUSTED channel where
                injection payloads live). Its position/liveness is inherited from the
                call it answers (matched by tool_call_id), so a dead injection-bearing
                output is recoverable as a benign_mention event.

Each event records its span_type so per-class histograms can be audited separately
(reviewer ask, 2026-06-07). Default span_types=("arg_values",) preserves the step-1
behavior the D_gen scoring path relies on; the calibration batch opts into tool_output.

Reads the staged artifact (the pool unit) and resolves its source carrier run for the
text, which normalization does not retain on the artifact.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence, Union

from parapet_data.p3.detectors.interface import EventContext

SPAN_ARG_VALUES = "arg_values"
SPAN_TOOL_OUTPUT = "tool_output"
DEFAULT_SPAN_TYPES = (SPAN_ARG_VALUES,)


@dataclass
class ExtractedEvent:
    event_text: str
    context: EventContext
    staged_path: str
    position: Optional[int]
    call_id: Optional[str]
    span_type: str = SPAN_ARG_VALUES


@dataclass
class SkippedEvent:
    reason: str
    staged_path: str
    position: Optional[int]
    span_type: str = SPAN_ARG_VALUES


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
    """Ordered tool calls with args, mirroring parapet_data.p3.carriers normalization order."""
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


def _tool_output_spans(
    messages: list, call_position: dict
) -> Iterator[tuple]:
    """Yield (position, call_id, function, content) for each tool-result message.

    position/function are inherited from the call the output answers (matched by
    tool_call_id via call_position); position is None for an unmatched output.
    """
    for m in messages:
        if m.get("role") != "tool":
            continue
        content = m.get("content")
        call_id = m.get("tool_call_id")
        pos, fn = call_position.get(call_id, (None, None))
        yield pos, call_id, fn, content


def events_from_staged(
    staged_path: str,
    repos_root: str,
    *,
    span_types: Sequence[str] = DEFAULT_SPAN_TYPES,
) -> Iterator[Union[ExtractedEvent, SkippedEvent]]:
    """Yield one ExtractedEvent or SkippedEvent per selected span in the carrier.

    span_types selects which span kinds to emit (arg_values and/or tool_output). With
    both, arg_values are yielded first (in call order), then tool_output spans (in
    message order). Silently yields nothing if the staged artifact lacks a run_path or
    its source carrier is missing/unreadable (the checkpoint is the integrity gate for
    that; here we just skip).
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

    messages = run.get("messages", [])
    calls = _source_tool_calls(messages)

    if SPAN_ARG_VALUES in span_types:
        for tc in calls:
            pos = tc["position"]
            values = string_values(tc.get("args"))
            if not values:
                yield SkippedEvent("no_event_text", staged_path, pos, SPAN_ARG_VALUES)
                continue
            ctx = EventContext(function=tc.get("function"), position=pos)
            yield ExtractedEvent(
                "\n".join(values), ctx, staged_path, pos, tc.get("call_id"), SPAN_ARG_VALUES,
            )

    if SPAN_TOOL_OUTPUT in span_types:
        call_position = {c["call_id"]: (c["position"], c["function"]) for c in calls if c["call_id"]}
        for pos, call_id, fn, content in _tool_output_spans(messages, call_position):
            if not isinstance(content, str) or not content.strip():
                yield SkippedEvent("no_event_text", staged_path, pos, SPAN_TOOL_OUTPUT)
                continue
            ctx = EventContext(function=fn, provenance_tag="tool_output", position=pos)
            yield ExtractedEvent(content, ctx, staged_path, pos, call_id, SPAN_TOOL_OUTPUT)
