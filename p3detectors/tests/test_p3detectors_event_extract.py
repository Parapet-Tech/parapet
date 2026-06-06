"""Tests for tool-call argument extraction from staged carriers."""
import json

from p3detectors.event_extract import (
    ExtractedEvent,
    SkippedEvent,
    events_from_staged,
    string_values,
)


def _make(tmp_path, tool_calls):
    """tool_calls: list of (function, args). Writes a source run + staged artifact."""
    run_rel = "AgentDyn/runs/m1/shopping/user_task_1/none/none.json"
    src = tmp_path / "repos" / run_rel
    src.parent.mkdir(parents=True, exist_ok=True)
    messages = [{
        "role": "assistant",
        "tool_calls": [{"function": f, "id": f"c{i}", "args": a}],
    } for i, (f, a) in enumerate(tool_calls)]
    src.write_text(json.dumps({"messages": messages}))

    staged_dir = tmp_path / "staged"
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_path = staged_dir / "art.json"
    staged_path.write_text(json.dumps({"source": {"run_path": run_rel}}))
    return str(staged_path), str(tmp_path / "repos")


def test_extract_arg_string_values(tmp_path):
    sp, repos = _make(tmp_path, [("send_money", {"recipient": "trustme@gmail.com", "amount": 100})])
    evs = list(events_from_staged(sp, repos))
    assert len(evs) == 1
    ev = evs[0]
    assert isinstance(ev, ExtractedEvent)
    assert ev.event_text == "trustme@gmail.com"   # amount (int) excluded
    assert ev.context.function == "send_money"     # function in context
    assert "send_money" not in ev.event_text       # tool name not scored as text
    assert ev.position == 0 and ev.call_id == "c0"


def test_multiple_string_values_joined(tmp_path):
    sp, repos = _make(tmp_path, [("post", {"to": "a@x.com", "body": "hello"})])
    ev = list(events_from_staged(sp, repos))[0]
    assert ev.event_text == "a@x.com\nhello"


def test_skip_no_string_args(tmp_path):
    sp, repos = _make(tmp_path, [("get_balance", {"account_id": 42})])
    evs = list(events_from_staged(sp, repos))
    assert len(evs) == 1 and isinstance(evs[0], SkippedEvent)
    assert evs[0].reason == "no_event_text" and evs[0].position == 0


def test_missing_source_yields_nothing(tmp_path):
    sp, repos = _make(tmp_path, [("send_money", {"recipient": "x@y.com"})])
    # point at a repos root with no carrier
    evs = list(events_from_staged(sp, str(tmp_path / "empty")))
    assert evs == []


def test_string_values_nested():
    assert string_values({"a": "x", "b": {"c": "y", "d": 3}, "e": ["z"]}) == ["x", "y", "z"]
    assert string_values({"n": 1, "blank": "  "}) == []  # whitespace-only excluded
