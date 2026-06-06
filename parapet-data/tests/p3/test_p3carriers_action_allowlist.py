"""Tests for the reviewed action allowlist loader/validator."""
from parapet_data.p3.carriers.action_allowlist import (
    load_action_set,
    load_allowlist,
    validate_allowlist,
)


def test_load_action_set_returns_only_reviewed_true():
    actions, sha = load_action_set()
    entries = load_allowlist()
    expected = {e["tool"] for e in entries if e["reviewed_action"] is True}
    assert actions == expected
    assert actions, "expected a non-empty action set"
    # reads / near-misses must not leak into the action set
    reads = {e["tool"] for e in entries if e["reviewed_action"] is False}
    assert actions.isdisjoint(reads)
    assert len(sha) == 64


def test_committed_allowlist_is_well_formed():
    entries = load_allowlist()
    assert validate_allowlist(entries) == []
    tools = [e["tool"] for e in entries]
    assert len(tools) == len(set(tools)), "duplicate tool entries"


def test_validate_catches_duplicates_and_missing_fields():
    bad = [
        {"tool": "x", "suites": ["shopping"], "candidate_mutation_signature": True,
         "candidate_in_ground_truth": False, "reviewed_action": True, "rationale": "ok"},
        {"tool": "x", "suites": [], "candidate_mutation_signature": "yes",
         "reviewed_action": True, "rationale": ""},  # dup, empty suites, bad bool, missing field, empty rationale
    ]
    problems = validate_allowlist(bad)
    joined = " | ".join(problems)
    assert "duplicate tool entry" in joined
    assert "suites must be a non-empty list" in joined
    assert "candidate_mutation_signature must be a boolean" in joined
    assert "missing field 'candidate_in_ground_truth'" in joined
    assert "rationale must be a non-empty string" in joined
