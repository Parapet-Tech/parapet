"""Shared constants, the locked liveness rule, and default data paths for P3 carriers.

Reference: local-llm/.local/multiturn/generation_distribution_spec.md (section 4,
2026-06-06 addendum) and carrier_integration_research.md in that directory.
"""
from __future__ import annotations

import os

# Liveness rule, LOCKED 2026-06-06 (author): linear-downstream.
LIVENESS_RULE = (
    "Event i is live iff a reviewed suite-action tool call appears at "
    "position >= i in the executed linear trace."
)

SCHEMA_VERSION = "p3-carrier-normalized/1"

# Carrier suites we use, keyed by the source repo dir holding the clone.
REPO_SUITES = {
    "agentdojo": {"workspace", "banking", "slack", "travel"},
    "AgentDyn": {"shopping", "dailylife", "github"},
}

# Method names treated as in-place mutations for the candidate-only mutation
# signal (deliberately imprecise; the reviewed allowlist is the authority).
MUT_METHODS = frozenset({
    "append", "extend", "insert", "remove", "pop", "clear",
    "update", "add", "discard", "setdefault", "popitem",
})


def parapet_root() -> str:
    """Repo root that holds p3carriers/ and parapet-runner/."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_repos_root() -> str:
    """Where the AgentDojo/AgentDyn clones live (git-ignored data pool).

    Kept under parapet-runner/runs/p3_pilot to avoid moving ~170k already-staged
    files; overridable on every CLI. The clones are regenerable via git and the
    staged pool via the normalizer, so this is data, not source.
    """
    return os.path.join(parapet_root(), "parapet-runner", "runs", "p3_pilot", "sources", "_repos")


def default_staged_out() -> str:
    """Where normalized carrier artifacts are written (git-ignored data pool)."""
    return os.path.join(parapet_root(), "parapet-runner", "runs", "p3_pilot", "staged")
