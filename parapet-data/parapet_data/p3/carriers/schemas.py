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
    """Repo root (holds parapet-data/ and parapet-runner/).

    This module lives at parapet-data/parapet_data/p3/carriers/schemas.py, so the
    repo root is five directories up.
    """
    here = os.path.abspath(__file__)
    for _ in range(5):
        here = os.path.dirname(here)
    return here


# Default data pool remains under parapet-runner/runs/p3_pilot for continuity with the
# already materialized ignored carrier pool. The package owns the logic; the data path
# is overrideable on every CLI.
def default_repos_root() -> str:
    """Where the AgentDojo/AgentDyn clones live (git-ignored data pool)."""
    return os.path.join(parapet_root(), "parapet-runner", "runs", "p3_pilot", "sources", "_repos")


def default_staged_out() -> str:
    """Where normalized carrier artifacts are written (git-ignored data pool)."""
    return os.path.join(parapet_root(), "parapet-runner", "runs", "p3_pilot", "staged")
