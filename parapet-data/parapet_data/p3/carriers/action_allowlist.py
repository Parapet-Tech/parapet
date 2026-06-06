"""Load and validate the reviewed P3 action-tool allowlist.

The allowlist (configs/p3_action_tools.yaml) is the authority for which tools are
state-mutating "actions" in the liveness label, NOT ground_truth() union and NOT
AST mutation inference. See that file's header for the inclusion principle.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required for parapet_data.p3.carriers.action_allowlist") from exc

REQUIRED_FIELDS = (
    "tool",
    "suites",
    "candidate_mutation_signature",
    "candidate_in_ground_truth",
    "reviewed_action",
    "rationale",
)


def default_allowlist_path() -> str:
    # configs/ is a sibling of carriers/ under parapet_data/p3/
    p3_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(p3_dir, "configs", "p3_action_tools.yaml")


def load_allowlist(path: str | None = None) -> list[dict[str, Any]]:
    """Return the raw list of tool entries from the allowlist YAML."""
    path = path or default_allowlist_path()
    with open(path) as fh:
        doc = yaml.safe_load(fh)
    return doc["tools"]


def load_action_set(path: str | None = None) -> tuple[set[str], str]:
    """Return (set of tools with reviewed_action true, sha256 of the file).

    Only reviewed_action: true tools are returned; reads and near-misses are
    excluded. The sha travels into every normalized artifact for provenance.
    """
    path = path or default_allowlist_path()
    with open(path, "rb") as fh:
        raw = fh.read()
    doc = yaml.safe_load(raw)
    actions = {t["tool"] for t in doc["tools"] if t.get("reviewed_action") is True}
    return actions, hashlib.sha256(raw).hexdigest()


def validate_allowlist(entries: list[dict[str, Any]]) -> list[str]:
    """Return a list of human-readable problems; empty means valid.

    Checks: no duplicate tools, every entry has all required fields with the
    right types (suites a non-empty list, the three candidate/review flags
    booleans, rationale a non-empty string).
    """
    problems: list[str] = []
    seen: set[str] = set()
    for i, e in enumerate(entries):
        tool = e.get("tool", f"<entry {i}>")
        for field in REQUIRED_FIELDS:
            if field not in e:
                problems.append(f"{tool}: missing field '{field}'")
        if e.get("tool") in seen:
            problems.append(f"{tool}: duplicate tool entry")
        seen.add(e.get("tool"))
        if not isinstance(e.get("suites"), list) or not e.get("suites"):
            problems.append(f"{tool}: suites must be a non-empty list")
        for flag in ("candidate_mutation_signature", "candidate_in_ground_truth", "reviewed_action"):
            if not isinstance(e.get(flag), bool):
                problems.append(f"{tool}: {flag} must be a boolean")
        rationale = e.get("rationale")
        if not (isinstance(rationale, str) and rationale.strip()):
            problems.append(f"{tool}: rationale must be a non-empty string")
    return problems
