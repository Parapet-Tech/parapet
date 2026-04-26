"""
Adjudication ledger: persistent, hash-keyed decisions about data quality.

The ledger is the memory layer for curation. Once a row is adjudicated,
the decision persists across reruns — bad data stays excluded, routing
corrections apply deterministically, and reviewed-valid rows stop
being surfaced for re-review.

Ledger file: parapet-data/adjudication/ledger.yaml
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, model_validator

from .filters import content_hash

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LedgerAction(str, Enum):
    """What to do with the row during verified sync or curation."""

    DROP = "drop"
    QUARANTINE = "quarantine"
    REROUTE_REASON = "reroute_reason"
    RELABEL_CLASS = "relabel_class"
    KEEP_HARD_CASE = "keep_hard_case"

    @property
    def excludes_from_training(self) -> bool:
        return self in (LedgerAction.DROP, LedgerAction.QUARANTINE)


class AdjudicationReason(str, Enum):
    """Why the row was flagged — must be a data-quality ground, not difficulty."""

    MISLABEL = "mislabel"
    NON_ATTACK_IN_ATTACK_SET = "non_attack_in_attack_set"
    BENIGN_CONTAMINATION = "benign_contamination"
    ROUTING_DEFECT = "routing_defect"
    EXTRACTION_DEFECT = "extraction_defect"
    MALFORMED_TEXT = "malformed_text"
    DUPLICATE_LEAKAGE = "duplicate_leakage"
    HOLDOUT_LEAKAGE = "holdout_leakage"
    OUT_OF_SCOPE = "out_of_scope"


# ---------------------------------------------------------------------------
# Entry model
# ---------------------------------------------------------------------------


class LedgerEntry(BaseModel):
    """One adjudication decision keyed by content_hash."""

    content_hash: str
    source: str
    label_at_time: Optional[str] = None
    reason_at_time: Optional[str] = None
    action: LedgerAction
    adjudication: AdjudicationReason
    relabel_to: Optional[str] = None
    reroute_to: Optional[str] = None
    rationale: Optional[str] = None
    first_seen_run: Optional[str] = None
    reviewer: Optional[str] = None
    updated_at: Optional[str] = None

    @model_validator(mode="after")
    def _action_requires_targets(self) -> LedgerEntry:
        if self.action == LedgerAction.REROUTE_REASON and not self.reroute_to:
            raise ValueError(
                "reroute_to is required when action is reroute_reason"
            )
        if self.action == LedgerAction.RELABEL_CLASS and self.relabel_to not in {
            "malicious",
            "benign",
        }:
            raise ValueError(
                "relabel_to must be 'malicious' or 'benign' when action is relabel_class"
            )
        return self


# ---------------------------------------------------------------------------
# Ledger loader
# ---------------------------------------------------------------------------


class Ledger:
    """In-memory index of adjudication decisions, keyed by content_hash."""

    def __init__(self, entries: list[LedgerEntry]) -> None:
        self._index: dict[str, LedgerEntry] = {}
        for entry in entries:
            if entry.content_hash in self._index:
                raise ValueError(
                    f"duplicate content_hash in ledger: {entry.content_hash}"
                )
            self._index[entry.content_hash] = entry

    def lookup(self, content_hash: str) -> LedgerEntry | None:
        return self._index.get(content_hash)

    def relabel_hashes(self) -> frozenset[str]:
        """Return content hashes with cross-class relabel actions."""
        return frozenset(
            content_hash
            for content_hash, entry in self._index.items()
            if entry.action == LedgerAction.RELABEL_CLASS
        )

    def __len__(self) -> int:
        return len(self._index)

    @classmethod
    def load(cls, path: Path) -> Ledger:
        """Load ledger from a YAML file. Missing file returns empty ledger."""
        if not path.exists():
            log.info("Ledger file not found at %s — using empty ledger", path)
            return cls([])
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not raw:
            return cls([])
        entries = [LedgerEntry(**item) for item in raw]
        return cls(entries)


@dataclass(frozen=True)
class LedgerRowResult:
    """Outcome of applying the ledger to a single row."""

    row: dict | None
    content_hash: str
    action: LedgerAction | None
    passed: int = 0
    dropped: int = 0
    quarantined: int = 0
    rerouted: int = 0
    relabeled: int = 0


def apply_ledger_to_row(
    row: dict,
    ledger: Ledger,
) -> LedgerRowResult:
    """Apply the ledger to one row and return the transformed outcome.

    The returned row is a shallow copy so callers can safely reuse the
    input row in multiple pipeline stages without in-place mutation.
    """
    resolved_hash = content_hash(row["content"])
    entry = ledger.lookup(resolved_hash)
    normalized = dict(row)
    normalized["content_hash"] = resolved_hash

    if entry is None or entry.action == LedgerAction.KEEP_HARD_CASE:
        return LedgerRowResult(
            row=normalized,
            content_hash=resolved_hash,
            action=entry.action if entry else None,
            passed=1,
        )

    if entry.action == LedgerAction.DROP:
        return LedgerRowResult(
            row=None,
            content_hash=resolved_hash,
            action=LedgerAction.DROP,
            dropped=1,
        )

    if entry.action == LedgerAction.QUARANTINE:
        return LedgerRowResult(
            row=None,
            content_hash=resolved_hash,
            action=LedgerAction.QUARANTINE,
            quarantined=1,
        )

    if entry.action == LedgerAction.REROUTE_REASON:
        original_reason = normalized.get("reason")
        normalized["reason"] = entry.reroute_to
        return LedgerRowResult(
            row=normalized,
            content_hash=resolved_hash,
            action=LedgerAction.REROUTE_REASON,
            passed=1,
            rerouted=0 if original_reason == entry.reroute_to else 1,
        )

    if entry.action == LedgerAction.RELABEL_CLASS:
        original_label = normalized.get("label")
        original_reason = normalized.get("reason")
        normalized["label"] = entry.relabel_to
        rerouted = 0
        if entry.reroute_to is not None:
            normalized["reason"] = entry.reroute_to
            rerouted = 0 if original_reason == entry.reroute_to else 1
        return LedgerRowResult(
            row=normalized,
            content_hash=resolved_hash,
            action=LedgerAction.RELABEL_CLASS,
            passed=1,
            rerouted=rerouted,
            relabeled=0 if original_label == entry.relabel_to else 1,
        )

    raise ValueError(f"Unhandled ledger action: {entry.action}")
