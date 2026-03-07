"""
Data quality filters: attack signature detection, dedup, label filtering.

These run on benign candidates BEFORE sampling to prevent contamination.
All filtering decisions are logged so the manifest captures what was dropped.
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

# ---------------------------------------------------------------------------
# Attack signature patterns
# ---------------------------------------------------------------------------

ATTACK_SIGNATURE_PATTERNS: list[re.Pattern] = [
    # HackAPrompt-style canary payloads
    re.compile(r"(?i)\bi\s+have\s+been\s+p[\W_]*w[\W_]*n[\W_]*e[\W_]*d\b"),
    re.compile(r"(?i)\bp[\W_]*w[\W_]*n[\W_]*e[\W_]*d\b"),
    # Instruction-override directives
    re.compile(
        r"(?is)\b(ignore|disregard|forget)\b.{0,60}\b(previous|prior|all|above)\b"
        r".{0,60}\b(instruction|instructions|prompt|rules?)\b"
    ),
    # System prompt exfiltration
    re.compile(
        r"(?is)\b(reveal|show|print|dump|leak|expose)\b.{0,80}\b"
        r"(system\s*prompt|developer\s*message|hidden\s*prompt|instructions?)\b"
    ),
    # Secret/canary extraction
    re.compile(
        r"(?is)\b(output|return|write)\b.{0,80}\b(canary|secret|api key|password)\b"
    ),
]

SYSTEM_LIKE_RE = re.compile(
    r"(?i)(^\s*system\s*:|\byou are\b|\brole\s*:\s*system\b|"
    r"\bassistant\b|\binstructions?\b|\bact as\b)"
)


def looks_like_attack(text: str) -> bool:
    """Heuristic check for obvious prompt-injection payload text."""
    t = text.strip()
    if not t:
        return False
    return any(p.search(t) for p in ATTACK_SIGNATURE_PATTERNS)


def is_system_like(text: str) -> bool:
    """Check if text resembles a system prompt."""
    return bool(SYSTEM_LIKE_RE.search(text))


# ---------------------------------------------------------------------------
# Label filtering
# ---------------------------------------------------------------------------


def passes_label_filter(
    row: dict,
    label_filter: dict[str, object] | None,
) -> bool:
    """Check if a row passes the label filter.

    label_filter is {"column": str, "allowed": list[value]} or None (pass all).
    Raises ValueError on malformed filter config to fail closed — a typo
    in the filter must not silently disable filtering.
    """
    if label_filter is None:
        return True
    col = label_filter.get("column")
    allowed = label_filter.get("allowed")
    if col is None or allowed is None:
        raise ValueError(
            f"Malformed label_filter: must have 'column' and 'allowed' keys, "
            f"got {sorted(label_filter.keys())}"
        )
    if not col or not allowed:
        raise ValueError(
            f"label_filter 'column' and 'allowed' must be non-empty, "
            f"got column={col!r}, allowed={allowed!r}"
        )
    val = row.get(col)
    # Normalize: compare string representations for mixed-type labels
    return str(val) in {str(a) for a in allowed}


# ---------------------------------------------------------------------------
# Content deduplication
# ---------------------------------------------------------------------------


def content_hash(text: str) -> str:
    """SHA256 hash of content for dedup and provenance.

    Strips leading/trailing whitespace before hashing so that
    trailing newlines or padding from YAML parsing don't create
    phantom-distinct hashes.
    """
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


class ContentDeduplicator:
    """Track seen content hashes to prevent duplicates across sources.

    Also supports cross-set dedup: register attack hashes first, then
    reject any benign sample whose content matches an attack.
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self._attack_hashes: set[str] = set()
        self.duplicates_dropped: int = 0
        self.cross_contamination_dropped: int = 0

    def register_attack_hashes(self, hashes: Sequence[str]) -> None:
        """Register attack content hashes for cross-set dedup."""
        self._attack_hashes.update(hashes)

    def is_duplicate(self, text: str) -> bool:
        """Check if content has been seen before (within benign set)."""
        h = content_hash(text)
        if h in self._seen:
            self.duplicates_dropped += 1
            return True
        self._seen.add(h)
        return False

    def is_cross_contaminated(self, text: str) -> bool:
        """Check if benign content matches a known attack."""
        h = content_hash(text)
        if h in self._attack_hashes:
            self.cross_contamination_dropped += 1
            return True
        return False

    def check(self, text: str) -> bool:
        """Combined check: returns True if the sample should be KEPT."""
        if self.is_cross_contaminated(text):
            return False
        if self.is_duplicate(text):
            return False
        return True
