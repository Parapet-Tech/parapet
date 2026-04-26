# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0

"""Specialist training-data generators.

Streaming utilities for turning staged attack artifacts into per-reason
specialist training files. Bounded retention: only deduped publishable
candidates are held in memory, never the full input.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from parapet_data.staged_artifact import write_staged_rows

INDIRECT_INJECTION_REASON = "indirect_injection"
INDIRECT_INJECTION_LAYER = "l3_inbound"
INDIRECT_INJECTION_DESCRIPTION = "LLMail staged indirect injection"
INDIRECT_INJECTION_ID_PREFIX = "indirect-llmail"

_LLMAIL_PUBLISHABLE_TOKENS: tuple[str, ...] = (
    "contact@",
    "contact[at]",
    "send email",
    "send an email",
    "email to",
    "sendmail",
    "tool call",
    "api_call",
    "api call",
    'body "confirmation"',
    "body is the string",
    "profit forecast",
)

_MOJIBAKE_MARKERS: tuple[str, ...] = ("â", "ðŸ", "\ufffd")

_MIN_CONTENT_LEN = 120
_MAX_CONTENT_LEN = 12000
_MAX_NON_ASCII_RATIO = 0.06


def is_publishable_candidate(row: dict[str, Any]) -> bool:
    """Return True when an LLMail row is publishable indirect-injection content."""
    if row.get("label") != "malicious":
        return False
    if row.get("reason") != INDIRECT_INJECTION_REASON:
        return False

    text = str(row.get("content", "")).strip()
    if not text:
        return False
    if len(text) < _MIN_CONTENT_LEN or len(text) > _MAX_CONTENT_LEN:
        return False

    lowered = text.lower()
    if not any(token in lowered for token in _LLMAIL_PUBLISHABLE_TOKENS):
        return False

    if any(marker in text for marker in _MOJIBAKE_MARKERS):
        return False

    non_ascii = sum(1 for char in text if ord(char) > 127)
    if non_ascii / len(text) > _MAX_NON_ASCII_RATIO:
        return False

    return True


def content_hash(text: str) -> str:
    """Stable sha256 of the trimmed content (the dedup key)."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _build_indirect_injection_case(row: dict[str, Any]) -> dict[str, Any]:
    digest = content_hash(str(row["content"]))
    return {
        "id": f"{INDIRECT_INJECTION_ID_PREFIX}-{digest[:12]}",
        "layer": INDIRECT_INJECTION_LAYER,
        "label": "malicious",
        "description": INDIRECT_INJECTION_DESCRIPTION,
        "content": row["content"],
    }


def build_indirect_injection_cases(
    rows: Iterable[dict[str, Any]],
    max_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stream rows into deduped, deterministically shuffled specialist cases.

    Memory bound: ``O(publishable candidates)``. Input rows are not retained
    after they are filtered; only deduped survivors stay in memory.
    """
    input_rows = 0
    reason_rows = 0
    publishable = 0
    deduped: dict[str, dict[str, Any]] = {}

    for row in rows:
        input_rows += 1
        if row.get("reason") == INDIRECT_INJECTION_REASON:
            reason_rows += 1
        if not is_publishable_candidate(row):
            continue
        publishable += 1
        deduped.setdefault(content_hash(str(row["content"])), row)

    survivors = list(deduped.values())
    random.Random(seed).shuffle(survivors)
    selected = survivors[:max_samples]

    cases = [_build_indirect_injection_case(row) for row in selected]
    summary = {
        "input_rows": input_rows,
        "reason_rows": reason_rows,
        "publishable_candidates": publishable,
        "deduped_candidates": len(deduped),
        "selected_rows": len(cases),
        "source_distribution": dict(
            sorted(Counter(row.get("source", "") for row in selected).items())
        ),
        "seed": seed,
        "max_samples": max_samples,
    }
    return cases, summary


def write_specialist_output(
    out_path: Path,
    cases: Iterable[dict[str, Any]],
    summary: dict[str, Any],
    *,
    title: str,
    generator: str,
) -> Path:
    """Write JSONL cases plus a ``.summary.json`` sidecar; return sidecar path."""
    write_staged_rows(out_path, cases)

    sidecar = out_path.with_suffix(out_path.suffix + ".summary.json")
    sidecar.write_text(
        json.dumps(
            {
                "title": title,
                "generator": generator,
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return sidecar
