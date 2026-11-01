"""Calibration contract for a specialist paired with frozen generalist scores."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _retention(kept: int, total: int) -> dict[str, Any]:
    if not total:
        return {"kept": kept, "total": total, "retention": None, "wilson95": None}
    z = 1.959963984540054
    p = kept / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return {"kept": kept, "total": total, "retention": p, "wilson95": [max(0., center - half), min(1., center + half)]}


def constrained_retention(
    rows: Sequence[Mapping[str, Any]], *, retention_floor: float = .99,
    generalist_threshold: float = 0., generalist_block_alone: float = 3.,
) -> dict[str, Any]:
    """Maximize benign relief over distinct specialist scores; ties use lowest t.

    Each row supplies label (benign/malicious), specialist_score, generalist_score.
    Empty calibration or no generalist-blocked attacks cannot define retention.
    With finite scores, the lowest candidate retains every blocked attack, so a
    feasible threshold always exists for a floor in [0, 1].
    """
    if not math.isfinite(retention_floor) or not 0 <= retention_floor <= 1:
        raise ValueError("retention floor must be in [0, 1]")
    if not all(map(math.isfinite, [generalist_threshold, generalist_block_alone])) or generalist_block_alone < generalist_threshold:
        raise ValueError("invalid generalist thresholds")
    for row in rows:
        if row["label"] not in ("malicious", "benign"):
            raise ValueError("invalid calibration label")
        if not all(math.isfinite(row[k]) for k in ("generalist_score", "specialist_score")):
            raise ValueError("calibration scores must be finite")
    attacks = [r for r in rows if r["label"] == "malicious" and r["generalist_score"] >= generalist_threshold]
    band = [r for r in attacks if r["generalist_score"] < generalist_block_alone]
    best = None
    for threshold in sorted({r["specialist_score"] for r in rows}):
        kept = sum(r["generalist_score"] >= generalist_block_alone or r["specialist_score"] >= threshold for r in attacks)
        if not attacks or kept / len(attacks) < retention_floor:
            continue
        relief = sum(r["label"] == "benign" and generalist_threshold <= r["generalist_score"] < generalist_block_alone and r["specialist_score"] < threshold for r in rows)
        if best is None or relief > best["relief_count"]:
            best = {"threshold": threshold, "relief_count": relief,
                    "total": _retention(kept, len(attacks)),
                    "deciding_band": _retention(sum(r["specialist_score"] >= threshold for r in band), len(band))}
    if best is None:
        raise ValueError("no feasible threshold: calibration requires generalist-blocked attacks")
    return best


def calibrate_pinned_scores(config: Any, validation_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Verify frozen row alignment and score bytes before selecting a threshold."""
    path = Path(config.calibration_scores)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != config.calibration_scores_sha256:
        raise ValueError("calibration score sha256 mismatch")
    scores = [json.loads(line) for line in data.splitlines() if line.strip()]
    expected = {r["id"]: r["label"] for r in validation_rows}
    actual = {r["id"]: r["label"] for r in scores}
    if len(expected) != len(validation_rows) or len(actual) != len(scores) or actual != expected:
        raise ValueError("calibration scores must match validation ids and labels exactly")
    result = constrained_retention(scores, retention_floor=config.retention_floor,
        generalist_threshold=config.generalist_threshold, generalist_block_alone=config.generalist_block_alone)
    return dict(result, scores_sha256=config.calibration_scores_sha256)
