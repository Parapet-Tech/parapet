"""tau_event calibration sweep helpers for P3 D_eval.

The calibration batch itself is local/private run data. This module is public,
reproducible harness code: load labeled event JSONL, score every event through a
D_eval-like detector, and summarize per-class event-level histograms plus a
threshold sweep for the author's tau_event LOCK.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional, Sequence

from parapet_data.p3.detectors.interface import Detector, DetectorResult, EventContext

DEFAULT_THRESHOLDS = (0.20, 0.35, 0.50, 0.65, 0.80)
DEFAULT_BINS = 10


@dataclass
class CalibrationEvent:
    event_id: str
    class_label: str
    event_text: str
    trajectory_id: Optional[str] = None
    context: Optional[EventContext] = None
    source: Optional[str] = None


def event_from_record(record: dict, *, line_no: int = 0) -> CalibrationEvent:
    """Parse one calibration event record.

    Required keys: event_text and class_label. event_id defaults to ev<line_no>
    so hand-built JSONL can stay minimal during calibration assembly.
    """
    text = record.get("event_text")
    label = record.get("class_label") or record.get("class")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"line {line_no}: event_text must be a non-empty string")
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"line {line_no}: class_label must be a non-empty string")
    ctx_raw = record.get("context") or {}
    if ctx_raw and not isinstance(ctx_raw, dict):
        raise ValueError(f"line {line_no}: context must be an object")
    ctx = EventContext(
        function=ctx_raw.get("function"),
        provenance_tag=ctx_raw.get("provenance_tag"),
        position=ctx_raw.get("position"),
        prior_text=ctx_raw.get("prior_text"),
    ) if ctx_raw else None
    return CalibrationEvent(
        event_id=str(record.get("event_id") or f"ev{line_no}"),
        class_label=label,
        event_text=text,
        trajectory_id=record.get("trajectory_id"),
        context=ctx,
        source=record.get("source"),
    )


def load_events_jsonl(path: str) -> list[CalibrationEvent]:
    events: list[CalibrationEvent] = []
    with open(path) as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            events.append(event_from_record(json.loads(line), line_no=line_no))
    return events


def _member_payload(result: DetectorResult) -> dict:
    return {
        "score": result.score,
        "error": result.error,
        "family": result.family,
        "model_id": result.model_id,
        "rationale": result.rationale,
    }


def score_calibration_events(events: Sequence[CalibrationEvent], detector: Detector) -> list[dict]:
    """Score all events and return JSON-serializable records.

    If the detector exposes score_batch_with_members(), use it so the expensive
    members run once while retaining per-member diagnostics for the histogram
    report. Otherwise, store only the aggregate result.
    """
    texts = [e.event_text for e in events]
    contexts = [e.context for e in events]
    if hasattr(detector, "score_batch_with_members"):
        aggregate, per_member = detector.score_batch_with_members(texts, contexts)
    else:
        aggregate = detector.score_batch(texts, contexts)
        per_member = {}
    out: list[dict] = []
    for i, (event, result) in enumerate(zip(events, aggregate)):
        rec = {
            "event_id": event.event_id,
            "trajectory_id": event.trajectory_id,
            "class_label": event.class_label,
            "source": event.source,
            "event_text_len": len(event.event_text),
            "score": result.score,
            "error": result.error,
            "family": result.family,
            "model_id": result.model_id,
            "detector_id": result.detector_id,
            "rationale": result.rationale,
            "members": {
                member_id: _member_payload(member_results[i])
                for member_id, member_results in per_member.items()
            },
        }
        out.append(rec)
    if len(out) != len(events):
        raise ValueError(f"detector returned {len(out)} results for {len(events)} events")
    return out


def write_jsonl(path: str, records: Sequence[dict]) -> None:
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")


def _quantile(sorted_values: list[float], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def _stats(values: list[float]) -> dict:
    vals = sorted(values)
    return {
        "n": len(vals),
        "min": vals[0] if vals else None,
        "p25": _quantile(vals, 0.25),
        "median": _quantile(vals, 0.50),
        "p75": _quantile(vals, 0.75),
        "max": vals[-1] if vals else None,
    }


def _bin_index(score: float, bins: int) -> int:
    if score >= 1.0:
        return bins - 1
    if score <= 0.0:
        return 0
    return int(score * bins)


def summarize_scores(
    scored_records: Sequence[dict],
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    bins: int = DEFAULT_BINS,
) -> dict:
    """Return per-class histograms and threshold counts over event-level scores."""
    classes = sorted({r["class_label"] for r in scored_records})
    histograms = {}
    sweep = {}
    for label in classes:
        rows = [r for r in scored_records if r["class_label"] == label]
        scored = [float(r["score"]) for r in rows if r.get("score") is not None]
        errors = [r for r in rows if r.get("score") is None]
        hist = [0] * bins
        for s in scored:
            hist[_bin_index(s, bins)] += 1
        histograms[label] = {
            "n_events": len(rows),
            "n_scored": len(scored),
            "n_errors": len(errors),
            "stats": _stats(scored),
            "bins": [{"lo": i / bins, "hi": (i + 1) / bins, "count": c}
                     for i, c in enumerate(hist)],
        }
        sweep[label] = []
        for tau in thresholds:
            below = sum(1 for s in scored if s < tau)
            at_or_above = len(scored) - below
            sweep[label].append({
                "tau_event": tau,
                "below_tau": below,
                "at_or_above_tau": at_or_above,
                "n_scored": len(scored),
                "n_errors": len(errors),
                "frac_below_tau": (below / len(scored)) if scored else None,
            })
    return {
        "thresholds": list(thresholds),
        "bins": bins,
        "classes": classes,
        "histograms": histograms,
        "sweep": sweep,
    }
