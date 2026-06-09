"""Pure structural validation for `p3-generated-trajectory/1` records.

A generation worker is UNTRUSTED. This module validates record SHAPE only and
computes NO scores and reads NO files. It enforces:

- a strict key WHITELIST at every level (any unknown key is rejected by name, which
  subsumes a denylist: it catches `score`, `d_eval`, `slow_burn`, `liveness`,
  `is_action`, `origin`, `label`, or any final-looking judgment smuggled under a new
  name). Only `s_hat` is an allowed score-like field (worker loop diagnostic).
- unique event positions, unless the trajectory's shape is in
  SHAPES_ALLOWING_DUP_POSITIONS (duplicate positions otherwise inflate persistence).
- non-empty event_text after normalization, with per-event length stats recorded for
  graft-scar and dilution analysis downstream.
- refine_meta (tau_event / epsilon / max_iter) MUST match the locked values; a
  mismatch is REJECTED, not warned. The locked values are injected (loaded from
  tau_event_lock.json by the caller) so this module stays I/O-free.

Liveness, origin, label, and detector scores are decided in validate.py against the
round-tripped carrier and the cross-family detectors, never here and never by the
worker. See local-llm/.local/multiturn/generation_worker_output_contract.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

SCHEMA_VERSION = "p3-generated-trajectory/1"

# Strict key whitelists. Any key outside these is a rejection (forbidden_key:<name>).
TOP_KEYS = frozenset(
    {"schema_version", "batch_id", "generator", "requested_class", "carrier", "shape", "T", "events", "refine_meta"}
)
CARRIER_KEYS = frozenset({"staged_path", "source_run_path", "repo"})
EVENT_KEYS = frozenset({"position", "grafted", "event_text", "quotation", "s_hat", "payload_seed_ref"})
EVENT_REQUIRED = frozenset({"position", "grafted", "event_text"})
REFINE_KEYS = frozenset({"max_iter", "epsilon", "tau_event", "drops"})
REFINE_REQUIRED = frozenset({"max_iter", "epsilon", "tau_event"})

VALID_CLASSES = frozenset({"slow_burn_attack", "benign_mention", "benign_other"})
VALID_SHAPES = frozenset({"uniform", "front", "back", "bursty", "oscillating", "late-onset"})
VALID_QUOTATION = frozenset({"quoted", "principal_discussion"})  # plus None

# Shapes that explicitly permit more than one graft at the same carrier position.
# Empty today (the mechanism exists so this is a one-line change if a shape needs it);
# even when a shape is added here, validate.py collapses same-position events before
# accumulation so persistence cannot be inflated by duplicates.
SHAPES_ALLOWING_DUP_POSITIONS: frozenset[str] = frozenset()


@dataclass
class SchemaResult:
    """Outcome of pure structural validation. reasons carry CODES only, never payloads."""
    ok: bool
    reasons: list[str] = field(default_factory=list)
    # per-event derived facts for downstream use + dilution/scar stats (no payloads):
    # [{position, grafted, len_chars, len_normalized, n_words}], in event order.
    event_stats: list[dict] = field(default_factory=list)


def normalize_text(s: str) -> str:
    """Normalization used for the non-empty check. Whitespace strip only; semantic
    normalization (NFKC etc.) is an L0 concern and not re-applied here."""
    return s.strip()


def text_stats(s: str) -> dict:
    """Length stats for graft-scar / dilution analysis. n_words is a whitespace-token
    proxy (a real tokenizer count is a pluggable downstream refinement)."""
    return {"len_chars": len(s), "len_normalized": len(normalize_text(s)), "n_words": len(s.split())}


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_record(record, *, tau_event: float, epsilon: float, max_iter: int) -> SchemaResult:
    """Validate one generated-trajectory record's SHAPE against the locked params.

    Pure: no I/O, no scoring. Fail-collecting (returns every violation found, not just
    the first) so a worker gets a complete rejection report. Locked params are injected.
    """
    reasons: list[str] = []
    event_stats: list[dict] = []

    if not isinstance(record, dict):
        return SchemaResult(False, ["record_not_object"])

    extra = set(record.keys()) - TOP_KEYS
    for k in sorted(extra):
        reasons.append(f"forbidden_key:{k}")
    missing = TOP_KEYS - set(record.keys())
    for k in sorted(missing):
        reasons.append(f"missing_key:{k}")
    # Without the required keys we cannot meaningfully check the rest.
    if missing:
        return SchemaResult(False, reasons)

    if record["schema_version"] != SCHEMA_VERSION:
        reasons.append("bad_schema_version")
    if not (isinstance(record["batch_id"], str) and record["batch_id"].strip()):
        reasons.append("bad_batch_id")
    if not (isinstance(record["generator"], str) and record["generator"].strip()):
        reasons.append("bad_generator")
    if record["requested_class"] not in VALID_CLASSES:
        reasons.append("bad_requested_class")
    if record["shape"] not in VALID_SHAPES:
        reasons.append("bad_shape")

    _validate_carrier(record["carrier"], reasons)
    _validate_refine_meta(
        record["refine_meta"], reasons, tau_event=tau_event, epsilon=epsilon, max_iter=max_iter
    )

    positions = _validate_events(record["events"], record["T"], record.get("shape"), reasons, event_stats)

    # Duplicate-position guard (persistence-inflation footgun).
    shape = record.get("shape")
    if shape not in SHAPES_ALLOWING_DUP_POSITIONS:
        seen: set[int] = set()
        dups: set[int] = set()
        for p in positions:
            if p in seen:
                dups.add(p)
            seen.add(p)
        for p in sorted(dups):
            reasons.append(f"duplicate_position:{p}")

    return SchemaResult(not reasons, reasons, event_stats)


def _validate_carrier(carrier, reasons: list[str]) -> None:
    if not isinstance(carrier, dict):
        reasons.append("carrier_not_object")
        return
    for k in sorted(set(carrier.keys()) - CARRIER_KEYS):
        reasons.append(f"carrier.forbidden_key:{k}")
    for k in sorted(CARRIER_KEYS - set(carrier.keys())):
        reasons.append(f"carrier.missing_key:{k}")
    for k in CARRIER_KEYS & set(carrier.keys()):
        if not (isinstance(carrier[k], str) and carrier[k].strip()):
            reasons.append(f"carrier.bad_{k}")


def _validate_refine_meta(meta, reasons: list[str], *, tau_event: float, epsilon: float, max_iter: int) -> None:
    if not isinstance(meta, dict):
        reasons.append("refine_meta_not_object")
        return
    for k in sorted(set(meta.keys()) - REFINE_KEYS):
        reasons.append(f"refine_meta.forbidden_key:{k}")
    for k in sorted(REFINE_REQUIRED - set(meta.keys())):
        reasons.append(f"refine_meta.missing_key:{k}")
    if REFINE_REQUIRED - set(meta.keys()):
        return
    # Must MATCH the lock (reject mismatch, do not warn).
    if not (_is_number(meta["tau_event"]) and math.isclose(meta["tau_event"], tau_event, rel_tol=1e-9, abs_tol=1e-12)):
        reasons.append("refine_meta_mismatch:tau_event")
    if not (_is_number(meta["epsilon"]) and math.isclose(meta["epsilon"], epsilon, rel_tol=1e-9, abs_tol=1e-12)):
        reasons.append("refine_meta_mismatch:epsilon")
    if not (isinstance(meta["max_iter"], int) and not isinstance(meta["max_iter"], bool) and meta["max_iter"] == max_iter):
        reasons.append("refine_meta_mismatch:max_iter")
    if "drops" in meta and not (isinstance(meta["drops"], int) and not isinstance(meta["drops"], bool) and meta["drops"] >= 0):
        reasons.append("refine_meta.bad_drops")


def _validate_events(events, T, shape, reasons: list[str], event_stats: list[dict]) -> list[int]:
    positions: list[int] = []
    if not isinstance(events, list):
        reasons.append("events_not_list")
        return positions
    if not (isinstance(T, int) and not isinstance(T, bool)):
        reasons.append("bad_T")
    elif T != len(events):
        reasons.append(f"T_mismatch:{T}!={len(events)}")

    for i, ev in enumerate(events):
        if not isinstance(ev, dict):
            reasons.append(f"event[{i}].not_object")
            continue
        for k in sorted(set(ev.keys()) - EVENT_KEYS):
            reasons.append(f"event[{i}].forbidden_key:{k}")
        for k in sorted(EVENT_REQUIRED - set(ev.keys())):
            reasons.append(f"event[{i}].missing_key:{k}")

        pos = ev.get("position")
        if isinstance(pos, int) and not isinstance(pos, bool):
            positions.append(pos)
        elif "position" in ev:
            reasons.append(f"event[{i}].position_not_int")

        if "grafted" in ev and not isinstance(ev["grafted"], bool):
            reasons.append(f"event[{i}].grafted_not_bool")

        txt = ev.get("event_text")
        if isinstance(txt, str):
            if not normalize_text(txt):
                reasons.append(f"event[{i}].empty_event_text")
            event_stats.append({"position": pos, "grafted": ev.get("grafted"), **text_stats(txt)})
        elif "event_text" in ev:
            reasons.append(f"event[{i}].event_text_not_str")

        q = ev.get("quotation")
        if q is not None and q not in VALID_QUOTATION:
            reasons.append(f"event[{i}].bad_quotation")

        sh = ev.get("s_hat")
        if sh is not None and not _is_number(sh):
            reasons.append(f"event[{i}].bad_s_hat")

        seed = ev.get("payload_seed_ref")
        if seed is not None and not isinstance(seed, str):
            reasons.append(f"event[{i}].bad_payload_seed_ref")

    return positions
