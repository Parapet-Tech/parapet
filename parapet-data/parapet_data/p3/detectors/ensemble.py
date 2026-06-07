"""D_eval ensemble: conservative max over cross-family per-event members.

The evaluated detector for the viability gate (detector_ensemble_spec.md sections
1, 4, 5). It combines the D_eval members (L1 linear + embedding retrieval; an external
judge is the deferred third, section 8) into one per-event surface_signal by taking the
MAX over the members' calibrated [0,1] scores. "Conservative" because an event counts
as low-signal (slow-burn) only if NO member flags it: s_i = max_m s_{i,m} < tau_event.
The check-8 disposition and the headline 2x2 cells read this aggregate.

Cross-family is load-bearing. D_gen (the held-out cloaking target) must be a DIFFERENT
family than every D_eval member, else cloaking against D_gen and "missing" under D_eval
share a blind spot and the per-event axis goes circular (verification_design.md P7).
Two guards: the ensemble requires its own members to have DISTINCT families at
construction (a real cross-family ensemble, not two of one family); assert_cross_family
checks a given D_gen against the members and is meant to run before any generation.

FAILS CLOSED: if ANY member errors on an event, the ensemble scores that event as an
error (score=None), never a silent max over the surviving members. A silent max could
UNDERSTATE s_i and falsely label an event slow-burn, which would fabricate support for
the paper's claim. The one benign case is an all-empty event (every member returns
empty_event_text), reported as empty_event_text.
"""
from __future__ import annotations

from typing import Optional

from parapet_data.p3.detectors.interface import (
    FAMILY_ENSEMBLE,
    DetectorResult,
    EventContext,
)

DETECTOR_ID = "deval_ensemble_max"
EMPTY_ERROR = "empty_event_text"


def cross_family_ok(dgen, deval_members) -> tuple:
    """(ok, message): ok is True iff dgen.family is not among the D_eval member families.

    The anti-circularity invariant for the per-event axis: the construction detector
    must be held out from the evaluated ensemble by FAMILY, not just by instance.
    """
    deval_families = sorted({m.family for m in deval_members})
    if dgen.family in deval_families:
        return False, (
            f"cross-family violation: D_gen family {dgen.family!r} also appears in "
            f"D_eval members {deval_families}"
        )
    return True, f"cross-family OK: D_gen={dgen.family!r} disjoint from D_eval={deval_families}"


def assert_cross_family(dgen, deval_members) -> None:
    """Raise ValueError if D_gen shares a family with any D_eval member. Run before
    generation: the held-out principle is a hard gate, not a soft preference."""
    ok, msg = cross_family_ok(dgen, deval_members)
    if not ok:
        raise ValueError(msg)


class DEvalEnsemble:
    """Per-event D_eval: conservative max over distinct-family member detectors."""

    family = FAMILY_ENSEMBLE
    detector_id = DETECTOR_ID

    def __init__(self, members: list):
        members = list(members or [])
        if not members:
            raise ValueError("DEvalEnsemble requires at least one member detector")
        families = [m.family for m in members]
        if len(set(families)) != len(families):
            raise ValueError(
                f"D_eval members must be cross-family (distinct families); got {families}"
            )
        self.members = members
        self.model_id = "max(" + ",".join(m.detector_id for m in members) + ")"

    def _result(self, score, error, rationale=None) -> DetectorResult:
        return DetectorResult(
            score=score, family=self.family, model_id=self.model_id,
            detector_id=self.detector_id, rationale=rationale, error=error,
        )

    def _member_error(self, member, error: str) -> DetectorResult:
        return DetectorResult(
            score=None,
            family=member.family,
            model_id=getattr(member, "model_id", member.detector_id),
            detector_id=member.detector_id,
            error=error,
        )

    def score(self, event_text: str, context: Optional[EventContext] = None) -> DetectorResult:
        return self.score_batch([event_text], [context])[0]

    def score_members(self, texts: list, contexts=None) -> dict:
        """Raw per-member results: {detector_id -> list[DetectorResult]} aligned to texts.

        Each member is scored ONCE in batch (the L1 binary, e.g., runs a single
        subprocess for the whole batch). For the tau_event sweep and per-member
        diagnostics (detector_ensemble_spec.md section 6, per-class histograms).
        """
        n = len(texts)
        ctxs = contexts if contexts is not None else [None] * len(texts)
        if len(ctxs) != n:
            raise ValueError(f"contexts length {len(ctxs)} does not match texts length {n}")
        out = {}
        for m in self.members:
            try:
                member_results = list(m.score_batch(list(texts), ctxs))
            except Exception as exc:  # member boundary: fail closed for every event
                member_results = [
                    self._member_error(m, f"member_exception: {type(exc).__name__}") for _ in range(n)
                ]
            if len(member_results) != n:
                member_results = [
                    self._member_error(
                        m, f"member_count_mismatch: got {len(member_results)} for {n}"
                    )
                    for _ in range(n)
                ]
            out[m.detector_id] = member_results
        return out

    def score_batch_with_members(self, texts: list, contexts=None) -> tuple:
        """Return (aggregate_results, per_member_results) from one member batch pass."""
        n = len(texts)
        per_member = self.score_members(texts, contexts)  # detector_id -> list (len n)
        member_ids = [m.detector_id for m in self.members]
        results: list = [None] * n
        for i in range(n):
            row = [(mid, per_member[mid][i]) for mid in member_ids]
            errored = [(mid, r.error) for (mid, r) in row if r.score is None]
            if errored:
                # All members agree an event is empty -> report empty, not a member error.
                if len(errored) == len(row) and all(e == EMPTY_ERROR for (_, e) in errored):
                    results[i] = self._result(None, EMPTY_ERROR)
                else:
                    detail = "; ".join(f"{mid}={err}" for (mid, err) in errored)
                    results[i] = self._result(None, f"member_error: {detail}")
                continue
            best_id, best = max(row, key=lambda kv: kv[1].score)
            breakdown = ", ".join(f"{mid}={r.score:.4f}" for (mid, r) in row)
            results[i] = self._result(
                best.score, None,
                rationale=f"max {best.score:.4f} from {best_id}; members: {breakdown}",
            )
        return results, per_member

    def score_batch(self, texts: list, contexts=None) -> list:
        results, _ = self.score_batch_with_members(texts, contexts)
        return results
