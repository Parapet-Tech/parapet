"""Peak + persistence trajectory accumulator for P3 (the decision rule the per-event
foil is meant to lose to).

Per-event D_eval gives a weak, individually-sub-threshold surface_signal for slow-burn
control-plane attacks (detector_ensemble_spec.md). The accumulator reduces a trajectory's
ordered per-event scores to two statistics, byte-identical across the 2x2 ablation cells:

- peak        : max_i s_i. This is the per-event foil lifted to the trajectory level
                (flag the trajectory iff ANY single event trips tau_event). A single
                benign spike makes peak a false positive.
- persistence : a CUSUM-style running sum of elevation above a drift k, reset at zero:
                g_t = max(0, g_{t-1} + (s_t - k)); persistence = max_t g_t. Many small,
                individually-sub-threshold elevations ACCUMULATE (slow burn), while an
                isolated benign spike decays, so persistence suppresses peak's spike FPs
                and recovers distributed signal peak misses.

This module is pure and deterministic (no detector, no IO): it operates on score lists,
so it is unit-testable and reused identically by the calibration sanity check and the gate.
"""
from __future__ import annotations

from typing import Sequence


def peak(scores: Sequence[float]) -> float:
    """max_i s_i (0.0 for an empty trajectory)."""
    return max(scores) if scores else 0.0


def persistence_cusum(scores: Sequence[float], drift: float) -> float:
    """max_t g_t where g_t = max(0, g_{t-1} + (s_t - drift)), g_0 = 0.

    drift k is the per-event 'normal' level: events above k add to the running sum,
    events below k decay it. Distributed elevation accumulates; isolated spikes decay.
    """
    g = 0.0
    best = 0.0
    for s in scores:
        g = max(0.0, g + (s - drift))
        if g > best:
            best = g
    return best


def count_over(scores: Sequence[float], tau: float) -> int:
    """Number of events at or above tau (persistence as a raw count)."""
    return sum(1 for s in scores if s >= tau)


def fraction_over(scores: Sequence[float], tau: float) -> float:
    """Fraction of events at or above tau (0.0 for an empty trajectory)."""
    return (count_over(scores, tau) / len(scores)) if scores else 0.0


def accumulate(scores: Sequence[float], *, tau: float, drift: float) -> dict:
    """Reduce a trajectory's ordered per-event scores to the accumulator statistics.

    tau: the per-event operating point (tau_event); drift: the CUSUM baseline k.
    Returns peak, persistence (CUSUM), n_over_tau, frac_over_tau, n_events.
    """
    scores = list(scores)
    return {
        "peak": peak(scores),
        "persistence": persistence_cusum(scores, drift),
        "n_over_tau": count_over(scores, tau),
        "frac_over_tau": fraction_over(scores, tau),
        "n_events": len(scores),
    }
