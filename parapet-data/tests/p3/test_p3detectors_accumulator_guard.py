"""Cycle-1 accumulator regression guard.

Standing rung-2 differential guard for the public accumulator math. Built and
verified sensor-side (cycle1_accumulator_guard, rung-2 reconciliation lane);
landed by local-lab as PR slice #1 of the rung-2 integration plan
(Option 3 / Cycle 1: guard the tracked accumulator subset only; ordering and
median-drift stay research-lane, no tracked home).

CONTENT-SAFETY: synthetic float fixtures only; imports ONLY the public accumulator
module. No receipts, no corpora, no models. Safe for the public package + CI.

These are DIFFERENTIAL oracles: each fixture has a known-correct AND a known-wrong
answer, engineered so a real regression (broken g_0 reset, wrong peak, strict-vs-
inclusive tau boundary) trips the assertion. A clean pass therefore means the
tracked math still matches the rung-2-validated behavior, not just "returns
something". Mirrors the rung2_dark_path_fixtures.py style (hermetic, exit-code in
script mode; test_* functions for pytest/CI collection).
"""
from parapet_data.p3.detectors.accumulator import (
    accumulate,
    count_over,
    fraction_over,
    peak,
    persistence_cusum,
)


def _no_floor_cusum(scores, drift):
    """Deliberately-buggy reference: a CUSUM WITHOUT the g_t = max(0, .) reset.

    This is the most likely regression of persistence_cusum (drop the floor, or
    init g_0 to something other than 0). Used only to PROVE the differential gap
    is real - the guard asserts the correct value, which differs from this.
    """
    g = 0.0
    best = 0.0
    for s in scores:
        g = g + (s - drift)          # no max(0, .) reset
        if g > best:
            best = g
    return best


def test_cusum_g0_zero_and_reset_is_load_bearing():
    # A leading sub-drift event must NOT bank negative credit against a later rise.
    # correct (g_0=0, reset-at-zero): g1=max(0,-0.5)=0, g2=max(0,0+0.2)=0.2 -> 0.2
    # buggy (no floor):               g1=-0.5,         g2=-0.3            -> <= 0
    scores, drift = [0.0, 0.7], 0.5
    assert abs(persistence_cusum(scores, drift) - 0.2) < 1e-9
    # the regression would land here; prove the gap is real and the guard separates them
    assert _no_floor_cusum(scores, drift) <= 0.0
    assert persistence_cusum(scores, drift) != _no_floor_cusum(scores, drift)


def test_cusum_distributed_accumulation_value():
    # six events 0.05 above drift accumulate to exactly 0.30 (slow-burn signal)
    assert abs(persistence_cusum([0.55] * 6, 0.50) - 0.30) < 1e-9


def test_peak_is_max_not_last_or_mean():
    scores = [0.20, 0.80, 0.50]
    assert peak(scores) == 0.80          # max
    assert peak(scores) != scores[-1]    # not last
    assert peak([]) == 0.0               # empty -> 0.0


def test_frac_and_count_use_inclusive_tau_boundary():
    # tau boundary is >= (inclusive), NOT strict >. The two events AT 0.50 count.
    # correct (>=): 3/4 = 0.75 ; buggy (>): 1/4 = 0.25
    scores, tau = [0.50, 0.50, 0.40, 0.60], 0.50
    assert count_over(scores, tau) == 3
    assert abs(fraction_over(scores, tau) - 0.75) < 1e-9
    strict_count = sum(1 for s in scores if s > tau)   # the regression
    assert strict_count == 1 and count_over(scores, tau) != strict_count
    assert fraction_over([], tau) == 0.0


def test_accumulate_contract_shape_and_values():
    a = accumulate([0.40, 0.60, 0.60], tau=0.50, drift=0.50)
    assert set(a) == {"peak", "persistence", "n_over_tau", "frac_over_tau", "n_events"}
    assert a["peak"] == 0.60
    assert a["n_over_tau"] == 2 and a["n_events"] == 3
    assert abs(a["frac_over_tau"] - 2 / 3) < 1e-9
    # persistence: g1=max(0,-0.1)=0, g2=0.1, g3=0.2 -> 0.2
    assert abs(a["persistence"] - 0.20) < 1e-9


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for fn in fns:
        try:
            fn()
        except AssertionError as exc:  # pragma: no cover - reporting path
            failures.append(f"{fn.__name__}: {exc}")
    if failures:
        print("FAIL (accumulator regressed):")
        for f in failures:
            print("  -", f)
        return 1
    print(f"OK: {len(fns)} accumulator differential guards passed")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_run())
