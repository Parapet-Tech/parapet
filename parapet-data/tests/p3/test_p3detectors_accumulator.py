"""Tests for the peak + persistence trajectory accumulator."""
from parapet_data.p3.detectors.accumulator import (
    accumulate,
    count_over,
    fraction_over,
    peak,
    persistence_cusum,
)


def test_peak_and_empty():
    assert peak([0.1, 0.9, 0.4]) == 0.9
    assert peak([]) == 0.0


def test_count_and_fraction_over():
    assert count_over([0.4, 0.5, 0.6], 0.5) == 2
    assert fraction_over([0.4, 0.5, 0.6], 0.5) == 2 / 3
    assert fraction_over([], 0.5) == 0.0


def test_cusum_accumulates_distributed_elevation():
    # six events each just above the drift: individually small, but they accumulate
    scores = [0.55] * 6
    drift = 0.50
    # g grows by 0.05 each step -> peak running sum 0.30
    assert abs(persistence_cusum(scores, drift) - 0.30) < 1e-9


def test_cusum_decays_isolated_spike():
    # one big spike then nothing: persistence captures the single 0.4 bump, then decays
    spike = persistence_cusum([0.9, 0.1, 0.1, 0.1], drift=0.5)
    distributed = persistence_cusum([0.6, 0.6, 0.6, 0.6], drift=0.5)
    # the spike's persistence is just its own excess (0.4); distributed accumulates 4*0.1=0.4
    assert abs(spike - 0.4) < 1e-9
    assert abs(distributed - 0.4) < 1e-9
    # a longer distributed run overtakes the single spike -> persistence > peak-driven value
    longer = persistence_cusum([0.6] * 8, drift=0.5)
    assert longer > spike


def test_cusum_resets_at_zero_no_negative_drift():
    # all events below drift -> running sum never goes negative -> persistence 0
    assert persistence_cusum([0.1, 0.2, 0.0], drift=0.5) == 0.0


def test_persistence_separates_slowburn_from_benign_spike():
    # the load-bearing property: a slow-burn trajectory (many sub-tau elevations) has
    # LOWER peak but HIGHER persistence than a benign trajectory with one spike.
    tau = 0.65
    drift = 0.45
    slowburn = [0.55, 0.58, 0.56, 0.60, 0.57, 0.59]   # every event BELOW tau
    benign_spike = [0.10, 0.90, 0.12, 0.08, 0.11]      # one event ABOVE tau
    a_slow = accumulate(slowburn, tau=tau, drift=drift)
    a_benign = accumulate(benign_spike, tau=tau, drift=drift)
    assert a_slow["peak"] < tau and a_benign["peak"] >= tau     # peak FALSELY favors benign
    assert a_slow["persistence"] > a_benign["persistence"]      # persistence recovers slow-burn


def test_accumulate_shape():
    a = accumulate([0.4, 0.6, 0.6], tau=0.5, drift=0.5)
    assert set(a) == {"peak", "persistence", "n_over_tau", "frac_over_tau", "n_events"}
    assert a["n_over_tau"] == 2 and a["n_events"] == 3
