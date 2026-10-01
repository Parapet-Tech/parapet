"""Tests for the frozen A.5 reference / held-out split and its shared derivations."""
import hashlib
import itertools

import pytest

from parapet_data.p3.reference.split import (
    A5_REGRESSION_BUCKET,
    A5_REGRESSION_KEY,
    A5_SALT,
    A5_SPLIT,
    FLOOR_SAMPLE_SALT,
    SaltedBucketSplit,
    assert_regression_pin,
    canonical_key,
    eligible_carriers,
    inventory_sha256,
    is_benign_carrier,
    is_benign_swe_carrier,
    partition,
    pick_carrier,
    salted_path_order_key,
    straddle,
)

# Keys found by search whose A.5 buckets sit either side of the 300 cut.
KEY_AT_299 = "owner646/repo646"
KEY_AT_300 = "owner2807/repo2807"


def test_regression_pin_holds_for_frozen_split():
    assert A5_SPLIT.bucket(A5_REGRESSION_KEY) == A5_REGRESSION_BUCKET == 854
    assert_regression_pin()


def test_regression_pin_fails_loud_on_salt_drift():
    drifted = SaltedBucketSplit(salt="p3-a5", held_out_milli=300)
    assert drifted.bucket(A5_REGRESSION_KEY) != A5_REGRESSION_BUCKET
    with pytest.raises(ValueError, match="regression pin"):
        assert_regression_pin(drifted)


def test_held_out_cut_is_strictly_below_held_out_milli():
    assert A5_SPLIT.bucket(KEY_AT_299) == 299
    assert A5_SPLIT.bucket(KEY_AT_300) == 300
    assert A5_SPLIT.is_held_out(KEY_AT_299) is True
    assert A5_SPLIT.is_held_out(KEY_AT_300) is False


def test_bucket_canonicalizes_case_and_composition():
    assert canonical_key("Owner0/Repo0") == "owner0/repo0"
    assert A5_SPLIT.bucket("Owner0/Repo0") == A5_SPLIT.bucket("owner0/repo0") == 232
    assert A5_SPLIT.bucket("café/repo") == A5_SPLIT.bucket("café/repo") == 261
    assert A5_SPLIT.bucket("STRASSE/x") == A5_SPLIT.bucket("straße/x")


def test_uncanonicalized_hash_would_flip_a_repo_across_the_cut():
    # Why the canon is load-bearing: hashing the raw spelling puts this repo on the other side.
    raw = hashlib.sha256(f"{A5_SALT}|Owner0/Repo0".encode()).hexdigest()
    assert int(raw[:8], 16) % 1000 == 807
    assert A5_SPLIT.is_held_out("Owner0/Repo0") is True
    assert A5_SPLIT.is_held_out("owner0/repo0") is True


def test_partition_keeps_spellings_of_one_repo_on_one_side():
    rows = [{"split_key": k} for k in ("Owner0/Repo0", "owner0/repo0", "OWNER0/REPO0")]
    parts = partition(rows)
    assert len(parts.held_out) == 3 and parts.reference_in == ()


def test_bucket_is_deterministic_and_in_range():
    keys = [f"o{i}/r{i}" for i in range(500)]
    buckets = [A5_SPLIT.bucket(k) for k in keys]
    assert buckets == [A5_SPLIT.bucket(k) for k in keys]
    assert all(0 <= b < 1000 for b in buckets)


@pytest.mark.parametrize("milli", [0, 1000, -1, 1001])
def test_split_rejects_out_of_range_fraction(milli):
    with pytest.raises(ValueError):
        SaltedBucketSplit(salt="s", held_out_milli=milli)


def test_split_rejects_empty_salt_and_non_int_fraction():
    with pytest.raises(ValueError):
        SaltedBucketSplit(salt="", held_out_milli=300)
    with pytest.raises(TypeError):
        SaltedBucketSplit(salt="s", held_out_milli=0.3)
    with pytest.raises(TypeError):
        SaltedBucketSplit(salt="s", held_out_milli=True)


def test_benign_filters_use_strict_identity():
    ok = {"usable_carrier": True, "attack_type": None, "scaffold": "openhands"}
    assert is_benign_carrier(ok) and is_benign_swe_carrier(ok)
    assert not is_benign_carrier({**ok, "usable_carrier": "True"})
    assert not is_benign_carrier({**ok, "attack_type": "important_instructions"})
    assert not is_benign_carrier({"attack_type": None})
    assert is_benign_carrier({**ok, "scaffold": "our-runner"})
    assert not is_benign_swe_carrier({**ok, "scaffold": "our-runner"})
    assert is_benign_swe_carrier({**ok, "scaffold": "swe-agent"})


def test_partition_sends_every_row_of_a_key_to_one_side():
    rows = [{"split_key": k, "out_path": f"{k}/{n}"} for k in (KEY_AT_299, KEY_AT_300) for n in range(3)]
    parts = partition(rows)
    assert {r["split_key"] for r in parts.held_out} == {KEY_AT_299}
    assert {r["split_key"] for r in parts.reference_in} == {KEY_AT_300}
    assert len(parts.held_out) + len(parts.reference_in) == len(rows)
    assert straddle(parts, "split_key") == set()


def test_straddle_reports_a_cell_split_across_repos():
    rows = [
        {"split_key": KEY_AT_299, "cell_id": "shared-cell"},
        {"split_key": KEY_AT_300, "cell_id": "shared-cell"},
        {"split_key": KEY_AT_300, "cell_id": "ref-only"},
    ]
    assert straddle(partition(rows), "cell_id") == {"shared-cell"}


def test_partition_fails_closed_on_missing_key():
    with pytest.raises(KeyError):
        partition([{"out_path": "x"}])


def test_inventory_is_order_independent_and_pinned():
    expected = hashlib.sha256(b"a\nb").hexdigest()
    assert inventory_sha256(["b", "a"]) == inventory_sha256(["a", "b"]) == expected
    assert inventory_sha256([]) == hashlib.sha256(b"").hexdigest()


def test_inventory_rejects_duplicate_paths():
    with pytest.raises(ValueError, match="duplicate"):
        inventory_sha256(["a", "b", "a"])


def test_salted_path_order_key_casefolds_and_depends_on_salt():
    assert salted_path_order_key("Trajectory/Owner/Repo") == salted_path_order_key("trajectory/owner/repo")
    assert salted_path_order_key("t/x") == hashlib.sha256(f"{FLOOR_SAMPLE_SALT}|t/x".encode()).hexdigest()
    assert salted_path_order_key("t/x", salt="other") != salted_path_order_key("t/x")


def _carrier(out_path, n_tool_calls=40, n_live_positions=30):
    return {"out_path": out_path, "n_tool_calls": n_tool_calls, "n_live_positions": n_live_positions}


CELL_PATHS = ["trajectory/a/one", "trajectory/a/two", "trajectory/a/three", "trajectory/a/four"]


def test_pick_carrier_is_order_independent_and_pinned():
    picks = {
        pick_carrier([_carrier(p) for p in perm], h_c=6)["out_path"]
        for perm in itertools.permutations(CELL_PATHS)
    }
    assert picks == {"trajectory/a/one"}


def test_pick_carrier_single_row_and_errors():
    assert pick_carrier([_carrier("only")], h_c=6)["out_path"] == "only"
    with pytest.raises(ValueError, match="no carrier"):
        pick_carrier([], h_c=6)
    with pytest.raises(ValueError, match="duplicate"):
        pick_carrier([_carrier("x"), _carrier("x")], h_c=6)


def test_pick_carrier_skips_an_ineligible_least_hash():
    # "trajectory/a/one" has the least hash; once it is too shallow the pick moves on.
    deep = [_carrier(p) for p in CELL_PATHS[1:]]
    runner_up = pick_carrier(deep, h_c=6)["out_path"]
    assert runner_up != "trajectory/a/one"
    for shallow in (_carrier("trajectory/a/one", n_tool_calls=17), _carrier("trajectory/a/one", n_live_positions=5)):
        assert pick_carrier([shallow, *deep], h_c=6)["out_path"] == runner_up


def test_eligibility_floor_boundaries():
    at_floor = _carrier("at", n_tool_calls=18, n_live_positions=6)
    assert eligible_carriers([at_floor], h_c=6) == (at_floor,)
    assert eligible_carriers([_carrier("events", n_tool_calls=17)], h_c=6) == ()
    assert eligible_carriers([_carrier("live", n_live_positions=9)], h_c=10) == ()
    # The native-event floor follows h_min, not the cell budget.
    assert eligible_carriers([_carrier("deep-budget", n_tool_calls=18, n_live_positions=10)], h_c=10) != ()


def test_pick_carrier_fails_loud_when_no_carrier_is_eligible():
    rows = [_carrier("a", n_tool_calls=17), _carrier("b", n_live_positions=5)]
    with pytest.raises(ValueError, match="no eligible carrier"):
        pick_carrier(rows, h_c=6)


def test_eligibility_fails_closed_on_bad_inputs():
    with pytest.raises(KeyError):
        pick_carrier([{"out_path": "x", "n_tool_calls": 40}], h_c=6)
    with pytest.raises(TypeError):
        pick_carrier([_carrier("x", n_tool_calls="40")], h_c=6)
    with pytest.raises(TypeError):
        pick_carrier([_carrier("x", n_live_positions=True)], h_c=6)
    with pytest.raises(ValueError, match="h_min"):
        pick_carrier([_carrier("x")], h_c=5)
    with pytest.raises(TypeError):
        pick_carrier([_carrier("x")], h_c=6.0)
    with pytest.raises(TypeError):
        pick_carrier([_carrier("x")])
