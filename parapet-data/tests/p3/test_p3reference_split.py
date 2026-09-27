"""Tests for the frozen A.5 reference / held-out split and its shared derivations."""
import hashlib
import itertools

import pytest

from parapet_data.p3.reference.split import (
    A5_REGRESSION_BUCKET,
    A5_REGRESSION_KEY,
    A5_SPLIT,
    FLOOR_SAMPLE_SALT,
    SaltedBucketSplit,
    assert_regression_pin,
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


def test_pick_carrier_is_order_independent_and_pinned():
    paths = ["trajectory/a/one", "trajectory/a/two", "trajectory/a/three", "trajectory/a/four"]
    picks = {pick_carrier([{"out_path": p} for p in perm])["out_path"] for perm in itertools.permutations(paths)}
    assert picks == {"trajectory/a/one"}


def test_pick_carrier_single_row_and_errors():
    assert pick_carrier([{"out_path": "only"}])["out_path"] == "only"
    with pytest.raises(ValueError, match="no carrier"):
        pick_carrier([])
    with pytest.raises(ValueError, match="duplicate"):
        pick_carrier([{"out_path": "x"}, {"out_path": "x"}])
