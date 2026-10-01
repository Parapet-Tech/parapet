"""Tests for the R2 / R5 backstop: name-twin exclusion, shingle Jaccard scan, null flags, verdict."""
import hashlib

import pytest

from parapet_data.p3.reference.backstop import (
    FLAG_SATURATED,
    FLAG_UNTESTED,
    NULL_SAMPLE_SIZE,
    THRESHOLD,
    Nearest,
    b2_verdict,
    build_query_index,
    event_hash,
    fires,
    jaccard,
    merge_nearest,
    name_component,
    name_twin_exclusion,
    nearest_rank_quantile,
    nearest_scan,
    null_sample,
    scan_chunk,
    shingles,
    summarize_null,
)
from parapet_data.p3.reference.split import salted_path_order_key


def _ev(function, arguments):
    return event_hash({"function": function, "arguments": arguments})


# A 20-event base trajectory and the variants the fixture spec names.
BASE = [_ev(f"fn{i % 4}", {"k": i}) for i in range(20)]
DUPLICATE = list(BASE)
# One event prepended, last dropped: 17 of 18 shingles survive, J = 17/19. Must fire.
TRAJECTORY_SHIFT = [_ev("fnX", {"k": -1})] + BASE[:-1]
# Each call keeps its function and position and takes the next call's arguments. Every event
# hash changes, J = 0. Must not fire; a function-only hash would score 1.0 here.
ARGUMENT_SHIFT = [_ev(f"fn{i % 4}", {"k": (i + 1) % 20}) for i in range(20)]
UNRELATED = [_ev("other", {"k": i}) for i in range(20)]

QUERY = (shingles(BASE), "held/q")
CORPUS = {"b/dup": DUPLICATE, "b/shift": ARGUMENT_SHIFT, "b/pos": TRAJECTORY_SHIFT, "held/q": BASE}


def _score(sequence):
    other = shingles(sequence)
    return jaccard(len(QUERY[0] & other), len(QUERY[0]), len(other))


def _scan(build, exact_events=frozenset(), corpus=CORPUS):
    return nearest_scan(build_query_index([QUERY]), build, corpus.__getitem__, exact_events)


# ---------------------------------------------------------------- B1

REFERENCE_ROWS = [
    {"split_key": "alice/Widget", "n_tool_calls": 7},
    {"split_key": "alice/widget", "n_tool_calls": 3},
    {"split_key": "carol/gadget", "n_tool_calls": 5},
]


def test_name_twin_is_excluded_across_owners_and_case():
    excluded = name_twin_exclusion(REFERENCE_ROWS, {"bob/widget", "dave/sprocket"})
    assert set(excluded) == {"alice/Widget", "alice/widget"}


def test_non_twin_repo_is_kept():
    assert "carol/gadget" not in name_twin_exclusion(REFERENCE_ROWS, {"bob/widget", "dave/sprocket"})


def test_exclusion_counts_carriers_and_events_per_repo():
    rows = REFERENCE_ROWS + [{"split_key": "alice/widget", "n_tool_calls": 4}]
    excluded = name_twin_exclusion(rows, {"bob/widget", "erin/WIDGET"})
    assert excluded["alice/Widget"] == {
        "repo": "alice/Widget", "held_out_twins": ["bob/widget", "erin/WIDGET"], "carriers": 1, "events": 7}
    assert excluded["alice/widget"]["carriers"] == 2 and excluded["alice/widget"]["events"] == 7
    assert list(excluded) == sorted(excluded)


def test_name_component_strips_owner_and_casefolds_only():
    assert name_component("Owner/Sub/Repo") == "repo"
    assert name_component("no-owner") == "no-owner"
    assert name_component("o/STRASSE") == name_component("o/straße")
    # Casefold only: composed and decomposed spellings stay distinct here, unlike the split key.
    assert name_component("o/café") != name_component("o/café")


def test_exclusion_fails_closed_on_bad_rows():
    with pytest.raises(KeyError):
        name_twin_exclusion([{"n_tool_calls": 1}], {"bob/widget"})
    with pytest.raises(KeyError):
        name_twin_exclusion([{"split_key": "alice/widget"}], {"bob/widget"})
    with pytest.raises(TypeError):
        name_twin_exclusion([{"split_key": "alice/widget", "n_tool_calls": "3"}], {"bob/widget"})
    with pytest.raises(TypeError):
        name_twin_exclusion([{"split_key": "alice/widget", "n_tool_calls": True}], {"bob/widget"})


def test_no_held_out_keys_excludes_nothing():
    assert name_twin_exclusion(REFERENCE_ROWS, set()) == {}


# ---------------------------------------------------------------- hashing and Jaccard

def test_event_hash_is_the_frozen_recipe():
    assert _ev("f", {"k": 1}) == hashlib.sha256(b"f\x00{'k': 1}").digest()
    assert event_hash({}) == hashlib.sha256(b"None\x00None").digest()
    # The separator keeps function and arguments from running together.
    assert _ev("ab", "c") != _ev("a", "bc")
    with pytest.raises(TypeError):
        event_hash("not an event")


def test_base_trajectory_has_one_shingle_per_window():
    assert len(QUERY[0]) == 18
    assert all(0 <= s < 2 ** 64 for s in QUERY[0])


@pytest.mark.parametrize("sequence, expect_fire", [
    (DUPLICATE, True),
    (TRAJECTORY_SHIFT, True),
    (ARGUMENT_SHIFT, False),
    (UNRELATED, False),
], ids=["duplicate", "trajectory_shift_positive_control", "argument_shift_negative_control", "unrelated"])
def test_threshold_fires_only_on_near_duplicates(sequence, expect_fire):
    assert (_score(sequence) >= THRESHOLD) is expect_fire


def test_duplicate_scores_exactly_one_and_shift_scores_17_of_19():
    assert _score(DUPLICATE) == 1.0
    assert _score(TRAJECTORY_SHIFT) == 17 / 19
    assert _score(ARGUMENT_SHIFT) == 0.0


def test_short_trajectories_have_no_shingles():
    assert shingles(BASE[:2]) == set()
    assert shingles([]) == set()
    assert len(shingles(BASE[:3])) == 1


def test_shingles_are_order_sensitive():
    assert shingles(BASE[:3]) != shingles(BASE[:3][::-1])


def test_jaccard_of_two_empty_sets_is_zero():
    assert jaccard(0, 0, 0) == 0.0
    assert jaccard(1, 2, 3) == 0.25


# ---------------------------------------------------------------- scan

def test_scan_finds_the_duplicate_as_nearest():
    best, _ = _scan([("b/shift", "b"), ("b/dup", "b"), ("b/pos", "b")])
    assert best[0] == Nearest(jaccard=1.0, out_path="b/dup", split_key="b", n_events=20, shared_shingles=18)


def test_scan_fires_on_a_trajectory_shift():
    best, _ = _scan([("b/pos", "b")])
    assert best[0].out_path == "b/pos" and best[0].jaccard >= THRESHOLD


def test_scan_does_not_find_an_argument_shift():
    best, _ = _scan([("b/shift", "b")])
    assert 0 not in best


def test_scan_skips_candidates_in_the_query_repo():
    best, _ = _scan([("held/q", "held/q")])
    assert 0 not in best


def test_scan_reports_exact_event_hits():
    _, hits = _scan([("b/shift", "b"), ("b/dup", "b"), ("b/pos", "b")], exact_events=set(BASE))
    assert hits == set(BASE)
    _, hits = _scan([("b/shift", "b")], exact_events=set(BASE))
    assert hits == set()
    # Exact hits are counted even for a carrier too short to shingle.
    _, hits = _scan([("b/two", "b")], exact_events=set(BASE), corpus={"b/two": BASE[:2]})
    assert hits == set(BASE[:2])


def test_scan_tie_goes_to_the_smaller_path_in_any_order():
    corpus = {"b/zz": DUPLICATE, "a/aa": DUPLICATE, "m/mm": DUPLICATE}
    build = [("b/zz", "b"), ("a/aa", "a"), ("m/mm", "m")]
    for order in (build, build[::-1]):
        best, _ = _scan(order, corpus=corpus)
        assert best[0].out_path == "a/aa"


def test_chunked_scan_merges_to_the_single_pass_result():
    build = [("b/shift", "b"), ("b/pos", "b"), ("b/dup", "b")]
    index = build_query_index([QUERY, (shingles(UNRELATED), "held/u")])
    whole, whole_hits = nearest_scan(index, build, CORPUS.__getitem__, set(BASE))
    merged, merged_hits = {}, set()
    for carrier in reversed(build):
        part, hits = scan_chunk(index, [carrier], CORPUS.__getitem__, set(BASE))
        merge_nearest(merged, part)
        merged_hits |= hits
    assert merged == whole and merged_hits == whole_hits
    assert 1 not in whole


def test_scan_fails_closed_when_the_loader_cannot_read_a_carrier():
    with pytest.raises(KeyError):
        _scan([("b/missing", "b")])


def test_query_index_rejects_a_shingle_list():
    with pytest.raises(TypeError):
        build_query_index([(sorted(QUERY[0]), "held/q")])


# ---------------------------------------------------------------- verdicts and null

def test_fires_is_inclusive_at_each_threshold():
    assert fires(0.5) == {"0.30": True, "0.50": True, "0.70": False}
    assert fires(0.299999) == {"0.30": False, "0.50": False, "0.70": False}
    assert fires(0.7) == {"0.30": True, "0.50": True, "0.70": True}


def test_nearest_rank_quantile():
    values = [float(v) for v in range(1, 11)]
    assert nearest_rank_quantile(values, 0.50) == 5.0
    assert nearest_rank_quantile(values, 0.90) == 9.0
    assert nearest_rank_quantile(values, 0.999) == 10.0
    assert nearest_rank_quantile(values, 0.0) == 1.0
    assert nearest_rank_quantile([], 0.5) is None


def test_null_saturation_needs_more_than_one_percent_at_threshold():
    at_one_percent = summarize_null([0.5] + [0.4] * 99)
    assert at_one_percent.share["0.50"] == 0.01 and FLAG_SATURATED not in at_one_percent.flags
    above = summarize_null([0.5, 0.5] + [0.4] * 98)
    assert above.flags == (FLAG_SATURATED,)


def test_null_that_never_reaches_the_low_threshold_is_untested():
    quiet = summarize_null([0.0, 0.1, 0.29])
    assert quiet.share == {"0.30": 0.0, "0.50": 0.0, "0.70": 0.0}
    assert quiet.flags == (FLAG_UNTESTED,)
    assert summarize_null([0.0, 0.3]).flags == ()


def test_empty_null_has_no_share_and_no_flags():
    empty = summarize_null([])
    assert empty.share == {"0.30": None, "0.50": None, "0.70": None}
    assert empty.flags == ()


def test_verdict_order():
    assert b2_verdict(False, ()) == "PASS"
    assert b2_verdict(False, (FLAG_UNTESTED,)) == "PASS"
    assert b2_verdict(True, (FLAG_UNTESTED,)) == "BLOCKER"
    # A saturated null overrides a fired carrier: the gate cannot tell it from an ordinary one.
    assert b2_verdict(True, (FLAG_SATURATED,)) == "NON-GATING"
    assert b2_verdict(False, (FLAG_SATURATED,)) == "NON-GATING"


def test_null_sample_takes_a_prefix_of_floor_sample_order():
    rows = [{"out_path": f"trajectory/o/r/{i}"} for i in range(50)]
    expected = sorted(rows, key=lambda r: salted_path_order_key(r["out_path"]))
    assert null_sample(rows, n=7) == expected[:7]
    assert null_sample(reversed(rows), n=7) == expected[:7]
    assert null_sample(rows) == expected
    assert NULL_SAMPLE_SIZE == 2000
