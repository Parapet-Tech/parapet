from __future__ import annotations

import pytest

from parapet_runner.manifest import EvalResult, compute_metric_delta, compute_semantic_parity_hash

try:
    from parapet_data.models import CellFillRecord, compute_semantic_hash as compute_data_semantic_hash
except Exception:  # pragma: no cover - parapet-data may be absent in some environments
    CellFillRecord = None
    compute_data_semantic_hash = None


def test_semantic_hash_is_order_insensitive_for_inputs() -> None:
    h1 = compute_semantic_parity_hash(
        ["b", "a", "c"],
        {
            "roleplay": {"target": 10, "actual": 9, "backfilled": 1},
            "meta_probe": {"target": 5, "actual": 5, "backfilled": 0},
        },
    )
    h2 = compute_semantic_parity_hash(
        ["c", "b", "a"],
        {
            "meta_probe": {"actual": 5, "target": 5, "backfilled": 0},
            "roleplay": {"actual": 9, "target": 10, "backfilled": 1},
        },
    )
    assert h1 == h2


def test_semantic_hash_changes_when_counts_change() -> None:
    h1 = compute_semantic_parity_hash(
        ["a", "b"],
        {"roleplay": {"target": 10, "actual": 9, "backfilled": 1}},
    )
    h2 = compute_semantic_parity_hash(
        ["a", "b"],
        {"roleplay": {"target": 10, "actual": 8, "backfilled": 2}},
    )
    assert h1 != h2


def test_metric_delta() -> None:
    eval_result = EvalResult(
        f1=0.84,
        precision=0.86,
        recall=0.82,
        false_positives=3,
        false_negatives=4,
        threshold=-0.5,
        holdout_size=100,
    )
    baseline = EvalResult(
        f1=0.80,
        precision=0.79,
        recall=0.81,
        false_positives=5,
        false_negatives=6,
        threshold=0.0,
        holdout_size=100,
    )
    delta = compute_metric_delta(eval_result, baseline)
    assert delta["f1_delta"] == pytest.approx(0.04)
    assert delta["precision_delta"] == pytest.approx(0.07)
    assert delta["recall_delta"] == pytest.approx(0.01)


def test_semantic_hash_matches_parapet_data_contract() -> None:
    if compute_data_semantic_hash is None or CellFillRecord is None:
        pytest.skip("parapet-data not available")

    content_hashes = ["b", "a"]
    per_cell = {"instruction_override": {"target": 10, "actual": 9, "backfilled": 1}}

    runner_hash = compute_semantic_parity_hash(content_hashes, per_cell)
    data_hash = compute_data_semantic_hash(
        content_hashes,
        {
            "instruction_override": CellFillRecord(
                target=10, actual=9, backfilled=1, backfill_sources=[]
            )
        },
    )
    assert runner_hash == data_hash


def test_semantic_hash_rejects_missing_required_cell_fill_keys() -> None:
    with pytest.raises(ValueError, match="missing keys"):
        compute_semantic_parity_hash(
            ["a"],
            {"instruction_override": {"target": 10, "actual": 9}},
        )


def test_semantic_hash_rejects_unknown_cell_fill_keys() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        compute_semantic_parity_hash(
            ["a"],
            {"instruction_override": {"target": 10, "actual": 9, "backfilled": 1, "oops": 123}},
        )


def test_semantic_hash_rejects_string_backfill_sources() -> None:
    with pytest.raises(ValueError, match="backfill_sources"):
        compute_semantic_parity_hash(
            ["a"],
            {
                "instruction_override": {
                    "target": 10,
                    "actual": 9,
                    "backfilled": 1,
                    "backfill_sources": "not-a-list",
                }
            },
        )
