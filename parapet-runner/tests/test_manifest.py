from __future__ import annotations

from pathlib import Path

import pytest

from parapet_runner.config import TrainConfig
from parapet_runner.manifest import (
    CurationManifest,
    EvalResult,
    RunManifest,
    RuntimeIdentity,
    compute_metric_delta,
    compute_semantic_parity_hash,
)

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


def test_semantic_hash_ignores_unknown_cell_fill_keys() -> None:
    base = compute_semantic_parity_hash(
        ["a"],
        {"instruction_override": {"target": 10, "actual": 9, "backfilled": 1}},
    )
    with_extra = compute_semantic_parity_hash(
        ["a"],
        {
            "instruction_override": {
                "target": 10,
                "actual": 9,
                "backfilled": 1,
                "by_format": {"prose": 9},
                "by_length": {"short": 5, "medium": 4},
                "by_language": {"EN": 9},
                "degraded": False,
                "degraded_mode": None,
                "oops": 123,
            }
        },
    )
    assert with_extra == base


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


def _curation_manifest() -> CurationManifest:
    return CurationManifest(
        spec_name="mirror_v1",
        spec_version="1.0.0",
        spec_hash="spec-hash",
        seed=42,
        timestamp="2026-03-01T00:00:00Z",
        source_hashes={"a": "b"},
        output_path=Path("curated.jsonl"),
        output_hash="deadbeef",
        semantic_hash="seeded",
        total_samples=100,
        attack_samples=50,
        benign_samples=50,
        splits={
            "train": {
                "name": "train",
                "sample_count": 80,
                "content_hashes": ["h1"],
                "artifact_path": Path("train.jsonl"),
            },
            "val": {
                "name": "val",
                "sample_count": 10,
                "content_hashes": ["h2"],
                "artifact_path": Path("val.jsonl"),
            },
            "holdout": {
                "name": "holdout",
                "sample_count": 10,
                "content_hashes": ["h3"],
                "artifact_path": Path("holdout.jsonl"),
            },
        },
        cell_fills={},
        gaps=[],
        cross_contamination_dropped=0,
    )


def _runtime_identity() -> RuntimeIdentity:
    return RuntimeIdentity(
        git_sha="abc",
        trainer_script_hash="trainer",
        parapet_eval_hash="eval",
        pg2_model_id="pg2",
        eval_config_hash="cfg",
        env_hash="env",
    )


def _eval_result() -> EvalResult:
    return EvalResult(
        f1=0.8,
        precision=0.8,
        recall=0.8,
        false_positives=1,
        false_negatives=2,
        threshold=-0.5,
        holdout_size=10,
    )


def test_run_manifest_rejects_partial_baseline_identity() -> None:
    with pytest.raises(ValueError, match="must be set together"):
        RunManifest(
            run_id="run-1",
            runtime=_runtime_identity(),
            curation=_curation_manifest(),
            train_config=TrainConfig(mode="iteration", cv_folds=0, max_features=15_000),
            eval_result=_eval_result(),
            baseline_family="protectai_size_matched",
        )


def test_run_manifest_accepts_complete_baseline_identity() -> None:
    manifest = RunManifest(
        run_id="run-2",
        runtime=_runtime_identity(),
        curation=_curation_manifest(),
        train_config=TrainConfig(mode="iteration", cv_folds=0, max_features=15_000),
        eval_result=_eval_result(),
        baseline_family="protectai_size_matched",
        baseline_recipe_hash="r" * 64,
        baseline_data_hash="d" * 64,
        baseline_data_size=24000,
    )
    assert manifest.baseline_family == "protectai_size_matched"
    assert manifest.baseline_data_size == 24000
