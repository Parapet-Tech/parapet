from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from parapet_runner.temporal import (
    TemporalAccumulatorConfig,
    TemporalEvent,
    TemporalLabels,
    TemporalReceiptMetadata,
    build_temporal_receipt,
    canonical_json_sha256,
    compute_aucs,
    load_temporal_events_jsonl,
    mann_whitney_auc,
    score_temporal_events,
    summarize_trajectories,
    write_temporal_receipt,
)


def _labels(
    *,
    attack: bool,
    event_attack: bool | None = None,
) -> TemporalLabels:
    return TemporalLabels(
        event_attack_label=attack if event_attack is None else event_attack,
        trajectory_label="attack" if attack else "benign",
        cell_label="attack" if attack else "benign",
        label_source_ref="fixture",
        label_provenance="construction",
    )


def _event(
    event_id: str,
    *,
    trajectory_id: str = "t1",
    cell_id: str | None = None,
    event_index: int,
    surprise: float,
    keys: dict[str, object] | None = None,
    attack: bool = True,
    event_attack: bool | None = None,
    layer_id: str = "l1",
    hard_trigger: bool = False,
) -> TemporalEvent:
    return TemporalEvent(
        event_id=event_id,
        trajectory_id=trajectory_id,
        cell_id=cell_id or trajectory_id,
        event_index=event_index,
        layer_id=layer_id,
        event_score=surprise,
        surprise=surprise,
        population="attack_eval" if attack else "benign_eval",
        score_context={"detector": "fixture"},
        cdf_fallback_level="exact",
        continuity_keys=keys
        if keys is not None
        else {
            "tool_target": "mail",
            "source_document_id": "doc-a",
            "instruction_channel": "retrieval",
            "task_epoch": 0,
        },
        labels=_labels(attack=attack, event_attack=event_attack),
        task_epoch_provenance=(
            "controller_deterministic"
            if (keys is None or keys.get("task_epoch") is not None)
            else None
        ),
        hard_trigger=hard_trigger,
        hard_trigger_source_ref="fixture-trigger" if hard_trigger else None,
        source_receipt_ref="fixture-source-receipt",
    )


def _config() -> TemporalAccumulatorConfig:
    return TemporalAccumulatorConfig(
        k_u_b=1.0,
        k_u_c_strict=1.0,
        k_u_c_broad=1.0,
        peak_alert_level=5.0,
    )


def _new_output_dir(case_name: str) -> Path:
    output_dir = Path("tests/.tmp_outputs") / case_name
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def test_b_accumulates_across_missing_key_but_c_missing_is_singleton() -> None:
    events = [
        _event("e1", event_index=0, surprise=1.6),
        _event("e2", event_index=1, surprise=1.6, keys={}),
        _event("e3", event_index=2, surprise=1.6),
    ]

    scored = score_temporal_events(events, _config())

    assert [score.b_persistence for score in scored] == pytest.approx([0.6, 1.2, 1.8])
    assert [score.c_strict for score in scored] == pytest.approx([0.6, 0.6, 0.6])
    assert [score.c_strict_missing for score in scored] == [False, True, False]
    assert [score.c_strict_break for score in scored] == [False, True, False]
    assert [score.c_strict_accumulation_break for score in scored] == [True, True, True]
    result = summarize_trajectories(events, scored)[0]
    assert result.strict_reset_count == 1


def test_consecutive_missing_key_events_do_not_accumulate_with_each_other() -> None:
    events = [
        _event("e1", event_index=0, surprise=1.6, keys={}),
        _event("e2", event_index=1, surprise=1.6, keys={}),
        _event("e3", event_index=2, surprise=1.6, keys={}),
    ]

    scored = score_temporal_events(events, _config())

    assert [score.c_strict for score in scored] == pytest.approx([0.6, 0.6, 0.6])
    assert [score.c_strict_missing for score in scored] == [True, True, True]
    assert [score.c_strict_break for score in scored] == [False, False, False]
    assert [score.c_strict_accumulation_break for score in scored] == [True, True, True]
    result = summarize_trajectories(events, scored)[0]
    assert result.strict_reset_count == 0


def test_broad_continues_when_source_document_pivots_but_strict_resets() -> None:
    events = [
        _event("e1", event_index=0, surprise=1.6),
        _event(
            "e2",
            event_index=1,
            surprise=1.6,
            keys={
                "tool_target": "mail",
                "source_document_id": "doc-b",
                "instruction_channel": "retrieval",
                "task_epoch": 0,
            },
        ),
    ]

    scored = score_temporal_events(events, _config())

    assert [score.c_strict for score in scored] == pytest.approx([0.6, 0.6])
    assert [score.c_broad for score in scored] == pytest.approx([0.6, 1.2])
    assert [score.c_strict_break for score in scored] == [False, True]
    assert [score.c_broad_break for score in scored] == [False, False]


def test_single_layer_v1_receipts_fail_closed() -> None:
    events = [
        _event("e1", event_index=0, surprise=1.6, layer_id="l1"),
        _event("e2", event_index=1, surprise=1.6, layer_id="tool"),
    ]

    with pytest.raises(ValueError, match="single layer_id"):
        score_temporal_events(events, _config())


def test_duplicate_event_index_in_trajectory_fails_closed() -> None:
    events = [
        _event("e1", event_index=0, surprise=1.6),
        _event("e2", event_index=0, surprise=1.6),
    ]

    with pytest.raises(ValueError, match="duplicate event_index"):
        score_temporal_events(events, _config())


def test_duplicate_event_id_fails_closed() -> None:
    events = [
        _event("duplicate", trajectory_id="t1", event_index=0, surprise=1.6),
        _event("duplicate", trajectory_id="t2", event_index=0, surprise=1.6),
    ]

    with pytest.raises(ValueError, match="duplicate event_id"):
        score_temporal_events(events, _config())


def test_inconsistent_cell_or_labels_within_trajectory_fails_closed() -> None:
    first = _event("e1", event_index=0, surprise=1.6)
    wrong_cell = _event("e2", event_index=1, surprise=1.6, cell_id="other-cell")

    with pytest.raises(ValueError, match="inconsistent cell_id or labels"):
        score_temporal_events([first, wrong_cell], _config())

    wrong_labels = _event("e3", event_index=1, surprise=1.6, attack=False)
    with pytest.raises(ValueError, match="inconsistent cell_id or labels"):
        score_temporal_events([first, wrong_labels], _config())


def test_v1_rejects_multiple_trajectories_for_one_cell() -> None:
    events = [
        _event("e1", trajectory_id="t1", cell_id="shared", event_index=0, surprise=1.6),
        _event("e2", trajectory_id="t2", cell_id="shared", event_index=0, surprise=1.6),
    ]

    with pytest.raises(ValueError, match="maps to multiple trajectories"):
        score_temporal_events(events, _config())


def test_continuous_scores_accumulate_across_drift_boundary() -> None:
    events = [
        _event("e1", event_index=0, surprise=0.999999),
        _event("e2", event_index=1, surprise=1.000001),
        _event("e3", event_index=2, surprise=1.25),
    ]

    scored = score_temporal_events(events, _config())

    assert [score.b_persistence for score in scored] == pytest.approx(
        [0.0, 0.000001, 0.250001]
    )
    assert [score.c_strict for score in scored] == pytest.approx(
        [0.0, 0.000001, 0.250001]
    )


@pytest.mark.parametrize(
    ("changed_key", "changed_value"),
    [
        ("instruction_channel", "user"),
        ("task_epoch", 1),
    ],
)
def test_channel_and_epoch_transitions_reset_both_c_variants(
    changed_key: str, changed_value: object
) -> None:
    first_keys = {
        "tool_target": "mail",
        "source_document_id": "doc-a",
        "instruction_channel": "retrieval",
        "task_epoch": 0,
    }
    second_keys = {**first_keys, changed_key: changed_value}
    events = [
        _event("e1", event_index=0, surprise=1.6, keys=first_keys),
        _event("e2", event_index=1, surprise=1.6, keys=second_keys),
    ]

    scored = score_temporal_events(events, _config())

    assert [score.c_strict for score in scored] == pytest.approx([0.6, 0.6])
    assert [score.c_broad for score in scored] == pytest.approx([0.6, 0.6])
    assert [score.c_strict_break for score in scored] == [False, True]
    assert [score.c_broad_break for score in scored] == [False, True]


def test_annotation_labels_require_audit_ref() -> None:
    with pytest.raises(ValueError, match="label_audit_ref"):
        TemporalLabels(
            event_attack_label=True,
            trajectory_label="attack",
            cell_label="attack",
            label_source_ref="annotation-fixture",
            label_provenance="annotation",
        )


def test_construction_labels_require_source_ref() -> None:
    with pytest.raises(ValueError, match="construction labels require label_source_ref"):
        TemporalLabels(
            event_attack_label=True,
            trajectory_label="attack",
            cell_label="attack",
            label_provenance="construction",
        )


def test_claim_receipt_metadata_fails_closed_when_incomplete() -> None:
    with pytest.raises(ValueError, match="p3_temporal_validation is not supported"):
        TemporalReceiptMetadata(receipt_kind="p3_temporal_validation")


def test_temporal_v1_rejects_nonzero_c_lambda() -> None:
    with pytest.raises(ValueError, match="requires c_hard_lambda = 0"):
        TemporalAccumulatorConfig(
            k_u_b=1.0,
            k_u_c_strict=1.0,
            k_u_c_broad=1.0,
            peak_alert_level=5.0,
            c_hard_lambda=0.5,
        )


def test_annotation_labels_with_source_and_audit_build_receipt() -> None:
    attack = _event("a1", trajectory_id="attack-a", event_index=0, surprise=1.4)
    attack = attack.model_copy(
        update={
            "labels": TemporalLabels(
                event_attack_label=True,
                trajectory_label="attack",
                cell_label="attack",
                label_source_ref="annotation-source",
                label_provenance="annotation",
                label_audit_ref="annotation-audit",
            )
        }
    )
    benign = _event(
        "b1", trajectory_id="benign-a", event_index=0, surprise=0.8, attack=False
    )

    receipt = build_temporal_receipt(
        [attack, benign], _config(), min_cells_positive_b_mass=1
    )

    assert receipt.events[0].label_source_ref == "annotation-source"
    assert receipt.events[0].label_audit_ref == "annotation-audit"


def test_mann_whitney_auc_uses_half_tie_credit() -> None:
    assert mann_whitney_auc([2.0, 1.0], [1.0, 0.0]) == pytest.approx(0.875)


def test_receipt_reports_trajectory_aucs_and_substrate_gate() -> None:
    events = [
        _event("a1", trajectory_id="attack-a", event_index=0, surprise=1.4, attack=True),
        _event("a2", trajectory_id="attack-a", event_index=1, surprise=1.4, attack=True),
        _event("a3", trajectory_id="attack-b", event_index=0, surprise=6.0, attack=True),
        _event("b1", trajectory_id="benign-a", event_index=0, surprise=0.8, attack=False),
        _event("b2", trajectory_id="benign-b", event_index=0, surprise=0.7, attack=False),
    ]

    receipt = build_temporal_receipt(
        events,
        _config(),
        min_productive_band_fraction=0.5,
        min_cells_positive_b_mass=1,
    )

    assert receipt.receipt_kind == "generic_temporal_scorer"
    assert receipt.receipt_version == "temporal_accumulation/1"
    assert receipt.aucs["a_peak"] == pytest.approx(1.0)
    assert receipt.aucs["b_persistence"] == pytest.approx(1.0)
    assert receipt.substrate_gate_result.productive_band_fraction == pytest.approx(2 / 3)
    assert receipt.substrate_gate_result.n_cells_positive_b_mass == 2
    assert receipt.substrate_gate_result.passed is True
    assert receipt.events[0].event_attack_label is True
    assert receipt.events[0].cdf_fallback_level == "exact"
    assert receipt.headline_results.n_independent_units == 4
    assert (
        receipt.headline_results.criterion_provenance["positive_b_mass_cell"]
        == "attack_cell_with_any_event_b_persistence_gt_zero"
    )


def test_positive_b_mass_counts_attack_cell_benign_filler_mass() -> None:
    events = [
        _event(
            "a1",
            trajectory_id="attack-a",
            event_index=0,
            surprise=0.5,
            attack=True,
            event_attack=True,
        ),
        _event(
            "a2",
            trajectory_id="attack-a",
            event_index=1,
            surprise=1.6,
            attack=True,
            event_attack=False,
        ),
        _event("b1", trajectory_id="benign-a", event_index=0, surprise=0.5, attack=False),
    ]

    receipt = build_temporal_receipt(
        events,
        _config(),
        min_productive_band_fraction=0.0,
        min_cells_positive_b_mass=1,
    )

    assert receipt.substrate_gate_result.productive_band_fraction == 0.0
    assert receipt.substrate_gate_result.n_cells_positive_b_mass == 1
    assert receipt.substrate_gate_result.passed is True


def test_canonical_key_hash_distinguishes_missing_from_empty_string() -> None:
    missing = [_event("e1", event_index=0, surprise=1.6, keys={})]
    empty = [
        _event(
            "e1",
            event_index=0,
            surprise=1.6,
            keys={
                "tool_target": "",
                "source_document_id": "",
                "instruction_channel": "",
                "task_epoch": "",
            },
        )
    ]

    missing_score = score_temporal_events(missing, _config())[0]
    empty_score = score_temporal_events(empty, _config())[0]

    assert missing_score.continuity_key_values_hash_strict is None
    assert empty_score.continuity_key_values_hash_strict == canonical_json_sha256(
        {
            "tool_target": "",
            "source_document_id": "",
            "instruction_channel": "",
            "task_epoch": "",
        }
    )


def test_temporal_jsonl_round_trip_writes_deterministic_receipt() -> None:
    tmp_path = _new_output_dir("temporal_jsonl_round_trip")
    events = [
        _event("a1", trajectory_id="attack-a", event_index=0, surprise=1.4, attack=True),
        _event("b1", trajectory_id="benign-a", event_index=0, surprise=0.8, attack=False),
    ]
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        "".join(json.dumps(event.model_dump(mode="json"), sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )

    loaded = load_temporal_events_jsonl(events_path)
    receipt = build_temporal_receipt(
        loaded,
        _config(),
        min_productive_band_fraction=0.0,
        min_cells_positive_b_mass=1,
    )
    out_path = tmp_path / "receipt.json"
    write_temporal_receipt(out_path, receipt)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["events_sha256"] == canonical_json_sha256(
        [event.model_dump(mode="json") for event in loaded]
    )
    assert payload["scored_events"][0]["event_id"] == "a1"


def test_receipt_artifact_id_changes_with_accumulator_config() -> None:
    events = [
        _event("a1", trajectory_id="attack-a", event_index=0, surprise=1.4, attack=True),
        _event("b1", trajectory_id="benign-a", event_index=0, surprise=0.8, attack=False),
    ]
    first = build_temporal_receipt(events, _config(), min_cells_positive_b_mass=1)
    changed = _config().model_copy(update={"k_u_b": 1.1})
    second = build_temporal_receipt(events, changed, min_cells_positive_b_mass=1)

    assert first.events_sha256 == second.events_sha256
    assert first.artifact_id != second.artifact_id


def test_receipt_artifact_id_changes_with_substrate_thresholds() -> None:
    events = [
        _event("a1", trajectory_id="attack-a", event_index=0, surprise=1.4, attack=True),
        _event("b1", trajectory_id="benign-a", event_index=0, surprise=0.8, attack=False),
    ]
    first = build_temporal_receipt(events, _config(), min_cells_positive_b_mass=1)
    second = build_temporal_receipt(events, _config(), min_cells_positive_b_mass=15)

    assert first.substrate_gate_result != second.substrate_gate_result
    assert first.artifact_id != second.artifact_id


def test_temporal_score_cli_writes_receipt(capsys) -> None:
    from parapet_runner.runner import main

    tmp_path = _new_output_dir("temporal_score_cli")
    events = [
        _event("a1", trajectory_id="attack-a", event_index=0, surprise=1.4, attack=True),
        _event("b1", trajectory_id="benign-a", event_index=0, surprise=0.8, attack=False),
    ]
    events_path = tmp_path / "events.jsonl"
    receipt_path = tmp_path / "receipt.json"
    events_path.write_text(
        "".join(json.dumps(event.model_dump(mode="json"), sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "temporal-score",
            "--events-jsonl",
            str(events_path),
            "--output-receipt",
            str(receipt_path),
            "--k-u-b",
            "1.0",
            "--k-u-c-strict",
            "1.0",
            "--k-u-c-broad",
            "1.0",
            "--peak-alert-level",
            "5.0",
            "--min-cells-positive-b-mass",
            "1",
        ]
    )

    assert exit_code == 0
    assert str(receipt_path.resolve()) in capsys.readouterr().out
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["receipt_kind"] == "generic_temporal_scorer"
    assert payload["events"][0]["label_source_ref"] == "fixture"
    assert payload["aucs"]["a_peak"] == pytest.approx(1.0)


def test_compute_aucs_requires_attack_and_benign_cells() -> None:
    events = [_event("a1", trajectory_id="attack-a", event_index=0, surprise=1.4, attack=True)]
    scored = score_temporal_events(events, _config())
    results = summarize_trajectories(events, scored)

    with pytest.raises(ValueError, match="one positive and one negative"):
        compute_aucs(results)
