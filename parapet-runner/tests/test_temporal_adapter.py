"""Behavior tests for the claim-bearing raw-envelope to TemporalEvent adapter."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from parapet_runner.runner import main as runner_main
from parapet_runner.temporal_adapter import (
    DetectorObservation,
    DetectorObservationProvenance,
    FrozenEventLabels,
    P3DetectorPins,
    P3_INDEX_SHA256,
    P3SurpriseReferencePins,
    P3TemporalAdapterPins,
    RawEnvelopeProvenance,
    RawEnvelopeEvent,
    TemporalAdapterContractError,
    assemble_temporal_events,
    make_temporal_event_id,
)
from parapet_runner.temporal_adapter_io import adapt_temporal_events_jsonl


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _labels(kind: str, *, event_attack: bool = False) -> FrozenEventLabels:
    return FrozenEventLabels(
        event_attack_label=event_attack,
        trajectory_label=kind,
        cell_label=kind,
        label_source_ref="manifest.json#cells/c1",
        label_provenance="construction",
    )


def _raw(
    *,
    trajectory_id: str = "trajectory/source/run.json",
    cell_id: str = "source|suite|cell",
    source_event_ordinal: int = 4,
    event_text: str = "exact payload",
    instruction_channel: str = "tool",
    population: str = "attack_eval",
    labels: FrozenEventLabels | None = None,
    tool_target: str | None = "read_file",
    source_document_id: str | None = None,
    task_epoch: int | str | None = None,
    task_epoch_provenance: str | None = None,
) -> RawEnvelopeEvent:
    effective_labels = labels or _labels("attack", event_attack=True)
    trajectory_strata = {"cohort_surface": "swe_coding"}
    if effective_labels.trajectory_label == "attack":
        trajectory_strata.update(
            generator="fixture-generator",
            mechanism="fixture-mechanism",
            surface="filesystem",
        )
    return RawEnvelopeEvent(
        trajectory_id=trajectory_id,
        cell_id=cell_id,
        source_event_ordinal=source_event_ordinal,
        event_text=event_text,
        event_text_sha256=_sha(event_text),
        span_type="tool_output",
        instruction_channel=instruction_channel,
        tool_target=tool_target,
        tool_target_provenance_ref=(
            "raw-trace.json#messages/4/tool_calls/0/function" if tool_target else None
        ),
        source_document_id=source_document_id,
        source_document_id_provenance_ref=(
            "runtime-receipt.json#documents/1" if source_document_id else None
        ),
        task_epoch=task_epoch,
        task_epoch_provenance=task_epoch_provenance,
        population=population,
        labels=effective_labels,
        trajectory_strata=trajectory_strata,
        source_receipt_ref="raw-envelope-receipt.json",
        provenance=RawEnvelopeProvenance(
            source_artifact_ref="raw-trace.json",
            source_artifact_sha256="6" * 64,
            source_message_ordinal=source_event_ordinal,
            source_tool_ordinal=0,
            source_call_id=None,
            extractor_id="lab-envelope-extractor/1",
            extractor_sha256="7" * 64,
        ),
    )


def _observation(raw: RawEnvelopeEvent) -> DetectorObservation:
    return DetectorObservation(
        trajectory_id=raw.trajectory_id,
        source_event_ordinal=raw.source_event_ordinal,
        event_text_sha256=raw.event_text_sha256,
        event_score=0.42,
        surprise=1.25,
        score_context={"tool_family": "filesystem", "action": "nonaction"},
        cdf_fallback_level="exact",
        detector_provenance=DetectorObservationProvenance(
            detector_receipt_ref="deval-receipt.json",
            surprise_receipt_ref="cdf-receipt.json",
            member_scores={"l1": 0.42, "contrastive": 0.05},
        ),
    )


def _pins() -> P3TemporalAdapterPins:
    return P3TemporalAdapterPins(
        index_sha256=P3_INDEX_SHA256,
        detector=P3DetectorPins(
            detector_id="DEvalEnsemble[L1,EmbedContrastive]",
            detector_code_sha256="2" * 64,
            l1_weights_sha256="3" * 64,
            contrastive_bank_id="p3-contrastive-reference/3",
            contrastive_bank_sha256="4" * 64,
            threshold=0.5,
        ),
        surprise_reference=P3SurpriseReferencePins(
            artifact_sha256="5" * 64,
        ),
        input_artifact_shas={"index.v2.json": P3_INDEX_SHA256},
        code_refs={"raw_envelope_extractor": "lab-owned/adapter@fixture"},
        contract_ref="temporal_event_adapter_plan_2026-07-09.md",
    )


def test_assemble_preserves_order_roles_and_honest_missingness() -> None:
    first = _raw(source_event_ordinal=4, event_text="one")
    second = _raw(
        source_event_ordinal=9,
        event_text="two",
        instruction_channel="assistant",
        labels=_labels("attack", event_attack=False),
    )
    benign = _raw(
        trajectory_id="trajectory/benign/run.json",
        cell_id="benign|suite|cell",
        source_event_ordinal=2,
        event_text="three",
        population="benign_eval",
        labels=_labels("benign"),
        tool_target=None,
    )

    events = assemble_temporal_events(
        [first, second, benign],
        [_observation(second), _observation(benign), _observation(first)],
        _pins(),
    )

    assert [event.event_index for event in events] == [0, 1, 0]
    assert [event.event_id for event in events[:2]] == [
        make_temporal_event_id(first.trajectory_id, 0, _pins().layer_id),
        make_temporal_event_id(first.trajectory_id, 1, _pins().layer_id),
    ]
    assert events[0].continuity_keys == {
        "tool_target": "read_file",
        "instruction_channel": "tool",
    }
    assert events[1].continuity_keys["instruction_channel"] == "assistant"
    assert events[0].labels.event_attack_label is True
    assert events[1].labels.event_attack_label is False
    assert events[0].trajectory_strata is not None
    assert events[0].trajectory_strata.cohort_surface == "swe_coding"
    assert events[0].trajectory_strata.surface == "filesystem"
    assert events[2].trajectory_strata is not None
    assert events[2].trajectory_strata.cohort_surface == "swe_coding"
    assert events[2].trajectory_strata.generator is None
    assert events[2].trajectory_strata.model_dump(mode="json") == {
        "cohort_surface": "swe_coding"
    }
    assert events[2].continuity_keys == {"instruction_channel": "tool"}
    assert events[0].provenance["envelope"]["source_event_ordinal"] == 4
    assert events[0].provenance["envelope"]["event_text_sha256"] == _sha("one")
    assert "event_text" not in events[0].model_dump()
    assert "event_text" not in events[0].provenance["envelope"]


def test_continuity_values_require_owner_provenance() -> None:
    with pytest.raises(ValidationError, match="tool_target requires tool_target_provenance_ref"):
        _raw().model_copy(update={"tool_target_provenance_ref": None}).model_validate(
            _raw().model_copy(update={"tool_target_provenance_ref": None}).model_dump()
        )

    with pytest.raises(
        ValidationError,
        match="source_document_id requires source_document_id_provenance_ref",
    ):
        _raw(source_document_id="doc-1").model_copy(
            update={"source_document_id_provenance_ref": None}
        ).model_validate(
            _raw(source_document_id="doc-1").model_copy(
                update={"source_document_id_provenance_ref": None}
            ).model_dump()
        )


def test_task_epoch_requires_owner_provenance() -> None:
    with pytest.raises(ValidationError, match="task_epoch requires task_epoch_provenance"):
        _raw(task_epoch=3)

    with pytest.raises(ValidationError, match="task_epoch_provenance requires task_epoch"):
        _raw(task_epoch_provenance="controller_deterministic")


def test_benign_events_and_hard_trigger_refs_are_consistent() -> None:
    with pytest.raises(ValidationError, match="benign trajectories cannot"):
        _labels("benign", event_attack=True)

    raw = _raw()
    with pytest.raises(ValidationError, match="hard_trigger_source_ref requires"):
        DetectorObservation.model_validate(
            _observation(raw).model_copy(
                update={"hard_trigger_source_ref": "receipt.json#trigger/0"}
            ).model_dump()
        )


@pytest.mark.parametrize("field_name", ["generator", "mechanism", "surface"])
def test_attack_trajectory_requires_each_attack_only_stratum(
    field_name: str,
) -> None:
    payload = _raw().model_dump(mode="json")
    del payload["trajectory_strata"][field_name]

    with pytest.raises(ValidationError, match=field_name):
        RawEnvelopeEvent.model_validate(payload)


@pytest.mark.parametrize("field_value", [None, "fabricated-stratum"])
@pytest.mark.parametrize("field_name", ["generator", "mechanism", "surface"])
def test_benign_trajectory_requires_attack_only_strata_to_be_absent(
    field_name: str,
    field_value: str | None,
) -> None:
    benign = _raw(
        trajectory_id="trajectory/benign/run.json",
        population="benign_eval",
        labels=_labels("benign"),
    )
    payload = benign.model_dump(mode="json")
    assert payload["trajectory_strata"] == {"cohort_surface": "swe_coding"}

    payload["trajectory_strata"][field_name] = field_value
    with pytest.raises(ValidationError, match=field_name):
        RawEnvelopeEvent.model_validate(payload)


def test_surface_relation_is_not_a_live_p3_v1_field() -> None:
    payload = _raw().model_dump(mode="json")
    payload["trajectory_strata"]["surface_relation"] = "within_surface"

    with pytest.raises(ValidationError, match="surface_relation"):
        RawEnvelopeEvent.model_validate(payload)


def test_contract_models_reject_unknown_fields() -> None:
    payload = _raw().model_dump(mode="json")
    payload["unreviewed_payload_copy"] = "must not be ignored"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RawEnvelopeEvent.model_validate(payload)

    observation_payload = _observation(_raw()).model_dump(mode="json")
    observation_payload["score_context"]["event_text"] = "must not reach output"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DetectorObservation.model_validate(observation_payload)


@pytest.mark.parametrize(
    ("trajectory_id", "message"),
    [
        ("/absolute/run.json", "absolute"),
        ("trajectory/../run.json", "parent traversal"),
    ],
)
def test_unsafe_trajectory_id_fails_closed(trajectory_id: str, message: str) -> None:
    raw = _raw(trajectory_id=trajectory_id)
    with pytest.raises(TemporalAdapterContractError, match=message):
        assemble_temporal_events([raw], [_observation(raw)], _pins())


def test_payload_hash_mismatch_fails_closed() -> None:
    raw = _raw()
    observation = _observation(raw).model_copy(update={"event_text_sha256": "f" * 64})
    with pytest.raises(TemporalAdapterContractError, match="payload hash mismatch"):
        assemble_temporal_events([raw], [observation], _pins())


def test_unknown_cdf_fallback_level_fails_closed() -> None:
    raw = _raw()
    observation = _observation(raw).model_copy(update={"cdf_fallback_level": "invented"})
    with pytest.raises(TemporalAdapterContractError, match="unsupported CDF fallback level"):
        assemble_temporal_events([raw], [observation], _pins())


def test_missing_and_extra_observations_fail_closed() -> None:
    raw = _raw()
    with pytest.raises(TemporalAdapterContractError, match="missing detector observation"):
        assemble_temporal_events([raw], [], _pins())

    extra_raw = _raw(source_event_ordinal=8, event_text="extra")
    with pytest.raises(TemporalAdapterContractError, match="unused detector observation"):
        assemble_temporal_events([raw], [_observation(raw), _observation(extra_raw)], _pins())


def test_empty_adapter_input_fails_closed() -> None:
    with pytest.raises(TemporalAdapterContractError, match="at least one raw envelope event"):
        assemble_temporal_events([], [], _pins())


def test_source_order_and_trajectory_drift_fail_closed() -> None:
    later = _raw(source_event_ordinal=9, event_text="later")
    earlier = _raw(source_event_ordinal=4, event_text="earlier")
    with pytest.raises(TemporalAdapterContractError, match="strictly increasing"):
        assemble_temporal_events(
            [later, earlier], [_observation(later), _observation(earlier)], _pins()
        )

    drift = _raw(source_event_ordinal=10, event_text="drift", cell_id="other-cell")
    with pytest.raises(TemporalAdapterContractError, match="cell_id drift"):
        assemble_temporal_events(
            [earlier, drift], [_observation(earlier), _observation(drift)], _pins()
        )

    strata_drift = _raw(source_event_ordinal=10, event_text="strata-drift")
    strata_drift = strata_drift.model_copy(
        update={
            "trajectory_strata": strata_drift.trajectory_strata.model_copy(
                update={"mechanism": "other-mechanism"}
            )
        }
    )
    with pytest.raises(TemporalAdapterContractError, match="trajectory_strata drift"):
        assemble_temporal_events(
            [earlier, strata_drift],
            [_observation(earlier), _observation(strata_drift)],
            _pins(),
        )


def test_p3_pins_fail_closed_on_recipe_or_hash_drift() -> None:
    with pytest.raises(ValidationError, match="input_artifact_shas"):
        _pins().model_copy(update={"input_artifact_shas": {}}).model_validate(
            _pins().model_copy(update={"input_artifact_shas": {}}).model_dump()
        )

    with pytest.raises(ValidationError, match="alpha must remain 1"):
        P3SurpriseReferencePins(artifact_sha256="5" * 64, alpha=2)

    with pytest.raises(ValidationError, match="exact_context_floor must remain 50"):
        P3SurpriseReferencePins(artifact_sha256="5" * 64, exact_context_floor=49)

    with pytest.raises(ValidationError, match="frozen P3 index"):
        _pins().model_copy(update={"index_sha256": "0" * 64}).model_validate(
            _pins().model_copy(update={"index_sha256": "0" * 64}).model_dump()
        )


def test_jsonl_adapter_is_deterministic_and_payload_free() -> None:
    with tempfile.TemporaryDirectory(prefix="parapet-temporal-adapter-") as temp_dir:
        temp_path = Path(temp_dir)
        raw = _raw()
        observation = _observation(raw)
        raw_path = temp_path / "raw.jsonl"
        observation_path = temp_path / "observations.jsonl"
        pins_path = temp_path / "pins.json"
        output_path = temp_path / "events.jsonl"
        receipt_path = temp_path / "receipt.json"

        raw_path.write_text(raw.model_dump_json() + "\n", encoding="utf-8")
        observation_path.write_text(observation.model_dump_json() + "\n", encoding="utf-8")
        pins_path.write_text(_pins().model_dump_json(indent=2) + "\n", encoding="utf-8")

        receipt = adapt_temporal_events_jsonl(
            raw_events_path=raw_path,
            detector_observations_path=observation_path,
            pins_path=pins_path,
            output_events_path=output_path,
            output_receipt_path=receipt_path,
        )
        output_bytes = output_path.read_bytes()
        receipt_bytes = receipt_path.read_bytes()

        assert b"exact payload" not in output_bytes
        assert receipt.output.sha256 == hashlib.sha256(output_bytes).hexdigest()
        assert receipt.counts.events == 1
        assert set(receipt.adapter_code_shas.model_dump()) == {
            "assembly",
            "io",
            "temporal_contract",
        }
        assert receipt.continuity_coverage.complete_strict_events == 0
        assert receipt.continuity_coverage.complete_broad_events == 0

        adapt_temporal_events_jsonl(
            raw_events_path=raw_path,
            detector_observations_path=observation_path,
            pins_path=pins_path,
            output_events_path=output_path,
            output_receipt_path=receipt_path,
        )
        assert output_path.read_bytes() == output_bytes
        assert receipt_path.read_bytes() == receipt_bytes


def test_jsonl_adapter_rejects_output_over_input() -> None:
    with tempfile.TemporaryDirectory(prefix="parapet-temporal-adapter-") as temp_dir:
        temp_path = Path(temp_dir)
        raw = _raw()
        raw_path = temp_path / "raw.jsonl"
        observation_path = temp_path / "observations.jsonl"
        pins_path = temp_path / "pins.json"
        receipt_path = temp_path / "receipt.json"
        raw_path.write_text(raw.model_dump_json() + "\n", encoding="utf-8")
        observation_path.write_text(
            _observation(raw).model_dump_json() + "\n", encoding="utf-8"
        )
        pins_path.write_text(_pins().model_dump_json() + "\n", encoding="utf-8")

        with pytest.raises(TemporalAdapterContractError, match="must not overwrite an input"):
            adapt_temporal_events_jsonl(
                raw_events_path=raw_path,
                detector_observations_path=observation_path,
                pins_path=pins_path,
                output_events_path=raw_path,
                output_receipt_path=receipt_path,
            )


def test_jsonl_validation_error_does_not_echo_payload() -> None:
    with tempfile.TemporaryDirectory(prefix="parapet-temporal-adapter-error-") as temp_dir:
        temp_path = Path(temp_dir)
        secret_payload = "private payload must not appear in errors"
        raw = _raw(event_text=secret_payload)
        raw_payload = raw.model_dump(mode="json")
        raw_payload["event_text_sha256"] = "f" * 64
        raw_path = temp_path / "raw.jsonl"
        observation_path = temp_path / "observations.jsonl"
        pins_path = temp_path / "pins.json"
        raw_path.write_text(json.dumps(raw_payload) + "\n", encoding="utf-8")
        observation_path.write_text(
            _observation(raw).model_dump_json() + "\n", encoding="utf-8"
        )
        pins_path.write_text(_pins().model_dump_json() + "\n", encoding="utf-8")

        with pytest.raises(TemporalAdapterContractError) as raised:
            adapt_temporal_events_jsonl(
                raw_events_path=raw_path,
                detector_observations_path=observation_path,
                pins_path=pins_path,
                output_events_path=temp_path / "events.jsonl",
                output_receipt_path=temp_path / "receipt.json",
            )
        assert secret_payload not in str(raised.value)


def test_temporal_adapt_cli_emits_events_and_receipt() -> None:
    with tempfile.TemporaryDirectory(prefix="parapet-temporal-adapter-cli-") as temp_dir:
        temp_path = Path(temp_dir)
        raw = _raw()
        raw_path = temp_path / "raw.jsonl"
        observation_path = temp_path / "observations.jsonl"
        pins_path = temp_path / "pins.json"
        output_path = temp_path / "events.jsonl"
        receipt_path = temp_path / "receipt.json"
        raw_path.write_text(raw.model_dump_json() + "\n", encoding="utf-8")
        observation_path.write_text(
            _observation(raw).model_dump_json() + "\n", encoding="utf-8"
        )
        pins_path.write_text(_pins().model_dump_json() + "\n", encoding="utf-8")

        assert runner_main(
            [
                "temporal-adapt",
                "--raw-events-jsonl",
                str(raw_path),
                "--detector-observations-jsonl",
                str(observation_path),
                "--pins-json",
                str(pins_path),
                "--output-events-jsonl",
                str(output_path),
                "--output-receipt",
                str(receipt_path),
            ]
        ) == 0
        assert output_path.is_file()
        assert receipt_path.is_file()


def test_known_answer_event_id() -> None:
    assert make_temporal_event_id("fixtures/known_answer.json", 0, "p3_d_eval") == (
        "te1:fd5f4925f4c7d2d7f74e95bfc17808d849a6805aecb7cdd42717d0dc5ec7ed22"
    )
