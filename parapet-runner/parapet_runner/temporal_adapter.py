"""Fail-closed assembly of claim-bearing raw-envelope detector events.

Source-specific parsing and D_eval execution stay behind owning-team boundaries.
This module owns the routing seam: it joins exact envelope payloads to frozen
detector/CDF observations, validates identity and provenance, and emits payload-
free :class:`TemporalEvent` rows ready for the temporal scorer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .temporal import TemporalEvent, TemporalLabels
from .temporal_claim import TemporalTrajectoryStrata


EVENT_ID_SCHEMA = "temporal-event-id/1"
RAW_ENVELOPE_SCHEMA = "temporal-raw-envelope-event/1"
DETECTOR_OBSERVATION_SCHEMA = "temporal-detector-observation/1"
ADAPTER_PINS_SCHEMA = "p3-temporal-adapter-pins/1"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
P3_INDEX_SHA256 = "95ab8177282f65f4f9007617f3480f13e557ea9e835c01e69cbcd56bd1cdcab1"
P3_DETECTOR_ID = "DEvalEnsemble[L1,EmbedContrastive]"


class TemporalAdapterContractError(ValueError):
    """Raised when claim-bearing adapter inputs violate the frozen contract."""


class StrictContractModel(BaseModel):
    """Fail closed on unknown fields and type coercion at adapter boundaries."""

    model_config = ConfigDict(extra="forbid", strict=True)


class TemporalEventAssembler(Protocol):
    """Interface for joining envelope events to detector observations."""

    def __call__(
        self,
        raw_events: Sequence["RawEnvelopeEvent"],
        observations: Sequence["DetectorObservation"],
        pins: "P3TemporalAdapterPins",
    ) -> list[TemporalEvent]:
        ...


class RawEnvelopeProvenance(StrictContractModel):
    """Content-free provenance for the exact source envelope location."""

    source_artifact_ref: str
    source_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    source_message_ordinal: int = Field(ge=0)
    source_tool_ordinal: int | None = Field(default=None, ge=0)
    source_call_id: str | None = None
    extractor_id: str
    extractor_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def require_named_refs(self) -> "RawEnvelopeProvenance":
        for field_name in ("source_artifact_ref", "extractor_id"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        return self


class FrozenEventLabels(StrictContractModel):
    """Strict adapter-side labels converted to the scorer's label model."""

    event_attack_label: bool = False
    trajectory_label: Literal["attack", "benign"]
    cell_label: Literal["attack", "benign"]
    label_source_ref: str
    label_provenance: Literal["construction", "annotation"]
    label_audit_ref: str | None = None

    @model_validator(mode="after")
    def validate_labels(self) -> "FrozenEventLabels":
        if not self.label_source_ref.strip():
            raise ValueError("label_source_ref must be non-empty")
        if self.trajectory_label != self.cell_label:
            raise ValueError("trajectory_label and cell_label must match in V1")
        if self.trajectory_label == "benign" and self.event_attack_label:
            raise ValueError("benign trajectories cannot contain attack-labeled events")
        if self.label_provenance == "annotation" and not self.label_audit_ref:
            raise ValueError("annotation labels require label_audit_ref")
        return self


class RawEnvelopeEvent(StrictContractModel):
    """Exact scored payload and owner-supplied envelope/label provenance.

    ``source_event_ordinal`` is the original source order for scoreable events.
    It may have gaps, but must be strictly increasing within a trajectory. The
    adapter assigns the separate zero-based contiguous ``TemporalEvent`` index.
    """

    schema_version: Literal["temporal-raw-envelope-event/1"] = RAW_ENVELOPE_SCHEMA
    trajectory_id: str
    cell_id: str
    source_event_ordinal: int = Field(ge=0)
    event_text: str
    event_text_sha256: str = Field(pattern=SHA256_PATTERN)
    span_type: Literal["arg_values", "tool_output", "message_text"]
    instruction_channel: Literal["system", "developer", "user", "assistant", "tool"]
    tool_target: str | None = None
    tool_target_provenance_ref: str | None = None
    source_document_id: str | None = None
    source_document_id_provenance_ref: str | None = None
    task_epoch: int | str | None = None
    task_epoch_provenance: (
        Literal["user_explicit", "controller_deterministic"] | None
    ) = None
    population: Literal["attack_eval", "benign_eval"]
    labels: FrozenEventLabels
    trajectory_strata: TemporalTrajectoryStrata
    source_receipt_ref: str
    provenance: RawEnvelopeProvenance

    @model_validator(mode="after")
    def validate_envelope_contract(self) -> "RawEnvelopeEvent":
        if not self.trajectory_id.strip() or not self.cell_id.strip():
            raise ValueError("trajectory_id and cell_id must be non-empty")
        if not self.event_text.strip():
            raise ValueError("event_text must be non-empty")
        if sha256_text(self.event_text) != self.event_text_sha256:
            raise ValueError("event_text_sha256 does not match event_text")
        if not self.instruction_channel.strip():
            raise ValueError("instruction_channel must be the non-empty raw envelope role")
        for field_name in ("tool_target", "source_document_id"):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must be non-empty when present")
        if self.tool_target is not None and not self.tool_target_provenance_ref:
            raise ValueError("tool_target requires tool_target_provenance_ref")
        if self.tool_target is None and self.tool_target_provenance_ref is not None:
            raise ValueError("tool_target_provenance_ref requires tool_target")
        if self.source_document_id is not None and not self.source_document_id_provenance_ref:
            raise ValueError(
                "source_document_id requires source_document_id_provenance_ref"
            )
        if self.source_document_id is None and self.source_document_id_provenance_ref is not None:
            raise ValueError("source_document_id_provenance_ref requires source_document_id")
        if self.task_epoch is not None and self.task_epoch_provenance is None:
            raise ValueError("task_epoch requires task_epoch_provenance")
        if self.task_epoch is None and self.task_epoch_provenance is not None:
            raise ValueError("task_epoch_provenance requires task_epoch")
        expected_population = (
            "attack_eval" if self.labels.trajectory_label == "attack" else "benign_eval"
        )
        if self.population != expected_population:
            raise ValueError("population must match trajectory_label")
        self.trajectory_strata.validate_for_trajectory_label(
            self.labels.trajectory_label
        )
        if not self.source_receipt_ref.strip():
            raise ValueError("source_receipt_ref must be non-empty")
        return self


class P3MemberScores(StrictContractModel):
    """The two frozen D_eval ensemble member outputs."""

    l1: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    contrastive: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class P3ScoreContext(StrictContractModel):
    """Exact payload-free P3 CDF context; unknown fields fail closed."""

    tool_family: str
    action: Literal["action", "nonaction"]

    @model_validator(mode="after")
    def require_named_tool_family(self) -> "P3ScoreContext":
        if not self.tool_family.strip():
            raise ValueError("tool_family must be non-empty")
        return self


class DetectorObservationProvenance(StrictContractModel):
    """Content-free per-event detector and surprise provenance."""

    detector_receipt_ref: str
    surprise_receipt_ref: str
    member_scores: P3MemberScores

    @model_validator(mode="after")
    def validate_provenance(self) -> "DetectorObservationProvenance":
        if not self.detector_receipt_ref.strip() or not self.surprise_receipt_ref.strip():
            raise ValueError("detector and surprise receipt refs must be non-empty")
        return self


class DetectorObservation(StrictContractModel):
    """Frozen D_eval and CDF observation for one exact envelope payload."""

    schema_version: Literal["temporal-detector-observation/1"] = (
        DETECTOR_OBSERVATION_SCHEMA
    )
    trajectory_id: str
    source_event_ordinal: int = Field(ge=0)
    event_text_sha256: str = Field(pattern=SHA256_PATTERN)
    event_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    surprise: float = Field(ge=0.0, allow_inf_nan=False)
    score_context: P3ScoreContext
    cdf_fallback_level: Literal["exact", "tool_family", "action", "global"]
    detector_provenance: DetectorObservationProvenance
    hard_trigger: bool = False
    hard_trigger_source_ref: str | None = None

    @model_validator(mode="after")
    def validate_observation_contract(self) -> "DetectorObservation":
        if not self.trajectory_id.strip():
            raise ValueError("trajectory_id must be non-empty")
        if self.hard_trigger and not self.hard_trigger_source_ref:
            raise ValueError("hard triggers require hard_trigger_source_ref")
        if not self.hard_trigger and self.hard_trigger_source_ref is not None:
            raise ValueError("hard_trigger_source_ref requires hard_trigger")
        return self


class P3DetectorPins(StrictContractModel):
    """Frozen detector-of-record identity required by the adapter contract."""

    detector_id: Literal["DEvalEnsemble[L1,EmbedContrastive]"] = P3_DETECTOR_ID
    detector_code_sha256: str = Field(pattern=SHA256_PATTERN)
    l1_weights_sha256: str = Field(pattern=SHA256_PATTERN)
    contrastive_bank_id: str
    contrastive_bank_sha256: str = Field(pattern=SHA256_PATTERN)
    threshold: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def require_named_artifacts(self) -> "P3DetectorPins":
        if not self.contrastive_bank_id.strip():
            raise ValueError("contrastive_bank_id must be non-empty")
        return self


class P3SurpriseReferencePins(StrictContractModel):
    """Frozen P3 benign-CDF recipe and artifact identity."""

    artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    alpha: int = 1
    exact_context_floor: int = 50
    context_fields: tuple[str, str] = ("tool_family", "action")
    fallback_order: tuple[str, str, str, str] = (
        "exact",
        "tool_family",
        "action",
        "global",
    )

    @model_validator(mode="after")
    def require_frozen_recipe(self) -> "P3SurpriseReferencePins":
        if self.alpha != 1:
            raise ValueError("alpha must remain 1")
        if self.exact_context_floor != 50:
            raise ValueError("exact_context_floor must remain 50")
        if self.context_fields != ("tool_family", "action"):
            raise ValueError("context_fields drifted from the frozen P3 recipe")
        if self.fallback_order != (
            "exact",
            "tool_family",
            "action",
            "global",
        ):
            raise ValueError("fallback_order drifted from the frozen P3 recipe")
        return self


class P3TemporalAdapterPins(StrictContractModel):
    """Explicit claim-bearing adapter configuration."""

    schema_version: Literal["p3-temporal-adapter-pins/1"] = ADAPTER_PINS_SCHEMA
    layer_id: Literal["p3_d_eval"] = "p3_d_eval"
    index_sha256: str = Field(pattern=SHA256_PATTERN)
    detector: P3DetectorPins
    surprise_reference: P3SurpriseReferencePins
    input_artifact_shas: dict[str, str]
    code_refs: dict[str, str]
    contract_ref: str

    @model_validator(mode="after")
    def validate_pins(self) -> "P3TemporalAdapterPins":
        if self.index_sha256 != P3_INDEX_SHA256:
            raise ValueError("index_sha256 does not match the frozen P3 index.v2 artifact")
        if self.input_artifact_shas.get("index.v2.json") != self.index_sha256:
            raise ValueError("input_artifact_shas must pin index.v2.json to index_sha256")
        if not self.code_refs:
            raise ValueError("code_refs must name the source extractor")
        if not self.contract_ref.strip():
            raise ValueError("contract_ref must be non-empty")
        for name, digest in self.input_artifact_shas.items():
            if not name or not _is_sha256(digest):
                raise ValueError(f"invalid input artifact SHA for {name!r}")
        for name, code_ref in self.code_refs.items():
            if not name.strip() or not code_ref.strip():
                raise ValueError("code_refs names and values must be non-empty")
        return self


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON bytes used by event identity and deterministic JSONL."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_temporal_event_id(trajectory_id: str, event_index: int, layer_id: str) -> str:
    identity = {
        "schema": EVENT_ID_SCHEMA,
        "trajectory_id": trajectory_id,
        "event_index": event_index,
        "layer_id": layer_id,
    }
    return "te1:" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _validate_trajectory_id(trajectory_id: str) -> None:
    if not trajectory_id:
        raise TemporalAdapterContractError("trajectory_id is absent")
    path = PurePosixPath(trajectory_id)
    if path.is_absolute() or trajectory_id.startswith("/"):
        raise TemporalAdapterContractError(f"trajectory_id is absolute: {trajectory_id}")
    if ".." in path.parts:
        raise TemporalAdapterContractError(
            f"trajectory_id contains parent traversal: {trajectory_id}"
        )


def _event_key(event: RawEnvelopeEvent | DetectorObservation) -> tuple[str, int]:
    return (event.trajectory_id, event.source_event_ordinal)


def _continuity_keys(raw: RawEnvelopeEvent) -> dict[str, Any]:
    keys: dict[str, Any] = {}
    if raw.tool_target is not None:
        keys["tool_target"] = raw.tool_target
    if raw.source_document_id is not None:
        keys["source_document_id"] = raw.source_document_id
    keys["instruction_channel"] = raw.instruction_channel
    if raw.task_epoch is not None:
        keys["task_epoch"] = raw.task_epoch
    return keys


def _trajectory_label_contract(labels: FrozenEventLabels) -> dict[str, Any]:
    """Fields that must be stable while event_attack_label may vary by event."""

    return {
        "trajectory_label": labels.trajectory_label,
        "cell_label": labels.cell_label,
        "label_source_ref": labels.label_source_ref,
        "label_provenance": labels.label_provenance,
        "label_audit_ref": labels.label_audit_ref,
    }


def assemble_temporal_events(
    raw_events: Sequence[RawEnvelopeEvent],
    observations: Sequence[DetectorObservation],
    pins: P3TemporalAdapterPins,
) -> list[TemporalEvent]:
    """Join exact payloads to frozen observations and emit payload-free events.

    The input order is authoritative. Trajectories must be contiguous, and source
    ordinals must be strictly increasing within each trajectory. Detector rows may
    arrive in any order but must join one-to-one by trajectory and source ordinal.
    """

    if not raw_events:
        raise TemporalAdapterContractError("adapter requires at least one raw envelope event")

    observation_by_key: dict[tuple[str, int], DetectorObservation] = {}
    for observation in observations:
        key = _event_key(observation)
        if key in observation_by_key:
            raise TemporalAdapterContractError(
                f"duplicate detector observation for {key[0]} ordinal {key[1]}"
            )
        observation_by_key[key] = observation

    output: list[TemporalEvent] = []
    used_observations: set[tuple[str, int]] = set()
    seen_raw_keys: set[tuple[str, int]] = set()
    seen_event_ids: set[str] = set()
    closed_trajectories: set[str] = set()
    current_trajectory: str | None = None
    trajectory_index = 0
    previous_source_ordinal: int | None = None
    trajectory_contract: dict[str, Any] | None = None

    for raw in raw_events:
        _validate_trajectory_id(raw.trajectory_id)
        raw_key = _event_key(raw)
        if raw_key in seen_raw_keys:
            raise TemporalAdapterContractError(
                f"duplicate raw envelope event for {raw_key[0]} ordinal {raw_key[1]}"
            )
        seen_raw_keys.add(raw_key)

        if raw.trajectory_id != current_trajectory:
            if current_trajectory is not None:
                closed_trajectories.add(current_trajectory)
            if raw.trajectory_id in closed_trajectories:
                raise TemporalAdapterContractError(
                    f"trajectory is not contiguous in input: {raw.trajectory_id}"
                )
            current_trajectory = raw.trajectory_id
            trajectory_index = 0
            previous_source_ordinal = None
            trajectory_contract = {
                "cell_id": raw.cell_id,
                "population": raw.population,
                "labels": _trajectory_label_contract(raw.labels),
                "trajectory_strata": raw.trajectory_strata.model_dump(mode="json"),
                "source_receipt_ref": raw.source_receipt_ref,
            }
        else:
            assert trajectory_contract is not None
            for field_name, expected in trajectory_contract.items():
                actual: Any
                if field_name == "labels":
                    actual = _trajectory_label_contract(raw.labels)
                elif field_name == "trajectory_strata":
                    actual = raw.trajectory_strata.model_dump(mode="json")
                else:
                    actual = getattr(raw, field_name)
                if actual != expected:
                    raise TemporalAdapterContractError(
                        f"{field_name} drift within trajectory {raw.trajectory_id}"
                    )

        if previous_source_ordinal is not None and raw.source_event_ordinal <= previous_source_ordinal:
            raise TemporalAdapterContractError(
                f"source_event_ordinal must be strictly increasing within {raw.trajectory_id}"
            )
        previous_source_ordinal = raw.source_event_ordinal

        observation = observation_by_key.get(raw_key)
        if observation is None:
            raise TemporalAdapterContractError(
                f"missing detector observation for {raw.trajectory_id} ordinal "
                f"{raw.source_event_ordinal}"
            )
        used_observations.add(raw_key)
        if observation.event_text_sha256 != raw.event_text_sha256:
            raise TemporalAdapterContractError(
                f"payload hash mismatch for {raw.trajectory_id} ordinal "
                f"{raw.source_event_ordinal}"
            )
        if observation.cdf_fallback_level not in pins.surprise_reference.fallback_order:
            raise TemporalAdapterContractError(
                f"unsupported CDF fallback level {observation.cdf_fallback_level!r} for "
                f"{raw.trajectory_id} ordinal {raw.source_event_ordinal}"
            )

        event_id = make_temporal_event_id(raw.trajectory_id, trajectory_index, pins.layer_id)
        if event_id in seen_event_ids:
            raise TemporalAdapterContractError(f"duplicate event_id: {event_id}")
        seen_event_ids.add(event_id)

        event = TemporalEvent(
            event_id=event_id,
            trajectory_id=raw.trajectory_id,
            cell_id=raw.cell_id,
            event_index=trajectory_index,
            layer_id=pins.layer_id,
            event_score=observation.event_score,
            surprise=observation.surprise,
            population=raw.population,
            score_context=observation.score_context.model_dump(mode="json"),
            cdf_fallback_level=observation.cdf_fallback_level,
            continuity_keys=_continuity_keys(raw),
            labels=TemporalLabels.model_validate(raw.labels.model_dump(mode="json")),
            provenance={
                "adapter": {
                    "schema_version": ADAPTER_PINS_SCHEMA,
                    "contract_ref": pins.contract_ref,
                    "index_sha256": pins.index_sha256,
                },
                "envelope": {
                    "source_event_ordinal": raw.source_event_ordinal,
                    "span_type": raw.span_type,
                    "instruction_channel": raw.instruction_channel,
                    "event_text_sha256": raw.event_text_sha256,
                    "tool_target_provenance_ref": raw.tool_target_provenance_ref,
                    "source_document_id_provenance_ref": (
                        raw.source_document_id_provenance_ref
                    ),
                    "source": raw.provenance.model_dump(mode="json"),
                },
                "detector": observation.detector_provenance.model_dump(mode="json"),
            },
            task_epoch_provenance=raw.task_epoch_provenance,
            hard_trigger=observation.hard_trigger,
            hard_trigger_source_ref=observation.hard_trigger_source_ref,
            source_receipt_ref=raw.source_receipt_ref,
            trajectory_strata=raw.trajectory_strata,
        )
        output.append(event)
        trajectory_index += 1

    unused = sorted(set(observation_by_key) - used_observations)
    if unused:
        trajectory_id, ordinal = unused[0]
        raise TemporalAdapterContractError(
            f"unused detector observation for {trajectory_id} ordinal {ordinal}"
        )
    return output
