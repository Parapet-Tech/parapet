"""Deterministic JSONL and receipt I/O for the temporal event adapter."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .temporal import DEFAULT_BROAD_KEYS, DEFAULT_STRICT_KEYS, TemporalEvent
from .temporal_adapter import (
    DetectorObservation,
    P3TemporalAdapterPins,
    RawEnvelopeEvent,
    StrictContractModel,
    TemporalAdapterContractError,
    assemble_temporal_events,
    canonical_json_bytes,
)


ADAPTER_RECEIPT_SCHEMA = "temporal-event-adapter-receipt/1"
ADAPTER_RESULT_LABEL = "temporal_event_adapter"
SHA256_PATTERN = r"^[0-9a-f]{64}$"


class AdapterArtifactRef(StrictContractModel):
    path: str
    sha256: str = Field(pattern=SHA256_PATTERN)


class TemporalAdapterCounts(StrictContractModel):
    events: int = Field(ge=0)
    trajectories: int = Field(ge=0)
    attack_events: int = Field(ge=0)
    benign_events: int = Field(ge=0)


class ContinuityCoverage(StrictContractModel):
    complete_strict_events: int = Field(ge=0)
    complete_broad_events: int = Field(ge=0)
    present_by_key: dict[str, int]


class AdapterCodeShas(StrictContractModel):
    """Hashes for every local module that affects adapter output semantics."""

    assembly: str = Field(pattern=SHA256_PATTERN)
    io: str = Field(pattern=SHA256_PATTERN)
    temporal_contract: str = Field(pattern=SHA256_PATTERN)


class TemporalAdapterReceipt(StrictContractModel):
    """Payload-free deterministic construction receipt for adapter output."""

    receipt_version: Literal["temporal-event-adapter-receipt/1"] = (
        ADAPTER_RECEIPT_SCHEMA
    )
    result_label: Literal["temporal_event_adapter"] = ADAPTER_RESULT_LABEL
    claim_scope: str
    contract_ref: str
    adapter_code_shas: AdapterCodeShas
    pins: P3TemporalAdapterPins
    inputs: dict[str, AdapterArtifactRef]
    output: AdapterArtifactRef
    counts: TemporalAdapterCounts
    continuity_coverage: ContinuityCoverage
    checks: tuple[str, ...]


def _load_jsonl_models(path: Path, model: type[BaseModel]) -> list[BaseModel]:
    if not path.is_file():
        raise TemporalAdapterContractError(f"input file does not exist: {path}")
    rows: list[BaseModel] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise TemporalAdapterContractError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
        try:
            rows.append(model.model_validate(value))
        except Exception as exc:
            raise TemporalAdapterContractError(
                f"invalid {model.__name__} at {path}:{line_no}: "
                f"{_validation_error_summary(exc)}"
            ) from exc
    return rows


def _validation_error_summary(exc: Exception) -> str:
    """Return validation diagnostics without echoing raw payload values."""

    if isinstance(exc, ValidationError):
        errors = [
            {
                "type": error.get("type"),
                "loc": list(error.get("loc", ())),
                "msg": error.get("msg"),
            }
            for error in exc.errors(include_input=False, include_url=False)
        ]
        return json.dumps(errors, ensure_ascii=False, sort_keys=True)
    return type(exc).__name__


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl_bytes(events: Sequence[TemporalEvent]) -> bytes:
    return b"".join(
        canonical_json_bytes(event.model_dump(mode="json")) + b"\n" for event in events
    )


def _continuity_coverage(events: Sequence[TemporalEvent]) -> ContinuityCoverage:
    present_by_key = {key: 0 for key in dict.fromkeys((*DEFAULT_STRICT_KEYS, *DEFAULT_BROAD_KEYS))}
    strict_complete = 0
    broad_complete = 0
    for event in events:
        for key in present_by_key:
            if key in event.continuity_keys and event.continuity_keys[key] is not None:
                present_by_key[key] += 1
        if all(event.continuity_keys.get(key) is not None for key in DEFAULT_STRICT_KEYS):
            strict_complete += 1
        if all(event.continuity_keys.get(key) is not None for key in DEFAULT_BROAD_KEYS):
            broad_complete += 1
    return ContinuityCoverage(
        complete_strict_events=strict_complete,
        complete_broad_events=broad_complete,
        present_by_key=present_by_key,
    )


def adapt_temporal_events_jsonl(
    *,
    raw_events_path: Path,
    detector_observations_path: Path,
    pins_path: Path,
    output_events_path: Path,
    output_receipt_path: Path,
) -> TemporalAdapterReceipt:
    """Read explicit inputs, emit deterministic TemporalEvent JSONL and receipt."""

    raw_events_path = raw_events_path.resolve()
    detector_observations_path = detector_observations_path.resolve()
    pins_path = pins_path.resolve()
    output_events_path = output_events_path.resolve()
    output_receipt_path = output_receipt_path.resolve()
    input_paths = {raw_events_path, detector_observations_path, pins_path}
    if output_events_path in input_paths or output_receipt_path in input_paths:
        raise TemporalAdapterContractError("adapter outputs must not overwrite an input")
    if output_events_path == output_receipt_path:
        raise TemporalAdapterContractError("events and receipt outputs must be distinct")

    raw_events = [
        row
        for row in _load_jsonl_models(raw_events_path, RawEnvelopeEvent)
        if isinstance(row, RawEnvelopeEvent)
    ]
    observations = [
        row
        for row in _load_jsonl_models(detector_observations_path, DetectorObservation)
        if isinstance(row, DetectorObservation)
    ]
    try:
        pins = P3TemporalAdapterPins.model_validate_json(pins_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise TemporalAdapterContractError(
            f"invalid adapter pins at {pins_path}: {_validation_error_summary(exc)}"
        ) from exc

    events = assemble_temporal_events(raw_events, observations, pins)
    output_bytes = _jsonl_bytes(events)
    output_events_path.parent.mkdir(parents=True, exist_ok=True)
    output_events_path.write_bytes(output_bytes)

    counts = TemporalAdapterCounts(
        events=len(events),
        trajectories=len({event.trajectory_id for event in events}),
        attack_events=sum(event.population == "attack_eval" for event in events),
        benign_events=sum(event.population == "benign_eval" for event in events),
    )
    receipt = TemporalAdapterReceipt(
        claim_scope=(
            "Construction receipt only: exact raw-envelope payload hashes joined one-to-one "
            "to frozen detector/CDF observations. Contains no payload text and makes no "
            "performance or complete-key continuity claim."
        ),
        contract_ref=pins.contract_ref,
        adapter_code_shas=AdapterCodeShas(
            assembly=_sha256_file(Path(__file__).with_name("temporal_adapter.py")),
            io=_sha256_file(Path(__file__)),
            temporal_contract=_sha256_file(Path(__file__).with_name("temporal.py")),
        ),
        pins=pins,
        inputs={
            "raw_events": AdapterArtifactRef(
                path=str(raw_events_path), sha256=_sha256_file(raw_events_path)
            ),
            "detector_observations": AdapterArtifactRef(
                path=str(detector_observations_path),
                sha256=_sha256_file(detector_observations_path),
            ),
            "pins": AdapterArtifactRef(path=str(pins_path), sha256=_sha256_file(pins_path)),
        },
        output=AdapterArtifactRef(
            path=str(output_events_path), sha256=hashlib.sha256(output_bytes).hexdigest()
        ),
        counts=counts,
        continuity_coverage=_continuity_coverage(events),
        checks=(
            "canonical_event_id",
            "zero_based_contiguous_event_index",
            "strict_source_order",
            "one_to_one_detector_join",
            "exact_payload_hash_join",
            "raw_instruction_channel_preserved",
            "missing_continuity_not_fabricated",
            "payload_text_absent_from_output",
        ),
    )
    receipt_bytes = (
        json.dumps(
            receipt.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    output_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    output_receipt_path.write_bytes(receipt_bytes)
    return receipt
