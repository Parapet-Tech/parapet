"""Reproducibility manifest models and deterministic hashing utilities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .config import TrainConfig

try:
    # Provided by parapet-data once that package lands.
    from parapet_data.models import (  # type: ignore
        CellFillRecord,
        CurationManifest,
        compute_semantic_hash as _data_semantic_hash,
    )
except Exception:  # pragma: no cover - exercised only before parapet-data exists.
    CellFillRecord = None  # type: ignore[assignment]
    _data_semantic_hash = None

    class CurationManifest(BaseModel):  # type: ignore[no-redef]
        """Minimal fallback contract so runner work can proceed independently."""

        spec_name: str
        spec_version: str
        output_path: Path
        output_hash: str
        cell_fills: dict[str, dict[str, int]] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Evaluation metrics for one split."""

    f1: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    threshold: float
    holdout_size: int = Field(ge=0)


class RuntimeIdentity(BaseModel):
    """Pin runtime/build artifacts needed for replay."""

    git_sha: str
    trainer_script_hash: str
    parapet_eval_hash: str
    pg2_model_id: str
    eval_config_hash: str
    env_hash: str


class RunManifest(BaseModel):
    """One complete experiment run."""

    run_id: str
    runtime: RuntimeIdentity
    curation: CurationManifest
    train_config: TrainConfig
    eval_result: EvalResult
    pg2_baseline: EvalResult | None = None
    delta: dict[str, float] | None = None
    error_file: Path | None = None
    semantic_parity_hash: str | None = None

    @model_validator(mode="after")
    def ensure_delta_consistency(self) -> "RunManifest":
        if self.pg2_baseline is None and self.delta is not None:
            raise ValueError("delta cannot be set without pg2_baseline")
        return self


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _canonicalize(value[k]) for k in sorted(value, key=lambda x: str(x))}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported value for semantic hash canonicalization: {type(value)}")


def compute_semantic_parity_hash(
    content_hashes: Sequence[str], per_cell_counts: Mapping[str, Any]
) -> str:
    """
    Compute semantic parity hash as:
    SHA256(sorted content hashes + per-cell counts).
    """

    # Use parapet-data's implementation whenever available to keep one canonical
    # cross-package hash contract.
    if _data_semantic_hash is not None and CellFillRecord is not None:
        normalized: dict[str, Any] = {}
        for reason, raw in per_cell_counts.items():
            if isinstance(raw, CellFillRecord):
                normalized[str(reason)] = raw
                continue
            if isinstance(raw, Mapping):
                required_keys = {"target", "actual", "backfilled"}
                optional_keys = {"backfill_sources"}
                present_keys = {str(k) for k in raw.keys()}
                missing = required_keys - present_keys
                unknown = present_keys - (required_keys | optional_keys)
                if missing:
                    raise ValueError(
                        f"Invalid per-cell counts for '{reason}': missing keys {sorted(missing)}"
                    )
                if unknown:
                    raise ValueError(
                        f"Invalid per-cell counts for '{reason}': unknown keys {sorted(unknown)}"
                    )
                backfill_sources = raw.get("backfill_sources", [])
                if isinstance(backfill_sources, (str, bytes)):
                    raise ValueError(
                        f"Invalid per-cell counts for '{reason}': backfill_sources must be a sequence, "
                        f"got {type(backfill_sources).__name__}"
                    )
                if not isinstance(backfill_sources, Sequence):
                    raise ValueError(
                        f"Invalid per-cell counts for '{reason}': backfill_sources must be a sequence, "
                        f"got {type(backfill_sources).__name__}"
                    )
                normalized[str(reason)] = CellFillRecord(
                    target=int(raw["target"]),
                    actual=int(raw["actual"]),
                    backfilled=int(raw["backfilled"]),
                    backfill_sources=list(backfill_sources),
                )
                continue
            raise TypeError(
                f"Unsupported per-cell count value for '{reason}': {type(raw)}"
            )
        return str(_data_semantic_hash(list(content_hashes), normalized))

    # Fallback for environments where parapet-data is unavailable.
    payload = {
        "content_hashes": sorted(str(h) for h in content_hashes),
        "per_cell_counts": _canonicalize(per_cell_counts),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def compute_metric_delta(eval_result: EvalResult, baseline_result: EvalResult) -> dict[str, float]:
    """Compute metric deltas against PG2 baseline."""

    return {
        "f1_delta": eval_result.f1 - baseline_result.f1,
        "precision_delta": eval_result.precision - baseline_result.precision,
        "recall_delta": eval_result.recall - baseline_result.recall,
    }
