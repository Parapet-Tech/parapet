"""Strict P3 claim-envelope contracts and deterministic CI computation."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator


SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ClaimContractModel(BaseModel):
    """Fail closed on unknown fields at the claim-bearing boundary."""

    model_config = ConfigDict(extra="forbid")


class TemporalResultLike(Protocol):
    """Interface consumed by paired cell-level CI estimation."""

    cell_label: Literal["attack", "benign"]
    a_peak: float
    b_peak: float
    c_strict_peak: float
    c_broad_peak: float


class AucEstimator(Protocol):
    """Interface for the frozen Mann-Whitney AUC convention."""

    def __call__(
        self,
        positives: Sequence[float],
        negatives: Sequence[float],
        *,
        tie_credit: float = 0.5,
    ) -> float:
        ...


def claim_contract_sha256(value: Any) -> str:
    """Hash one claim contract with the receipt's canonical JSON policy."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_nonempty_strings(model: BaseModel, field_names: Sequence[str]) -> None:
    for field_name in field_names:
        value = getattr(model, field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be non-empty")


class P3CalibrationBlock(ClaimContractModel):
    """Frozen calibration inputs and thresholds required by a P3 claim."""

    benign_cdf_artifact_sha: str = Field(pattern=SHA256_PATTERN)
    cdf_fallback_ladder_sha: str = Field(pattern=SHA256_PATTERN)
    calibration_pool_sha: str = Field(pattern=SHA256_PATTERN)
    target_fpr: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    k_u_b: float = Field(ge=0.0, allow_inf_nan=False)
    k_u_c_strict: float = Field(ge=0.0, allow_inf_nan=False)
    k_u_c_broad: float = Field(ge=0.0, allow_inf_nan=False)
    h_a_peak: float = Field(ge=0.0, allow_inf_nan=False)
    h_b: float = Field(ge=0.0, allow_inf_nan=False)
    h_c_strict: float = Field(ge=0.0, allow_inf_nan=False)
    h_c_broad: float = Field(ge=0.0, allow_inf_nan=False)
    peak_alert_level: float = Field(ge=0.0, allow_inf_nan=False)
    peak_alert_level_provenance: str
    surprise_transform_sha: str = Field(pattern=SHA256_PATTERN)
    surprise_alpha: float = Field(gt=0.0, allow_inf_nan=False)
    surprise_clip_cap: float = Field(gt=0.0, allow_inf_nan=False)
    k_u_calibration_procedure_sha: str = Field(pattern=SHA256_PATTERN)
    productive_band_lower: float = Field(ge=0.0, allow_inf_nan=False)
    productive_band_upper: float = Field(ge=0.0, allow_inf_nan=False)
    length_stratification_schema_sha: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )

    @model_validator(mode="after")
    def validate_productive_band(self) -> "P3CalibrationBlock":
        _require_nonempty_strings(self, ("peak_alert_level_provenance",))
        if self.productive_band_lower != self.k_u_b:
            raise ValueError("productive_band_lower must equal k_u_b")
        if self.productive_band_upper != self.peak_alert_level:
            raise ValueError("productive_band_upper must equal peak_alert_level")
        if self.productive_band_lower >= self.productive_band_upper:
            raise ValueError("productive band lower bound must be below upper bound")
        return self


class P3EvaluationRefs(ClaimContractModel):
    """Frozen evaluation artifacts and mechanical disjointness references."""

    attack_eval_artifact_sha: str = Field(pattern=SHA256_PATTERN)
    benign_eval_artifact_sha: str = Field(pattern=SHA256_PATTERN)
    label_schema_sha: str = Field(pattern=SHA256_PATTERN)
    calibration_eval_disjointness_ref: str
    calibration_pool_composition_ref: str
    benign_eval_composition_ref: str
    score_context_schema_sha: str = Field(pattern=SHA256_PATTERN)
    continuity_key_canonicalization_sha: str = Field(pattern=SHA256_PATTERN)
    ci_config_sha: str = Field(pattern=SHA256_PATTERN)
    float_policy_sha: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def require_evidence_refs(self) -> "P3EvaluationRefs":
        _require_nonempty_strings(
            self,
            (
                "calibration_eval_disjointness_ref",
                "calibration_pool_composition_ref",
                "benign_eval_composition_ref",
            ),
        )
        return self


class P3ValidationContract(ClaimContractModel):
    """Pre-registered claim scope and DoD-6 independence contract."""

    contract_sha: str = Field(pattern=SHA256_PATTERN)
    contract_name: str
    claim_scope: str
    independence_rung: str
    residual_named: str
    criterion_provenance_ref: str

    @model_validator(mode="after")
    def require_named_contract(self) -> "P3ValidationContract":
        _require_nonempty_strings(
            self,
            (
                "contract_name",
                "claim_scope",
                "independence_rung",
                "residual_named",
                "criterion_provenance_ref",
            ),
        )
        return self


class P3RegistrationRefs(ClaimContractModel):
    """Ordering witnesses proving the declared artifacts predate the run."""

    break_schema_registration_ref: str
    label_schema_registration_ref: str
    calibration_registration_ref: str
    threshold_registration_ref: str
    ci_config_registration_ref: str
    receipt_schema_registration_ref: str
    registration_log_kind: str
    registration_log_entry: str
    registration_observed_before_run: Literal[True]

    @model_validator(mode="after")
    def require_ordering_witnesses(self) -> "P3RegistrationRefs":
        _require_nonempty_strings(
            self,
            (
                "break_schema_registration_ref",
                "label_schema_registration_ref",
                "calibration_registration_ref",
                "threshold_registration_ref",
                "ci_config_registration_ref",
                "receipt_schema_registration_ref",
                "registration_log_kind",
                "registration_log_entry",
            ),
        )
        return self


class P3CIConfig(ClaimContractModel):
    """Deterministic paired cell-level bootstrap configuration."""

    method: Literal["paired_cell_bootstrap"] = "paired_cell_bootstrap"
    resample_count: int = Field(gt=0)
    seed: int = Field(ge=0)
    interval_type: Literal["percentile"] = "percentile"
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    pairing: Literal["paired_by_cell_id"] = "paired_by_cell_id"
    stratification: Literal[
        "independent_attack_benign_cell_resampling"
    ] = "independent_attack_benign_cell_resampling"


class P3FloatPolicy(ClaimContractModel):
    """Declared numeric comparison policy for receipt gate recomputation."""

    mode: Literal["exact", "absolute_tolerance"]
    absolute_tolerance: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_tolerance(self) -> "P3FloatPolicy":
        if self.mode == "exact" and self.absolute_tolerance is not None:
            raise ValueError("exact float policy cannot declare a tolerance")
        if self.mode == "absolute_tolerance" and self.absolute_tolerance is None:
            raise ValueError("absolute_tolerance float policy requires a tolerance")
        return self


class P3CriterionProvenance(ClaimContractModel):
    """Frozen threshold and interpretation provenance carried in headlines."""

    productive_band_definition: Literal[
        "k_u_b < surprise < peak_alert_level"
    ] = "k_u_b < surprise < peak_alert_level"
    positive_b_mass_cell_definition: Literal[
        "attack_cell_with_any_event_b_persistence_gt_zero"
    ] = "attack_cell_with_any_event_b_persistence_gt_zero"
    substrate_threshold_source: str
    min_productive_band_fraction: float = Field(ge=0.0, le=1.0)
    min_cells_positive_b_mass: int = Field(ge=0)
    c_shape_interpretation_ref: str
    delta_ci_interpretation_ref: str

    @model_validator(mode="after")
    def require_interpretation_refs(self) -> "P3CriterionProvenance":
        _require_nonempty_strings(
            self,
            (
                "substrate_threshold_source",
                "c_shape_interpretation_ref",
                "delta_ci_interpretation_ref",
            ),
        )
        return self


class P3ClaimInputs(ClaimContractModel):
    """P3-only inputs used to produce self-contained headline statistics."""

    ci_config: P3CIConfig
    float_policy: P3FloatPolicy
    calibration_n_events: tuple[int, ...] = Field(min_length=1)
    length_comparison_method: Literal["median_second_minus_first"] = (
        "median_second_minus_first"
    )
    criterion_provenance: P3CriterionProvenance

    @model_validator(mode="after")
    def require_positive_lengths(self) -> "P3ClaimInputs":
        if any(length <= 0 for length in self.calibration_n_events):
            raise ValueError("calibration_n_events values must be positive")
        return self


class TemporalTrajectoryStrata(ClaimContractModel):
    """Class-symmetric cohort identity plus attack-only construction strata.

    P3 V1 has one governed cohort surface, ``swe_coding``. ``generator``,
    ``mechanism``, and ``surface`` exist only for the attack-side 5B census and
    no-single-carrier checks; they are not cross-class matching fields.

    ``surface_relation`` is deliberately absent from V1. A future multi-surface
    study may reintroduce it only from preregistered per-event surface IDs, with
    a class-blind derivation that is recomputable from receipt events.
    """

    cohort_surface: Literal["swe_coding"]
    generator: str | None = None
    mechanism: str | None = None
    surface: str | None = None

    @model_validator(mode="after")
    def require_named_strata(self) -> "TemporalTrajectoryStrata":
        for field_name in ("generator", "mechanism", "surface"):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must be non-empty when present")
        return self

    def validate_for_trajectory_label(
        self,
        trajectory_label: Literal["attack", "benign"],
    ) -> None:
        """Enforce attack-required and benign-absent construction metadata."""

        attack_only_fields = ("generator", "mechanism", "surface")
        if trajectory_label == "attack":
            for field_name in attack_only_fields:
                if getattr(self, field_name) is None:
                    raise ValueError(f"attack trajectory requires {field_name}")
            return

        for field_name in attack_only_fields:
            if field_name in self.model_fields_set:
                raise ValueError(f"benign trajectory must not include {field_name}")

    @model_serializer(mode="wrap")
    def omit_absent_attack_only_fields(self, handler: Any) -> dict[str, Any]:
        """Serialize benign strata without invented null construction fields."""

        return {key: value for key, value in handler(self).items() if value is not None}


class TemporalConfidenceInterval(ClaimContractModel):
    """One deterministic percentile interval over a paired delta."""

    lower: float = Field(allow_inf_nan=False)
    upper: float = Field(allow_inf_nan=False)
    confidence_level: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def require_ordered_bounds(self) -> "TemporalConfidenceInterval":
        if self.lower > self.upper:
            raise ValueError("confidence interval lower bound exceeds upper bound")
        return self


class P3DeltaIntervals(ClaimContractModel):
    """Paired bootstrap intervals for all pre-registered A/B/C deltas."""

    delta_b_minus_peak: TemporalConfidenceInterval
    delta_c_strict_minus_b: TemporalConfidenceInterval
    delta_c_broad_minus_b: TemporalConfidenceInterval


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (
        ordered[upper_index] - ordered[lower_index]
    )


def compute_paired_delta_intervals(
    results: Sequence[TemporalResultLike],
    config: P3CIConfig,
    *,
    auc_estimator: AucEstimator,
) -> P3DeltaIntervals:
    """Bootstrap attack/benign cells independently and pair metrics per draw."""

    attack = sorted(
        (result for result in results if result.cell_label == "attack"),
        key=lambda result: result.trajectory_id,
    )
    benign = sorted(
        (result for result in results if result.cell_label == "benign"),
        key=lambda result: result.trajectory_id,
    )
    if not attack or not benign:
        raise ValueError("paired delta CIs require attack and benign cells")

    rng = random.Random(config.seed)
    deltas: dict[str, list[float]] = {
        "delta_b_minus_peak": [],
        "delta_c_strict_minus_b": [],
        "delta_c_broad_minus_b": [],
    }
    for _ in range(config.resample_count):
        attack_draw = [attack[rng.randrange(len(attack))] for _ in attack]
        benign_draw = [benign[rng.randrange(len(benign))] for _ in benign]

        def auc(field_name: str) -> float:
            return auc_estimator(
                [float(getattr(result, field_name)) for result in attack_draw],
                [float(getattr(result, field_name)) for result in benign_draw],
                tie_credit=0.5,
            )

        auc_peak = auc("a_peak")
        auc_b = auc("b_peak")
        auc_c_strict = auc("c_strict_peak")
        auc_c_broad = auc("c_broad_peak")
        deltas["delta_b_minus_peak"].append(auc_b - auc_peak)
        deltas["delta_c_strict_minus_b"].append(auc_c_strict - auc_b)
        deltas["delta_c_broad_minus_b"].append(auc_c_broad - auc_b)

    tail = (1.0 - config.confidence_level) / 2.0

    def interval(values: Sequence[float]) -> TemporalConfidenceInterval:
        return TemporalConfidenceInterval(
            lower=_percentile(values, tail),
            upper=_percentile(values, 1.0 - tail),
            confidence_level=config.confidence_level,
        )

    return P3DeltaIntervals(
        delta_b_minus_peak=interval(deltas["delta_b_minus_peak"]),
        delta_c_strict_minus_b=interval(deltas["delta_c_strict_minus_b"]),
        delta_c_broad_minus_b=interval(deltas["delta_c_broad_minus_b"]),
    )
