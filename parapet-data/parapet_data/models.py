"""
Core types for mirror-based corpus curation.

The organizing principle: every benign sample exists because a specific mirror
category exists. The benign corpus mirrors the attack corpus along every
dimension except the injection signal itself, forcing the classifier to learn
actual discriminative features instead of spurious correlations.

Shared contract between parapet-data (curation) and parapet-runner (orchestration).
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums — the axes of the cell matrix
# ---------------------------------------------------------------------------


class Language(str, Enum):
    """Languages the classifier must handle without using language as signal."""

    EN = "EN"
    RU = "RU"
    ZH = "ZH"
    AR = "AR"


class FormatBin(str, Enum):
    """Content format — mirrors must match format distribution."""

    PROSE = "prose"  # natural language text
    STRUCTURED = "structured"  # JSON/MD/HTML/XML
    CODE = "code"  # programming language content


class LengthBin(str, Enum):
    """Length bucket — mirrors must match length distribution."""

    SHORT = "short"  # 1-2 sentences
    MEDIUM = "medium"  # 1 paragraph
    LONG = "long"  # multi-paragraph


class AttackReason(str, Enum):
    """Legacy/default prompt-injection mirror taxonomy."""

    INSTRUCTION_OVERRIDE = "instruction_override"
    ROLEPLAY_JAILBREAK = "roleplay_jailbreak"
    META_PROBE = "meta_probe"
    EXFILTRATION = "exfiltration"
    ADVERSARIAL_SUFFIX = "adversarial_suffix"
    INDIRECT_INJECTION = "indirect_injection"
    OBFUSCATION = "obfuscation"
    CONSTRAINT_BYPASS = "constraint_bypass"


class LanguageQuotaMode(str, Enum):
    """Language quota behavior for multi-language sampling."""

    STRICT = "strict"
    BEST_EFFORT = "best_effort"


class SourceGroundingMode(str, Enum):
    """How trustworthy a source's reason assignment is."""

    REASON_GROUNDED = "reason_grounded"
    POOLED = "pooled"


class SourceRoutePolicy(str, Enum):
    """Which lane a source is intended to feed."""

    MIRROR = "mirror"
    RESIDUAL = "residual"
    BACKGROUND = "background"
    QUARANTINE = "quarantine"


class ReasonProvenance(str, Enum):
    """How a source's reason label was established."""

    SOURCE_LABEL = "source_label"
    MANUAL_MAP = "manual_map"
    ADJUDICATED = "adjudicated"
    HEURISTIC = "heuristic"
    NONE = "none"


class ApplicabilityScope(str, Enum):
    """How well the source matches the generalist prompt-attack task."""

    IN_DOMAIN = "in_domain"
    MIXED = "mixed"
    UNKNOWN = "unknown"


DEFAULT_REASON_CATEGORIES: tuple[str, ...] = tuple(reason.value for reason in AttackReason)


def _normalize_category_name(value: str | AttackReason, *, field_name: str) -> str:
    category = str(value).strip()
    if not category:
        raise ValueError(f"{field_name} must not be empty")
    return category


def _normalize_category_list(
    values: list[str | AttackReason],
    *,
    field_name: str,
) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []

    for value in values:
        category = _normalize_category_name(value, field_name=field_name)
        if category in seen:
            duplicates.append(category)
            continue
        seen.add(category)
        normalized.append(category)

    if duplicates:
        duplicate_list = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"{field_name} contains duplicates: {duplicate_list}")

    return normalized


# ---------------------------------------------------------------------------
# Source references
# ---------------------------------------------------------------------------


class SourceRef(BaseModel):
    """A pointer to a data source with extraction instructions.

    Each source lives in TheWall or schema/eval and has a specific format
    that requires a dedicated extractor.
    """

    name: str
    path: Path
    language: Language
    extractor: str  # registry key for extraction function
    max_samples: int | None = None
    label_filter: dict[str, object] | None = None
    grounding_mode: SourceGroundingMode | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
    )
    route_policy: SourceRoutePolicy | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
    )
    reason_provenance: ReasonProvenance | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
    )
    applicability_scope: ApplicabilityScope | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
    )


class SourceMetadata(BaseModel):
    """Manifest snapshot of source selection and routing, keyed by source name.

    The manifest needs enough information to explain which rows were eligible
    from a physical source, not just where the source lived. That includes
    selection instructions like label_filter/max_samples in addition to lane
    routing and provenance metadata.
    """

    path: Path
    language: Language
    extractor: str
    max_samples: int | None = None
    label_filter: dict[str, object] | None = None
    grounding_mode: SourceGroundingMode | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
    )
    route_policy: SourceRoutePolicy | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
    )
    reason_provenance: ReasonProvenance | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
    )
    applicability_scope: ApplicabilityScope | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
    )

    @classmethod
    def from_source_ref(cls, source: SourceRef) -> SourceMetadata:
        """Project spec-facing source fields into manifest metadata."""
        return cls(
            path=source.path,
            language=source.language,
            extractor=source.extractor,
            max_samples=source.max_samples,
            label_filter=source.label_filter,
            grounding_mode=source.grounding_mode,
            route_policy=source.route_policy,
            reason_provenance=source.reason_provenance,
            applicability_scope=source.applicability_scope,
        )


_STRICT_MIRROR_REASON_PROVENANCE = {
    ReasonProvenance.SOURCE_LABEL,
    ReasonProvenance.MANUAL_MAP,
    ReasonProvenance.ADJUDICATED,
}


def _route_policy_violation(
    source: SourceRef,
    *,
    expected: SourceRoutePolicy,
    context: str,
) -> str | None:
    actual = source.route_policy.value if source.route_policy is not None else "unset"
    if source.route_policy != expected:
        return (
            f"{context}: source {source.name} must declare route_policy={expected.value}, got {actual}"
        )
    return None


# ---------------------------------------------------------------------------
# Mirror cells — the core abstraction
# ---------------------------------------------------------------------------


class MirrorCell(BaseModel):
    """One mirror category and its benign twin.

    The cell defines WHAT the classifier should learn: the teaching_goal
    describes the spurious correlation that the mirror neutralizes. Both
    sides must be populated — a mirror with only one side is not a mirror.
    """

    reason: str
    attack_sources: list[SourceRef]
    benign_sources: list[SourceRef]
    teaching_goal: str
    languages: list[Language]
    format_distribution: dict[FormatBin, float]
    length_distribution: dict[LengthBin, float]

    @model_validator(mode="after")
    def reason_non_empty(self) -> MirrorCell:
        self.reason = _normalize_category_name(self.reason, field_name="reason")
        return self

    @model_validator(mode="after")
    def both_sides_populated(self) -> MirrorCell:
        if not self.attack_sources:
            raise ValueError(f"{self.reason}: no attack sources")
        if not self.benign_sources:
            raise ValueError(f"{self.reason}: no benign sources")
        return self

    @model_validator(mode="after")
    def languages_non_empty(self) -> MirrorCell:
        if not self.languages:
            raise ValueError(f"{self.reason}: languages must not be empty")
        return self

    @model_validator(mode="after")
    def distributions_valid(self) -> MirrorCell:
        for name, dist in [
            ("format_distribution", self.format_distribution),
            ("length_distribution", self.length_distribution),
        ]:
            if any(v < 0 for v in dist.values()):
                raise ValueError(
                    f"{self.reason}: {name} contains negative values"
                )
            total = sum(dist.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"{self.reason}: {name} sums to {total}, expected 1.0"
                )
        return self

    @property
    def cell_id(self) -> str:
        """Stable cell identifier including reason and language set."""
        langs = ",".join(sorted({lang.value for lang in self.languages}))
        return f"{self.reason}__{langs}"


# ---------------------------------------------------------------------------
# Backfill and supplements
# ---------------------------------------------------------------------------


class BackfillPolicy(BaseModel):
    """What to do when a cell can't hit its target.

    Hard mirror is the intent. Backfill is the escape hatch, but every
    use is logged in the manifest so the mirror's imperfections are visible.
    """

    strategy: Literal[
        "same_reason_any_language",
        "oversample",
        "fail",
        "ngram_safe",
        "mirror_reason_first",
    ]
    max_oversample_ratio: float = 2.0
    log_gaps: bool = True


class LanguageQuota(BaseModel):
    """Global language budget profile applied per cell in multi-language specs."""

    mode: LanguageQuotaMode = LanguageQuotaMode.BEST_EFFORT
    profile: dict[Language, float]

    @model_validator(mode="after")
    def profile_sums_to_one(self) -> LanguageQuota:
        total = sum(self.profile.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Language profile sums to {total}, expected 1.0")
        return self


class Supplement(BaseModel):
    """Handcrafted data targeting a known weakness, outside the mirror.

    Supplements are not part of the mirror taxonomy — they exist to close
    specific gaps that the mirror can't address (e.g., GCG entropy samples
    that don't naturally occur in any source dataset).
    """

    name: str
    weakness: str
    attack_sources: list[SourceRef]
    benign_sources: list[SourceRef]
    max_samples: int  # hard cap per side


class BackgroundLane(BaseModel):
    """Benign-only background samples outside the mirror.

    Background samples don't match any attack reason by surface similarity.
    They represent the true prior of benign traffic, anchoring the classifier's
    decision boundary so it doesn't overfit to mirror edge-cases.

    Budget comes from the benign allocation — mirror cells receive slightly
    less benign to make room for background, keeping total_target unchanged.
    """

    sources: list[SourceRef]
    budget_fraction: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description="Fraction of total benign budget allocated to background",
    )


# ---------------------------------------------------------------------------
# MirrorSpec — the experiment's data contract
# ---------------------------------------------------------------------------


class MirrorSpec(BaseModel):
    """Complete experiment data contract. The spec IS the experiment design.

    A MirrorSpec defines what data to curate and how. It is serializable,
    diffable, and versionable. Every experiment is a different spec.
    The validator guarantees structural soundness before any data is sampled.
    """

    name: str
    version: str
    cells: list[MirrorCell]
    supplements: list[Supplement] = Field(default_factory=list)
    supplement_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of total budget allocated to supplements (for sweep experiments)",
    )
    ratio: float = Field(
        default=1.0,
        gt=0.0,
        description="Benign:malicious ratio per cell",
    )
    total_target: int | None = None
    backfill: BackfillPolicy = Field(
        default_factory=lambda: BackfillPolicy(strategy="same_reason_any_language")
    )
    language_quota: LanguageQuota | None = None
    background: BackgroundLane | None = None
    seed: int = 42
    allow_partial_mirror: bool = False
    enforce_source_contracts: bool = Field(
        default=False,
        exclude_if=lambda v: v is False,
    )
    reason_categories: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("reason_categories", "entity_categories"),
        exclude_if=lambda v: not v,
    )
    holdout_only_reasons: list[str] = Field(default_factory=list)

    @property
    def mirror_reason_categories(self) -> list[str]:
        explicit = _normalize_category_list(
            self.reason_categories,
            field_name="reason_categories",
        )
        if explicit:
            return explicit

        cell_reasons = _normalize_category_list(
            [cell.reason for cell in self.cells],
            field_name="cells.reason",
        )
        if any(reason not in DEFAULT_REASON_CATEGORIES for reason in cell_reasons):
            return cell_reasons
        return list(DEFAULT_REASON_CATEGORIES)

    @model_validator(mode="after")
    def reason_categories_consistent(self) -> MirrorSpec:
        self.holdout_only_reasons = _normalize_category_list(
            self.holdout_only_reasons,
            field_name="holdout_only_reasons",
        )
        if self.reason_categories:
            self.reason_categories = _normalize_category_list(
                self.reason_categories,
                field_name="reason_categories",
            )

        seen: set[str] = set()
        duplicates: list[str] = []
        for cell in self.cells:
            if cell.reason in seen:
                duplicates.append(cell.reason)
                continue
            seen.add(cell.reason)
        if duplicates:
            duplicate_list = ", ".join(sorted(set(duplicates)))
            raise ValueError(f"cells contain duplicate reasons: {duplicate_list}")

        declared_categories = self.mirror_reason_categories
        declared_set = set(declared_categories)
        covered = {cell.reason for cell in self.cells}

        unexpected = covered - declared_set
        if unexpected:
            unexpected_list = ", ".join(sorted(unexpected))
            raise ValueError(
                f"cells declare reasons outside reason_categories: {unexpected_list}"
            )

        undeclared_holdout = set(self.holdout_only_reasons) - declared_set
        if undeclared_holdout:
            undeclared_list = ", ".join(sorted(undeclared_holdout))
            raise ValueError(
                "holdout_only_reasons reference undeclared categories: "
                f"{undeclared_list}"
            )

        if self.allow_partial_mirror:
            return self

        missing = declared_set - covered
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(
                f"Mirror incomplete — uncovered categories: {missing_list}"
            )
        return self

    @model_validator(mode="after")
    def supplement_consistency(self) -> MirrorSpec:
        if self.supplement_ratio > 0 and not self.supplements:
            raise ValueError("supplement_ratio > 0 but no supplements defined")
        if self.supplements and self.total_target:
            supplement_budget = int(self.total_target * self.supplement_ratio)
            supplement_cap = sum(s.max_samples * 2 for s in self.supplements)
            if supplement_budget > supplement_cap:
                raise ValueError(
                    f"supplement_ratio implies {supplement_budget} samples "
                    f"but supplements cap at {supplement_cap}"
                )
        return self

    @model_validator(mode="after")
    def source_contracts_consistent(self) -> MirrorSpec:
        if not self.enforce_source_contracts:
            return self

        violations: list[str] = []

        for cell in self.cells:
            reason = cell.reason
            for source in cell.attack_sources:
                context = f"{reason}: attack source"
                violation = _route_policy_violation(
                    source,
                    expected=SourceRoutePolicy.MIRROR,
                    context=context,
                )
                if violation is not None:
                    violations.append(violation)
                if source.grounding_mode != SourceGroundingMode.REASON_GROUNDED:
                    violations.append(
                        f"{context} {source.name} must declare grounding_mode=reason_grounded"
                    )
                if source.reason_provenance not in _STRICT_MIRROR_REASON_PROVENANCE:
                    actual = (
                        source.reason_provenance.value
                        if source.reason_provenance is not None
                        else "unset"
                    )
                    violations.append(
                        f"{context} {source.name} must declare non-heuristic reason_provenance, got {actual}"
                    )
                if source.applicability_scope != ApplicabilityScope.IN_DOMAIN:
                    actual = (
                        source.applicability_scope.value
                        if source.applicability_scope is not None
                        else "unset"
                    )
                    violations.append(
                        f"{context} {source.name} must declare applicability_scope=in_domain, got {actual}"
                    )
            for source in cell.benign_sources:
                violation = _route_policy_violation(
                    source,
                    expected=SourceRoutePolicy.MIRROR,
                    context=f"{reason}: benign source",
                )
                if violation is not None:
                    violations.append(violation)

        for supplement in self.supplements:
            for source in supplement.attack_sources + supplement.benign_sources:
                violation = _route_policy_violation(
                    source,
                    expected=SourceRoutePolicy.RESIDUAL,
                    context=f"supplement {supplement.name}",
                )
                if violation is not None:
                    violations.append(violation)

        if self.background is not None:
            for source in self.background.sources:
                violation = _route_policy_violation(
                    source,
                    expected=SourceRoutePolicy.BACKGROUND,
                    context="background",
                )
                if violation is not None:
                    violations.append(violation)

        if violations:
            raise ValueError("source contract violations:\n" + "\n".join(violations))

        return self

    def spec_hash(self) -> str:
        """Deterministic hash of the spec for provenance tracking."""
        canonical = self.model_dump_json(indent=None)
        return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Provenance — what was produced and from what
# ---------------------------------------------------------------------------


class SplitManifest(BaseModel):
    """Per-split provenance — enables auditing that calibration never touches holdout."""

    name: str  # "train", "val", "holdout"
    sample_count: int
    content_hashes: list[str]  # SHA256 of each sample's content, sorted
    artifact_path: Path


class VerifiedSyncManifest(BaseModel):
    """Optional receipt of a verified-sync preflight run."""

    staging_dir: Path
    verified_dir: Path
    files_processed: int
    total_input: int
    passed: int
    dropped: int
    quarantined: int
    rerouted: int
    relabeled: int


class CurationManifest(BaseModel):
    """Immutable receipt of a curation run.

    The manifest captures everything needed to audit and reproduce the
    curated dataset: what spec was used, what sources were read, what
    was produced, and where the mirror fell short.
    """

    spec_name: str
    spec_version: str
    spec_hash: str
    seed: int
    timestamp: str  # ISO 8601
    source_hashes: dict[str, str]  # source name -> Merkle hash
    source_metadata: dict[str, SourceMetadata] = Field(default_factory=dict)
    output_hash: str  # SHA256 of the full curated dataset
    semantic_hash: str  # SHA256(sorted content hashes + per-cell counts)
    total_samples: int
    attack_samples: int
    benign_samples: int
    splits: dict[str, SplitManifest]
    cell_fills: dict[str, CellFillRecord]
    gaps: list[str]
    duplicates_dropped: int = 0
    cross_contamination_dropped: int
    background_requested: int = 0
    background_actual: int = 0
    source_alias_warnings: list[str] = Field(default_factory=list)
    feature_coverage_warnings: list[str] = Field(default_factory=list)
    ledger_dropped: int = 0
    ledger_quarantined: int = 0
    ledger_rerouted: int = 0
    ledger_relabeled: int = 0
    verified_sync: VerifiedSyncManifest | None = None
    output_path: Path


class CellFillRecord(BaseModel):
    """How well a cell hit its target — the mirror's report card."""

    target: int
    actual: int
    backfilled: int
    backfill_sources: list[str] = Field(default_factory=list)
    by_format: dict[str, int] = Field(default_factory=dict)
    by_length: dict[str, int] = Field(default_factory=dict)
    by_language: dict[str, int] = Field(default_factory=dict)
    degraded: bool = False
    degraded_mode: str | None = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def merkle_hash_directory(directory: Path) -> str:
    """Compute a Merkle hash over all files in a directory.

    Sorted by relative path to ensure determinism regardless of filesystem
    enumeration order. Captures file additions, removals, and content changes.
    """
    file_hashes: list[str] = []
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file():
            h = hashlib.sha256(file_path.read_bytes()).hexdigest()
            rel = file_path.relative_to(directory).as_posix()
            file_hashes.append(f"{rel}:{h}")
    combined = "\n".join(file_hashes)
    return hashlib.sha256(combined.encode()).hexdigest()


def merkle_hash_file(file_path: Path) -> str:
    """Hash a single file for source provenance."""
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def compute_source_hash(source: SourceRef, base_dir: Path | None = None) -> str:
    """Compute provenance hash for a source — Merkle for directories, SHA256 for files.

    If source.path is relative, resolves against base_dir to avoid
    cwd-sensitivity across environments.
    """
    path = source.path
    if not path.is_absolute() and base_dir is not None:
        path = (base_dir / path).resolve()
    if path.is_dir():
        return merkle_hash_directory(path)
    return merkle_hash_file(path)


def compute_semantic_hash(
    content_hashes: list[str],
    cell_fills: dict[str, CellFillRecord] | dict[str, dict[str, int]],
) -> str:
    """Order-independent, serialization-independent hash of curation output.

    This is the CANONICAL semantic hash implementation. Both parapet-data
    and parapet-runner MUST use this function — do not reimplement.

    CI gates on this hash. It is insensitive to serialization format and
    key ordering, but sensitive to data content and cell fill values.

    Accepts either CellFillRecord or plain dict for cross-package compatibility.
    """
    payload: dict[str, object] = {
        "content_hashes": sorted(str(h) for h in content_hashes),
        "per_cell_counts": {
            str(k): _canonicalize_cell_fill(v)
            for k in sorted(cell_fills.keys())
            for v in [cell_fills[k]]
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _canonicalize_cell_fill(fill: CellFillRecord | dict) -> dict[str, object]:
    """Normalize cell fill to a canonical dict for hashing."""
    if isinstance(fill, CellFillRecord):
        return {"target": fill.target, "actual": fill.actual, "backfilled": fill.backfilled}
    # Plain dict from runner side — canonicalize key order
    return {str(k): v for k, v in sorted(fill.items(), key=lambda x: str(x[0]))}
