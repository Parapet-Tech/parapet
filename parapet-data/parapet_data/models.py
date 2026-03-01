"""
Core types for mirror-based corpus curation.

The organizing principle: every benign sample exists because a specific attack
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

from pydantic import BaseModel, Field, model_validator


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
    """Why an attack exists — the signal the classifier must isolate.

    Each reason maps to a benign mirror that shares the same surface
    characteristics (vocabulary, format, length) but lacks the injection signal.
    """

    INSTRUCTION_OVERRIDE = "instruction_override"
    ROLEPLAY_JAILBREAK = "roleplay_jailbreak"
    META_PROBE = "meta_probe"
    EXFILTRATION = "exfiltration"
    ADVERSARIAL_SUFFIX = "adversarial_suffix"
    INDIRECT_INJECTION = "indirect_injection"
    OBFUSCATION = "obfuscation"
    CONSTRAINT_BYPASS = "constraint_bypass"


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


# ---------------------------------------------------------------------------
# Mirror cells — the core abstraction
# ---------------------------------------------------------------------------


class MirrorCell(BaseModel):
    """One attack reason and its benign twin.

    The cell defines WHAT the classifier should learn: the teaching_goal
    describes the spurious correlation that the mirror neutralizes. Both
    sides must be populated — a mirror with only one side is not a mirror.
    """

    reason: AttackReason
    attack_sources: list[SourceRef]
    benign_sources: list[SourceRef]
    teaching_goal: str
    languages: list[Language]
    format_distribution: dict[FormatBin, float]
    length_distribution: dict[LengthBin, float]

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


# ---------------------------------------------------------------------------
# Backfill and supplements
# ---------------------------------------------------------------------------


class BackfillPolicy(BaseModel):
    """What to do when a cell can't hit its target.

    Hard mirror is the intent. Backfill is the escape hatch, but every
    use is logged in the manifest so the mirror's imperfections are visible.
    """

    strategy: Literal["same_reason_any_language", "oversample", "fail"]
    max_oversample_ratio: float = 2.0
    log_gaps: bool = True


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
    seed: int = 42
    allow_partial_mirror: bool = False
    holdout_only_reasons: list[AttackReason] = Field(default_factory=list)

    @model_validator(mode="after")
    def all_reasons_covered(self) -> MirrorSpec:
        if self.allow_partial_mirror:
            return self
        covered = {c.reason for c in self.cells}
        missing = set(AttackReason) - covered
        if missing:
            raise ValueError(
                f"Mirror incomplete — uncovered attack reasons: {missing}"
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
    output_hash: str  # SHA256 of the full curated dataset
    semantic_hash: str  # SHA256(sorted content hashes + per-cell counts)
    total_samples: int
    attack_samples: int
    benign_samples: int
    splits: dict[str, SplitManifest]
    cell_fills: dict[str, CellFillRecord]
    gaps: list[str]
    cross_contamination_dropped: int
    output_path: Path


class CellFillRecord(BaseModel):
    """How well a cell hit its target — the mirror's report card."""

    target: int
    actual: int
    backfilled: int
    backfill_sources: list[str] = Field(default_factory=list)


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
