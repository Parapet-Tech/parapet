"""
Cell matrix sampling with hard mirror and backfill.

The sampler reads sources via extractors, classifies each sample by
format and length bins, samples per-cell according to MirrorSpec
distributions, and applies the backfill policy when cells can't hit
targets. All filtering (attack sig, dedup, cross-contamination) runs
before sampling so rejected samples are never counted toward targets.

This module owns the transition from MirrorSpec -> list[Sample].
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import yaml

from .extractors import get_extractor
from .filters import ContentDeduplicator, looks_like_attack, passes_label_filter
from .models import (
    AttackReason,
    BackfillPolicy,
    CellFillRecord,
    FormatBin,
    LengthBin,
    MirrorCell,
    MirrorSpec,
    SourceRef,
    Supplement,
)

logger = logging.getLogger(__name__)

try:
    _YAML_LOADER = yaml.CSafeLoader
except AttributeError:
    _YAML_LOADER = yaml.SafeLoader


# ---------------------------------------------------------------------------
# Sample record — the unit produced by sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """One curated training sample with full provenance."""

    content: str
    content_hash: str
    label: str  # "malicious" or "benign"
    reason: str  # AttackReason value or supplement name
    source_name: str
    language: str
    format_bin: str
    length_bin: str


# ---------------------------------------------------------------------------
# Bin classification
# ---------------------------------------------------------------------------

# Heuristic thresholds for length binning (character count)
_SHORT_MAX = 200
_MEDIUM_MAX = 800

_CODE_INDICATORS = re.compile(
    r"(?:def |class |import |function |const |var |let |"
    r"if\s*\(|for\s*\(|while\s*\(|#include|package |"
    r"\{[\s\S]*\}|=>|->|::|&&|\|\|)"
)

_STRUCTURED_INDICATORS = re.compile(
    r"(?:^\s*[\[{]|</?\w+>|^\s*#+ |^\|.*\||^\s*-\s+\w+:)", re.MULTILINE
)


def classify_length(text: str) -> LengthBin:
    """Assign a length bin based on character count."""
    n = len(text)
    if n <= _SHORT_MAX:
        return LengthBin.SHORT
    if n <= _MEDIUM_MAX:
        return LengthBin.MEDIUM
    return LengthBin.LONG


def classify_format(text: str) -> FormatBin:
    """Heuristic format classification."""
    if _CODE_INDICATORS.search(text):
        return FormatBin.CODE
    if _STRUCTURED_INDICATORS.search(text):
        return FormatBin.STRUCTURED
    return FormatBin.PROSE


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> list[dict]:
    """Load a YAML file and return list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def load_source(
    source: SourceRef,
    base_dir: Path | None = None,
) -> list[dict]:
    """Load raw rows from a source, resolving relative paths."""
    path = source.path
    if not path.is_absolute() and base_dir is not None:
        path = (base_dir / path).resolve()
    if not path.exists():
        logger.warning("Source %s not found at %s, skipping", source.name, path)
        return []
    if path.is_dir():
        rows: list[dict] = []
        for yaml_file in sorted(path.glob("*.yaml")) + sorted(path.glob("*.yml")):
            rows.extend(_load_yaml(yaml_file))
        return rows
    return _load_yaml(path)


def extract_samples_from_source(
    source: SourceRef,
    label: str,
    reason: str,
    base_dir: Path | None = None,
) -> list[Sample]:
    """Load a source, extract text, classify bins, return Sample list."""
    rows = load_source(source, base_dir=base_dir)
    extractor = get_extractor(source.extractor)
    samples: list[Sample] = []

    for row in rows:
        # Apply label filter if configured
        if not passes_label_filter(row, source.label_filter):
            continue

        text = extractor(row)
        if not text:
            continue

        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        samples.append(Sample(
            content=text,
            content_hash=content_hash,
            label=label,
            reason=reason,
            source_name=source.name,
            language=source.language.value,
            format_bin=classify_format(text).value,
            length_bin=classify_length(text).value,
        ))

    if source.max_samples and len(samples) > source.max_samples:
        samples = samples[:source.max_samples]

    return samples


# ---------------------------------------------------------------------------
# Distribution-aware sampling
# ---------------------------------------------------------------------------


def _sample_by_distribution(
    pool: list[Sample],
    target_n: int,
    distribution: dict[str, float],
    bin_key: str,
    rng: random.Random,
) -> list[Sample]:
    """Sample from pool respecting a distribution over a bin dimension.

    Fills each bin proportionally, then backfills remainder from leftovers.
    """
    by_bin: dict[str, list[Sample]] = defaultdict(list)
    for s in pool:
        by_bin[getattr(s, bin_key)].append(s)

    sampled: list[Sample] = []
    sampled_hashes: set[str] = set()

    for bin_val, proportion in distribution.items():
        bin_target = int(target_n * proportion)
        available = by_bin.get(bin_val, [])
        take = min(bin_target, len(available))
        if take > 0:
            chosen = rng.sample(available, take)
            sampled.extend(chosen)
            sampled_hashes.update(s.content_hash for s in chosen)

    # Backfill remainder from unused pool
    remaining = target_n - len(sampled)
    if remaining > 0:
        leftovers = [s for s in pool if s.content_hash not in sampled_hashes]
        if leftovers:
            take = min(remaining, len(leftovers))
            sampled.extend(rng.sample(leftovers, take))

    return sampled[:target_n]


# ---------------------------------------------------------------------------
# Cell sampling
# ---------------------------------------------------------------------------


@dataclass
class CellSampleResult:
    """Result of sampling one cell."""

    samples: list[Sample]
    fill: CellFillRecord
    gaps: list[str] = field(default_factory=list)


def sample_cell(
    cell: MirrorCell,
    label: str,
    target_per_side: int | None,
    dedup: ContentDeduplicator,
    rng: random.Random,
    backfill: BackfillPolicy,
    base_dir: Path | None = None,
    filter_benign_attacks: bool = True,
) -> CellSampleResult:
    """Sample one side (attack or benign) of a mirror cell.

    1. Load all sources for this side
    2. Apply dedup + attack-sig filtering (benign side only)
    3. Sample according to length distribution
    4. Apply backfill if under target
    """
    sources = cell.attack_sources if label == "malicious" else cell.benign_sources
    reason = cell.reason.value

    # Load all candidate samples
    all_candidates: list[Sample] = []
    for source in sources:
        extracted = extract_samples_from_source(
            source, label=label, reason=reason, base_dir=base_dir,
        )
        all_candidates.extend(extracted)

    # Filter
    filtered: list[Sample] = []
    for s in all_candidates:
        if not dedup.check(s.content):
            continue
        if filter_benign_attacks and label == "benign" and looks_like_attack(s.content):
            continue
        filtered.append(s)

    # Sample with length distribution
    take_all = target_per_side is None
    effective_target = len(filtered) if take_all else target_per_side

    length_dist = {k.value: v for k, v in cell.length_distribution.items()}
    sampled = _sample_by_distribution(
        filtered, effective_target, length_dist, "length_bin", rng,
    )

    actual = len(sampled)
    backfilled = 0
    gaps: list[str] = []

    if not take_all and actual < target_per_side:
        gap = target_per_side - actual
        if backfill.strategy == "fail":
            gaps.append(
                f"{reason}/{label}: need {target_per_side}, got {actual}, "
                f"policy=fail"
            )
        elif backfill.strategy == "oversample" and filtered:
            max_extra = int(len(filtered) * (backfill.max_oversample_ratio - 1))
            extra_take = min(gap, max_extra)
            if extra_take > 0:
                extra = rng.choices(filtered, k=extra_take)
                sampled.extend(extra)
                backfilled = extra_take
            remaining_gap = gap - extra_take
            if remaining_gap > 0:
                gaps.append(
                    f"{reason}/{label}: oversampled {extra_take}, "
                    f"still short {remaining_gap}"
                )
        # same_reason_any_language is handled at the spec level (below)

    fill = CellFillRecord(
        target=effective_target,
        actual=len(sampled),
        backfilled=backfilled,
    )

    if backfill.log_gaps and gaps:
        for g in gaps:
            logger.warning(g)

    return CellSampleResult(samples=sampled, fill=fill, gaps=gaps)


# ---------------------------------------------------------------------------
# Spec-level sampling
# ---------------------------------------------------------------------------


@dataclass
class SamplingResult:
    """Complete result of sampling a MirrorSpec."""

    attack_samples: list[Sample]
    benign_samples: list[Sample]
    cell_fills: dict[str, CellFillRecord]
    gaps: list[str]
    cross_contamination_dropped: int


def sample_spec(
    spec: MirrorSpec,
    base_dir: Path | None = None,
) -> SamplingResult:
    """Sample all cells in a MirrorSpec, producing labeled samples.

    Execution:
    1. Sample all attack sides first (to build cross-contamination set)
    2. Register attack hashes in dedup
    3. Sample all benign sides
    4. Sample supplements
    5. Apply same_reason_any_language backfill for benign shortfalls
    """
    rng = random.Random(spec.seed)
    dedup = ContentDeduplicator()

    # Calculate per-cell budget
    if spec.total_target:
        supplement_budget = int(spec.total_target * spec.supplement_ratio)
        mirror_budget = spec.total_target - supplement_budget
        n_cells = len(spec.cells)
        per_cell_total = mirror_budget // n_cells if n_cells else 0
        # ratio = benign:malicious, so per_side = total / (1 + ratio)
        per_cell_attack = int(per_cell_total / (1 + spec.ratio))
        per_cell_benign = per_cell_total - per_cell_attack
    else:
        # No budget — take everything from sources
        per_cell_attack = None
        per_cell_benign = None

    all_attack: list[Sample] = []
    all_benign: list[Sample] = []
    cell_fills: dict[str, CellFillRecord] = {}
    all_gaps: list[str] = []

    # --- Phase 1: Attack sampling ---
    for cell in spec.cells:
        result = sample_cell(
            cell, label="malicious", target_per_side=per_cell_attack,
            dedup=dedup, rng=rng, backfill=spec.backfill,
            base_dir=base_dir, filter_benign_attacks=False,
        )
        all_attack.extend(result.samples)
        cell_fills[f"{cell.reason.value}_attack"] = result.fill
        all_gaps.extend(result.gaps)

    # --- Phase 2: Register attack hashes for cross-contamination ---
    dedup.register_attack_hashes([s.content_hash for s in all_attack])

    # --- Phase 3: Benign sampling ---
    benign_by_reason: dict[str, list[Sample]] = {}
    for cell in spec.cells:
        result = sample_cell(
            cell, label="benign", target_per_side=per_cell_benign,
            dedup=dedup, rng=rng, backfill=spec.backfill,
            base_dir=base_dir, filter_benign_attacks=True,
        )
        benign_by_reason[cell.reason.value] = result.samples
        all_benign.extend(result.samples)
        cell_fills[f"{cell.reason.value}_benign"] = result.fill
        all_gaps.extend(result.gaps)

    # --- Phase 4: same_reason_any_language backfill ---
    if spec.backfill.strategy == "same_reason_any_language" and per_cell_benign is not None:
        for cell in spec.cells:
            fill = cell_fills[f"{cell.reason.value}_benign"]
            if fill.actual >= per_cell_benign:
                continue
            gap = per_cell_benign - fill.actual
            # Try pulling from other cells' unused benign pool
            # (same reason concept — the backfill is logged)
            other_benign = [
                s for reason, samples in benign_by_reason.items()
                if reason != cell.reason.value
                for s in samples
            ]
            if other_benign:
                take = min(gap, len(other_benign))
                extra = rng.sample(other_benign, take)
                all_benign.extend(extra)
                cell_fills[f"{cell.reason.value}_benign"] = CellFillRecord(
                    target=fill.target,
                    actual=fill.actual + take,
                    backfilled=fill.backfilled + take,
                    backfill_sources=list(fill.backfill_sources) + ["cross_reason_backfill"],
                )
                if take < gap:
                    all_gaps.append(
                        f"{cell.reason.value}/benign: backfilled {take}, "
                        f"still short {gap - take}"
                    )

    # --- Phase 5: Supplements ---
    if spec.supplements:
        supplement_per = (
            int(spec.total_target * spec.supplement_ratio / len(spec.supplements) / 2)
            if spec.total_target
            else None
        )
        for supp in spec.supplements:
            cap = min(supplement_per, supp.max_samples) if supplement_per is not None else supp.max_samples
            for sources, label in [
                (supp.attack_sources, "malicious"),
                (supp.benign_sources, "benign"),
            ]:
                supp_samples: list[Sample] = []
                for source in sources:
                    extracted = extract_samples_from_source(
                        source, label=label, reason=f"supplement:{supp.name}",
                        base_dir=base_dir,
                    )
                    for s in extracted:
                        if dedup.check(s.content):
                            supp_samples.append(s)
                if len(supp_samples) > cap:
                    supp_samples = rng.sample(supp_samples, cap)
                if label == "malicious":
                    all_attack.extend(supp_samples)
                else:
                    all_benign.extend(supp_samples)

    return SamplingResult(
        attack_samples=all_attack,
        benign_samples=all_benign,
        cell_fills=cell_fills,
        gaps=all_gaps,
        cross_contamination_dropped=dedup.cross_contamination_dropped,
    )
