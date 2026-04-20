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

import logging
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from .classifiers import classify_reason
from .extractors import get_extractor
from .filters import ContentDeduplicator, content_hash, looks_like_attack, passes_label_filter
from .ledger import Ledger, apply_ledger_to_row
from .models import (
    BackfillPolicy,
    BackgroundLane,
    CellFillRecord,
    FormatBin,
    Language,
    LanguageQuotaMode,
    LengthBin,
    MirrorCell,
    MirrorSpec,
    ReasonProvenance,
    SourceRef,
    Supplement,
)
from .staged_artifact import iter_staged_artifact_paths, iter_staged_rows

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sample record - the unit produced by sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """One curated training sample with full provenance."""

    content: str
    content_hash: str
    label: str  # "malicious" or "benign"
    reason: str  # mirror category or supplement name
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


def load_source(
    source: SourceRef,
    base_dir: Path | None = None,
) -> Iterator[dict]:
    """Yield raw rows from a source, resolving relative paths.

    Returns an iterator (not a list) so per-source extraction can early-exit
    at ``source.max_samples`` without materializing the rest of the file.
    Note: this is bounded retention — for each source we still load the
    underlying file once, but rows are not retained across sources and
    there is no cross-run cache. PyYAML cannot stream a top-level list,
    so within a single source the file is fully parsed once before rows
    are yielded.
    """
    path = source.path
    if not path.is_absolute() and base_dir is not None:
        path = (base_dir / path).resolve()
    if not path.exists():
        logger.warning("Source %s not found at %s, skipping", source.name, path)
        return

    if path.is_dir():
        for staged_file in iter_staged_artifact_paths(path):
            yield from iter_staged_rows(staged_file)
        return

    yield from iter_staged_rows(path)


@dataclass
class SourceExtractResult:
    """Samples extracted from one source plus ledger counters."""

    samples: list[Sample]
    ledger_dropped: int = 0
    ledger_quarantined: int = 0
    ledger_rerouted: int = 0
    ledger_relabeled: int = 0


def _source_extract_cache_key(
    source: SourceRef,
    *,
    label: str,
    reason: str,
    base_dir: Path | None,
) -> tuple[str, str, str, str, str, str, str]:
    path = source.path
    if not path.is_absolute() and base_dir is not None:
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return (
        source.name,
        str(path),
        label,
        reason,
        repr(source.label_filter),
        str(source.max_samples),
        source.reason_provenance.value if source.reason_provenance is not None else "unset",
    )


def _extract_samples_from_source_with_ledger(
    source: SourceRef,
    label: str,
    reason: str,
    base_dir: Path | None = None,
    ledger: Ledger | None = None,
) -> SourceExtractResult:
    """Load a source, apply optional ledger actions, classify bins."""
    rows = load_source(source, base_dir=base_dir)
    extractor = get_extractor(source.extractor)
    samples: list[Sample] = []
    ledger_dropped = 0
    ledger_quarantined = 0
    ledger_rerouted = 0
    ledger_relabeled = 0

    max_samples = source.max_samples
    for row in rows:
        if not passes_label_filter(row, source.label_filter):
            continue

        text = extractor(row)
        if not text:
            continue

        assigned_reason = reason
        if (
            label == "malicious"
            and source.reason_provenance == ReasonProvenance.HEURISTIC
        ):
            classification = classify_reason(text)
            if classification is None:
                continue
            classified_reason = (
                classification.reason.value
                if hasattr(classification.reason, "value")
                else str(classification.reason)
            )
            if classified_reason != reason:
                continue
            assigned_reason = classified_reason

        normalized_row = {
            "content": text,
            "label": label,
            "reason": assigned_reason,
            "source": source.name,
            "language": source.language.value,
        }
        # Canonicalize from extracted content so stale staged metadata
        # cannot poison split manifests or ledger lookups.
        normalized_row["content_hash"] = content_hash(text)

        if ledger is not None:
            ledger_result = apply_ledger_to_row(normalized_row, ledger)
            ledger_dropped += ledger_result.dropped
            ledger_quarantined += ledger_result.quarantined
            ledger_rerouted += ledger_result.rerouted
            ledger_relabeled += ledger_result.relabeled
            if ledger_result.row is None:
                continue
            normalized_row = ledger_result.row

        text = normalized_row["content"]
        samples.append(Sample(
            content=text,
            content_hash=normalized_row["content_hash"],
            label=normalized_row["label"],
            reason=normalized_row["reason"],
            source_name=source.name,
            language=source.language.value,
            format_bin=classify_format(text).value,
            length_bin=classify_length(text).value,
        ))

        if max_samples is not None and len(samples) >= max_samples:
            break

    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]

    return SourceExtractResult(
        samples=samples,
        ledger_dropped=ledger_dropped,
        ledger_quarantined=ledger_quarantined,
        ledger_rerouted=ledger_rerouted,
        ledger_relabeled=ledger_relabeled,
    )


def extract_samples_from_source(
    source: SourceRef,
    label: str,
    reason: str,
    base_dir: Path | None = None,
    ledger: Ledger | None = None,
) -> list[Sample]:
    """Load a source, extract text, classify bins, return Sample list."""
    return _extract_samples_from_source_with_ledger(
        source=source,
        label=label,
        reason=reason,
        base_dir=base_dir,
        ledger=ledger,
    ).samples


def _get_source_extract_result(
    source: SourceRef,
    *,
    label: str,
    reason: str,
    base_dir: Path | None,
    ledger: Ledger | None,
    source_extract_cache: dict[tuple[str, str, str, str, str, str, str], SourceExtractResult] | None,
) -> SourceExtractResult:
    """Get extracted samples, optionally reusing a per-run cache."""
    if source_extract_cache is None:
        return _extract_samples_from_source_with_ledger(
            source=source,
            label=label,
            reason=reason,
            base_dir=base_dir,
            ledger=ledger,
        )

    key = _source_extract_cache_key(
        source,
        label=label,
        reason=reason,
        base_dir=base_dir,
    )
    cached = source_extract_cache.get(key)
    if cached is not None:
        return cached

    extracted = _extract_samples_from_source_with_ledger(
        source=source,
        label=label,
        reason=reason,
        base_dir=base_dir,
        ledger=ledger,
    )
    source_extract_cache[key] = extracted
    return extracted


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


def _compute_integer_targets(
    total: int,
    weights: dict[str | tuple[str, str], float],
) -> dict[str | tuple[str, str], int]:
    """Allocate integer targets with largest-remainder rounding."""
    if total <= 0:
        return {k: 0 for k in weights}

    positive = {k: v for k, v in weights.items() if v > 0}
    if not positive:
        return {k: 0 for k in weights}

    norm = sum(positive.values())
    raw = {k: (total * (v / norm)) for k, v in positive.items()}
    floor = {k: int(x) for k, x in raw.items()}
    allocated = sum(floor.values())
    remainder = total - allocated

    fractions = sorted(
        ((k, raw[k] - floor[k]) for k in positive),
        key=lambda kv: kv[1],
        reverse=True,
    )
    for i in range(remainder):
        floor[fractions[i % len(fractions)][0]] += 1

    return {k: floor.get(k, 0) for k in weights}


def _sample_by_joint_distribution(
    pool: list[Sample],
    target_n: int,
    format_dist: dict[str, float],
    length_dist: dict[str, float],
    rng: random.Random,
    cell_id: str,
) -> tuple[list[Sample], str | None]:
    """Sample with 2D format x length stratification plus sparse fallbacks."""
    if target_n <= 0 or not pool:
        return [], None

    if len(pool) < 20:
        take = min(target_n, len(pool))
        return rng.sample(pool, take), "unstratified"

    if len(pool) < 50:
        sampled = _sample_by_distribution(
            pool, target_n, format_dist, "format_bin", rng,
        )
        logger.info(
            "%s: sparse cell (%d), using 1D format fallback",
            cell_id,
            len(pool),
        )
        return sampled, "1d_format"

    by_joint: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    for sample in pool:
        by_joint[(sample.format_bin, sample.length_bin)].append(sample)

    joint_weights: dict[tuple[str, str], float] = {}
    for fmt, fmt_pct in format_dist.items():
        for length, len_pct in length_dist.items():
            joint_weights[(fmt, length)] = fmt_pct * len_pct

    joint_targets = _compute_integer_targets(target_n, joint_weights)
    sampled: list[Sample] = []
    sampled_hashes: set[str] = set()

    for joint_bin, bin_target in joint_targets.items():
        if bin_target <= 0:
            continue
        available = by_joint.get(joint_bin, [])
        if not available:
            continue
        take = min(bin_target, len(available))
        if take == 0:
            continue
        chosen = rng.sample(available, take)
        sampled.extend(chosen)
        sampled_hashes.update(s.content_hash for s in chosen)

    remaining = target_n - len(sampled)
    if remaining > 0:
        leftovers = [s for s in pool if s.content_hash not in sampled_hashes]
        if leftovers:
            sampled.extend(rng.sample(leftovers, min(remaining, len(leftovers))))

    realized_by_format = Counter(s.format_bin for s in sampled)
    realized_by_length = Counter(s.length_bin for s in sampled)
    logger.debug(
        "%s: joint sampling target=%d realized=%d format=%s length=%s",
        cell_id,
        target_n,
        len(sampled),
        dict(realized_by_format),
        dict(realized_by_length),
    )
    return sampled[:target_n], None


def _dimension_counts(samples: list[Sample]) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Build format/length/language counts for manifest reporting."""
    return (
        dict(Counter(s.format_bin for s in samples)),
        dict(Counter(s.length_bin for s in samples)),
        dict(Counter(s.language for s in samples)),
    )


# ---------------------------------------------------------------------------
# Cell sampling
# ---------------------------------------------------------------------------


@dataclass
class CellSampleResult:
    """Result of sampling one cell."""

    samples: list[Sample]
    fill: CellFillRecord
    gaps: list[str] = field(default_factory=list)
    available_pool: list[Sample] = field(default_factory=list)


def _all_candidates_from_reason_map(
    all_samples_by_reason: dict[str, list[Sample]],
) -> list[Sample]:
    """Flatten reason map into unique samples by content hash."""
    seen: set[str] = set()
    merged: list[Sample] = []
    for samples in all_samples_by_reason.values():
        for sample in samples:
            if sample.content_hash in seen:
                continue
            seen.add(sample.content_hash)
            merged.append(sample)
    return merged


def _apply_backfill_ladder(
    cell: MirrorCell,
    label: str,
    current_samples: list[Sample],
    target: int,
    all_samples_by_reason: dict[str, list[Sample]],
    all_samples_by_language: dict[str, list[Sample]],
    globally_assigned_hashes: set[str],
    rng: random.Random,
    strategy: str,
) -> tuple[list[Sample], list[str], list[str]]:
    """Apply v2 fallback ladders for sparse cells."""
    missing = target - len(current_samples)
    if missing <= 0:
        return [], [], []

    desired_reason = cell.reason
    desired_languages = {lang.value for lang in cell.languages}
    desired_formats = {
        fmt.value for fmt, pct in cell.format_distribution.items() if pct > 0
    }
    desired_lengths = {
        length.value for length, pct in cell.length_distribution.items() if pct > 0
    }
    excluded_hashes = {s.content_hash for s in current_samples}
    excluded_hashes.update(globally_assigned_hashes)
    extras: list[Sample] = []
    tags: list[str] = []
    gaps: list[str] = []

    all_candidates = _all_candidates_from_reason_map(all_samples_by_reason)
    if not all_candidates:
        gaps.append(
            f"{cell.cell_id}/{label}: no candidates available for backfill, short {missing}"
        )
        return extras, tags, gaps

    def _select(pool: list[Sample], tag: str | None = None) -> None:
        nonlocal missing
        if missing <= 0:
            return
        candidates = [s for s in pool if s.content_hash not in excluded_hashes]
        if not candidates:
            return
        take = min(missing, len(candidates))
        chosen = rng.sample(candidates, take)
        extras.extend(chosen)
        excluded_hashes.update(s.content_hash for s in chosen)
        missing -= take
        if tag and take > 0:
            tags.append(tag)

    # Step 1: same reason + same language + same format/length.
    _select(
        [
            s for s in all_samples_by_reason.get(desired_reason, [])
            if s.language in desired_languages
            and s.format_bin in desired_formats
            and s.length_bin in desired_lengths
        ]
    )
    # Step 2: same reason + same language.
    _select(
        [
            s for s in all_samples_by_reason.get(desired_reason, [])
            if s.language in desired_languages
        ]
    )

    if strategy == "mirror_reason_first":
        # Step 3: same reason + any language.
        _select(
            list(all_samples_by_reason.get(desired_reason, [])),
            tag="degraded_cross_language_same_reason",
        )
        # Step 4: cross-reason + same language.
        _select(
            [
                s for lang, samples in all_samples_by_language.items()
                if lang in desired_languages
                for s in samples
                if s.reason != desired_reason
            ],
            tag="degraded_cross_reason",
        )
        # Step 5: cross-reason + cross-language.
        _select(
            [
                s for s in all_candidates
                if s.reason != desired_reason and s.language not in desired_languages
            ],
            tag="degraded_random_fill",
        )
    else:
        # ngram_safe Step 3: cross-reason + same language.
        _select(
            [
                s for lang, samples in all_samples_by_language.items()
                if lang in desired_languages
                for s in samples
                if s.reason != desired_reason
            ],
            tag="degraded_cross_reason",
        )
        # ngram_safe Step 4: cross-reason + cross-language.
        _select(
            [
                s for s in all_candidates
                if s.reason != desired_reason and s.language not in desired_languages
            ],
            tag="degraded_random_fill",
        )

    if missing > 0:
        gaps.append(
            f"{cell.cell_id}/{label}: backfilled {target - len(current_samples) - missing}, "
            f"still short {missing}"
        )

    return extras, tags, gaps


def sample_cell(
    cell: MirrorCell,
    label: str,
    target_per_side: int | None,
    dedup: ContentDeduplicator,
    rng: random.Random,
    backfill: BackfillPolicy,
    base_dir: Path | None = None,
    filter_benign_attacks: bool = True,
    allowed_languages: set[str] | None = None,
    ledger: Ledger | None = None,
    ledger_totals: dict[str, int] | None = None,
    source_extract_cache: dict[tuple[str, str, str, str, str, str, str], SourceExtractResult] | None = None,
    extra_candidates: Sequence[Sample] | None = None,
) -> CellSampleResult:
    """Sample one side (attack or benign) of a mirror cell.

    1. Load all sources for this side
    2. Apply dedup + attack-sig filtering (benign side only)
    3. Sample according to length distribution
    4. Apply backfill if under target
    """
    sources = cell.attack_sources if label == "malicious" else cell.benign_sources
    reason = cell.reason

    # Load all candidate samples
    all_candidates: list[Sample] = []
    if extra_candidates:
        all_candidates.extend(sample for sample in extra_candidates if sample.label == label)
    for source in sources:
        # In language-quota mode, skip non-target source languages early so
        # they cannot consume dedup budget for other language passes.
        if allowed_languages is not None and source.language.value not in allowed_languages:
            continue
        extracted = _get_source_extract_result(
            source,
            label=label,
            reason=reason,
            base_dir=base_dir,
            ledger=ledger,
            source_extract_cache=source_extract_cache,
        )
        if ledger_totals is not None and source_extract_cache is None:
            ledger_totals["dropped"] += extracted.ledger_dropped
            ledger_totals["quarantined"] += extracted.ledger_quarantined
            ledger_totals["rerouted"] += extracted.ledger_rerouted
            ledger_totals["relabeled"] += extracted.ledger_relabeled
        all_candidates.extend(sample for sample in extracted.samples if sample.label == label)

    # Filter
    filtered: list[Sample] = []
    for s in all_candidates:
        # Apply language filter before dedup so per-language quota passes do
        # not mark other-language rows as already seen.
        if allowed_languages is not None and s.language not in allowed_languages:
            continue
        if filter_benign_attacks and label == "benign" and looks_like_attack(s.content):
            continue
        if not dedup.check(s.content):
            continue
        filtered.append(s)

    # Sample with length distribution
    take_all = target_per_side is None
    effective_target = len(filtered) if take_all else target_per_side

    degraded_mode: str | None = None
    if take_all:
        sampled = list(filtered)
    else:
        format_dist = {k.value: v for k, v in cell.format_distribution.items()}
        length_dist = {k.value: v for k, v in cell.length_distribution.items()}
        sampled, degraded_mode = _sample_by_joint_distribution(
            filtered,
            effective_target,
            format_dist,
            length_dist,
            rng,
            cell_id=cell.cell_id,
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

    by_format, by_length, by_language = _dimension_counts(sampled)
    sampled_hashes = {s.content_hash for s in sampled}
    available_pool = [s for s in filtered if s.content_hash not in sampled_hashes]
    fill = CellFillRecord(
        target=effective_target,
        actual=len(sampled),
        backfilled=backfilled,
        by_format=by_format,
        by_length=by_length,
        by_language=by_language,
        degraded=degraded_mode is not None,
        degraded_mode=degraded_mode,
    )

    if backfill.log_gaps and gaps:
        for g in gaps:
            logger.warning(g)

    return CellSampleResult(
        samples=sampled,
        fill=fill,
        gaps=gaps,
        available_pool=available_pool,
    )


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
    duplicates_dropped: int
    cross_contamination_dropped: int
    background_requested: int = 0
    background_actual: int = 0
    discussion_requested: int = 0
    discussion_actual: int = 0
    ledger_dropped: int = 0
    ledger_quarantined: int = 0
    ledger_rerouted: int = 0
    ledger_relabeled: int = 0


def _language_targets_for_cell(
    cell: MirrorCell,
    target: int,
    profile: dict[Language, float],
) -> dict[str, int]:
    """Compute per-language integer targets for one cell."""
    profile_by_lang = {lang.value: pct for lang, pct in profile.items()}
    languages = [lang.value for lang in cell.languages]
    if not languages:
        return {}

    weights = {lang: profile_by_lang.get(lang, 0.0) for lang in languages}
    if sum(weights.values()) <= 0:
        weights = {lang: 1.0 for lang in languages}

    allocated = _compute_integer_targets(target, weights)
    return {str(k): int(v) for k, v in allocated.items()}


def _merge_degraded_mode(existing: str | None, tags: list[str]) -> str | None:
    modes: set[str] = set()
    if existing:
        modes.update(m for m in existing.split("+") if m)
    modes.update(t for t in tags if t)
    if not modes:
        return None
    return "+".join(sorted(modes))


def _remove_samples_from_maps(
    samples: Sequence[Sample],
    by_reason: dict[str, list[Sample]],
    by_language: dict[str, list[Sample]],
) -> None:
    """Remove consumed samples from candidate maps by content hash."""
    consumed = {s.content_hash for s in samples}
    if not consumed:
        return

    for reason in list(by_reason.keys()):
        remaining = [s for s in by_reason[reason] if s.content_hash not in consumed]
        if remaining:
            by_reason[reason] = remaining
        else:
            by_reason.pop(reason, None)

    for language in list(by_language.keys()):
        remaining = [s for s in by_language[language] if s.content_hash not in consumed]
        if remaining:
            by_language[language] = remaining
        else:
            by_language.pop(language, None)


def sample_spec(
    spec: MirrorSpec,
    base_dir: Path | None = None,
    ledger: Ledger | None = None,
) -> SamplingResult:
    """Sample all cells in a MirrorSpec, producing labeled samples."""
    rng = random.Random(spec.seed)
    dedup = ContentDeduplicator()

    # Calculate per-cell budget
    background_budget = 0
    discussion_budget = 0
    if spec.total_target:
        supplement_budget = int(spec.total_target * spec.supplement_ratio)
        mirror_budget = spec.total_target - supplement_budget
        n_cells = len(spec.cells)
        per_cell_total = mirror_budget // n_cells if n_cells else 0
        # ratio = benign:malicious, so per_side = total / (1 + ratio)
        per_cell_attack = int(per_cell_total / (1 + spec.ratio))
        per_cell_benign = per_cell_total - per_cell_attack
        # Carve benign-only side lanes from the benign budget while keeping
        # total_target unchanged.
        total_benign = n_cells * per_cell_benign
        if spec.discussion_benign:
            discussion_budget = int(
                total_benign * spec.discussion_benign.budget_fraction
            )
        if spec.background:
            background_budget = int(total_benign * spec.background.budget_fraction)
        mirror_benign = max(total_benign - background_budget - discussion_budget, 0)
        per_cell_benign = mirror_benign // n_cells if n_cells else 0
    else:
        per_cell_attack = None
        per_cell_benign = None

    all_attack: list[Sample] = []
    all_benign: list[Sample] = []
    assigned_hashes: set[str] = set()
    cell_fills: dict[str, CellFillRecord] = {}
    per_cell_samples: dict[tuple[str, str], list[Sample]] = {}
    by_reason: dict[str, dict[str, list[Sample]]] = {
        "malicious": defaultdict(list),
        "benign": defaultdict(list),
    }
    by_language: dict[str, dict[str, list[Sample]]] = {
        "malicious": defaultdict(list),
        "benign": defaultdict(list),
    }
    available_by_reason: dict[str, dict[str, list[Sample]]] = {
        "malicious": defaultdict(list),
        "benign": defaultdict(list),
    }
    available_by_language: dict[str, dict[str, list[Sample]]] = {
        "malicious": defaultdict(list),
        "benign": defaultdict(list),
    }
    all_gaps: list[str] = []
    ledger_totals = {
        "dropped": 0,
        "quarantined": 0,
        "rerouted": 0,
        "relabeled": 0,
    }
    source_extract_cache: dict[
        tuple[str, str, str, str, str, str, str],
        SourceExtractResult,
    ] = {}
    relabeled_candidates: dict[str, dict[str, list[Sample]]] = {
        "malicious": defaultdict(list),
        "benign": defaultdict(list),
    }

    def _record_extract_counts(extracted: SourceExtractResult) -> None:
        ledger_totals["dropped"] += extracted.ledger_dropped
        ledger_totals["quarantined"] += extracted.ledger_quarantined
        ledger_totals["rerouted"] += extracted.ledger_rerouted
        ledger_totals["relabeled"] += extracted.ledger_relabeled

    def _register_relabeled_candidates(
        *,
        source: SourceRef,
        label: str,
        reason: str,
    ) -> None:
        key = _source_extract_cache_key(
            source,
            label=label,
            reason=reason,
            base_dir=base_dir,
        )
        extracted = source_extract_cache.get(key)
        if extracted is None:
            extracted = _get_source_extract_result(
                source,
                label=label,
                reason=reason,
                base_dir=base_dir,
                ledger=ledger,
                source_extract_cache=source_extract_cache,
            )
            _record_extract_counts(extracted)
        for sample in extracted.samples:
            if sample.label != label:
                relabeled_candidates[sample.label][sample.reason].append(sample)

    for cell in spec.cells:
        for source in cell.attack_sources:
            _register_relabeled_candidates(
                source=source,
                label="malicious",
                reason=cell.reason,
            )
        for source in cell.benign_sources:
            _register_relabeled_candidates(
                source=source,
                label="benign",
                reason=cell.reason,
            )
    for supp in spec.supplements:
        supp_reason = f"supplement:{supp.name}"
        for source in supp.attack_sources:
            _register_relabeled_candidates(
                source=source,
                label="malicious",
                reason=supp_reason,
            )
        for source in supp.benign_sources:
            _register_relabeled_candidates(
                source=source,
                label="benign",
                reason=supp_reason,
            )
    if spec.background:
        for source in spec.background.sources:
            _register_relabeled_candidates(
                source=source,
                label="benign",
                reason="background",
            )
    if spec.discussion_benign:
        for source in spec.discussion_benign.sources:
            _register_relabeled_candidates(
                source=source,
                label="benign",
                reason="discussion_benign",
            )

    def _sample_supplement_sources(
        *,
        sources: Sequence[SourceRef],
        label: str,
        reason: str,
    ) -> list[Sample]:
        samples: list[Sample] = []
        for sample in relabeled_candidates[label].get(reason, []):
            if sample.label != label:
                continue
            if not dedup.check(sample.content):
                continue
            samples.append(sample)
        for source in sources:
            extracted = _get_source_extract_result(
                source,
                label=label,
                reason=reason,
                base_dir=base_dir,
                ledger=ledger,
                source_extract_cache=source_extract_cache,
            )
            for sample in extracted.samples:
                if sample.label != label:
                    continue
                if not dedup.check(sample.content):
                    continue
                samples.append(sample)
        return samples

    def _sample_benign_lane(
        *,
        lane_name: str,
        sources: Sequence[SourceRef],
        budget: int,
        filter_attacklike: bool,
    ) -> list[Sample]:
        samples: list[Sample] = []
        for sample in relabeled_candidates["benign"].get(lane_name, []):
            if sample.label != "benign":
                continue
            if filter_attacklike and looks_like_attack(sample.content):
                continue
            if dedup.check(sample.content):
                samples.append(sample)
        for source in sources:
            extracted = _get_source_extract_result(
                source,
                label="benign",
                reason=lane_name,
                base_dir=base_dir,
                ledger=ledger,
                source_extract_cache=source_extract_cache,
            )
            for sample in extracted.samples:
                if sample.label != "benign":
                    continue
                if filter_attacklike and looks_like_attack(sample.content):
                    continue
                if dedup.check(sample.content):
                    samples.append(sample)
        if len(samples) > budget:
            samples = rng.sample(samples, budget)
        return samples

    def _sample_cell_with_optional_quota(
        cell: MirrorCell,
        label: str,
        target_per_side: int | None,
        filter_benign_attacks: bool,
        dedup_override: ContentDeduplicator | None = None,
    ) -> CellSampleResult:
        effective_dedup = dedup_override if dedup_override is not None else dedup
        allowed_languages = {lang.value for lang in cell.languages}
        if target_per_side is None or spec.language_quota is None:
            return sample_cell(
                cell=cell,
                label=label,
                target_per_side=target_per_side,
                dedup=effective_dedup,
                rng=rng,
                backfill=spec.backfill,
                base_dir=base_dir,
                filter_benign_attacks=filter_benign_attacks,
                allowed_languages=allowed_languages,
                ledger=ledger,
                source_extract_cache=source_extract_cache,
                extra_candidates=relabeled_candidates[label].get(cell.reason, []),
            )

        language_targets = _language_targets_for_cell(
            cell=cell,
            target=target_per_side,
            profile=spec.language_quota.profile,
        )
        merged_samples: list[Sample] = []
        merged_available: list[Sample] = []
        merged_gaps: list[str] = []
        merged_backfill_sources: list[str] = []
        merged_backfilled = 0
        merged_degraded_mode: str | None = None

        for lang, lang_target in language_targets.items():
            if lang_target <= 0:
                continue
            lang_result = sample_cell(
                cell=cell,
                label=label,
                target_per_side=lang_target,
                dedup=effective_dedup,
                rng=rng,
                backfill=spec.backfill,
                base_dir=base_dir,
                filter_benign_attacks=filter_benign_attacks,
                allowed_languages={lang},
                ledger=ledger,
                source_extract_cache=source_extract_cache,
                extra_candidates=relabeled_candidates[label].get(cell.reason, []),
            )
            merged_samples.extend(lang_result.samples)
            merged_available.extend(lang_result.available_pool)
            merged_gaps.extend(lang_result.gaps)
            merged_backfilled += lang_result.fill.backfilled
            merged_backfill_sources.extend(lang_result.fill.backfill_sources)
            merged_degraded_mode = _merge_degraded_mode(
                merged_degraded_mode,
                [lang_result.fill.degraded_mode] if lang_result.fill.degraded_mode else [],
            )

            actual_lang = lang_result.fill.by_language.get(lang, 0)
            if actual_lang < lang_target:
                shortfall = lang_target - actual_lang
                pct = (shortfall / lang_target) * 100 if lang_target else 0.0
                msg = (
                    f"{cell.cell_id}/{label}: {lang} quota target {lang_target}, "
                    f"actual {actual_lang} (-{pct:.1f}%)"
                )
                if spec.language_quota.mode == LanguageQuotaMode.STRICT:
                    raise ValueError(msg)
                merged_gaps.append(msg)

        by_fmt, by_len, by_lang = _dimension_counts(merged_samples)
        fill = CellFillRecord(
            target=target_per_side,
            actual=len(merged_samples),
            backfilled=merged_backfilled,
            backfill_sources=merged_backfill_sources,
            by_format=by_fmt,
            by_length=by_len,
            by_language=by_lang,
            degraded=merged_degraded_mode is not None,
            degraded_mode=merged_degraded_mode,
        )
        return CellSampleResult(
            samples=merged_samples,
            fill=fill,
            gaps=merged_gaps,
            available_pool=merged_available,
        )

    # Phase 1: Attack sampling
    for cell in spec.cells:
        result = _sample_cell_with_optional_quota(
            cell=cell,
            label="malicious",
            target_per_side=per_cell_attack,
            filter_benign_attacks=False,
        )
        all_attack.extend(result.samples)
        assigned_hashes.update(sample.content_hash for sample in result.samples)
        key = f"{cell.cell_id}_malicious"
        cell_fills[key] = result.fill
        per_cell_samples[(cell.cell_id, "malicious")] = list(result.samples)
        for sample in result.samples:
            by_reason["malicious"][sample.reason].append(sample)
            by_language["malicious"][sample.language].append(sample)
        for sample in result.available_pool:
            available_by_reason["malicious"][sample.reason].append(sample)
            available_by_language["malicious"][sample.language].append(sample)
        all_gaps.extend(result.gaps)

    # Phase 2: Register attack hashes for cross-contamination
    attack_hashes = [s.content_hash for s in all_attack]
    dedup.register_attack_hashes(attack_hashes)

    # Phase 3: Benign sampling
    # Use per-cell dedup to prevent starvation of shared multilingual
    # benign pools.  The global dedup marks ALL candidates as seen during
    # filtering (even those not ultimately sampled), so shared benign
    # sources like wikipedia/xquad get entirely consumed by the first
    # cell, leaving subsequent cells with 0 benign for that language.
    # Per-cell dedup still prevents within-cell duplicates and
    # attack↔benign cross-contamination via registered attack hashes.
    benign_duplicates_dropped = 0
    benign_cross_contamination_dropped = 0
    for cell in spec.cells:
        cell_benign_dedup = ContentDeduplicator()
        cell_benign_dedup.register_attack_hashes(attack_hashes)
        result = _sample_cell_with_optional_quota(
            cell=cell,
            label="benign",
            target_per_side=per_cell_benign,
            filter_benign_attacks=True,
            dedup_override=cell_benign_dedup,
        )
        benign_duplicates_dropped += cell_benign_dedup.duplicates_dropped
        benign_cross_contamination_dropped += cell_benign_dedup.cross_contamination_dropped
        all_benign.extend(result.samples)
        assigned_hashes.update(sample.content_hash for sample in result.samples)
        key = f"{cell.cell_id}_benign"
        cell_fills[key] = result.fill
        per_cell_samples[(cell.cell_id, "benign")] = list(result.samples)
        for sample in result.samples:
            by_reason["benign"][sample.reason].append(sample)
            by_language["benign"][sample.language].append(sample)
        for sample in result.available_pool:
            available_by_reason["benign"][sample.reason].append(sample)
            available_by_language["benign"][sample.language].append(sample)
        all_gaps.extend(result.gaps)

    # Phase 4a: legacy same_reason_any_language backfill (kept for compatibility)
    if spec.backfill.strategy == "same_reason_any_language" and per_cell_benign is not None:
        for cell in spec.cells:
            key = f"{cell.cell_id}_benign"
            fill = cell_fills[key]
            if fill.actual >= per_cell_benign:
                continue
            gap = per_cell_benign - fill.actual
            other_benign = [
                s for reason, samples in available_by_reason["benign"].items()
                if reason != cell.reason
                for s in samples
                if s.content_hash not in assigned_hashes
            ]
            if other_benign:
                take = min(gap, len(other_benign))
                extra = rng.sample(other_benign, take)
                current_samples = per_cell_samples.get((cell.cell_id, "benign"), [])
                current_samples.extend(extra)
                per_cell_samples[(cell.cell_id, "benign")] = current_samples
                all_benign.extend(extra)
                assigned_hashes.update(sample.content_hash for sample in extra)
                for sample in extra:
                    by_reason["benign"][sample.reason].append(sample)
                    by_language["benign"][sample.language].append(sample)
                _remove_samples_from_maps(
                    extra,
                    available_by_reason["benign"],
                    available_by_language["benign"],
                )
                by_fmt, by_len, by_lang = _dimension_counts(current_samples)
                cell_fills[key] = CellFillRecord(
                    target=fill.target,
                    actual=len(current_samples),
                    backfilled=fill.backfilled + take,
                    backfill_sources=list(fill.backfill_sources) + ["cross_reason_backfill"],
                    by_format=by_fmt,
                    by_length=by_len,
                    by_language=by_lang,
                    degraded=True,
                    degraded_mode=_merge_degraded_mode(fill.degraded_mode, ["cross_reason_backfill"]),
                )
                if take < gap:
                    all_gaps.append(
                        f"{cell.cell_id}/benign: backfilled {take}, still short {gap - take}"
                    )

    # Phase 4b: v2 backfill ladders
    if spec.backfill.strategy in {"ngram_safe", "mirror_reason_first"}:
        for label, target in [("malicious", per_cell_attack), ("benign", per_cell_benign)]:
            if target is None:
                continue
            for cell in spec.cells:
                key = f"{cell.cell_id}_{label}"
                fill = cell_fills[key]
                if fill.actual >= target:
                    continue
                current = per_cell_samples.get((cell.cell_id, label), [])
                extras, tags, gaps = _apply_backfill_ladder(
                    cell=cell,
                    label=label,
                    current_samples=current,
                    target=target,
                    all_samples_by_reason=available_by_reason[label],
                    all_samples_by_language=available_by_language[label],
                    globally_assigned_hashes=assigned_hashes,
                    rng=rng,
                    strategy=spec.backfill.strategy,
                )
                if extras:
                    current.extend(extras)
                    per_cell_samples[(cell.cell_id, label)] = current
                    if label == "malicious":
                        all_attack.extend(extras)
                    else:
                        all_benign.extend(extras)
                    assigned_hashes.update(sample.content_hash for sample in extras)
                    for sample in extras:
                        by_reason[label][sample.reason].append(sample)
                        by_language[label][sample.language].append(sample)
                    _remove_samples_from_maps(
                        extras,
                        available_by_reason[label],
                        available_by_language[label],
                    )
                    by_fmt, by_len, by_lang = _dimension_counts(current)
                    cell_fills[key] = CellFillRecord(
                        target=fill.target,
                        actual=len(current),
                        backfilled=fill.backfilled + len(extras),
                        backfill_sources=list(fill.backfill_sources) + tags,
                        by_format=by_fmt,
                        by_length=by_len,
                        by_language=by_lang,
                        degraded=fill.degraded or bool(tags),
                        degraded_mode=_merge_degraded_mode(fill.degraded_mode, tags),
                    )
                if gaps:
                    all_gaps.extend(gaps)

    # Phase 5a: Supplement attack lane
    if spec.supplements:
        supplement_per = (
            int(spec.total_target * spec.supplement_ratio / len(spec.supplements) / 2)
            if spec.total_target
            else None
        )
        for supp in spec.supplements:
            cap = (
                min(supplement_per, supp.max_samples)
                if supplement_per is not None
                else supp.max_samples
            )
            supp_attack = _sample_supplement_sources(
                sources=supp.attack_sources,
                label="malicious",
                reason=f"supplement:{supp.name}",
            )
            if len(supp_attack) > cap:
                supp_attack = rng.sample(supp_attack, cap)
            all_attack.extend(supp_attack)

    if spec.supplements:
        dedup.register_attack_hashes([s.content_hash for s in all_attack])

    # Phase 5b: Discussion-benign lane
    # Reserve attack-adjacent benign discussion before background/supplement
    # benign so this scarce slice is measured explicitly.
    discussion_actual = 0
    if spec.discussion_benign and discussion_budget > 0:
        discussion_candidates = _sample_benign_lane(
            lane_name="discussion_benign",
            sources=spec.discussion_benign.sources,
            budget=discussion_budget,
            filter_attacklike=False,
        )
        all_benign.extend(discussion_candidates)
        discussion_actual = len(discussion_candidates)
        logger.info(
            "discussion_benign lane: requested=%d actual=%d",
            discussion_budget,
            discussion_actual,
        )
        if discussion_actual < discussion_budget:
            all_gaps.append(
                f"discussion_benign: requested {discussion_budget}, "
                f"got {discussion_actual} (-{discussion_budget - discussion_actual})"
            )

    # Phase 5c: Background benign lane
    background_actual = 0
    if spec.background and background_budget > 0:
        bg_candidates = _sample_benign_lane(
            lane_name="background",
            sources=spec.background.sources,
            budget=background_budget,
            filter_attacklike=True,
        )
        all_benign.extend(bg_candidates)
        background_actual = len(bg_candidates)
        logger.info(
            "background lane: requested=%d actual=%d",
            background_budget,
            background_actual,
        )
        if background_actual < background_budget:
            all_gaps.append(
                f"background: requested {background_budget}, "
                f"got {background_actual} (-{background_budget - background_actual})"
            )

    # Phase 5d: Supplement benign lane
    if spec.supplements:
        supplement_per = (
            int(spec.total_target * spec.supplement_ratio / len(spec.supplements) / 2)
            if spec.total_target
            else None
        )
        for supp in spec.supplements:
            cap = (
                min(supplement_per, supp.max_samples)
                if supplement_per is not None
                else supp.max_samples
            )
            supp_benign = _sample_supplement_sources(
                sources=supp.benign_sources,
                label="benign",
                reason=f"supplement:{supp.name}",
            )
            if len(supp_benign) > cap:
                supp_benign = rng.sample(supp_benign, cap)
            all_benign.extend(supp_benign)

    return SamplingResult(
        attack_samples=all_attack,
        benign_samples=all_benign,
        cell_fills=cell_fills,
        gaps=all_gaps,
        duplicates_dropped=dedup.duplicates_dropped + benign_duplicates_dropped,
        cross_contamination_dropped=dedup.cross_contamination_dropped + benign_cross_contamination_dropped,
        background_requested=background_budget,
        background_actual=background_actual,
        discussion_requested=discussion_budget,
        discussion_actual=discussion_actual,
        ledger_dropped=ledger_totals["dropped"],
        ledger_quarantined=ledger_totals["quarantined"],
        ledger_rerouted=ledger_totals["rerouted"],
        ledger_relabeled=ledger_totals["relabeled"],
    )
