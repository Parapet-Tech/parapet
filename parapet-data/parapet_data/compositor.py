"""
Dataset composition: split, write, and manifest.

The compositor takes sampled data from the sampler, splits it into
train/val/holdout, writes output files, and produces a CurationManifest
with full provenance. This is the final step before data leaves
parapet-data and enters parapet-runner.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Sequence

import yaml

from .models import (
    CellFillRecord,
    CurationManifest,
    MirrorSpec,
    SplitManifest,
    SourceMetadata,
    VerifiedSyncManifest,
    compute_semantic_hash,
    compute_source_hash,
)
from .guardrails import check_feature_coverage
from .sampler import Sample, SamplingResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------

DEFAULT_SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "holdout": 0.1}


def _allocate_split_counts(
    n: int,
    *,
    val_ratio: float,
    holdout_ratio: float,
    min_stratum_for_floor: int = 3,
) -> tuple[int, int, int]:
    """Allocate train/val/holdout counts for one stratum."""
    if n <= 0:
        return 0, 0, 0

    n_holdout = int(n * holdout_ratio)
    n_val = int(n * val_ratio)

    if n >= min_stratum_for_floor and holdout_ratio > 0 and n_holdout == 0:
        n_holdout = 1
    if n >= min_stratum_for_floor and val_ratio > 0 and n_val == 0 and (n - n_holdout) >= 2:
        n_val = 1

    while n_holdout + n_val > n:
        if n_val >= n_holdout and n_val > 0:
            n_val -= 1
        elif n_holdout > 0:
            n_holdout -= 1
        else:
            break

    n_train = n - n_holdout - n_val
    return n_train, n_val, n_holdout


def split_samples(
    samples: list[Sample],
    ratios: dict[str, float] | None = None,
    holdout_only_reasons: Sequence[str] = (),
    seed: int = 42,
    stratified: bool = True,
) -> dict[str, list[Sample]]:
    """Split samples into train/val/holdout.

    Samples whose reason is in holdout_only_reasons go exclusively
    to holdout — they test generalization to unseen attack categories.
    Remaining samples are shuffled and split by ratio.
    """
    ratios = ratios or DEFAULT_SPLIT_RATIOS

    holdout_only: list[Sample] = []
    remainder: list[Sample] = []

    holdout_reasons = set(str(r) for r in holdout_only_reasons)
    for s in samples:
        if s.reason in holdout_reasons:
            holdout_only.append(s)
        else:
            remainder.append(s)

    rng = random.Random(seed)
    holdout_ratio = ratios.get("holdout", 0.1)
    val_ratio = ratios.get("val", 0.1)

    splits: dict[str, list[Sample]] = {"train": [], "val": [], "holdout": []}

    if not stratified:
        rng.shuffle(remainder)
        n = len(remainder)
        n_holdout = int(n * holdout_ratio)
        n_val = int(n * val_ratio)
        n_train = n - n_holdout - n_val
        splits["train"] = remainder[:n_train]
        splits["val"] = remainder[n_train : n_train + n_val]
        splits["holdout"] = remainder[n_train + n_val :]
    else:
        # Primary strata: preserve label/reason/language proportions.
        primary_groups: dict[tuple[str, str, str], list[Sample]] = {}
        for sample in remainder:
            key = (sample.label, sample.reason, sample.language)
            primary_groups.setdefault(key, []).append(sample)

        final_strata: list[list[Sample]] = []
        for primary_key in sorted(primary_groups.keys()):
            primary_samples = primary_groups[primary_key]
            secondary: dict[tuple[str, str], list[Sample]] = {}
            for sample in primary_samples:
                sec_key = (sample.format_bin, sample.length_bin)
                secondary.setdefault(sec_key, []).append(sample)

            collapsed: list[Sample] = []
            for sec_key in sorted(secondary.keys()):
                sec_samples = secondary[sec_key]
                if len(sec_samples) < 3:
                    collapsed.extend(sec_samples)
                    logger.debug(
                        "Collapsing tiny sub-stratum %s/%s size=%d to primary stratum",
                        primary_key,
                        sec_key,
                        len(sec_samples),
                    )
                else:
                    final_strata.append(sec_samples)
            if collapsed:
                final_strata.append(collapsed)

        for stratum in final_strata:
            local = list(stratum)
            rng.shuffle(local)
            n = len(local)
            n_train, n_val, n_holdout = _allocate_split_counts(
                n, val_ratio=val_ratio, holdout_ratio=holdout_ratio
            )
            splits["train"].extend(local[:n_train])
            splits["val"].extend(local[n_train : n_train + n_val])
            splits["holdout"].extend(local[n_train + n_val :])
            if n >= 3 and (n_val == 0 or n_holdout == 0):
                logger.warning(
                    "Stratum size=%d received sparse allocation (val=%d, holdout=%d)",
                    n,
                    n_val,
                    n_holdout,
                )

    splits["holdout"].extend(holdout_only)

    for name, split in splits.items():
        logger.info(
            "Split %s: %d samples (%d attack, %d benign)",
            name,
            len(split),
            sum(1 for s in split if s.label == "malicious"),
            sum(1 for s in split if s.label == "benign"),
        )

    return splits


# ---------------------------------------------------------------------------
# Output writing (YAML + JSONL)
# ---------------------------------------------------------------------------

OutputFormat = Literal["yaml", "jsonl"]

_FORMAT_EXTENSIONS: dict[OutputFormat, str] = {"yaml": ".yaml", "jsonl": ".jsonl"}


def _collect_source_alias_warnings(
    source_metadata: dict[str, SourceMetadata],
) -> list[str]:
    """Warn when different source names point at the same path with different lanes."""
    by_path: dict[str, list[tuple[str, SourceMetadata]]] = defaultdict(list)
    for name, metadata in source_metadata.items():
        by_path[str(metadata.path)].append((name, metadata))

    warnings: list[str] = []
    for path in sorted(by_path.keys()):
        entries = by_path[path]
        if len(entries) < 2:
            continue
        route_policies = {
            metadata.route_policy.value if metadata.route_policy is not None else "unset"
            for _, metadata in entries
        }
        if len(route_policies) < 2:
            continue
        aliases = ", ".join(sorted(name for name, _ in entries))
        lanes = ", ".join(sorted(route_policies))
        warnings.append(
            f"source path {path} is aliased by {aliases} with differing route_policies: {lanes}"
        )
    return warnings


def _sample_to_dict(sample: Sample) -> dict:
    """Convert a Sample to a serializable dict."""
    return {
        "content": sample.content,
        "label": sample.label,
        "reason": sample.reason,
        "source": sample.source_name,
        "language": sample.language,
        "format_bin": sample.format_bin,
        "length_bin": sample.length_bin,
    }


def write_split(
    samples: list[Sample],
    output_path: Path,
    fmt: OutputFormat = "yaml",
) -> SplitManifest:
    """Write samples to disk and return a SplitManifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content_hashes: list[str] = []
    rows = []
    for sample in samples:
        content_hashes.append(sample.content_hash)
        rows.append(_sample_to_dict(sample))

    if fmt == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(rows, f, default_flow_style=False, allow_unicode=True,
                      sort_keys=False, width=200)

    return SplitManifest(
        name=output_path.stem,
        sample_count=len(samples),
        content_hashes=sorted(content_hashes),
        artifact_path=Path(output_path.name),
    )


# ---------------------------------------------------------------------------
# Manifest generation
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """SHA256 of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compose(
    spec: MirrorSpec,
    sampling_result: SamplingResult,
    output_dir: Path,
    base_dir: Path | None = None,
    fmt: OutputFormat = "yaml",
    stratified_split: bool = True,
    min_df: int | None = None,
    max_features: int | None = None,
    verified_sync: VerifiedSyncManifest | None = None,
) -> CurationManifest:
    """Compose the final curated dataset from sampling results.

    1. Merge attack + benign samples
    2. Split into train/val/holdout
    3. Write split files (YAML or JSONL)
    4. Compute provenance hashes
    5. Return CurationManifest
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = _FORMAT_EXTENSIONS[fmt]

    all_samples = sampling_result.attack_samples + sampling_result.benign_samples

    # Split
    holdout_reasons = list(spec.holdout_only_reasons)
    splits = split_samples(
        all_samples,
        holdout_only_reasons=holdout_reasons,
        seed=spec.seed,
        stratified=stratified_split,
    )

    # Write per split
    split_manifests: dict[str, SplitManifest] = {}
    all_content_hashes: list[str] = []

    for split_name, split_samples_list in splits.items():
        path = output_dir / f"{split_name}{ext}"
        manifest = write_split(split_samples_list, path, fmt=fmt)
        split_manifests[split_name] = manifest
        all_content_hashes.extend(manifest.content_hashes)

    # Write combined dataset too
    combined_path = output_dir / f"curated{ext}"
    rows = [_sample_to_dict(s) for s in all_samples]
    if fmt == "jsonl":
        with open(combined_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        with open(combined_path, "w", encoding="utf-8") as f:
            yaml.dump(rows, f, default_flow_style=False, allow_unicode=True,
                      sort_keys=False, width=200)

    # Source hashes
    source_hashes: dict[str, str] = {}
    source_metadata: dict[str, SourceMetadata] = {}
    all_sources = []
    for cell in spec.cells:
        all_sources.extend(cell.attack_sources)
        all_sources.extend(cell.benign_sources)
    for supp in spec.supplements:
        all_sources.extend(supp.attack_sources)
        all_sources.extend(supp.benign_sources)
    if spec.background:
        all_sources.extend(spec.background.sources)
    for source in all_sources:
        if source.name not in source_metadata:
            source_metadata[source.name] = SourceMetadata.from_source_ref(source)
        if source.name not in source_hashes:
            try:
                source_hashes[source.name] = compute_source_hash(
                    source, base_dir=base_dir,
                )
            except (FileNotFoundError, OSError) as e:
                logger.warning("Could not hash source %s: %s", source.name, e)
                source_hashes[source.name] = "unavailable"

    # Semantic hash
    semantic_hash = compute_semantic_hash(
        all_content_hashes,
        sampling_result.cell_fills,
    )

    feature_coverage_warnings: list[str] = []
    if min_df is not None and max_features is not None:
        quota_profile = None
        if spec.language_quota is not None:
            quota_profile = {
                lang.value: pct for lang, pct in spec.language_quota.profile.items()
            }
        feature_coverage_warnings = check_feature_coverage(
            composition=composition_report(all_samples),
            min_df=min_df,
            max_features=max_features,
            language_quota=quota_profile,
        )
    source_alias_warnings = _collect_source_alias_warnings(source_metadata)

    return CurationManifest(
        spec_name=spec.name,
        spec_version=spec.version,
        spec_hash=spec.spec_hash(),
        seed=spec.seed,
        timestamp=datetime.now(timezone.utc).isoformat(),
        source_hashes=source_hashes,
        source_metadata=source_metadata,
        output_path=Path(combined_path.name),
        output_hash=_sha256_file(combined_path),
        semantic_hash=semantic_hash,
        total_samples=len(all_samples),
        attack_samples=len(sampling_result.attack_samples),
        benign_samples=len(sampling_result.benign_samples),
        splits=split_manifests,
        cell_fills=sampling_result.cell_fills,
        gaps=sampling_result.gaps,
        duplicates_dropped=sampling_result.duplicates_dropped,
        cross_contamination_dropped=sampling_result.cross_contamination_dropped,
        background_requested=sampling_result.background_requested,
        background_actual=sampling_result.background_actual,
        source_alias_warnings=source_alias_warnings,
        feature_coverage_warnings=feature_coverage_warnings,
        ledger_dropped=sampling_result.ledger_dropped,
        ledger_quarantined=sampling_result.ledger_quarantined,
        ledger_rerouted=sampling_result.ledger_rerouted,
        ledger_relabeled=sampling_result.ledger_relabeled,
        verified_sync=verified_sync,
    )


# ---------------------------------------------------------------------------
# Composition stats (for reporting)
# ---------------------------------------------------------------------------


def composition_report(samples: list[Sample]) -> dict:
    """Build composition statistics for reporting."""
    n = len(samples)
    if n == 0:
        return {"total": 0}

    def _rows(counter: Counter) -> list[dict]:
        return [
            {"name": k, "count": int(v), "pct": round((v / n) * 100, 2)}
            for k, v in counter.most_common()
        ]

    return {
        "total": n,
        "by_label": _rows(Counter(s.label for s in samples)),
        "by_reason": _rows(Counter(s.reason for s in samples)),
        "by_language": _rows(Counter(s.language for s in samples)),
        "by_format": _rows(Counter(s.format_bin for s in samples)),
        "by_length": _rows(Counter(s.length_bin for s in samples)),
        "by_source": _rows(Counter(s.source_name for s in samples)),
    }
