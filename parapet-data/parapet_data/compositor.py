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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Sequence

import yaml

from .models import (
    CellFillRecord,
    CurationManifest,
    MirrorSpec,
    SplitManifest,
    compute_semantic_hash,
    compute_source_hash,
)
from .sampler import Sample, SamplingResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------

DEFAULT_SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "holdout": 0.1}


def split_samples(
    samples: list[Sample],
    ratios: dict[str, float] | None = None,
    holdout_only_reasons: Sequence[str] = (),
    seed: int = 42,
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
    rng.shuffle(remainder)

    # Compute split sizes from remainder
    n = len(remainder)
    holdout_ratio = ratios.get("holdout", 0.1)
    val_ratio = ratios.get("val", 0.1)

    n_holdout = int(n * holdout_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_holdout - n_val

    splits: dict[str, list[Sample]] = {
        "train": remainder[:n_train],
        "val": remainder[n_train : n_train + n_val],
        "holdout": remainder[n_train + n_val :] + holdout_only,
    }

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
    holdout_reasons = [r.value for r in spec.holdout_only_reasons]
    splits = split_samples(
        all_samples,
        holdout_only_reasons=holdout_reasons,
        seed=spec.seed,
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
    all_sources = []
    for cell in spec.cells:
        all_sources.extend(cell.attack_sources)
        all_sources.extend(cell.benign_sources)
    for supp in spec.supplements:
        all_sources.extend(supp.attack_sources)
        all_sources.extend(supp.benign_sources)
    for source in all_sources:
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

    return CurationManifest(
        spec_name=spec.name,
        spec_version=spec.version,
        spec_hash=spec.spec_hash(),
        seed=spec.seed,
        timestamp=datetime.now(timezone.utc).isoformat(),
        source_hashes=source_hashes,
        output_path=Path(combined_path.name),
        output_hash=_sha256_file(combined_path),
        semantic_hash=semantic_hash,
        total_samples=len(all_samples),
        attack_samples=len(sampling_result.attack_samples),
        benign_samples=len(sampling_result.benign_samples),
        splits=split_manifests,
        cell_fills=sampling_result.cell_fills,
        gaps=sampling_result.gaps,
        cross_contamination_dropped=sampling_result.cross_contamination_dropped,
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
