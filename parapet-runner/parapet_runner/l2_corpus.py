"""Build a stratified L2 latency-bench corpus from curated v8 train/val.

direction.md Phase 0.6 needs representative residual-shaped text against
which to measure mDeBERTa CPU p50/p99. The Phase 1 residual export against
the freshly-clean L1 path is not yet generated, so the immediate corpus is
a stratified sample of curated v8 train/val that preserves the dimensions
that affect tokenizer behavior and inference cost — primarily language and
length, with reason / label / source recorded as observational metadata.

Output is a strict JSONL file (``{"content": "..."}`` per line) plus a
sidecar ``<output>.manifest.json`` recording:

* source paths and their content SHA-256 (so a future run confirms
  "same inputs"),
* stratification axes and length-bucket edges,
* random seed,
* input vs output cell counts and per-axis distributions,
* output content SHA-256 (cross-checks the bench's ``corpus_sha256``).

This module imports nothing ML-related — it is a pure data sampler.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .latency_corpus import (
    CorpusFormatError,
    compute_corpus_sha256,
    load_corpus_rows,
)
from .latency_runtime import sha256_file


# Default character-length bucket edges. Char length is a proxy for token
# count; coarser than the bench's histogram so cells stay populated.
_DEFAULT_LENGTH_BUCKETS: tuple[int, ...] = (128, 512, 2048)

# Default stratification axes. Latency is most sensitive to length and
# tokenizer behavior (which varies by language); other axes ride along
# in the manifest distributions.
_DEFAULT_STRATIFY_AXES: tuple[str, ...] = ("language", "length_bucket")

# Axes always recorded in distribution snapshots, whether or not they
# stratify. Keeps reviewers honest about what made it into the sample.
_RECORDED_AXES: tuple[str, ...] = (
    "language",
    "label",
    "reason",
    "source",
    "length_bucket",
)


@dataclass(frozen=True)
class StratifySpec:
    target_rows: int
    seed: int
    axes: tuple[str, ...] = _DEFAULT_STRATIFY_AXES
    length_buckets: tuple[int, ...] = _DEFAULT_LENGTH_BUCKETS
    text_field: str = "content"

    def __post_init__(self) -> None:
        if self.target_rows < 1:
            raise ValueError(f"target_rows must be >= 1, got {self.target_rows}")
        if not self.axes:
            raise ValueError("axes must be non-empty")
        for axis in self.axes:
            if axis not in _RECORDED_AXES:
                raise ValueError(
                    f"Unknown stratification axis {axis!r}; allowed: {_RECORDED_AXES}"
                )
        if not self.length_buckets:
            raise ValueError("length_buckets must be non-empty")
        if any(e <= 0 for e in self.length_buckets):
            raise ValueError(
                f"length_buckets must all be positive, got {self.length_buckets}"
            )
        if not self.text_field.strip():
            raise ValueError("text_field must be non-empty")


class CorpusBuildManifest(BaseModel):
    """Sidecar manifest describing a built latency corpus."""

    schema_version: int = 1
    source_paths: list[str]
    source_sha256s: list[str]
    stratify_axes: list[str]
    length_buckets: list[int]
    seed: int
    target_rows: int
    text_field: str
    n_input_rows: int
    n_output_rows: int
    output_sha256: str
    cell_counts_in: dict[str, int]
    cell_counts_out: dict[str, int]
    axis_distribution_in: dict[str, dict[str, int]]
    axis_distribution_out: dict[str, dict[str, int]]


def build_l2_latency_corpus(
    source_paths: list[Path],
    spec: StratifySpec,
    output_path: Path,
) -> CorpusBuildManifest:
    """Read sources, stratified-sample, write JSONL + sidecar manifest.

    Returns the manifest also written to ``{output_path}.manifest.json``.
    Output rows are JSONL with only the ``content`` field projected out;
    other metadata stays in the manifest's distributions for transparency
    but does not leak into what the bench actually consumes.
    """

    if not source_paths:
        raise ValueError("source_paths must be non-empty")

    rows = _collect_valid_rows(source_paths, spec)
    if not rows:
        raise CorpusFormatError(
            f"No valid rows with field {spec.text_field!r} found in {len(source_paths)} sources"
        )

    rng = random.Random(spec.seed)
    sampled = _stratified_sample(rows, spec, rng)

    cell_in = _cell_counts(rows, spec)
    cell_out = _cell_counts(sampled, spec)
    axis_in = {axis: _axis_distribution(rows, axis, spec) for axis in _RECORDED_AXES}
    axis_out = {axis: _axis_distribution(sampled, axis, spec) for axis in _RECORDED_AXES}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    texts: list[str] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for row in sampled:
            text = row[spec.text_field]
            handle.write(json.dumps({"content": text}, ensure_ascii=False) + "\n")
            texts.append(text)

    manifest = CorpusBuildManifest(
        source_paths=[str(p) for p in source_paths],
        source_sha256s=[sha256_file(p) for p in source_paths],
        stratify_axes=list(spec.axes),
        length_buckets=list(spec.length_buckets),
        seed=spec.seed,
        target_rows=spec.target_rows,
        text_field=spec.text_field,
        n_input_rows=len(rows),
        n_output_rows=len(sampled),
        output_sha256=compute_corpus_sha256(texts),
        cell_counts_in=cell_in,
        cell_counts_out=cell_out,
        axis_distribution_in=axis_in,
        axis_distribution_out=axis_out,
    )
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _collect_valid_rows(
    source_paths: list[Path], spec: StratifySpec
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for path in source_paths:
        for raw in load_corpus_rows(path):
            if not isinstance(raw, Mapping):
                # Bare YAML strings have no metadata for stratification.
                continue
            value = raw.get(spec.text_field)
            if not isinstance(value, str) or not value.strip():
                continue
            rows.append(raw)
    return rows


def _stratified_sample(
    rows: list[Mapping[str, Any]],
    spec: StratifySpec,
    rng: random.Random,
) -> list[Mapping[str, Any]]:
    """Largest-remainder proportional allocation by composite cell key."""

    cells: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(_axis_value(row, axis, spec) for axis in spec.axes)
        cells[key].append(row)

    n_in = len(rows)
    raw_alloc = {key: len(cell) * spec.target_rows / n_in for key, cell in cells.items()}
    int_alloc: dict[tuple[str, ...], int] = {key: int(v) for key, v in raw_alloc.items()}

    used = sum(int_alloc.values())
    deficit = spec.target_rows - used
    if deficit > 0:
        ranked = sorted(
            cells.keys(),
            key=lambda k: (-(raw_alloc[k] - int_alloc[k]), str(k)),
        )
        for key in ranked[:deficit]:
            int_alloc[key] += 1

    # Cap by cell capacity (can't sample more than the cell holds).
    for key in list(int_alloc):
        int_alloc[key] = min(int_alloc[key], len(cells[key]))

    sampled: list[Mapping[str, Any]] = []
    for key in sorted(cells.keys(), key=str):
        n = int_alloc.get(key, 0)
        if n == 0:
            continue
        sampled.extend(rng.sample(cells[key], n))

    rng.shuffle(sampled)
    return sampled


def _axis_value(row: Mapping[str, Any], axis: str, spec: StratifySpec) -> str:
    if axis == "length_bucket":
        text = row.get(spec.text_field, "")
        if not isinstance(text, str):
            return "unknown"
        return _bucket_label(len(text), spec.length_buckets)
    val = row.get(axis)
    if val is None:
        return "unknown"
    return str(val)


def _bucket_label(length: int, edges: tuple[int, ...]) -> str:
    sorted_edges = sorted(set(edges))
    prev = 0
    for edge in sorted_edges:
        if length <= edge:
            return f"{prev}-{edge}"
        prev = edge + 1
    return f"{prev}+"


def _cell_counts(
    rows: Iterable[Mapping[str, Any]], spec: StratifySpec
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        key_parts = [f"{axis}={_axis_value(row, axis, spec)}" for axis in spec.axes]
        counter["|".join(key_parts)] += 1
    return dict(counter)


def _axis_distribution(
    rows: Iterable[Mapping[str, Any]], axis: str, spec: StratifySpec
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter[_axis_value(row, axis, spec)] += 1
    return dict(counter)
