"""Corpus loaders and synthetic length-stratified text for the latency bench.

This module is intentionally policy-free: it does not know which paths are
eval-only or which sources are forbidden by direction.md Phase 3 split
discipline. That gate lives in the Phase 0.6 entry-point so the loader stays
reusable across legitimate corpora.

Loader contract (strict — fail-closed on ambiguity):

  * ``.jsonl`` — one JSON object per line. ``row[text_field]`` must exist and
    be a non-empty string. Blank lines are skipped. Malformed JSON, non-object
    rows, missing fields, or non-string / empty / whitespace-only values raise
    ``CorpusFormatError``.

  * ``.yaml`` / ``.yml`` — root must be a list. The first element pins the
    shape: either every element is a non-empty string, or every element is an
    object with ``row[text_field]`` non-empty string. Mixed shapes raise.

  * Other extensions raise. ``.json`` (single-document) is intentionally not
    supported; convert to ``.jsonl`` if needed.

Synthetic generator: ``synthetic_length_stratified`` draws character-length
samples from a histogram, deterministically seeded. Useful when no non-eval
corpus is available — guarantees zero leakage of eval-only artifacts. Note
this approximates token length via character length; it is correct for
order-of-magnitude latency estimates on English-like text but not exact for
multilingual or heavily-tokenized scripts. For exact token-length control,
the entry-point should sample from real (non-eval) curated text.
"""

from __future__ import annotations

import hashlib
import json
import random
import string
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_ALPHABET: str = string.ascii_lowercase + " "


class CorpusFormatError(ValueError):
    """Raised when a corpus file's shape is unsupported or ambiguous."""


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_corpus(path: Path, text_field: str = "content") -> Iterator[str]:
    """Yield text strings from a corpus file (projection of full-row loader)."""

    if not text_field:
        raise ValueError("text_field must be non-empty")

    for row in load_corpus_rows(path):
        if isinstance(row, str):
            # YAML list-of-strings shape — text is the row itself.
            if not row.strip():
                # Already validated in load_corpus_rows, but defensive.
                continue
            yield row
            continue
        # Mapping row — extract the requested text field.
        if text_field not in row:
            raise CorpusFormatError(
                f"{path} row missing field {text_field!r}"
            )
        value = row[text_field]
        if not isinstance(value, str):
            raise CorpusFormatError(
                f"{path} field {text_field!r} must be str, got {type(value).__name__}"
            )
        if not value.strip():
            raise CorpusFormatError(
                f"{path} field {text_field!r} is empty or whitespace-only"
            )
        yield value


def load_corpus_rows(path: Path) -> Iterator[Mapping[str, Any] | str]:
    """Yield full corpus rows preserving metadata (label, language, source, ...).

    Object rows come back as dicts; bare-string YAML rows come back as
    strings. Callers that only need text should use ``load_corpus`` instead.

    Strict shape validation matches ``load_corpus``: malformed JSON,
    non-mapping JSONL rows, mixed YAML list shapes all fail closed.
    """

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _load_jsonl_rows(path)
    elif suffix in (".yaml", ".yml"):
        yield from _load_yaml_rows(path)
    else:
        raise CorpusFormatError(
            f"Unsupported corpus extension {suffix!r} (path={path}); "
            "use .jsonl, .yaml, or .yml"
        )


def _load_jsonl_rows(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusFormatError(
                    f"{path}:{line_no} malformed JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, Mapping):
                raise CorpusFormatError(
                    f"{path}:{line_no} expected JSON object, got {type(row).__name__}"
                )
            yield row


def _load_yaml_rows(path: Path) -> Iterator[Mapping[str, Any] | str]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None:
        raise CorpusFormatError(f"{path} is empty or contains only YAML null")
    if not isinstance(data, list):
        raise CorpusFormatError(
            f"{path} root must be a list, got {type(data).__name__}"
        )
    if not data:
        return

    first = data[0]
    if isinstance(first, str):
        for i, row in enumerate(data):
            if not isinstance(row, str):
                raise CorpusFormatError(
                    f"{path}[{i}] mixed list shape: expected str (matched first row), "
                    f"got {type(row).__name__}"
                )
            if not row.strip():
                raise CorpusFormatError(f"{path}[{i}] empty or whitespace-only string")
            yield row
    elif isinstance(first, Mapping):
        for i, row in enumerate(data):
            if not isinstance(row, Mapping):
                raise CorpusFormatError(
                    f"{path}[{i}] mixed list shape: expected object (matched first row), "
                    f"got {type(row).__name__}"
                )
            yield row
    else:
        raise CorpusFormatError(
            f"{path}[0] expected str or object, got {type(first).__name__}"
        )


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------


def synthetic_length_stratified(
    char_length_histogram: Mapping[int, int],
    seed: int,
    alphabet: str = _DEFAULT_ALPHABET,
) -> Iterator[str]:
    """Generate strings whose character-length distribution matches the input.

    ``char_length_histogram`` maps character-length → number of samples to
    emit at that length. The generator is fully deterministic in ``seed``.
    """

    if not alphabet:
        raise ValueError("alphabet must be non-empty")

    plan: list[int] = []
    for length, count in char_length_histogram.items():
        if length < 0:
            raise ValueError(f"length must be non-negative, got {length}")
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if length == 0 and count > 0:
            raise ValueError(
                "length=0 produces empty strings, which the loader rejects; "
                "drop length=0 entries from the histogram"
            )
        plan.extend([length] * count)

    rng = random.Random(seed)
    rng.shuffle(plan)
    for length in plan:
        yield "".join(rng.choices(alphabet, k=length))


# ---------------------------------------------------------------------------
# Content-anchored corpus hashing
# ---------------------------------------------------------------------------


def compute_corpus_sha256(strings: Iterable[str]) -> str:
    """SHA-256 over the materialized corpus content.

    Format-agnostic: the same rows in .jsonl, .yaml, or in-memory-synthetic
    produce the same hash. Each row is encoded as a length-prefixed utf-8
    blob (8-byte big-endian length followed by bytes) so that boundaries
    between rows cannot alias to embedded payload bytes — e.g.
    ``["a\\0b"]`` and ``["a", "b"]`` produce different hashes.

    Reproducibility-relevant: the manifest's ``corpus_sha256`` is the
    identity anchor for which text a bench actually saw, regardless of how
    that text was sourced.
    """
    h = hashlib.sha256()
    for row in strings:
        if not isinstance(row, str):
            raise TypeError(f"corpus row must be str, got {type(row).__name__}")
        body = row.encode("utf-8")
        h.update(len(body).to_bytes(8, "big"))
        h.update(body)
    return h.hexdigest()
