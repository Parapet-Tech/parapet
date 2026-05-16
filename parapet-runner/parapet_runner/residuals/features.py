"""Mechanical feature extraction for residual analysis."""

from __future__ import annotations

import json
import math
import re
import unicodedata
import zlib
from collections import Counter
from typing import Any

from .geometry import scalar_auc_for_field


FEATURE_SEMANTICS = {
    "*_len": "Unicode scalar values; intended Rust analogue is s.chars().count().",
    "*_byte_*": "UTF-8 bytes; intended Rust analogue is s.len().",
    "graphemes": "Not used in pass one.",
}

ZERO_WIDTH_CHARS = frozenset({
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
    "\ufeff",
    "\u00ad",
})
ROLE_MARKER_HINTS = (
    "[INST]",
    "[/INST]",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    "<|endoftext|>",
)
BRACKET_CHARS = set("()[]{}<>")
QUOTE_CHARS = set("\"'`“”‘’«»")
OPERATOR_CHARS = set("+-=*/\\|_^~:%$#@!")


FEATURE_FAMILIES: dict[str, list[str]] = {
    "l2_geometry": [
        "raw_margin",
        "raw_margin_to_threshold",
        "unquoted_margin_to_threshold",
        "squash_margin_to_threshold",
        "squash_minus_raw",
        "squash_minus_unquoted",
        "raw_minus_unquoted",
        "mention_dampened_effective_margin",
    ],
    "l0_deltas": [
        "l0_char_delta",
        "l0_byte_delta",
        "l0_zero_width_removed_count",
        "l0_control_removed_count",
        "l0_html_stripped_hint",
        "l0_role_marker_hint",
    ],
    "mechanical_text_shape": [
        "char_len",
        "byte_len",
        "line_count",
        "max_line_len",
        "avg_line_len",
        "longest_no_whitespace_span",
        "whitespace_ratio",
        "punct_ratio",
        "digit_ratio",
        "quote_ratio",
        "bracket_ratio",
        "operator_ratio",
        "uppercase_ratio",
        "non_ascii_ratio",
        "mixed_script_hint",
        "letter_ratio",
        "control_count",
        "zero_width_count",
        "replacement_char_count",
        "json_like",
        "yaml_like",
        "markdown_fence_count",
        "url_count",
        "xml_tag_like_count",
    ],
    "entropy_compression": [
        "shannon_entropy_bytes",
        "zlib_compression_ratio",
    ],
}


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _threshold(row: dict[str, Any]) -> float:
    thresholds = row.get("l1_thresholds")
    if isinstance(thresholds, dict):
        value = _to_float(thresholds.get("l1"))
        if value is not None:
            return value
    return 0.0


def _longest_no_whitespace_span(text: str) -> int:
    longest = 0
    current = 0
    for char in text:
        if char.isspace():
            longest = max(longest, current)
            current = 0
        else:
            current += 1
    return max(longest, current)


def _script_name(char: str) -> str | None:
    if not char.isalpha() or ord(char) < 128:
        return None
    name = unicodedata.name(char, "")
    for script in ("ARABIC", "CYRILLIC", "CJK", "HIRAGANA", "KATAKANA", "HEBREW", "GREEK", "LATIN"):
        if script in name:
            return script
    return None


def _shannon_entropy_bytes(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _python_l0_normalize(text: str) -> str:
    """Approximate Rust L0 for offline feature discovery.

    Runtime promotion still requires a Rust parity fixture.
    """

    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"(?is)<script\b[^>]*>.*?</script\s*>", "", normalized)
    normalized = re.sub(r"(?is)<style\b[^>]*>.*?</style\s*>", "", normalized)
    normalized = re.sub(r"(?s)<[^>]*>", "", normalized)
    normalized = "".join(char for char in normalized if char not in ZERO_WIDTH_CHARS)
    return normalized


def compute_features(row: dict[str, Any]) -> dict[str, Any]:
    """Compute feature values for one residual row."""

    text = row.get("content")
    if not isinstance(text, str):
        text = ""

    chars = list(text)
    char_count = len(chars)
    byte_count = len(text.encode("utf-8"))
    lines = text.splitlines() or [text]
    categories = [unicodedata.category(char) for char in chars]
    scripts = {script for char in chars if (script := _script_name(char)) is not None}

    whitespace = sum(1 for char in chars if char.isspace())
    punctuation = sum(1 for cat in categories if cat.startswith("P"))
    digits = sum(1 for char in chars if char.isdigit())
    quotes = sum(1 for char in chars if char in QUOTE_CHARS)
    brackets = sum(1 for char in chars if char in BRACKET_CHARS)
    operators = sum(1 for char in chars if char in OPERATOR_CHARS)
    uppercase = sum(1 for char in chars if char.isupper())
    non_ascii = sum(1 for char in chars if ord(char) > 127)
    letters = sum(1 for char in chars if char.isalpha())
    controls = sum(1 for cat in categories if cat.startswith("C"))
    zero_width = sum(1 for char in chars if char in ZERO_WIDTH_CHARS)
    replacement = text.count("\ufffd")

    raw_score = _to_float(row.get("raw_score"))
    raw_unquoted = _to_float(row.get("raw_unquoted_score"))
    raw_squash = _to_float(row.get("raw_squash_score"))
    threshold = _threshold(row)

    l0_text = _python_l0_normalize(text)
    l0_removed_zero_width = max(0, zero_width - sum(1 for char in l0_text if char in ZERO_WIDTH_CHARS))
    l0_control_post = sum(1 for char in l0_text if unicodedata.category(char).startswith("C"))

    stripped = text.strip()
    encoded = text.encode("utf-8")
    compressed_len = len(zlib.compress(encoded)) if encoded else 0

    return {
        "char_len": char_count,
        "byte_len": byte_count,
        "line_count": len(lines),
        "max_line_len": max((len(line) for line in lines), default=0),
        "avg_line_len": _ratio(sum(len(line) for line in lines), len(lines)),
        "longest_no_whitespace_span": _longest_no_whitespace_span(text),
        "whitespace_ratio": _ratio(whitespace, char_count),
        "punct_ratio": _ratio(punctuation, char_count),
        "digit_ratio": _ratio(digits, char_count),
        "quote_ratio": _ratio(quotes, char_count),
        "bracket_ratio": _ratio(brackets, char_count),
        "operator_ratio": _ratio(operators, char_count),
        "uppercase_ratio": _ratio(uppercase, letters),
        "non_ascii_ratio": _ratio(non_ascii, char_count),
        "mixed_script_hint": len(scripts) > 1,
        "letter_ratio": _ratio(letters, char_count),
        "control_count": controls,
        "zero_width_count": zero_width,
        "replacement_char_count": replacement,
        "shannon_entropy_bytes": _shannon_entropy_bytes(encoded),
        "zlib_compression_ratio": _ratio(compressed_len, len(encoded)),
        "json_like": stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")),
        "yaml_like": bool(re.search(r"(?m)^\s*[-\w][^:\n]{0,80}:\s+", text)),
        "markdown_fence_count": text.count("```"),
        "url_count": len(re.findall(r"https?://", text)),
        "xml_tag_like_count": len(re.findall(r"</?[A-Za-z][^>\n]{0,120}>", text)),
        "raw_margin": None if raw_score is None else raw_score - threshold,
        "raw_margin_to_threshold": None if raw_score is None else raw_score - threshold,
        "unquoted_margin_to_threshold": None if raw_unquoted is None else raw_unquoted - threshold,
        "squash_margin_to_threshold": None if raw_squash is None else raw_squash - threshold,
        "squash_minus_raw": None if raw_squash is None or raw_score is None else raw_squash - raw_score,
        "squash_minus_unquoted": None if raw_squash is None or raw_unquoted is None else raw_squash - raw_unquoted,
        "raw_minus_unquoted": None if raw_score is None or raw_unquoted is None else raw_score - raw_unquoted,
        "mention_dampened_effective_margin": (
            None if raw_score is None else
            (raw_unquoted if row.get("quote_detected") is True and raw_unquoted is not None else raw_score) - threshold
        ),
        "l0_char_delta": len(text) - len(l0_text),
        "l0_byte_delta": len(text.encode("utf-8")) - len(l0_text.encode("utf-8")),
        "l0_zero_width_removed_count": l0_removed_zero_width,
        "l0_control_removed_count": max(0, controls - l0_control_post),
        "l0_html_stripped_hint": bool(re.search(r"(?is)<[A-Za-z!/][^>]*>", text)),
        "l0_role_marker_hint": any(marker in text for marker in ROLE_MARKER_HINTS),
    }


METADATA_FIELDS = (
    "content_hash",
    "fold_id",
    "label",
    "language",
    "source",
    "format_bin",
    "length_bin",
    "reason",
    "residual_category",
    "error_type",
    "l1_score",
    "raw_score",
    "raw_unquoted_score",
    "raw_squash_score",
    "raw_score_delta",
    "unquoted_score",
    "squash_score",
    "quote_detected",
    "l1_thresholds",
)


def feature_table_row(row: dict[str, Any], *, row_set: str) -> dict[str, Any]:
    out = {"row_set": row_set}
    for field in METADATA_FIELDS:
        if field in row:
            out[field] = row[field]
    out.update(compute_features(row))
    return out


def build_feature_table(
    residual_rows: list[dict[str, Any]],
    sidecar_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    table = [feature_table_row(row, row_set="residual") for row in residual_rows]
    table.extend(feature_table_row(row, row_set="baseline_correct") for row in sidecar_rows)
    return table


def feature_auc_tables(feature_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    residual_rows = [row for row in feature_rows if row.get("row_set") == "residual"]
    out: dict[str, list[dict[str, Any]]] = {}
    for family, fields in FEATURE_FAMILIES.items():
        rows = [scalar_auc_for_field(residual_rows, field) for field in fields]
        rows.sort(key=lambda row: (-1.0 if row["separation"] is None else -row["separation"], row["field"]))
        out[family] = rows
    return out


def top_decile_enrichment(feature_rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    residual_rows = [
        row for row in feature_rows
        if row.get("row_set") == "residual"
        and row.get("residual_category") in {"false_negative", "false_positive", "near_boundary_benign"}
        and isinstance(row.get(field), (int, float, bool))
        and math.isfinite(float(row.get(field)))
    ]
    residual_rows.sort(key=lambda row: float(row[field]), reverse=True)
    if not residual_rows:
        return {"field": field, "selected_n": 0, "false_negative_share": 0.0}
    selected_n = max(1, math.ceil(len(residual_rows) * 0.10))
    selected = residual_rows[:selected_n]
    positives = sum(1 for row in selected if row.get("residual_category") == "false_negative")
    return {
        "field": field,
        "selected_n": selected_n,
        "false_negative_count": positives,
        "false_negative_share": positives / selected_n,
    }


def semantics_receipt() -> dict[str, Any]:
    return {
        "feature_semantics": FEATURE_SEMANTICS,
        "l0_python_mirror": "analysis_only_unverified_against_rust",
        "grapheme_features": "deferred",
    }
