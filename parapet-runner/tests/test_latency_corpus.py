from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import pytest
import yaml

from parapet_runner.latency_corpus import (
    CorpusFormatError,
    compute_corpus_sha256,
    load_corpus,
    synthetic_length_stratified,
)


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Path:
    """Project-local replacement for the disabled pytest tmp_path fixture.

    Mirrors the existing convention in tests/test_intersect_residuals.py.
    """

    name = request.node.name
    output_dir = Path("tests/.tmp_outputs/latency_corpus") / name
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_jsonl_object_rows_yield_text_field(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "c.jsonl",
        '{"content": "hello"}\n{"content": "world"}\n',
    )
    assert list(load_corpus(path)) == ["hello", "world"]


def test_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "c.jsonl",
        '{"content": "a"}\n\n   \n{"content": "b"}\n',
    )
    assert list(load_corpus(path)) == ["a", "b"]


def test_jsonl_custom_text_field(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.jsonl", '{"prompt": "x"}\n{"prompt": "y"}\n')
    assert list(load_corpus(path, text_field="prompt")) == ["x", "y"]


def test_jsonl_missing_field_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.jsonl", '{"other": "x"}\n')
    with pytest.raises(CorpusFormatError, match="missing field 'content'"):
        list(load_corpus(path))


def test_jsonl_non_string_value_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.jsonl", '{"content": 123}\n')
    with pytest.raises(CorpusFormatError, match="must be str, got int"):
        list(load_corpus(path))


def test_jsonl_empty_string_value_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.jsonl", '{"content": ""}\n')
    with pytest.raises(CorpusFormatError, match="empty or whitespace-only"):
        list(load_corpus(path))


def test_jsonl_whitespace_only_value_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.jsonl", '{"content": "   \\n  "}\n')
    with pytest.raises(CorpusFormatError, match="empty or whitespace-only"):
        list(load_corpus(path))


def test_jsonl_non_object_row_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.jsonl", '"just a string"\n')
    with pytest.raises(CorpusFormatError, match="expected JSON object, got str"):
        list(load_corpus(path))


def test_jsonl_malformed_line_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.jsonl", '{"content": "ok"}\n{not json\n')
    with pytest.raises(CorpusFormatError, match=r"malformed JSON"):
        list(load_corpus(path))


def test_jsonl_empty_file_yields_nothing(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.jsonl", "")
    assert list(load_corpus(path)) == []


def test_jsonl_preserves_internal_whitespace(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.jsonl", json.dumps({"content": "  hello  world  "}) + "\n")
    assert list(load_corpus(path)) == ["  hello  world  "]


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------


def test_yaml_list_of_objects_yields_text_field(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "c.yaml",
        yaml.safe_dump([{"content": "alpha"}, {"content": "beta"}]),
    )
    assert list(load_corpus(path)) == ["alpha", "beta"]


def test_yaml_list_of_strings_yields_directly(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.yaml", yaml.safe_dump(["one", "two", "three"]))
    assert list(load_corpus(path)) == ["one", "two", "three"]


def test_yaml_yml_extension_works(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.yml", yaml.safe_dump(["a"]))
    assert list(load_corpus(path)) == ["a"]


def test_yaml_mixed_str_then_object_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.yaml", yaml.safe_dump(["x", {"content": "y"}]))
    with pytest.raises(CorpusFormatError, match="mixed list shape"):
        list(load_corpus(path))


def test_yaml_mixed_object_then_str_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.yaml", yaml.safe_dump([{"content": "x"}, "y"]))
    with pytest.raises(CorpusFormatError, match="mixed list shape"):
        list(load_corpus(path))


def test_yaml_object_missing_field_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.yaml", yaml.safe_dump([{"other": "x"}]))
    with pytest.raises(CorpusFormatError, match="missing field 'content'"):
        list(load_corpus(path))


def test_yaml_object_non_string_field_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.yaml", yaml.safe_dump([{"content": 42}]))
    with pytest.raises(CorpusFormatError, match="must be str"):
        list(load_corpus(path))


def test_yaml_empty_string_in_list_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.yaml", yaml.safe_dump(["valid", ""]))
    with pytest.raises(CorpusFormatError, match="empty or whitespace-only"):
        list(load_corpus(path))


def test_yaml_whitespace_string_in_list_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.yaml", yaml.safe_dump(["valid", "   "]))
    with pytest.raises(CorpusFormatError, match="empty or whitespace-only"):
        list(load_corpus(path))


def test_yaml_root_dict_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.yaml", yaml.safe_dump({"content": "x"}))
    with pytest.raises(CorpusFormatError, match="root must be a list"):
        list(load_corpus(path))


def test_yaml_empty_list_yields_nothing(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.yaml", "[]")
    assert list(load_corpus(path)) == []


def test_yaml_completely_empty_file_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.yaml", "")
    with pytest.raises(CorpusFormatError, match="empty or contains only YAML null"):
        list(load_corpus(path))


def test_yaml_first_row_unsupported_type_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.yaml", yaml.safe_dump([42, 43]))
    with pytest.raises(CorpusFormatError, match="expected str or object, got int"):
        list(load_corpus(path))


# ---------------------------------------------------------------------------
# Extension handling
# ---------------------------------------------------------------------------


def test_unknown_extension_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.txt", "hello\n")
    with pytest.raises(CorpusFormatError, match="Unsupported corpus extension"):
        list(load_corpus(path))


def test_single_document_json_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.json", '{"content": "x"}')
    with pytest.raises(CorpusFormatError, match="Unsupported corpus extension"):
        list(load_corpus(path))


def test_missing_file_raises_filenotfound(tmp_path: Path) -> None:
    path = tmp_path / "does_not_exist.jsonl"
    with pytest.raises(FileNotFoundError):
        list(load_corpus(path))


def test_blank_text_field_arg_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.jsonl", '{"content": "x"}\n')
    with pytest.raises(ValueError, match="text_field must be non-empty"):
        list(load_corpus(path, text_field=""))


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------


def test_synthetic_length_distribution_matches_histogram() -> None:
    histogram = {10: 3, 50: 5, 200: 2}
    out = list(synthetic_length_stratified(histogram, seed=7))
    lengths = Counter(len(s) for s in out)
    assert lengths == Counter({10: 3, 50: 5, 200: 2})


def test_synthetic_is_deterministic_for_same_seed() -> None:
    histogram = {10: 5, 32: 5}
    a = list(synthetic_length_stratified(histogram, seed=42))
    b = list(synthetic_length_stratified(histogram, seed=42))
    assert a == b


def test_synthetic_differs_across_seeds() -> None:
    histogram = {32: 100}
    a = list(synthetic_length_stratified(histogram, seed=1))
    b = list(synthetic_length_stratified(histogram, seed=2))
    assert a != b


def test_synthetic_empty_histogram_yields_nothing() -> None:
    assert list(synthetic_length_stratified({}, seed=0)) == []


def test_synthetic_zero_count_yields_nothing_for_that_length() -> None:
    out = list(synthetic_length_stratified({10: 3, 20: 0}, seed=0))
    assert all(len(s) == 10 for s in out)
    assert len(out) == 3


def test_synthetic_negative_length_raises() -> None:
    with pytest.raises(ValueError, match="length must be non-negative"):
        list(synthetic_length_stratified({-1: 1}, seed=0))


def test_synthetic_negative_count_raises() -> None:
    with pytest.raises(ValueError, match="count must be non-negative"):
        list(synthetic_length_stratified({10: -1}, seed=0))


def test_synthetic_zero_length_with_positive_count_raises() -> None:
    # Zero-length strings would be rejected by the loader; refuse here too.
    with pytest.raises(ValueError, match="length=0 produces empty strings"):
        list(synthetic_length_stratified({0: 1}, seed=0))


def test_synthetic_uses_only_alphabet_characters() -> None:
    out = list(synthetic_length_stratified({30: 10}, seed=0, alphabet="ABC"))
    for s in out:
        assert set(s) <= {"A", "B", "C"}


def test_synthetic_rejects_empty_alphabet() -> None:
    with pytest.raises(ValueError, match="alphabet must be non-empty"):
        list(synthetic_length_stratified({10: 1}, seed=0, alphabet=""))


# ---------------------------------------------------------------------------
# Corpus hashing
# ---------------------------------------------------------------------------


def test_corpus_hash_is_content_anchored() -> None:
    # Same content → same hash, regardless of source.
    a = compute_corpus_sha256(["hello", "world"])
    b = compute_corpus_sha256(iter(["hello", "world"]))
    assert a == b


def test_corpus_hash_differs_on_content_change() -> None:
    a = compute_corpus_sha256(["hello", "world"])
    b = compute_corpus_sha256(["hello", "earth"])
    assert a != b


def test_corpus_hash_uses_separator_to_disambiguate() -> None:
    # ["ab", "c"] vs ["a", "bc"] must produce different hashes; otherwise
    # different corpora would alias to the same identity.
    assert compute_corpus_sha256(["ab", "c"]) != compute_corpus_sha256(["a", "bc"])


def test_corpus_hash_is_unaliased_by_embedded_null_byte() -> None:
    # Regression: a NUL-only separator would alias ["a\0b"] to ["a", "b"].
    # Length-prefixing each row prevents that.
    assert compute_corpus_sha256(["a\0b"]) != compute_corpus_sha256(["a", "b"])


def test_corpus_hash_is_unaliased_by_embedded_length_lookalike() -> None:
    # A row whose bytes look like a length prefix followed by content must
    # not alias to two shorter rows. With 8-byte BE length prefixes:
    #   ["\0\0\0\0\0\0\0\1xY"]
    # would be: prefix=10, body=10 bytes
    # whereas the two-row form would be: prefix=1, body="x", prefix=1, body="Y"
    a = compute_corpus_sha256(["\x00\x00\x00\x00\x00\x00\x00\x01x" + "\x00\x00\x00\x00\x00\x00\x00\x01Y"])
    b = compute_corpus_sha256(["x", "Y"])
    assert a != b


def test_corpus_hash_deterministic_across_calls() -> None:
    rows = ["one", "two", "three"]
    assert compute_corpus_sha256(rows) == compute_corpus_sha256(rows)


def test_corpus_hash_returns_64_hex_chars() -> None:
    h = compute_corpus_sha256(["x"])
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_corpus_hash_empty_corpus_is_valid() -> None:
    h = compute_corpus_sha256([])
    assert len(h) == 64
    # Empty hash equals SHA-256 of empty bytes — well-defined.
    import hashlib as _hashlib

    assert h == _hashlib.sha256(b"").hexdigest()


def test_corpus_hash_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="must be str"):
        compute_corpus_sha256(["ok", 42])  # type: ignore[list-item]
