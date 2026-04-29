from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from parapet_runner.l2_corpus import (
    CorpusBuildManifest,
    StratifySpec,
    _bucket_label,
    _stratified_sample,
    build_l2_latency_corpus,
)


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Path:
    name = request.node.name
    output_dir = Path("tests/.tmp_outputs/l2_corpus") / name
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# StratifySpec validation
# ---------------------------------------------------------------------------


def test_spec_rejects_zero_target_rows() -> None:
    with pytest.raises(ValueError, match="target_rows must be >= 1"):
        StratifySpec(target_rows=0, seed=0)


def test_spec_rejects_unknown_axis() -> None:
    with pytest.raises(ValueError, match="Unknown stratification axis"):
        StratifySpec(target_rows=10, seed=0, axes=("unknown_axis",))


def test_spec_rejects_empty_axes() -> None:
    with pytest.raises(ValueError, match="axes must be non-empty"):
        StratifySpec(target_rows=10, seed=0, axes=())


def test_spec_rejects_zero_length_bucket_edge() -> None:
    with pytest.raises(ValueError, match="must all be positive"):
        StratifySpec(target_rows=10, seed=0, length_buckets=(0, 128))


def test_spec_rejects_blank_text_field() -> None:
    with pytest.raises(ValueError, match="text_field must be non-empty"):
        StratifySpec(target_rows=10, seed=0, text_field="   ")


def test_spec_accepts_default_axes_and_buckets() -> None:
    spec = StratifySpec(target_rows=10, seed=0)
    assert spec.axes == ("language", "length_bucket")
    assert spec.length_buckets == (128, 512, 2048)
    assert spec.text_field == "content"


# ---------------------------------------------------------------------------
# _bucket_label
# ---------------------------------------------------------------------------


def test_bucket_label_partitions_by_edges() -> None:
    edges = (128, 512, 2048)
    assert _bucket_label(0, edges) == "0-128"
    assert _bucket_label(128, edges) == "0-128"
    assert _bucket_label(129, edges) == "129-512"
    assert _bucket_label(512, edges) == "129-512"
    assert _bucket_label(2048, edges) == "513-2048"
    assert _bucket_label(2049, edges) == "2049+"
    assert _bucket_label(50_000, edges) == "2049+"


def test_bucket_label_sorts_unsorted_edges() -> None:
    assert _bucket_label(100, (512, 128)) == "0-128"
    assert _bucket_label(200, (512, 128)) == "129-512"


# ---------------------------------------------------------------------------
# _stratified_sample
# ---------------------------------------------------------------------------


import random


def _make_rows(distribution: dict[tuple[str, str], int]) -> list[dict[str, Any]]:
    """Build rows from a {(language, label): count} spec, with content sized
    to land deterministically in the 0-128 bucket."""

    rows: list[dict[str, Any]] = []
    for (lang, label), count in distribution.items():
        for i in range(count):
            rows.append(
                {
                    "content": f"{lang}-{label}-{i}".ljust(50, "x"),
                    "language": lang,
                    "label": label,
                    "reason": "uncategorized",
                    "source": f"{lang}_corpus",
                }
            )
    return rows


def test_stratified_sample_preserves_proportions_within_rounding() -> None:
    rows = _make_rows({("EN", "benign"): 600, ("RU", "benign"): 300, ("AR", "benign"): 100})
    spec = StratifySpec(
        target_rows=100, seed=42, axes=("language",), length_buckets=(128, 512, 2048)
    )
    sampled = _stratified_sample(rows, spec, random.Random(42))
    counts = Counter(r["language"] for r in sampled)
    # Largest-remainder allocation for proportions 0.6/0.3/0.1 of 100.
    assert counts["EN"] == 60
    assert counts["RU"] == 30
    assert counts["AR"] == 10


def test_stratified_sample_caps_at_cell_capacity() -> None:
    # Cell only has 5 rows; even if proportional allocation says 10, cap.
    rows = _make_rows({("EN", "benign"): 95, ("AR", "benign"): 5})
    spec = StratifySpec(target_rows=100, seed=42, axes=("language",))
    sampled = _stratified_sample(rows, spec, random.Random(42))
    # Should still be 100 since EN has plenty of headroom.
    assert len(sampled) == 100
    counts = Counter(r["language"] for r in sampled)
    assert counts["AR"] == 5  # capped — only 5 available
    assert counts["EN"] == 95


def test_stratified_sample_is_deterministic_for_same_seed() -> None:
    rows = _make_rows({("EN", "benign"): 100, ("RU", "benign"): 100})
    spec = StratifySpec(target_rows=20, seed=7, axes=("language",))

    a = _stratified_sample(rows, spec, random.Random(spec.seed))
    b = _stratified_sample(rows, spec, random.Random(spec.seed))
    # Order should match exactly (rng.shuffle is also seeded).
    assert [r["content"] for r in a] == [r["content"] for r in b]


def test_stratified_sample_differs_across_seeds() -> None:
    rows = _make_rows({("EN", "benign"): 100, ("RU", "benign"): 100})
    spec_a = StratifySpec(target_rows=20, seed=1, axes=("language",))
    spec_b = StratifySpec(target_rows=20, seed=2, axes=("language",))

    a = _stratified_sample(rows, spec_a, random.Random(spec_a.seed))
    b = _stratified_sample(rows, spec_b, random.Random(spec_b.seed))
    # Different seeds → different specific rows (counts the same per cell).
    assert [r["content"] for r in a] != [r["content"] for r in b]


# ---------------------------------------------------------------------------
# build_l2_latency_corpus end-to-end
# ---------------------------------------------------------------------------


def _v8_shaped_yaml(rows: list[dict[str, Any]]) -> str:
    return yaml.safe_dump(rows, allow_unicode=True, sort_keys=False)


def test_build_corpus_writes_jsonl_and_manifest(tmp_path: Path) -> None:
    rows = [
        {
            "content": f"text-{i}-" + ("x" * (i * 50)),
            "label": "benign" if i % 2 == 0 else "malicious",
            "language": ["EN", "RU", "AR", "ZH"][i % 4],
            "reason": "uncategorized",
            "source": "test_source",
        }
        for i in range(40)
    ]
    src = tmp_path / "v8_train.yaml"
    src.write_text(_v8_shaped_yaml(rows), encoding="utf-8")

    output = tmp_path / "out.jsonl"
    spec = StratifySpec(target_rows=20, seed=42)
    manifest = build_l2_latency_corpus([src], spec, output)

    # Output file: strict JSONL, only content.
    assert output.exists()
    output_lines = output.read_text(encoding="utf-8").splitlines()
    assert len(output_lines) == 20
    for line in output_lines:
        obj = json.loads(line)
        assert set(obj.keys()) == {"content"}
        assert isinstance(obj["content"], str)

    # Manifest sidecar.
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    assert manifest_path.exists()
    parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert parsed["n_output_rows"] == 20
    assert parsed["n_input_rows"] == 40
    assert parsed["seed"] == 42
    assert parsed["text_field"] == "content"
    assert parsed["stratify_axes"] == ["language", "length_bucket"]
    assert "language" in parsed["axis_distribution_in"]
    assert "label" in parsed["axis_distribution_in"]
    assert "reason" in parsed["axis_distribution_in"]
    assert "source" in parsed["axis_distribution_in"]
    assert "length_bucket" in parsed["axis_distribution_in"]
    assert len(parsed["source_paths"]) == 1
    assert len(parsed["source_sha256s"]) == 1
    assert len(parsed["source_sha256s"][0]) == 64

    # Output SHA matches what the bench would compute on the same content.
    from parapet_runner.latency_corpus import compute_corpus_sha256

    contents = [json.loads(line)["content"] for line in output_lines]
    assert manifest.output_sha256 == compute_corpus_sha256(contents)


def test_build_corpus_combines_multiple_sources(tmp_path: Path) -> None:
    train = [
        {"content": f"train-{i}-xxx" * 5, "language": "EN", "label": "benign", "reason": "uncategorized", "source": "s1"}
        for i in range(50)
    ]
    val = [
        {"content": f"val-{i}-xxx" * 5, "language": "RU", "label": "malicious", "reason": "uncategorized", "source": "s2"}
        for i in range(20)
    ]
    train_path = tmp_path / "train.yaml"
    val_path = tmp_path / "val.yaml"
    train_path.write_text(_v8_shaped_yaml(train), encoding="utf-8")
    val_path.write_text(_v8_shaped_yaml(val), encoding="utf-8")

    output = tmp_path / "combined.jsonl"
    spec = StratifySpec(target_rows=35, seed=1)
    manifest = build_l2_latency_corpus([train_path, val_path], spec, output)

    assert manifest.n_input_rows == 70
    assert manifest.n_output_rows == 35
    assert len(manifest.source_paths) == 2
    assert len(manifest.source_sha256s) == 2


def test_build_corpus_drops_rows_missing_content(tmp_path: Path) -> None:
    rows = [
        {"content": "ok-" + "x" * 30, "language": "EN", "label": "benign"},
        {"language": "RU", "label": "malicious"},  # missing content
        {"content": "", "language": "AR", "label": "benign"},  # empty content
        {"content": "   ", "language": "ZH", "label": "benign"},  # whitespace
        {"content": "fine-" + "x" * 30, "language": "EN", "label": "malicious"},
    ]
    src = tmp_path / "c.yaml"
    src.write_text(_v8_shaped_yaml(rows), encoding="utf-8")

    output = tmp_path / "out.jsonl"
    spec = StratifySpec(target_rows=2, seed=0)
    manifest = build_l2_latency_corpus([src], spec, output)

    assert manifest.n_input_rows == 2  # only the two valid rows


def test_build_corpus_raises_on_no_valid_rows(tmp_path: Path) -> None:
    rows = [{"content": ""}, {"content": "   "}]
    src = tmp_path / "c.yaml"
    src.write_text(_v8_shaped_yaml(rows), encoding="utf-8")

    output = tmp_path / "out.jsonl"
    spec = StratifySpec(target_rows=1, seed=0)
    with pytest.raises(Exception, match="No valid rows"):
        build_l2_latency_corpus([src], spec, output)


def test_build_corpus_raises_on_empty_source_list(tmp_path: Path) -> None:
    output = tmp_path / "out.jsonl"
    spec = StratifySpec(target_rows=1, seed=0)
    with pytest.raises(ValueError, match="source_paths must be non-empty"):
        build_l2_latency_corpus([], spec, output)


def test_build_corpus_handles_multilingual_content(tmp_path: Path) -> None:
    rows = [
        {"content": "Hello world " * 5, "language": "EN", "label": "benign", "reason": "x", "source": "s"},
        {"content": "Привет мир " * 5, "language": "RU", "label": "benign", "reason": "x", "source": "s"},
        {"content": "مرحبا بالعالم " * 5, "language": "AR", "label": "benign", "reason": "x", "source": "s"},
        {"content": "你好世界 " * 5, "language": "ZH", "label": "benign", "reason": "x", "source": "s"},
    ] * 10
    src = tmp_path / "ml.yaml"
    src.write_text(_v8_shaped_yaml(rows), encoding="utf-8")

    output = tmp_path / "ml.jsonl"
    spec = StratifySpec(target_rows=20, seed=42)
    manifest = build_l2_latency_corpus([src], spec, output)

    # All four languages should be represented (proportional to input).
    assert set(manifest.axis_distribution_out["language"].keys()) == {"EN", "RU", "AR", "ZH"}
    # Check Unicode round-trips correctly.
    output_text = output.read_text(encoding="utf-8")
    assert "你好世界" in output_text
    assert "Привет" in output_text
    assert "مرحبا" in output_text


def test_build_corpus_jsonl_uses_ensure_ascii_false(tmp_path: Path) -> None:
    rows = [{"content": "你好" * 50, "language": "ZH", "label": "benign", "reason": "x", "source": "s"}]
    src = tmp_path / "c.yaml"
    src.write_text(_v8_shaped_yaml(rows), encoding="utf-8")

    output = tmp_path / "out.jsonl"
    spec = StratifySpec(target_rows=1, seed=0)
    build_l2_latency_corpus([src], spec, output)

    raw = output.read_text(encoding="utf-8")
    # Unicode kept as Unicode, not escaped to \uXXXX.
    assert "你好" in raw
    assert "\\u" not in raw


def test_build_corpus_manifest_axis_counts_match_total(tmp_path: Path) -> None:
    rows = _make_rows({("EN", "benign"): 30, ("RU", "malicious"): 20, ("AR", "benign"): 10})
    src = tmp_path / "c.yaml"
    src.write_text(_v8_shaped_yaml(rows), encoding="utf-8")

    output = tmp_path / "out.jsonl"
    spec = StratifySpec(target_rows=30, seed=0)
    manifest = build_l2_latency_corpus([src], spec, output)

    # Per-axis distributions should each total to n_output_rows.
    for axis, dist in manifest.axis_distribution_out.items():
        assert sum(dist.values()) == manifest.n_output_rows, f"axis {axis}"
