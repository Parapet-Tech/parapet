from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from parapet_data.staged_artifact import (
    iter_staged_rows,
    load_staged_rows,
)


@pytest.fixture()
def tmp_dir() -> Path:
    root = Path(__file__).resolve().parent / ".tmp_temp"
    root.mkdir(exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="staged_artifact_", dir=root))
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _write_yaml(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(rows, sort_keys=False), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row, ensure_ascii=False)}\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_iter_staged_rows_from_yaml(tmp_dir: Path) -> None:
    path = _write_yaml(
        tmp_dir / "en_x_attacks_staged.yaml",
        [{"content": "a", "label": "malicious"}, {"content": "b", "label": "benign"}],
    )

    rows = list(iter_staged_rows(path))

    assert [row["content"] for row in rows] == ["a", "b"]


def test_iter_staged_rows_from_jsonl(tmp_dir: Path) -> None:
    path = _write_jsonl(
        tmp_dir / "en_x_attacks_staged.jsonl",
        [{"content": "a"}, {"content": "b"}, {"content": "c"}],
    )

    rows = list(iter_staged_rows(path))

    assert [row["content"] for row in rows] == ["a", "b", "c"]


def test_iter_staged_rows_handles_empty_yaml(tmp_dir: Path) -> None:
    path = _write_yaml(tmp_dir / "empty.yaml", [])

    rows = list(iter_staged_rows(path))

    assert rows == []


def test_iter_staged_rows_skips_blank_jsonl_lines(tmp_dir: Path) -> None:
    path = tmp_dir / "with_blanks.jsonl"
    path.write_text(
        '{"content": "a"}\n\n{"content": "b"}\n   \n',
        encoding="utf-8",
    )

    rows = list(iter_staged_rows(path))

    assert [row["content"] for row in rows] == ["a", "b"]


def test_iter_staged_rows_rejects_non_list_yaml(tmp_dir: Path) -> None:
    path = tmp_dir / "scalar.yaml"
    path.write_text("content: nope\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected top-level YAML list"):
        list(iter_staged_rows(path))


def test_iter_staged_rows_rejects_non_mapping_yaml_row(tmp_dir: Path) -> None:
    path = _write_yaml(tmp_dir / "bad_row.yaml", ["not a dict", {"content": "ok"}])

    with pytest.raises(ValueError, match="expected mapping row"):
        list(iter_staged_rows(path))


def test_iter_staged_rows_rejects_non_mapping_jsonl_row(tmp_dir: Path) -> None:
    path = tmp_dir / "bad_row.jsonl"
    path.write_text('["not", "a", "mapping"]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="expected JSON object row"):
        list(iter_staged_rows(path))


def test_iter_staged_rows_rejects_unsupported_suffix(tmp_dir: Path) -> None:
    path = tmp_dir / "data.csv"
    path.write_text("content\nfoo\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported staged artifact suffix"):
        list(iter_staged_rows(path))


def test_load_staged_rows_returns_full_list(tmp_dir: Path) -> None:
    path = _write_yaml(
        tmp_dir / "x.yaml",
        [{"content": "a"}, {"content": "b"}],
    )

    rows = load_staged_rows(path)

    assert isinstance(rows, list)
    assert [r["content"] for r in rows] == ["a", "b"]
