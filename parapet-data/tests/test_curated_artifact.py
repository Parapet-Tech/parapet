from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from parapet_data.curated_artifact import (
    iter_curated_entries,
    load_curated_entries,
    resolve_curated_path,
)


@pytest.fixture()
def tmp_dir() -> Path:
    root = Path(__file__).resolve().parent / ".tmp_temp"
    root.mkdir(exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="curated_artifact_", dir=root))
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _write_yaml(path: Path, rows: list[dict]) -> Path:
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


def test_resolve_curated_path_prefers_jsonl(tmp_dir: Path) -> None:
    artifact_dir = tmp_dir / "artifact"
    _write_yaml(artifact_dir / "curated.yaml", [{"content": "yaml"}])
    jsonl_path = _write_jsonl(artifact_dir / "curated.jsonl", [{"content": "jsonl"}])

    assert resolve_curated_path(artifact_dir) == jsonl_path


def test_load_curated_entries_from_yaml_file(tmp_dir: Path) -> None:
    path = _write_yaml(tmp_dir / "curated.yaml", [{"content": "a"}, {"content": "b"}])

    rows = load_curated_entries(path)

    assert [row["content"] for row in rows] == ["a", "b"]


def test_iter_curated_entries_from_jsonl_directory(tmp_dir: Path) -> None:
    artifact_dir = tmp_dir / "artifact"
    _write_jsonl(artifact_dir / "curated.jsonl", [{"content": "a"}, {"content": "b"}])

    rows = list(iter_curated_entries(artifact_dir))

    assert [row["content"] for row in rows] == ["a", "b"]


def test_load_curated_entries_rejects_non_list_yaml(tmp_dir: Path) -> None:
    path = tmp_dir / "curated.yaml"
    path.write_text("content: nope\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected top-level YAML list"):
        load_curated_entries(path)


def test_iter_curated_entries_rejects_non_mapping_jsonl_row(tmp_dir: Path) -> None:
    path = tmp_dir / "curated.jsonl"
    path.write_text('["not","a","mapping"]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="expected JSON object row"):
        list(iter_curated_entries(path))


def test_resolve_curated_path_rejects_missing_artifact(tmp_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Could not find curated artifact"):
        resolve_curated_path(tmp_dir / "missing")
