from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from parapet_runner.residuals.io import (
    input_receipts,
    load_inputs,
    load_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


def _tmp(case: str) -> Path:
    path = Path("tests/.tmp_outputs") / "residuals_io" / case
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_write_and_load_jsonl_round_trips_rows():
    out = _tmp("jsonl_round_trip")
    path = out / "rows.jsonl"
    rows = [{"b": 2, "a": 1}, {"x": "y"}]

    count = write_jsonl(path, rows)

    assert count == 2
    assert load_jsonl(path) == rows
    assert sha256_file(path) == sha256_file(path)


def test_load_jsonl_rejects_non_object_rows():
    out = _tmp("jsonl_non_object")
    path = out / "bad.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected JSON object"):
        load_jsonl(path)


def test_load_inputs_requires_baseline_sidecar_and_manifest():
    out = _tmp("bundle")
    residuals = out / "residuals.jsonl"
    baseline = out / "baseline_correct.jsonl"
    manifest = out / "manifest.json"
    write_jsonl(residuals, [{"content_hash": "a"}])
    write_jsonl(baseline, [{"content_hash": "b"}])
    write_json(manifest, {"run": "test"})

    bundle = load_inputs(residuals, baseline, manifest)
    receipts = input_receipts(bundle)

    assert len(bundle.residuals) == 1
    assert len(bundle.baseline_correct) == 1
    assert bundle.export_manifest == {"run": "test"}
    assert receipts["residuals"]["rows"] == 1
    assert receipts["baseline_correct"]["rows"] == 1
    assert len(receipts["export_manifest"]["sha256"]) == 64


def test_load_inputs_missing_sidecar_is_error():
    out = _tmp("missing_sidecar")
    residuals = out / "residuals.jsonl"
    baseline = out / "baseline_correct.jsonl"
    manifest = out / "manifest.json"
    residuals.write_text(json.dumps({"content_hash": "a"}) + "\n", encoding="utf-8")
    write_json(manifest, {"run": "test"})

    with pytest.raises(FileNotFoundError):
        load_inputs(residuals, baseline, manifest)
