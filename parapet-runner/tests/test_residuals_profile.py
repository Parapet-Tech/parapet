from __future__ import annotations

import json
import shutil
from pathlib import Path

from parapet_runner.residuals.profile import build_profile, main


def _tmp(case: str) -> Path:
    path = Path("tests/.tmp_outputs") / "residuals_profile" / case
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def _row(content_hash: str, category: str, raw: float, squash: float, label: str = "malicious") -> dict:
    return {
        "content": f"content {content_hash}",
        "content_hash": content_hash,
        "residual_category": category,
        "error_type": category if category != "near_boundary_benign" else "correct",
        "label": label,
        "language": "EN",
        "source": "synthetic",
        "format_bin": "plain",
        "length_bin": "short",
        "raw_score": raw,
        "raw_unquoted_score": raw,
        "raw_squash_score": squash,
        "raw_score_delta": 0.0,
        "quote_detected": False,
        "l1_thresholds": {"l1": 0.0},
    }


def test_build_profile_writes_receipt_outputs_and_no_raw_content_in_report():
    out = _tmp("build")
    residuals_path = out / "residuals.jsonl"
    baseline_path = out / "baseline_correct.jsonl"
    manifest_path = out / "export_manifest.json"
    profile_dir = out / "profile"
    _write_jsonl(residuals_path, [
        _row("fn", "false_negative", -0.1, 0.9),
        _row("hn", "near_boundary_benign", -0.2, 0.8, label="benign"),
    ])
    _write_jsonl(baseline_path, [_row("base", "baseline_correct", -2.0, -1.0, label="benign")])
    manifest_path.write_text(json.dumps({"run": "phase-test"}), encoding="utf-8")

    result = build_profile(
        residuals_path=residuals_path,
        baseline_correct_path=baseline_path,
        export_manifest_path=manifest_path,
        output_dir=profile_dir,
        borderline_band=0.5,
    )

    paths = {name: Path(path) for name, path in result["paths"].items()}
    assert paths["feature_table"].is_file()
    assert paths["feature_profile_json"].is_file()
    assert paths["feature_profile_md"].is_file()
    assert paths["manifest"].is_file()

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["inputs"]["baseline_correct"]["rows"] == 1
    assert len(manifest["outputs"]["feature_table"]["sha256"]) == 64
    assert manifest["command"] == "python -m parapet_runner.residuals.profile"

    md = paths["feature_profile_md"].read_text(encoding="utf-8")
    assert "content fn" not in md
    assert "squash_when_raw_borderline" in md
    assert "Policy Interpretation" in md
    assert "does not justify default enforcement" in md
    assert "Borderline Window Sweep" in md
    assert "min FN value for break-even" in md
    assert "Squash-Borderline Detail" in md
    assert "Blocks by residual category" in md
    assert "FN recovery by reason" in md


def test_main_returns_zero_and_writes_outputs():
    out = _tmp("main")
    residuals_path = out / "residuals.jsonl"
    baseline_path = out / "baseline_correct.jsonl"
    manifest_path = out / "export_manifest.json"
    profile_dir = out / "profile"
    _write_jsonl(residuals_path, [_row("fn", "false_negative", -0.1, 0.9)])
    _write_jsonl(baseline_path, [_row("base", "baseline_correct", -2.0, -1.0, label="benign")])
    manifest_path.write_text(json.dumps({"run": "phase-test"}), encoding="utf-8")

    rc = main([
        "--residuals", str(residuals_path),
        "--baseline-correct", str(baseline_path),
        "--manifest", str(manifest_path),
        "--output-dir", str(profile_dir),
    ])

    assert rc == 0
    assert (profile_dir / "feature_profile.json").is_file()
