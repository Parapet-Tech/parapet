"""Tests for parapet-runner/scripts/phase1_v8_residual_export.py.

Coverage:
  * eval.json -> curated/holdout.yaml join by case_id (no row data sourced from
    eval.json beyond verdict/scores).
  * fold_id derived from fold_*/ dir names; ignores invalid preview-style
    fold_assignments.json.
  * dataset/holdout.yaml vs curated/holdout.yaml content-equality guard.
  * reason normalization (canonical preserved, non-canonical -> uncategorized).
  * Partition logic uses raw_score for near-boundary, not l1_score/score.
  * Leakage check exits non-zero unless --allow-leakage and reports overlap.
  * Cross-checks total joined rows against summary.json + all_predictions.jsonl.
  * Manifest carries fold thresholds, sha256s, row counts, script args.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts" / "phase1_v8_residual_export.py"
    )
    spec = importlib.util.spec_from_file_location("phase1_v8_residual_export", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _new_output_dir(case_name: str) -> Path:
    """Per-test scratch dir. Pyproject disables pytest's tmpdir plugin."""
    output_dir = Path("tests/.tmp_outputs") / "phase1_v8" / case_name
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------


_REASONS = [
    "instruction_override", "roleplay_jailbreak", "meta_probe",
    "exfiltration", "adversarial_suffix", "indirect_injection",
    "obfuscation", "constraint_bypass",
]


def _build_fold(
    fold_dir: Path,
    *,
    rows: list[dict],
    threshold: float,
) -> None:
    """Lay out a single fold's on-disk artifacts the way the runner does.

    rows: list of dicts with keys content, label, language, reason, source,
          format_bin, length_bin, expected (blocked|allowed),
          actual (blocked|allowed), score (float), raw_score (float).
    """
    # The runner names this dir _eval_holdout_<calibrated>; the script
    # discovers any _eval_holdout_*. We pick a representative name per fold.
    eval_dir_name = "_eval_holdout_0p0" if threshold == 0.0 else "_eval_holdout_m0p5"

    eval_dir = fold_dir / "run" / eval_dir_name
    dataset_dir = eval_dir / "dataset"
    curated_dir = fold_dir / "curated"
    eval_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    curated_dir.mkdir(parents=True, exist_ok=True)

    # curated/holdout.yaml — rich §0.1 metadata
    curated_rows = [
        {
            "content": r["content"],
            "label": r["label"],
            "language": r["language"],
            "reason": r["reason"],
            "source": r["source"],
            "format_bin": r["format_bin"],
            "length_bin": r["length_bin"],
        }
        for r in rows
    ]
    curated_dir.joinpath("holdout.yaml").write_text(
        yaml.safe_dump(curated_rows, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    # dataset/holdout.yaml — harness-flattened, MUST be row-aligned to curated.
    dataset_rows = [
        {
            "id": f"holdout_{i+1}",
            "layer": "l1",
            "label": r["label"],
            "description": f"holdout sample {i+1}",
            "content": r["content"],
        }
        for i, r in enumerate(rows)
    ]
    dataset_dir.joinpath("holdout.yaml").write_text(
        yaml.safe_dump(dataset_rows, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    # eval.json — verdicts + l1_signals
    results = []
    for i, r in enumerate(rows):
        results.append({
            "case_id": f"holdout_{i+1}",
            "layer": "l1",
            "label": r["label"],
            "source": "holdout",
            "expected": r["expected"],
            "actual": r["actual"],
            "correct": r["expected"] == r["actual"],
            "detail": "status=200",
            "duration_ms": 1.0,
            "l1_signals": [{
                "message_index": 0,
                "role": "User",
                "score": r["score"],
                "raw_score": r["raw_score"],
                "raw_unquoted_score": r.get("raw_unquoted_score", r["raw_score"]),
                "raw_squash_score": r.get("raw_squash_score", r["raw_score"]),
                "unquoted_score": r.get("unquoted_score", r["score"]),
                "squash_score": r.get("squash_score", r["score"]),
                "quote_detected": False,
                "raw_score_delta": 0.0,
            }],
        })
    eval_dir.joinpath("eval.json").write_text(
        json.dumps({"results": results, "layers": [], "sources": [], "evidence": {}}),
        encoding="utf-8",
    )

    # eval_config_l1_threshold.yaml
    eval_dir.joinpath("eval_config_l1_threshold.yaml").write_text(
        yaml.safe_dump({"layers": {"L1": {"mode": "block", "threshold": float(threshold)}}}),
        encoding="utf-8",
    )


def _make_row(
    *, content: str, label: str, expected: str, actual: str,
    score: float, raw_score: float,
    language: str = "EN", reason: str = "instruction_override",
    source: str = "src_a", format_bin: str = "prose", length_bin: str = "short",
) -> dict:
    return {
        "content": content, "label": label, "expected": expected, "actual": actual,
        "score": score, "raw_score": raw_score,
        "language": language, "reason": reason, "source": source,
        "format_bin": format_bin, "length_bin": length_bin,
    }


def _build_run_dir(tmp_path: Path) -> tuple[Path, list[list[dict]]]:
    """Build a 2-fold synthetic source-run-dir. Returns (run_dir, per_fold_rows)."""
    run_dir = tmp_path / "src_run"
    run_dir.mkdir()

    fold0_rows = [
        _make_row(content="benign A", label="benign", expected="allowed", actual="allowed",
                  score=0.1, raw_score=0.2, source="src_a"),
        _make_row(content="attack B", label="malicious", expected="blocked", actual="allowed",
                  score=0.4, raw_score=-0.3, reason="meta_probe", source="src_b"),
        _make_row(content="benign C near", label="benign", expected="allowed", actual="allowed",
                  score=0.5, raw_score=0.4, language="RU", source="src_c"),
        _make_row(content="benign D fp", label="benign", expected="allowed", actual="blocked",
                  score=0.7, raw_score=2.5, language="ZH", source="src_d"),
        _make_row(content="benign E correct", label="benign", expected="allowed", actual="allowed",
                  score=0.0, raw_score=-3.0, language="AR", source="src_e"),
    ]
    fold1_rows = [
        _make_row(content="attack F", label="malicious", expected="blocked", actual="blocked",
                  score=0.9, raw_score=2.0, reason="exfiltration", source="src_f"),
        _make_row(content="attack G FN", label="malicious", expected="blocked", actual="allowed",
                  score=0.3, raw_score=-0.5, reason="obfuscation", source="src_g"),
    ]

    _build_fold(run_dir / "fold_0", rows=fold0_rows, threshold=0.0)
    _build_fold(run_dir / "fold_1", rows=fold1_rows, threshold=-0.5)

    # summary.json with the expected total — used for cross-check
    (run_dir / "summary.json").write_text(
        json.dumps({"total_predictions": len(fold0_rows) + len(fold1_rows)}),
        encoding="utf-8",
    )

    # Match the bug we saw in production: 10-row preview with a trailing
    # "... (N total)" suffix that breaks JSON parsing.
    (run_dir / "fold_assignments.json").write_text(
        '[{"index": 0, "fold": 0}]\n... (7 total)\n',
        encoding="utf-8",
    )

    # all_predictions.jsonl with matching row count
    with (run_dir / "all_predictions.jsonl").open("w", encoding="utf-8") as f:
        for i in range(len(fold0_rows) + len(fold1_rows)):
            f.write(json.dumps({"row": i}) + "\n")

    return run_dir, [fold0_rows, fold1_rows]


def _build_curation_dir(tmp_path: Path, *, train_contents: list[str]) -> Path:
    cur = tmp_path / "curated_v8"
    cur.mkdir()
    rows = [{"content": c, "label": "benign", "language": "EN"} for c in train_contents]
    (cur / "train.yaml").write_text(yaml.safe_dump(rows, allow_unicode=True), encoding="utf-8")
    (cur / "manifest.json").write_text(json.dumps({"spec_name": "test"}), encoding="utf-8")
    return cur


# ---------------------------------------------------------------------------
# Tests: pure functions
# ---------------------------------------------------------------------------


def test_normalize_reason_canonical_preserved():
    m = _load_module()
    for r in _REASONS + ["uncategorized"]:
        out, was_norm = m.normalize_reason(r)
        assert out == r
        assert was_norm is False


def test_normalize_reason_missing_or_noncanonical_becomes_uncategorized():
    m = _load_module()
    for bad in [None, "", "weird_thing", "Instruction_Override", 42, []]:
        out, was_norm = m.normalize_reason(bad)
        assert out == "uncategorized"
        assert was_norm is True


def test_to_phase1_row_uses_score_as_l1_score_and_drops_pred_label():
    m = _load_module()
    joined = {
        "content": "x", "label": "benign", "language": "EN",
        "reason_raw": "meta_probe", "source": "s", "format_bin": "prose",
        "length_bin": "short", "pred_label": "benign", "correct": True,
        "error_type": "correct",
        "score": 0.42, "raw_score": -0.1, "raw_unquoted_score": -0.1,
        "raw_squash_score": -0.1, "unquoted_score": 0.42, "squash_score": 0.42,
        "raw_score_delta": 0.0, "quote_detected": False,
        "fold_id": 3, "applied_threshold": 0.0,
    }
    out = m.to_phase1_row(joined)
    # §0.1 training-input fields
    for k in ("content", "label", "language", "reason", "source", "content_hash"):
        assert k in out
    # l1_score is the post-pipeline score, NOT raw_score
    assert out["l1_score"] == 0.42
    # raw_score retained as diagnostic
    assert out["raw_score"] == -0.1
    # decision derived from pred_label
    assert out["l1_decision"] == "allow"
    # threshold metadata structured
    assert out["l1_thresholds"] == {"l1": 0.0}
    # routed_reason reserved
    assert out["routed_reason"] is None
    # content_hash matches contract
    assert out["content_hash"] == m.content_hash("x")


def test_to_phase1_row_block_decision():
    m = _load_module()
    joined = {
        "content": "y", "label": "malicious", "language": "EN",
        "reason_raw": "meta_probe", "source": "s",
        "pred_label": "malicious", "score": 0.8, "raw_score": 1.0,
        "fold_id": 0, "applied_threshold": 0.0,
        "format_bin": "prose", "length_bin": "short",
        "error_type": "correct", "raw_unquoted_score": None,
        "raw_squash_score": None, "raw_score_delta": None,
        "unquoted_score": None, "squash_score": None, "quote_detected": None,
    }
    out = m.to_phase1_row(joined)
    assert out["l1_decision"] == "block"


def test_partition_uses_raw_score_for_near_boundary_not_score():
    m = _load_module()
    # benign correct with raw_score=0.5 (within band) but score=0.99 (outside band):
    # near_boundary should still pick it up because the selector is raw_score.
    rows = [{
        "content": "p", "label": "benign", "error_type": "correct",
        "raw_score": 0.5, "score": 0.99,
    }]
    buckets = m.partition_rows(
        rows, margin_band_raw_score=1.0, baseline_sample_rate=0.0, seed=0,
    )
    assert len(buckets["near_boundary_benign"]) == 1
    assert len(buckets["baseline_correct"]) == 0


def test_partition_baseline_correct_deterministic_with_seed():
    m = _load_module()
    rows = [
        {"content": f"r{i}", "label": "benign", "error_type": "correct",
         "raw_score": 5.0, "score": 0.99}
        for i in range(100)
    ]
    out_a = m.partition_rows(rows, margin_band_raw_score=1.0,
                             baseline_sample_rate=0.5, seed=42)
    out_b = m.partition_rows(rows, margin_band_raw_score=1.0,
                             baseline_sample_rate=0.5, seed=42)
    out_c = m.partition_rows(rows, margin_band_raw_score=1.0,
                             baseline_sample_rate=0.5, seed=99)
    a_ids = [r["content"] for r in out_a["baseline_correct"]]
    b_ids = [r["content"] for r in out_b["baseline_correct"]]
    c_ids = [r["content"] for r in out_c["baseline_correct"]]
    assert a_ids == b_ids
    assert a_ids != c_ids


# ---------------------------------------------------------------------------
# Tests: per-fold join + integrity
# ---------------------------------------------------------------------------


def test_join_fold_pulls_metadata_from_curated_not_eval():
    tmp_path = _new_output_dir("join_fold_pulls_metadata")
    m = _load_module()
    rows_in = [_make_row(
        content="hello world", label="benign", expected="allowed", actual="allowed",
        score=0.1, raw_score=-0.2, language="ZH", reason="obfuscation",
        source="zh_src", format_bin="code", length_bin="long",
    )]
    fold_dir = tmp_path / "fold_0"
    _build_fold(fold_dir, rows=rows_in, threshold=0.0)

    joined, threshold, sources = m.join_fold(0, fold_dir)
    assert threshold == 0.0
    assert len(joined) == 1
    j = joined[0]
    # Source is curated/holdout.yaml — language/reason/source/format_bin/length_bin
    # would be missing if the script wrongly read dataset/holdout.yaml instead.
    assert j["language"] == "ZH"
    assert j["reason_raw"] == "obfuscation"
    assert j["source"] == "zh_src"
    assert j["format_bin"] == "code"
    assert j["length_bin"] == "long"
    # Verdict + scores from eval.json
    assert j["score"] == 0.1
    assert j["raw_score"] == -0.2
    assert j["pred_label"] == "benign"
    assert j["fold_id"] == 0
    assert j["applied_threshold"] == 0.0
    # Sources dict for manifest
    assert sources["eval_json"].name == "eval.json"
    assert sources["dataset_holdout"].parent.name == "dataset"
    assert sources["curated_holdout"].parent.name == "curated"


def test_join_fold_rejects_dataset_curated_content_skew():
    tmp_path = _new_output_dir("join_fold_rejects_skew")
    m = _load_module()
    rows_in = [
        _make_row(content="A", label="benign", expected="allowed", actual="allowed",
                  score=0, raw_score=0),
        _make_row(content="B", label="benign", expected="allowed", actual="allowed",
                  score=0, raw_score=0),
    ]
    fold_dir = tmp_path / "fold_0"
    _build_fold(fold_dir, rows=rows_in, threshold=0.0)
    # Corrupt curated/holdout.yaml so row 1's content disagrees with dataset/.
    curated_path = fold_dir / "curated" / "holdout.yaml"
    curated = yaml.safe_load(curated_path.read_text(encoding="utf-8"))
    curated[1]["content"] = "DIFFERENT"
    curated_path.write_text(yaml.safe_dump(curated, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset/holdout.yaml content"):
        m.join_fold(0, fold_dir)


def test_load_eval_results_rejects_out_of_bounds_case_id():
    tmp_path = _new_output_dir("eval_oob_case_id")
    m = _load_module()
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps({"results": [
        {"case_id": "holdout_99", "expected": "allowed", "actual": "allowed",
         "correct": True, "l1_signals": []}
    ]}), encoding="utf-8")
    with pytest.raises(ValueError):
        m.load_eval_results_by_index(eval_path, expected_n=1)


def test_load_eval_results_rejects_duplicate_case_id():
    tmp_path = _new_output_dir("eval_dup_case_id")
    m = _load_module()
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps({"results": [
        {"case_id": "holdout_1", "expected": "allowed", "actual": "allowed",
         "correct": True, "l1_signals": []},
        {"case_id": "holdout_1", "expected": "allowed", "actual": "allowed",
         "correct": True, "l1_signals": []},
    ]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        m.load_eval_results_by_index(eval_path, expected_n=2)


def test_discover_fold_dirs_sorted_numerically():
    tmp_path = _new_output_dir("discover_fold_dirs")
    m = _load_module()
    for n in [10, 0, 2, 1]:
        (tmp_path / f"fold_{n}").mkdir()
    (tmp_path / "not_a_fold").mkdir()
    found = m.discover_fold_dirs(tmp_path)
    assert [fid for fid, _ in found] == [0, 1, 2, 10]


# ---------------------------------------------------------------------------
# Tests: leakage check
# ---------------------------------------------------------------------------


def test_leakage_check_reports_overlap():
    tmp_path = _new_output_dir("leakage_overlap")
    m = _load_module()
    train = tmp_path / "train.yaml"
    holdout = tmp_path / "holdout.yaml"
    train.write_text(yaml.safe_dump([{"content": "shared"}, {"content": "only_in_train"}]),
                     encoding="utf-8")
    holdout.write_text(yaml.safe_dump([{"content": "shared"}, {"content": "only_in_holdout"}]),
                       encoding="utf-8")
    train_h = m.hashes_for_yaml(train)
    holdout_h = m.hashes_for_yaml(holdout)
    overlap = m.compute_leakage(train_h, holdout_h)
    assert overlap == [m.content_hash("shared")]


def test_leakage_check_clean():
    tmp_path = _new_output_dir("leakage_clean")
    m = _load_module()
    train = tmp_path / "train.yaml"
    holdout = tmp_path / "holdout.yaml"
    train.write_text(yaml.safe_dump([{"content": "a"}, {"content": "b"}]), encoding="utf-8")
    holdout.write_text(yaml.safe_dump([{"content": "c"}]), encoding="utf-8")
    overlap = m.compute_leakage(m.hashes_for_yaml(train), m.hashes_for_yaml(holdout))
    assert overlap == []


def test_discover_challenge_yamls_recursive():
    tmp_path = _new_output_dir("discover_challenges")
    m = _load_module()
    (tmp_path / "challenges" / "subset_a").mkdir(parents=True)
    (tmp_path / "challenges" / "subset_b" / "deeper").mkdir(parents=True)
    (tmp_path / "challenges" / "_excluded").mkdir(parents=True)
    (tmp_path / "challenges" / "top.yaml").write_text("[]", encoding="utf-8")
    (tmp_path / "challenges" / "subset_a" / "a1.yaml").write_text("[]", encoding="utf-8")
    (tmp_path / "challenges" / "subset_b" / "deeper" / "b1.yaml").write_text("[]", encoding="utf-8")
    # Files in any dir whose name starts with `_` must be skipped (e.g.
    # `_excluded/` tombstones).
    (tmp_path / "challenges" / "_excluded" / "tombstone.yaml").write_text("[]", encoding="utf-8")
    # Files outside challenges/ must NOT be picked up.
    (tmp_path / "outside.yaml").write_text("[]", encoding="utf-8")
    found = [str(p) for p in m.discover_challenge_yamls(tmp_path)]
    assert any("top.yaml" in p for p in found)
    assert any("a1.yaml" in p for p in found)
    assert any("b1.yaml" in p for p in found)
    assert not any("tombstone.yaml" in p for p in found), \
        "_excluded/ subdir must be skipped"
    assert not any("outside.yaml" in p for p in found)


def test_main_tolerates_dict_shaped_canonical_eval():
    """Dict-shaped canonical-eval YAML must be skipped with reason, not crash."""
    tmp_path = _new_output_dir("main_tolerates_dict_yaml")
    m = _load_module()
    run_dir, _ = _build_run_dir(tmp_path)
    cur_dir = _build_curation_dir(tmp_path, train_contents=["A"])
    real = tmp_path / "real_eval.yaml"
    real.write_text(yaml.safe_dump([{"content": "challenge_row"}]), encoding="utf-8")
    dict_shaped = tmp_path / "dict_shaped.yaml"
    dict_shaped.write_text(
        yaml.safe_dump({"removed_from": "x", "reason": "y"}), encoding="utf-8"
    )
    output_dir = tmp_path / "out"

    rc = m.main([
        "--source-run-dir", str(run_dir),
        "--curation-dir", str(cur_dir),
        "--canonical-eval", str(real),
        "--canonical-eval", str(dict_shaped),
        "--output-dir", str(output_dir),
    ])
    assert rc == 0
    leak = json.loads((output_dir / "leakage_check.json").read_text(encoding="utf-8"))
    skipped = [s["path"] for s in leak["skipped_targets"]]
    assert any("dict_shaped.yaml" in p for p in skipped)
    target_paths = [t["path"] for t in leak["targets"]]
    assert any("real_eval.yaml" in p for p in target_paths)


# ---------------------------------------------------------------------------
# Tests: fold_assignments.json detection (informational only)
# ---------------------------------------------------------------------------


def test_detect_fold_assignments_status_preview_style():
    tmp_path = _new_output_dir("fold_assignments_preview")
    m = _load_module()
    (tmp_path / "fold_assignments.json").write_text(
        '[{"i": 0}]\n... (24202 total)\n', encoding="utf-8",
    )
    status = m.detect_fold_assignments_status(tmp_path)
    assert status["present"] is True
    assert status["machine_readable"] is False
    assert "preview" in status["note"].lower()


def test_detect_fold_assignments_status_absent():
    tmp_path = _new_output_dir("fold_assignments_absent")
    m = _load_module()
    status = m.detect_fold_assignments_status(tmp_path)
    assert status["present"] is False


# ---------------------------------------------------------------------------
# Tests: end-to-end main()
# ---------------------------------------------------------------------------


def test_main_end_to_end_clean_run_succeeds_no_canonical_eval():
    """Default invocation (no --canonical-eval) emits residuals + manifest with
    gate_status='no_canonical_eval_defined' and exits 0."""
    tmp_path = _new_output_dir("main_clean")
    m = _load_module()
    run_dir, _ = _build_run_dir(tmp_path)
    cur_dir = _build_curation_dir(tmp_path, train_contents=["A", "B", "C"])
    output_dir = tmp_path / "out"

    rc = m.main([
        "--source-run-dir", str(run_dir),
        "--curation-dir", str(cur_dir),
        "--output-dir", str(output_dir),
        "--seed", "42",
    ])
    assert rc == 0, "clean run should exit 0"

    # Outputs present (leakage_check.json is NOT written when no canonical eval)
    assert (output_dir / "residuals.jsonl").exists()
    assert (output_dir / "baseline_correct.jsonl").exists()
    assert (output_dir / "distribution_report.json").exists()
    assert (output_dir / "distribution_report.md").exists()
    assert not (output_dir / "leakage_check.json").exists()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    # Manifest contracts
    assert manifest["fold_thresholds"] == {"0": 0.0, "1": -0.5}
    assert "0" in manifest["fold_threshold_sources"]
    assert manifest["row_counts"]["total_predictions"] == 7  # 5 + 2 in fixtures
    assert manifest["row_counts"]["false_negative"] >= 1
    assert manifest["row_counts"]["false_positive"] >= 1
    assert manifest["near_boundary_definition"]["selector_field"] == "raw_score"
    assert manifest["schema_directionmd_0_1"]["routed_reason_note"]
    assert manifest["fold_assignments_json_status"]["machine_readable"] is False

    # Gate / usability contracts
    assert manifest["gate"]["status"] == "no_canonical_eval_defined"
    assert manifest["gate"]["overall_overlap_count"] == 0
    assert manifest["gate"]["leakage_check_path"] is None
    assert manifest["usability"]["for_l2_training"] is True
    assert manifest["usability"]["for_truncation_policy_spike"] is True
    assert manifest["usability"]["for_phase4_acceptance"] is False
    assert any("legacy" in c.lower() for c in manifest["usability"]["caveats"])

    # Residuals JSONL only carries §0.1 schema rows; validate first row
    with (output_dir / "residuals.jsonl").open(encoding="utf-8") as f:
        first = json.loads(f.readline())
    for k in ("content", "label", "language", "reason", "source", "content_hash",
              "l1_score", "l1_decision", "l1_thresholds", "routed_reason",
              "fold_id", "residual_category", "raw_score"):
        assert k in first, f"missing field {k} in residual row"
    assert first["routed_reason"] is None


def test_main_canonical_eval_clean_marks_phase4_usable():
    """When --canonical-eval is passed and overlap=0, gate_status='passed'
    and usability.for_phase4_acceptance becomes True."""
    tmp_path = _new_output_dir("main_canonical_clean")
    m = _load_module()
    run_dir, _ = _build_run_dir(tmp_path)
    cur_dir = _build_curation_dir(tmp_path, train_contents=["A", "B", "C"])
    eval_path = tmp_path / "canonical_eval.yaml"
    eval_path.write_text(yaml.safe_dump([{"content": "X_disjoint"}]), encoding="utf-8")
    output_dir = tmp_path / "out"

    rc = m.main([
        "--source-run-dir", str(run_dir),
        "--curation-dir", str(cur_dir),
        "--canonical-eval", str(eval_path),
        "--output-dir", str(output_dir),
    ])
    assert rc == 0
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["gate"]["status"] == "passed"
    assert manifest["usability"]["for_phase4_acceptance"] is True
    assert (output_dir / "leakage_check.json").exists()


def test_main_succeeds_when_fold_assignments_json_is_invalid():
    """Regression: existing on-disk fold_assignments.json is preview-style and
    not machine-readable. The exporter must still succeed because fold_id is
    derived from fold_*/run/_eval_holdout_*/ artifacts, not that file."""
    tmp_path = _new_output_dir("main_preview_assignments")
    m = _load_module()
    run_dir, _ = _build_run_dir(tmp_path)
    # _build_run_dir already wrote a preview-style fold_assignments.json.
    # Sanity check we haven't accidentally written valid JSON.
    fa = (run_dir / "fold_assignments.json").read_text(encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        json.loads(fa)

    cur_dir = _build_curation_dir(tmp_path, train_contents=["A"])
    output_dir = tmp_path / "out"

    rc = m.main([
        "--source-run-dir", str(run_dir),
        "--curation-dir", str(cur_dir),
        "--output-dir", str(output_dir),
    ])
    assert rc == 0


def test_main_rejects_fold_id_off_by_one():
    """Regression: corrupting a fold's curated/holdout.yaml by reordering it
    must be caught by the dataset/curated content-equality guard, so fold_id
    can never tag the wrong row metadata."""
    tmp_path = _new_output_dir("main_off_by_one")
    m = _load_module()
    run_dir, fold_rows = _build_run_dir(tmp_path)
    # Reverse curated/holdout.yaml in fold_0 to simulate a row-reorder bug.
    f0_curated = run_dir / "fold_0" / "curated" / "holdout.yaml"
    rows = yaml.safe_load(f0_curated.read_text(encoding="utf-8"))
    rows.reverse()
    f0_curated.write_text(yaml.safe_dump(rows, allow_unicode=True), encoding="utf-8")

    cur_dir = _build_curation_dir(tmp_path, train_contents=["A"])
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="dataset/holdout.yaml content"):
        m.main([
            "--source-run-dir", str(run_dir),
            "--curation-dir", str(cur_dir),
            "--output-dir", str(output_dir),
        ])


def test_main_rejects_summary_total_mismatch():
    """Cross-check: summary.json.total_predictions must match joined row count."""
    tmp_path = _new_output_dir("main_summary_mismatch")
    m = _load_module()
    run_dir, _ = _build_run_dir(tmp_path)
    # Corrupt summary.json to a wrong total.
    (run_dir / "summary.json").write_text(json.dumps({"total_predictions": 999}),
                                          encoding="utf-8")

    cur_dir = _build_curation_dir(tmp_path, train_contents=["A"])
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="summary.json"):
        m.main([
            "--source-run-dir", str(run_dir),
            "--curation-dir", str(cur_dir),
            "--output-dir", str(output_dir),
        ])


def test_main_canonical_eval_overlap_blocks():
    """Overlap with --canonical-eval is a hard fail (rc=2)."""
    tmp_path = _new_output_dir("main_canonical_blocks")
    m = _load_module()
    run_dir, _ = _build_run_dir(tmp_path)
    cur_dir = _build_curation_dir(tmp_path, train_contents=["LEAKED", "B"])
    eval_path = tmp_path / "canonical_eval.yaml"
    eval_path.write_text(yaml.safe_dump([{"content": "LEAKED"}]), encoding="utf-8")
    output_dir = tmp_path / "out"

    rc = m.main([
        "--source-run-dir", str(run_dir),
        "--curation-dir", str(cur_dir),
        "--canonical-eval", str(eval_path),
        "--output-dir", str(output_dir),
    ])
    assert rc == 2
    leak = json.loads((output_dir / "leakage_check.json").read_text(encoding="utf-8"))
    assert leak["overall_overlap_count"] == 1
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["gate"]["status"] == "failed_blocked"
    assert manifest["usability"]["for_phase4_acceptance"] is False


def test_main_canonical_eval_allow_leakage_returns_zero():
    tmp_path = _new_output_dir("main_canonical_allow")
    m = _load_module()
    run_dir, _ = _build_run_dir(tmp_path)
    cur_dir = _build_curation_dir(tmp_path, train_contents=["LEAKED"])
    eval_path = tmp_path / "canonical_eval.yaml"
    eval_path.write_text(yaml.safe_dump([{"content": "LEAKED"}]), encoding="utf-8")
    output_dir = tmp_path / "out"

    rc = m.main([
        "--source-run-dir", str(run_dir),
        "--curation-dir", str(cur_dir),
        "--canonical-eval", str(eval_path),
        "--output-dir", str(output_dir),
        "--allow-leakage",
    ])
    assert rc == 0
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["gate"]["status"] == "failed_allowed"
    # Allowed-leakage still doesn't certify for Phase 4.
    assert manifest["usability"]["for_phase4_acceptance"] is False
