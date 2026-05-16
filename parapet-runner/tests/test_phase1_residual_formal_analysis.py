"""Tests for parapet-runner/scripts/phase1_residual_formal_analysis.py.

Focuses on the pure analysis functions (quantile / concentration / margin /
hash quality / quote-squash). Avoids exercising the full main() since the
script is descriptive-only and the math is the load-bearing part.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts" / "phase1_residual_formal_analysis.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase1_residual_formal_analysis", script_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# pct
# ---------------------------------------------------------------------------


def test_pct_zero_total_returns_zero():
    m = _load_module()
    assert m.pct(0, 0) == 0.0
    assert m.pct(5, 0) == 0.0  # don't blow up on accidental div-by-zero


def test_pct_ordinary():
    m = _load_module()
    assert m.pct(1, 4) == 0.25
    assert m.pct(100, 100) == 1.0


# ---------------------------------------------------------------------------
# quantile
# ---------------------------------------------------------------------------


def test_quantile_empty_returns_none():
    m = _load_module()
    assert m.quantile([], 0.5) is None


def test_quantile_single_value():
    m = _load_module()
    assert m.quantile([7.0], 0.5) == 7.0
    assert m.quantile([7.0], 0.0) == 7.0
    assert m.quantile([7.0], 1.0) == 7.0


def test_quantile_exact_midpoint():
    m = _load_module()
    assert m.quantile([1.0, 2.0, 3.0], 0.5) == 2.0


def test_quantile_endpoints():
    m = _load_module()
    vals = [10.0, 20.0, 30.0, 40.0]
    assert m.quantile(vals, 0.0) == 10.0
    assert m.quantile(vals, 1.0) == 40.0


def test_quantile_linear_interpolation():
    m = _load_module()
    # For [1, 2, 3, 4], q=0.25 -> position 0.75 -> 1*0.25 + 2*0.75 = 1.75
    result = m.quantile([1.0, 2.0, 3.0, 4.0], 0.25)
    assert result is not None
    assert abs(result - 1.75) < 1e-9


# ---------------------------------------------------------------------------
# quantiles
# ---------------------------------------------------------------------------


def test_quantiles_filters_non_numeric():
    m = _load_module()
    result = m.quantiles([1.0, "not_numeric", None, 2.0, 3.0])
    assert result["n"] == 3
    assert result["min"] == 1.0
    assert result["max"] == 3.0
    assert result["p50"] == 2.0


def test_quantiles_filters_non_finite():
    m = _load_module()
    result = m.quantiles([1.0, math.inf, -math.inf, math.nan, 5.0])
    assert result["n"] == 2
    assert result["min"] == 1.0
    assert result["max"] == 5.0


def test_quantiles_empty_input():
    m = _load_module()
    result = m.quantiles([])
    assert result["n"] == 0
    assert result["p50"] is None
    assert result["max"] is None


# ---------------------------------------------------------------------------
# counter_table
# ---------------------------------------------------------------------------


def test_counter_table_counts_and_shares():
    m = _load_module()
    rows = [
        {"reason": "instruction_override"},
        {"reason": "instruction_override"},
        {"reason": "obfuscation"},
        {"reason": "instruction_override"},
    ]
    out = m.counter_table(rows, "reason")
    assert out[0] == {"value": "instruction_override", "count": 3, "share": 0.75}
    assert out[1] == {"value": "obfuscation", "count": 1, "share": 0.25}


def test_counter_table_top_n_truncates():
    m = _load_module()
    rows = [{"x": str(i)} for i in range(10)] + [{"x": "common"}] * 3
    out = m.counter_table(rows, "x", top=2)
    assert len(out) == 2
    assert out[0]["value"] == "common"


def test_counter_table_missing_field_treated_as_marker():
    m = _load_module()
    rows = [{"reason": "x"}, {}]  # second row missing the field
    out = m.counter_table(rows, "reason")
    values = {entry["value"] for entry in out}
    assert "<missing>" in values


# ---------------------------------------------------------------------------
# concentration (HHI)
# ---------------------------------------------------------------------------


def test_concentration_hhi_for_evenly_distributed_sources():
    """4 equal sources -> HHI = 4 * 0.25^2 = 0.25, effective_n = 4."""
    m = _load_module()
    rows = (
        [{"source": "a"}] * 25 + [{"source": "b"}] * 25
        + [{"source": "c"}] * 25 + [{"source": "d"}] * 25
    )
    result = m.concentration(rows, "source")
    assert result["unique"] == 4
    assert abs(result["hhi"] - 0.25) < 1e-9
    assert abs(result["effective_n"] - 4.0) < 1e-9


def test_concentration_hhi_for_single_source_is_one():
    m = _load_module()
    rows = [{"source": "only"}] * 50
    result = m.concentration(rows, "source")
    assert result["unique"] == 1
    assert result["hhi"] == 1.0
    assert result["top_share"] == 1.0
    assert result["effective_n"] == 1.0


def test_concentration_top5_top10_shares_are_cumulative():
    m = _load_module()
    # 6 sources, share descending
    rows = (
        [{"source": "a"}] * 50 + [{"source": "b"}] * 25
        + [{"source": "c"}] * 10 + [{"source": "d"}] * 8
        + [{"source": "e"}] * 5 + [{"source": "f"}] * 2
    )
    result = m.concentration(rows, "source")
    # top5_share covers the first 5 sources = 50+25+10+8+5 = 98 of 100.
    assert abs(result["top5_share"] - 0.98) < 1e-9


def test_concentration_zero_hhi_handled():
    """Empty input shouldn't crash; effective_n should be None."""
    m = _load_module()
    result = m.concentration([], "source")
    assert result["n"] == 0
    assert result["effective_n"] is None  # 1/0 not computed


# ---------------------------------------------------------------------------
# threshold + score_margin
# ---------------------------------------------------------------------------


def test_threshold_extracts_l1_value():
    m = _load_module()
    assert m.threshold({"l1_thresholds": {"l1": -0.5}}) == -0.5
    assert m.threshold({"l1_thresholds": {"l1": 0.0}}) == 0.0


def test_threshold_falls_back_to_zero_when_missing():
    m = _load_module()
    assert m.threshold({}) == 0.0
    assert m.threshold({"l1_thresholds": None}) == 0.0
    assert m.threshold({"l1_thresholds": {}}) == 0.0
    assert m.threshold({"l1_thresholds": {"l1": "not_numeric"}}) == 0.0


def test_score_margin_returns_distance_to_threshold():
    m = _load_module()
    margin = m.score_margin({"raw_score": 1.5, "l1_thresholds": {"l1": 0.5}})
    assert margin == 1.0


def test_score_margin_returns_none_when_raw_score_missing():
    m = _load_module()
    assert m.score_margin({}) is None
    assert m.score_margin({"raw_score": None}) is None
    assert m.score_margin({"raw_score": "not_numeric"}) is None


def test_score_margin_is_signed():
    m = _load_module()
    pos = m.score_margin({"raw_score": 0.7, "l1_thresholds": {"l1": 0.0}})
    neg = m.score_margin({"raw_score": -0.3, "l1_thresholds": {"l1": 0.0}})
    assert pos == 0.7
    assert neg == -0.3


# ---------------------------------------------------------------------------
# margin_analysis
# ---------------------------------------------------------------------------


def test_margin_analysis_bands_are_cumulative():
    """A row inside a 0.10 band is also inside the 0.50 and 1.00 bands."""
    m = _load_module()
    rows = [
        {"raw_score": 0.05, "l1_thresholds": {"l1": 0.0}, "residual_category": "fp",
         "label": "benign"},
        {"raw_score": 0.40, "l1_thresholds": {"l1": 0.0}, "residual_category": "fn",
         "label": "malicious"},
        {"raw_score": 1.50, "l1_thresholds": {"l1": 0.0}, "residual_category": "fn",
         "label": "malicious"},
    ]
    result = m.margin_analysis(rows)
    band_counts = {b["abs_margin_lte"]: b["count"] for b in result["by_band"]}
    assert band_counts[0.05] == 1
    assert band_counts[0.10] == 1
    assert band_counts[0.50] == 2
    assert band_counts[1.00] == 2
    assert band_counts[2.00] == 3


def test_margin_analysis_skips_rows_with_missing_score():
    m = _load_module()
    rows = [
        {"raw_score": None, "l1_thresholds": {"l1": 0.0}, "residual_category": "x",
         "label": "benign"},
        {"raw_score": 0.1, "l1_thresholds": {"l1": 0.0}, "residual_category": "x",
         "label": "benign"},
    ]
    result = m.margin_analysis(rows)
    assert result["n"] == 1


# ---------------------------------------------------------------------------
# hash_quality
# ---------------------------------------------------------------------------


def test_hash_quality_detects_duplicates_with_no_conflicts():
    m = _load_module()
    rows = [
        {"content_hash": "h1", "label": "benign", "residual_category": "fp"},
        {"content_hash": "h1", "label": "benign", "residual_category": "fp"},
        {"content_hash": "h2", "label": "malicious", "residual_category": "fn"},
    ]
    result = m.hash_quality(rows)
    assert result["unique_hashes"] == 2
    assert result["duplicate_hashes"] == 1
    assert result["duplicate_rows"] == 2
    assert result["label_conflicts"] == []
    assert result["category_conflicts"] == []


def test_hash_quality_flags_label_conflicts():
    m = _load_module()
    rows = [
        {"content_hash": "h1", "label": "benign", "residual_category": "fp"},
        {"content_hash": "h1", "label": "malicious", "residual_category": "fn"},
    ]
    result = m.hash_quality(rows)
    assert len(result["label_conflicts"]) == 1
    conflict = result["label_conflicts"][0]
    assert conflict["content_hash"] == "h1"
    assert sorted(conflict["labels"]) == ["benign", "malicious"]


def test_hash_quality_flags_category_conflicts():
    m = _load_module()
    rows = [
        {"content_hash": "h1", "label": "benign", "residual_category": "fp"},
        {"content_hash": "h1", "label": "benign", "residual_category": "near_boundary"},
    ]
    result = m.hash_quality(rows)
    assert len(result["category_conflicts"]) == 1
    assert result["label_conflicts"] == []  # labels agreed


def test_hash_quality_counts_missing_hashes():
    m = _load_module()
    rows = [
        {"content_hash": "h1", "label": "benign"},
        {"content_hash": "", "label": "benign"},
        {"content_hash": None, "label": "benign"},
        {"label": "benign"},
    ]
    result = m.hash_quality(rows)
    assert result["missing_hash"] == 3
    assert result["unique_hashes"] == 1


# ---------------------------------------------------------------------------
# missing_field_analysis
# ---------------------------------------------------------------------------


def test_missing_field_analysis_reports_only_missing():
    m = _load_module()
    rows = [
        {"a": 1, "b": "x"},
        {"a": 2, "b": ""},   # b empty -> counts as missing
        {"a": 3, "b": None},  # b None -> counts as missing
    ]
    result = m.missing_field_analysis(rows)
    assert "a" not in result  # never missing
    assert result["b"]["missing"] == 2
    assert abs(result["b"]["share"] - 2 / 3) < 1e-9


def test_missing_field_analysis_handles_disjoint_keys():
    m = _load_module()
    rows = [
        {"a": 1},
        {"b": 2},
    ]
    result = m.missing_field_analysis(rows)
    # Each row is missing exactly one of the union of fields.
    assert result["a"]["missing"] == 1
    assert result["b"]["missing"] == 1


# ---------------------------------------------------------------------------
# group_rows + numeric_by_group
# ---------------------------------------------------------------------------


def test_group_rows_partitions_by_field_value():
    m = _load_module()
    rows = [
        {"language": "EN", "x": 1},
        {"language": "RU", "x": 2},
        {"language": "EN", "x": 3},
    ]
    out = m.group_rows(rows, "language")
    assert sorted(out.keys()) == ["EN", "RU"]
    assert len(out["EN"]) == 2
    assert len(out["RU"]) == 1


def test_group_rows_uses_marker_for_missing_field():
    m = _load_module()
    rows = [{"language": "EN"}, {}]
    out = m.group_rows(rows, "language")
    assert "<missing>" in out


# ---------------------------------------------------------------------------
# quote_squash_analysis
# ---------------------------------------------------------------------------


def test_quote_squash_analysis_computes_squash_advantage():
    """squash_minus_unquoted = raw_squash_score - raw_unquoted_score."""
    m = _load_module()
    rows = [
        {"residual_category": "fn", "raw_squash_score": 0.8,
         "raw_unquoted_score": 0.2, "raw_score_delta": 0.6,
         "quote_detected": True, "label": "malicious", "source": "src_a"},
        {"residual_category": "fn", "raw_squash_score": 0.5,
         "raw_unquoted_score": 0.4, "raw_score_delta": 0.1,
         "quote_detected": False, "label": "malicious", "source": "src_a"},
    ]
    result = m.quote_squash_analysis(rows)
    fn = result["fn"]
    assert fn["n"] == 2
    assert fn["quote_detected_count"] == 1
    # squash advantage values are 0.6 and 0.1; quantiles n=2.
    assert fn["squash_minus_unquoted_quantiles"]["n"] == 2


def test_quote_squash_analysis_groups_by_residual_category():
    m = _load_module()
    rows = [
        {"residual_category": "fn", "raw_squash_score": 0.5,
         "raw_unquoted_score": 0.1, "quote_detected": True, "label": "malicious",
         "source": "src"},
        {"residual_category": "fp", "raw_squash_score": 0.3,
         "raw_unquoted_score": 0.2, "quote_detected": False, "label": "benign",
         "source": "src"},
    ]
    result = m.quote_squash_analysis(rows)
    assert sorted(result.keys()) == ["fn", "fp"]


# ---------------------------------------------------------------------------
# Integration smoke
# ---------------------------------------------------------------------------


def test_analyze_rows_runs_end_to_end_on_minimal_residual():
    """Smoke test: the orchestrator stitches all pieces together without error."""
    m = _load_module()
    rows = [
        {
            "content": "x", "content_hash": "h1", "label": "benign",
            "residual_category": "fp", "error_type": "false_positive",
            "language": "EN", "source": "src_a", "reason": "instruction_override",
            "format_bin": "prose", "length_bin": "short",
            "fold_id": 0, "raw_score": 0.5, "l1_score": 0.4,
            "raw_unquoted_score": 0.3, "raw_squash_score": 0.5,
            "raw_score_delta": 0.2, "unquoted_score": 0.3, "squash_score": 0.5,
            "quote_detected": False, "l1_thresholds": {"l1": 0.0},
        },
        {
            "content": "y", "content_hash": "h2", "label": "malicious",
            "residual_category": "fn", "error_type": "false_negative",
            "language": "RU", "source": "src_b", "reason": "obfuscation",
            "format_bin": "code", "length_bin": "long",
            "fold_id": 1, "raw_score": -0.4, "l1_score": -0.3,
            "raw_unquoted_score": -0.2, "raw_squash_score": -0.5,
            "raw_score_delta": -0.3, "unquoted_score": -0.2, "squash_score": -0.5,
            "quote_detected": True, "l1_thresholds": {"l1": -0.5},
        },
    ]
    out = m.analyze_rows(rows, "test")
    assert out["row_count"] == 2
    assert out["hash_quality"]["unique_hashes"] == 2
    assert "fp" in out["category_keys"] or "fn" in out["category_keys"]
    assert "EN" in {entry["value"] for entry in out["composition"]["language"]}
