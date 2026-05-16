from __future__ import annotations

from parapet_runner.residuals.features import (
    FEATURE_FAMILIES,
    FEATURE_SEMANTICS,
    build_feature_table,
    compute_features,
    feature_auc_tables,
    feature_table_row,
    semantics_receipt,
    top_decile_enrichment,
)


def test_feature_semantics_declares_unicode_scalar_and_byte_counts():
    assert "chars().count()" in FEATURE_SEMANTICS["*_len"]
    assert "s.len()" in FEATURE_SEMANTICS["*_byte_*"]
    assert FEATURE_SEMANTICS["graphemes"] == "Not used in pass one."


def test_compute_features_counts_lengths_ratios_and_entropy():
    row = {"content": "Hi\u200b!\nABC 123", "l1_thresholds": {"l1": 0.0}}

    features = compute_features(row)

    assert features["char_len"] == len("Hi\u200b!\nABC 123")
    assert features["byte_len"] == len("Hi\u200b!\nABC 123".encode("utf-8"))
    assert features["line_count"] == 2
    assert features["zero_width_count"] == 1
    assert features["l0_zero_width_removed_count"] == 1
    assert features["digit_ratio"] > 0
    assert features["shannon_entropy_bytes"] > 0


def test_compute_features_l0_html_delta_is_analysis_only():
    features = compute_features({"content": "<b>ignore</b>", "l1_thresholds": {"l1": 0.0}})

    assert features["l0_html_stripped_hint"] is True
    assert features["l0_char_delta"] > 0
    assert semantics_receipt()["l0_python_mirror"] == "analysis_only_unverified_against_rust"


def test_feature_table_row_excludes_raw_content():
    row = {
        "content": "secret raw text",
        "content_hash": "abc",
        "residual_category": "false_negative",
        "raw_score": -0.1,
        "raw_squash_score": 0.5,
        "l1_thresholds": {"l1": 0.0},
    }

    out = feature_table_row(row, row_set="residual")

    assert out["content_hash"] == "abc"
    assert "content" not in out
    assert out["squash_minus_raw"] == 0.6


def test_feature_auc_tables_are_separated_by_family():
    residuals = [
        {
            "content": "AAAA",
            "content_hash": "fn",
            "residual_category": "false_negative",
            "raw_score": 2.0,
            "raw_squash_score": 2.0,
            "raw_unquoted_score": 2.0,
            "l1_thresholds": {"l1": 0.0},
        },
        {
            "content": "bb",
            "content_hash": "hn",
            "residual_category": "near_boundary_benign",
            "raw_score": 0.1,
            "raw_squash_score": 0.1,
            "raw_unquoted_score": 0.1,
            "l1_thresholds": {"l1": 0.0},
        },
    ]
    table = build_feature_table(residuals, [])
    aucs = feature_auc_tables(table)

    assert set(FEATURE_FAMILIES).issubset(aucs)
    assert any(row["field"] == "raw_margin" for row in aucs["l2_geometry"])
    assert any(row["field"] == "char_len" for row in aucs["mechanical_text_shape"])
    json_like = next(row for row in aucs["mechanical_text_shape"] if row["field"] == "json_like")
    assert json_like["n"] == 2


def test_top_decile_enrichment_reports_false_negative_share():
    rows = [
        {"row_set": "residual", "residual_category": "false_negative", "x": 10.0},
        {"row_set": "residual", "residual_category": "near_boundary_benign", "x": 1.0},
    ]

    out = top_decile_enrichment(rows, "x")

    assert out["selected_n"] == 1
    assert out["false_negative_share"] == 1.0
