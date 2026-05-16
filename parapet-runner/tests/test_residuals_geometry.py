from __future__ import annotations

from parapet_runner.residuals.geometry import (
    borderline_squash_sweep,
    blocks_borderline_squash,
    blocks_current_effective,
    default_policy_specs,
    evaluate_policy,
    evaluate_policies,
    rank_auc,
    scalar_auc_for_field,
    threshold,
)


def _row(category: str, raw: float, *, squash: float | None = None, label: str = "malicious") -> dict:
    return {
        "residual_category": category,
        "label": label,
        "language": "EN",
        "reason": "obfuscation",
        "source": "src_a",
        "format_bin": "plain",
        "raw_score": raw,
        "raw_unquoted_score": raw,
        "raw_squash_score": raw if squash is None else squash,
        "raw_score_delta": 0.0,
        "quote_detected": False,
        "l1_thresholds": {"l1": 0.0},
    }


def test_threshold_falls_back_to_zero_for_missing_or_bad_values():
    assert threshold({}) == 0.0
    assert threshold({"l1_thresholds": {"l1": "bad"}}) == 0.0
    assert threshold({"l1_thresholds": {"l1": -0.5}}) == -0.5


def test_current_effective_uses_unquoted_when_quote_delta_is_large():
    row = {
        "raw_score": 0.8,
        "raw_unquoted_score": -0.4,
        "raw_score_delta": 1.3,
        "quote_detected": True,
        "l1_thresholds": {"l1": 0.0},
    }

    assert not blocks_current_effective(row)


def test_borderline_squash_recovers_only_borderline_rows():
    near = _row("false_negative", -0.1, squash=0.9)
    far = _row("false_negative", -2.0, squash=0.9)

    assert blocks_borderline_squash(near, borderline_band=0.5)
    assert not blocks_borderline_squash(far, borderline_band=0.5)


def test_evaluate_policy_reports_fn_recovery_sidecar_cost_and_source_hhi():
    policy = default_policy_specs(borderline_band=0.5)[2]  # squash_when_raw_borderline
    residuals = [
        _row("false_negative", -0.1, squash=0.9),
        _row("false_negative", -1.2, squash=0.9),
        _row("near_boundary_benign", -0.1, squash=0.7, label="benign"),
        _row("false_positive", 0.6, squash=0.8, label="benign"),
    ]
    sidecar = [
        _row("baseline_correct", -0.2, squash=0.9, label="benign"),
        _row("baseline_correct", -2.0, squash=0.9, label="benign"),
        _row("baseline_correct", 2.0, squash=2.0, label="malicious"),
    ]

    result = evaluate_policy(residuals, sidecar, policy)

    assert result["false_negative_recovered"] == 1
    assert result["false_negative_n"] == 2
    assert result["sidecar_total_n"] == 3
    assert result["sidecar_n"] == 2
    assert result["sidecar_blocks"] == 1
    assert result["sidecar_block_share"] == 0.5
    assert result["recovered_fn_source_concentration"]["hhi"] == 1.0
    assert result["by_language"][0]["false_negative_recovered"] == 1
    by_category = {row["residual_category"]: row for row in result["by_residual_category"]}
    assert by_category["false_positive"]["added_blocks"] == 0
    assert by_category["near_boundary_benign"]["added_blocks"] == 1
    assert result["by_reason"][0]["value"] == "obfuscation"
    assert result["by_reason"][0]["false_negative_recovered"] == 1


def test_evaluate_policies_keeps_current_and_squash_separate():
    residuals = [_row("false_negative", -0.1, squash=0.9)]
    sidecar = []

    results = evaluate_policies(residuals, sidecar, default_policy_specs(borderline_band=0.5))
    by_name = {row["name"]: row for row in results}

    assert by_name["current_effective"]["false_negative_recovered"] == 0
    assert by_name["squash_when_raw_borderline"]["false_negative_recovered"] == 1


def test_borderline_squash_sweep_reports_break_even_and_language_gate():
    residuals = [
        _row("false_negative", -0.1, squash=0.9),
        {**_row("false_negative", -0.1, squash=0.9), "language": "RU"},
        _row("near_boundary_benign", -0.1, squash=0.9, label="benign"),
    ]

    rows = borderline_squash_sweep(
        residuals,
        [],
        bands=(0.05, 0.2),
        languages=(None, "EN"),
    )
    by_gate_band = {(row["language_gate"], row["borderline_band"]): row for row in rows}

    assert by_gate_band[("ALL", 0.2)]["false_negative_recovered"] == 2
    assert by_gate_band[("ALL", 0.2)]["added_near_boundary_benign_blocks"] == 1
    assert by_gate_band[("ALL", 0.2)]["min_fn_value_for_break_even"] == 0.5
    assert by_gate_band[("EN", 0.2)]["false_negative_recovered"] == 1


def test_rank_auc_handles_ties_and_missing_classes():
    assert rank_auc([1.0, 2.0, 3.0], [False, False, False]) is None
    assert rank_auc([1.0, 1.0, 2.0], [False, True, True]) == 0.75


def test_scalar_auc_for_field_uses_fn_vs_hard_negatives():
    rows = [
        {"residual_category": "false_negative", "x": 3.0},
        {"residual_category": "near_boundary_benign", "x": 1.0},
        {"residual_category": "baseline_correct", "x": 100.0},
    ]

    out = scalar_auc_for_field(rows, "x")

    assert out["n"] == 2
    assert out["auc"] == 1.0
