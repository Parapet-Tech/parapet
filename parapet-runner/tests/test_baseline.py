from __future__ import annotations

from parapet_runner.baseline import parse_eval_result_json


def test_parse_eval_result_json_from_nested_payload() -> None:
    payload = {
        "summary": {
            "metrics": {
                "f1": 0.91,
                "precision": 0.92,
                "recall": 0.90,
                "false_positives": 4,
                "false_negatives": 6,
            }
        },
        "metadata": {"threshold": -0.4, "count": 222},
    }

    result = parse_eval_result_json(payload, threshold_fallback=-0.5)
    assert result.f1 == 0.91
    assert result.precision == 0.92
    assert result.recall == 0.90
    assert result.false_positives == 4
    assert result.false_negatives == 6
    assert result.threshold == -0.4
    assert result.holdout_size == 222
