from __future__ import annotations

from pathlib import Path

from parapet_runner.baseline import SubprocessCommandExecutor, parse_eval_result_json


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


def test_subprocess_executor_uses_utf8_with_replacement(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeCompleted:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeCompleted()

    monkeypatch.setattr("parapet_runner.baseline.subprocess.run", fake_run)

    result = SubprocessCommandExecutor().run(["echo", "hi"], cwd=Path("C:/tmp"))

    assert result.returncode == 0
    assert result.stdout == "ok"
    assert result.stderr == ""
    assert captured["kwargs"]["encoding"] == "utf-8"
    assert captured["kwargs"]["errors"] == "replace"
