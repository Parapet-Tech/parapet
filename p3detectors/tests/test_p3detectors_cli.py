"""CLI tests with an injected fake judge (no live model, no endpoint)."""
import json

from p3detectors.interface import FAMILY_GENERATIVE_MLX, DetectorResult
from p3detectors.score_cli import main


class FakeJudge:
    family = FAMILY_GENERATIVE_MLX
    detector_id = "dgen_mlx_judge"
    model_id = "qwen3-30b-a3b-2507-mlx-4bit"

    def __init__(self, *, fail_preflight=False):
        self.fail_preflight = fail_preflight
        self.calls = 0

    def preflight(self):
        return (False, "no server") if self.fail_preflight else (True, "ok")

    def score(self, event_text, context=None):
        self.calls += 1
        return DetectorResult(score=0.42, family=self.family, model_id=self.model_id,
                              detector_id=self.detector_id, rationale="fake")


def _fixture(tmp_path):
    run_rel = "AgentDyn/runs/m1/shopping/user_task_1/none/none.json"
    src = tmp_path / "repos" / run_rel
    src.parent.mkdir(parents=True, exist_ok=True)
    msgs = [{
        "role": "assistant",
        "tool_calls": [
            {"function": "send_money", "id": "c0", "args": {"recipient": "evil@example.com"}},
            {"function": "get_balance", "id": "c1", "args": {"account_id": 1}},  # skipped
        ],
    }]
    src.write_text(json.dumps({"messages": msgs}))
    staged = tmp_path / "staged"
    staged.mkdir(parents=True, exist_ok=True)
    (staged / "art.json").write_text(json.dumps({"source": {"run_path": run_rel}}))
    return str(staged), str(tmp_path / "repos")


def test_cli_scores_and_skips(tmp_path, capsys):
    staged, repos = _fixture(tmp_path)
    out = tmp_path / "scores.jsonl"
    judge = FakeJudge()
    rc = main(["--staged", staged, "--repos-root", repos, "--out", str(out)], judge=judge)
    assert rc == 0
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == 1  # one scored event; the no-string-arg call is skipped, not written
    rec = lines[0]
    assert rec["score"] == 0.42 and rec["family"] == "generative-mlx"
    assert rec["position"] == 0 and rec["call_id"] == "c0"
    assert "send_money" not in rec["event_text_preview"]
    assert judge.calls == 1
    out_text = capsys.readouterr().out
    assert "scored 1" in out_text and "skipped 1" in out_text


def test_cli_preflight_gate_blocks_scoring(tmp_path):
    staged, repos = _fixture(tmp_path)
    out = tmp_path / "scores.jsonl"
    judge = FakeJudge(fail_preflight=True)
    rc = main(["--staged", staged, "--repos-root", repos, "--out", str(out)], judge=judge)
    assert rc == 2
    assert judge.calls == 0       # never scored
    assert not out.exists()       # no output written


def test_cli_missing_staged_dir(tmp_path):
    rc = main(["--staged", str(tmp_path / "nope"), "--repos-root", str(tmp_path)], judge=FakeJudge())
    assert rc == 2


def test_cli_limit_caps_scoring(tmp_path):
    # two string-arg calls; limit 1 should score only one
    run_rel = "AgentDyn/runs/m1/shopping/user_task_2/none/none.json"
    src = tmp_path / "repos" / run_rel
    src.parent.mkdir(parents=True, exist_ok=True)
    msgs = [{"role": "assistant", "tool_calls": [
        {"function": "a", "id": "c0", "args": {"x": "one"}},
        {"function": "b", "id": "c1", "args": {"x": "two"}},
    ]}]
    src.write_text(json.dumps({"messages": msgs}))
    staged = tmp_path / "staged"
    staged.mkdir(parents=True, exist_ok=True)
    (staged / "art.json").write_text(json.dumps({"source": {"run_path": run_rel}}))
    out = tmp_path / "scores.jsonl"
    judge = FakeJudge()
    rc = main(["--staged", str(staged), "--repos-root", str(tmp_path / "repos"),
               "--out", str(out), "--limit", "1"], judge=judge)
    assert rc == 0 and judge.calls == 1
    assert len(out.read_text().splitlines()) == 1
