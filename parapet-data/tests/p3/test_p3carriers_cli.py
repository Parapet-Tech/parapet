"""CLI smoke test for the normalizer against a tiny fixture tree (not the real clones)."""
import json
import os

from parapet_data.p3.carriers.normalize import main


def _write_run(repos_root, repo, model, suite, task, attack, inj, funcs):
    d = os.path.join(repos_root, repo, "runs", model, suite, task, attack)
    os.makedirs(d, exist_ok=True)
    run = {
        "suite_name": suite, "pipeline_name": model, "user_task_id": task,
        "injection_task_id": None if inj == "none" else inj,
        "attack_type": None if attack == "none" else attack,
        "injections": {}, "error": None,
        "messages": [{"role": "assistant", "tool_calls": [{"function": f, "id": f"c{i}"}]}
                     for i, f in enumerate(funcs)],
    }
    with open(os.path.join(d, inj + ".json"), "w") as fh:
        json.dump(run, fh)


def test_cli_smoke_against_fixture_tree(tmp_path):
    repos = tmp_path / "repos"
    out = tmp_path / "staged"
    # uses real allowlist names: send_money is an action; search_product/get_balance are reads
    _write_run(str(repos), "AgentDyn", "m1", "shopping", "user_task_1", "none", "none",
               ["search_product", "send_money"])
    _write_run(str(repos), "agentdojo", "m1", "banking", "user_task_1", "none", "none",
               ["get_balance"])

    rc = main(["--repos-root", str(repos), "--out", str(out)])
    assert rc == 0

    idx = json.loads((out / "index.json").read_text())
    assert idx["counts"]["written"] == 2
    assert idx["counts"]["clean_carriers"] == 2
    # shopping carrier has one action -> graft-ready; banking is read-only
    shopping = next(c for c in idx["carriers"] if c["suite"] == "shopping")
    banking = next(c for c in idx["carriers"] if c["suite"] == "banking")
    assert shopping["n_action_calls"] == 1 and shopping["usable_carrier"] is True
    assert banking["n_action_calls"] == 0
    # the per-run artifact exists and carries the liveness rule
    art = json.loads((out / shopping["out_path"]).read_text())
    assert art["liveness_rule"]
    assert art["action_tools_present"] == ["send_money"]


def test_cli_clean_only_filter(tmp_path):
    repos = tmp_path / "repos"
    out = tmp_path / "staged"
    _write_run(str(repos), "AgentDyn", "m1", "github", "user_task_1", "none", "none", ["git_push"])
    _write_run(str(repos), "AgentDyn", "m1", "github", "user_task_1", "direct", "injection_task_0",
               ["git_push"])

    rc = main(["--repos-root", str(repos), "--out", str(out), "--clean-only"])
    assert rc == 0
    idx = json.loads((out / "index.json").read_text())
    assert idx["counts"]["written"] == 1  # only the clean none/none carrier
    assert idx["carriers"][0]["attack_type"] is None
