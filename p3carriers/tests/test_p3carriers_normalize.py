"""Tests for liveness labeling and run normalization."""
from p3carriers.normalize import normalize_run, tool_call_sequence

ACTIONS = {"send_money", "create_file"}


def _seq(funcs):
    msgs = [{"role": "assistant", "tool_calls": [{"function": f, "id": f"c{i}"}]}
            for i, f in enumerate(funcs)]
    return tool_call_sequence(msgs, ACTIONS)


def test_liveness_no_actions_all_dead():
    seq = _seq(["search_product", "get_balance", "read_file"])
    assert [e["live"] for e in seq] == [False, False, False]
    assert all(e["is_action"] is False for e in seq)


def test_liveness_action_at_end_all_prior_live():
    seq = _seq(["search_product", "get_balance", "send_money"])
    assert [e["live"] for e in seq] == [True, True, True]


def test_liveness_action_in_middle_reads_after_last_action_dead():
    seq = _seq(["search_product", "send_money", "get_balance", "read_file"])
    assert [e["is_action"] for e in seq] == [False, True, False, False]
    # earlier positions and the action are live; reads after the last action are not
    assert [e["live"] for e in seq] == [True, True, False, False]


def test_liveness_two_actions_live_through_last():
    seq = _seq(["create_file", "read_file", "send_money", "get_balance"])
    assert [e["live"] for e in seq] == [True, True, True, False]


def test_normalize_run_clean_usable_and_fields():
    run = {
        "suite_name": "shopping", "pipeline_name": "m", "user_task_id": "user_task_1",
        "injection_task_id": None, "attack_type": None, "injections": {},
        "error": None, "benchmark_version": "v1", "agentdojo_package_version": "x",
        "messages": [
            {"role": "user", "content": [{"type": "text", "content": "hi"}]},
            {"role": "assistant", "tool_calls": [{"function": "search_product", "id": "c0"}]},
            {"role": "assistant", "tool_calls": [{"function": "send_money", "id": "c1"}]},
        ],
    }
    art = normalize_run(
        run, rel_path="AgentDyn/runs/m/shopping/user_task_1/none/none.json",
        repo="AgentDyn", model="m", action_set=ACTIONS, action_sha="deadbeef", allowlist_path="x.yaml",
    )
    assert art["clean_carrier"] is True
    assert art["usable_carrier"] is True
    assert art["action_tools_present"] == ["send_money"]
    assert art["last_action_position"] == 1
    assert art["n_tool_calls"] == 2 and art["n_action_calls"] == 1
    assert art["n_live_positions"] == 2
    src = art["source"]
    assert src["repo"] == "AgentDyn" and src["model"] == "m" and src["suite"] == "shopping"
    assert src["user_task_id"] == "user_task_1" and src["run_path"].endswith("none.json")
    assert art["liveness_rule"] and art["action_set"]["sha256"] == "deadbeef"


def test_normalize_run_attacked_is_not_clean_or_usable():
    run = {
        "suite_name": "shopping", "pipeline_name": "m", "user_task_id": "user_task_1",
        "injection_task_id": "injection_task_0", "attack_type": "direct",
        "injections": {"injection_reviews_0": "payload"}, "error": None,
        "messages": [{"role": "assistant", "tool_calls": [{"function": "send_money", "id": "c0"}]}],
    }
    art = normalize_run(
        run, rel_path="x", repo="AgentDyn", model="m",
        action_set=ACTIONS, action_sha="s", allowlist_path="x",
    )
    assert art["clean_carrier"] is False
    assert art["usable_carrier"] is False  # attacked, even though it has an action call


def test_normalize_run_clean_but_no_calls_not_usable():
    run = {
        "suite_name": "banking", "pipeline_name": "m", "user_task_id": "user_task_2",
        "injection_task_id": None, "attack_type": None, "injections": {}, "error": None,
        "messages": [{"role": "user", "content": [{"type": "text", "content": "hi"}]}],
    }
    art = normalize_run(
        run, rel_path="x", repo="agentdojo", model="m",
        action_set=ACTIONS, action_sha="s", allowlist_path="x",
    )
    assert art["clean_carrier"] is True
    assert art["usable_carrier"] is False  # clean but zero tool calls
    assert art["last_action_position"] is None and art["action_tools_present"] == []
