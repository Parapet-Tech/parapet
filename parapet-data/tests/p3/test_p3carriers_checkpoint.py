"""Tests for the staged-artifact round-trip checkpoint.

Strategy: produce a faithful staged artifact from a fixture source run via the real
normalizer (so it matches by construction), then inject one fault at a time and assert
the checkpoint catches exactly that fault. Uses the real default allowlist so staged
and current shas agree (no sha noise) unless a test deliberately drifts the sha.
"""
import json
import os

from parapet_data.p3.carriers.action_allowlist import default_allowlist_path
from parapet_data.p3.carriers.checkpoint import main, run_checkpoint
from parapet_data.p3.carriers.normalize import run_normalize


def _write_source_run(repos_root, repo, model, suite, task, funcs,
                      *, attack="none", inj="none", injections=None):
    d = os.path.join(repos_root, repo, "runs", model, suite, task, attack)
    os.makedirs(d, exist_ok=True)
    run = {
        "suite_name": suite, "pipeline_name": model, "user_task_id": task,
        "injection_task_id": None if inj == "none" else inj,
        "attack_type": None if attack == "none" else attack,
        "injections": injections or {}, "error": None,
        "messages": [{"role": "assistant", "tool_calls": [{"function": f, "id": f"c{i}"}]}
                     for i, f in enumerate(funcs)],
    }
    with open(os.path.join(d, inj + ".json"), "w") as fh:
        json.dump(run, fh)


def _stage(tmp_path, funcs, **kw):
    """Write a source run + produce its faithful staged artifact via the real normalizer."""
    repos = str(tmp_path / "repos")
    out = str(tmp_path / "staged")
    _write_source_run(repos, "AgentDyn", "m1", "shopping", "user_task_1", funcs, **kw)
    run_normalize(repos, default_allowlist_path(), out)
    return repos, out


def _artifact_path(out):
    for root, _, files in os.walk(out):
        for fn in sorted(files):
            if fn.endswith(".json") and fn != "index.json":
                return os.path.join(root, fn)
    raise AssertionError("no staged artifact produced")


def _load(p):
    with open(p) as fh:
        return json.load(fh)


def _dump(p, d):
    with open(p, "w") as fh:
        json.dump(d, fh)


def test_passing_artifact(tmp_path):
    repos, out = _stage(tmp_path, ["search_product", "send_money"])
    s = run_checkpoint(out, repos, default_allowlist_path())
    assert s["checked"] == 1
    assert s["failed"] == 0 and s["skipped"] == 0
    assert s["staging_shas"] == [s["current_sha"]]  # staged with current allowlist -> no drift


def test_detects_wrong_live(tmp_path):
    repos, out = _stage(tmp_path, ["search_product", "send_money"])
    ap = _artifact_path(out)
    art = _load(ap)
    assert art["tool_call_sequence"][0]["live"] is True  # dead read, live via downstream action
    art["tool_call_sequence"][0]["live"] = False
    _dump(ap, art)
    s = run_checkpoint(out, repos, default_allowlist_path())
    assert s["failed"] == 1 and s["geometry_failed"] == 1
    assert "seq_field_mismatch:live@0" in s["failures"][0][1]


def test_detects_wrong_is_action(tmp_path):
    repos, out = _stage(tmp_path, ["search_product", "send_money"])
    ap = _artifact_path(out)
    art = _load(ap)
    assert art["tool_call_sequence"][1]["is_action"] is True  # send_money
    art["tool_call_sequence"][1]["is_action"] = False
    _dump(ap, art)
    s = run_checkpoint(out, repos, default_allowlist_path())
    assert s["failed"] == 1 and s["geometry_failed"] == 1
    assert "seq_field_mismatch:is_action@1" in s["failures"][0][1]


def test_detects_changed_allowlist_sha(tmp_path):
    repos, out = _stage(tmp_path, ["search_product", "send_money"])
    ap = _artifact_path(out)
    art = _load(ap)
    art["action_set"]["sha256"] = "deadbeef" * 8
    _dump(ap, art)
    s = run_checkpoint(out, repos, default_allowlist_path())
    assert s["failed"] == 1
    # geometry untouched -> classified sha-only, NOT geometry corruption
    assert s["geometry_failed"] == 0 and s["sha_only_failed"] == 1
    assert s["failures"][0][1] == ["action_sha_mismatch"]


def test_missing_source_is_failed_not_skipped(tmp_path):
    repos, out = _stage(tmp_path, ["send_money"])
    ap = _artifact_path(out)
    src_rel = _load(ap)["source"]["run_path"]
    os.remove(os.path.join(repos, src_rel))
    s = run_checkpoint(out, repos, default_allowlist_path())
    # explicit choice: unverifiable -> FAILED (reason source_missing), never a silent skip
    assert s["failed"] == 1 and s["skipped"] == 0
    assert s["failures"][0][1] == ["source_missing"]


def test_schema_version_is_skipped(tmp_path):
    repos, out = _stage(tmp_path, ["send_money"])
    ap = _artifact_path(out)
    art = _load(ap)
    art["schema_version"] = "p3-carrier-normalized/999"
    _dump(ap, art)
    s = run_checkpoint(out, repos, default_allowlist_path())
    assert s["checked"] == 0 and s["skipped"] == 1 and s["failed"] == 0


def test_injection_payload_not_emitted(tmp_path, capsys):
    secret_staged = "STAGED_SECRET_PAYLOAD"
    secret_source = "SOURCE_SECRET_PAYLOAD"
    repos, out = _stage(
        tmp_path, ["search_product", "send_money"],
        attack="important_instructions", inj="injection_task_0",
        injections={"injection_x": secret_staged},
    )
    # drift the SOURCE injections so recompute disagrees with the staged artifact
    ap = _artifact_path(out)
    src_path = os.path.join(repos, _load(ap)["source"]["run_path"])
    run = _load(src_path)
    run["injections"] = {"injection_x": secret_source}
    _dump(src_path, run)
    rc = main(["--staged", out, "--repos-root", repos, "--allowlist", default_allowlist_path()])
    captured = capsys.readouterr().out
    assert "source_meta_mismatch:injections" in captured
    assert secret_staged not in captured and secret_source not in captured
    assert rc == 1  # source-meta mismatch is a geometry failure -> nonzero


def test_cli_smoke(tmp_path, capsys):
    repos, out = _stage(tmp_path, ["search_product", "send_money"])
    rc = main(["--staged", out, "--repos-root", repos, "--allowlist", default_allowlist_path()])
    assert rc == 0
    assert "checked 1" in capsys.readouterr().out
    # introduce a geometry fault -> exit 1
    ap = _artifact_path(out)
    art = _load(ap)
    art["tool_call_sequence"][1]["is_action"] = False
    _dump(ap, art)
    rc = main(["--staged", out, "--repos-root", repos, "--allowlist", default_allowlist_path()])
    assert rc == 1
