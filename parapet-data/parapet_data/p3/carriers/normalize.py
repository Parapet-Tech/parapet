"""Normalize AgentDojo / AgentDyn pre-executed runs into P3 carrier artifacts.

Extracts the ordered tool-call sequence from each executed run, labels each call
position with the locked linear-downstream liveness rule using the reviewed
action allowlist, marks clean carriers, and writes one artifact per run plus an
index. Input runs are immutable; output is git-ignored.

Logic is here (tested); scripts/p3_normalize_carriers.py is a thin CLI.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

from parapet_data.p3.carriers import schemas
from parapet_data.p3.carriers.action_allowlist import default_allowlist_path, load_action_set


def tool_call_sequence(messages: list[dict[str, Any]], action_set: set[str]) -> list[dict[str, Any]]:
    """Ordered tool calls with linear-downstream liveness labels.

    live[i] is true iff some call at position >= i is an action (reverse scan).
    """
    seq: list[dict[str, Any]] = []
    for mi, m in enumerate(messages):
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function")
            seq.append({
                "position": len(seq),
                "message_index": mi,
                "function": fn,
                "call_id": tc.get("id"),
                "is_action": fn in action_set,
            })
    seen_action = False
    for ev in reversed(seq):
        if ev["is_action"]:
            seen_action = True
        ev["live"] = seen_action
    return seq


def normalize_run(
    run: dict[str, Any],
    *,
    rel_path: str,
    repo: str,
    model: str,
    action_set: set[str],
    action_sha: str,
    allowlist_path: str,
) -> dict[str, Any]:
    """Build a normalized carrier artifact from one loaded run dict."""
    injections = run.get("injections") or {}
    clean_carrier = (
        run.get("attack_type") is None
        and run.get("injection_task_id") is None
        and not injections
    )
    seq = tool_call_sequence(run.get("messages", []), action_set)
    action_positions = [e["position"] for e in seq if e["is_action"]]
    return {
        "schema_version": schemas.SCHEMA_VERSION,
        "liveness_rule": schemas.LIVENESS_RULE,
        "action_set": {
            "authority": allowlist_path,
            "sha256": action_sha,
            "n_action_tools": len(action_set),
        },
        "source": {
            "repo": repo,
            "run_path": rel_path,
            "model": model,
            "suite": run.get("suite_name"),
            "user_task_id": run.get("user_task_id"),
            "injection_task_id": run.get("injection_task_id"),
            "attack_type": run.get("attack_type"),
            "injections": injections,
            "benchmark_version": run.get("benchmark_version"),
            "agentdojo_package_version": run.get("agentdojo_package_version"),
            "error": run.get("error"),
        },
        "clean_carrier": clean_carrier,
        "usable_carrier": clean_carrier and run.get("error") is None and len(seq) > 0,
        "n_tool_calls": len(seq),
        "n_action_calls": len(action_positions),
        "n_live_positions": sum(1 for e in seq if e["live"]),
        "last_action_position": (action_positions[-1] if action_positions else None),
        "action_tools_present": sorted({e["function"] for e in seq if e["is_action"]}),
        "tool_call_sequence": seq,
    }


def out_name(artifact: dict[str, Any]) -> str:
    s = artifact["source"]
    attack = s["attack_type"] or "none"
    inj = s["injection_task_id"] or "none"
    return f"{s['user_task_id']}__{attack}__{inj}.json"


def _path_fields(abs_path: str, repos_root: str) -> tuple[str, str, str]:
    """Return (rel_path, repo, model) for a run file under repos_root."""
    rel = os.path.relpath(abs_path, repos_root)
    parts = rel.split(os.sep)
    return rel, parts[0], parts[2]  # <repo>/runs/<model>/...


def run_normalize(
    repos_root: str,
    allowlist_path: str,
    out: str,
    *,
    suites: list[str] | None = None,
    models: list[str] | None = None,
    clean_only: bool = False,
    limit: int = 0,
) -> dict[str, Any]:
    """Normalize every matching run; write artifacts + index.json; return summary."""
    action_set, action_sha = load_action_set(allowlist_path)

    paths: list[str] = []
    for repo in schemas.REPO_SUITES:
        paths.extend(sorted(glob.glob(f"{repos_root}/{repo}/runs/*/*/*/*/*.json")))
    if models:
        paths = [p for p in paths if any(m in p for m in models)]

    index: list[dict[str, Any]] = []
    written = skipped = 0
    for path in paths:
        try:
            rel, repo, model = _path_fields(path, repos_root)
            with open(path) as fh:
                run = json.load(fh)
            artifact = normalize_run(
                run, rel_path=rel, repo=repo, model=model,
                action_set=action_set, action_sha=action_sha, allowlist_path=allowlist_path,
            )
        except (json.JSONDecodeError, OSError, KeyError, IndexError):
            skipped += 1
            continue
        suite = artifact["source"]["suite"]
        if suite not in schemas.REPO_SUITES.get(repo, set()):
            skipped += 1
            continue
        if suites and suite not in suites:
            continue
        if clean_only and not artifact["clean_carrier"]:
            continue
        out_dir = os.path.join(out, repo, model, suite)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, out_name(artifact)), "w") as fh:
            json.dump(artifact, fh, indent=1)
        index.append({
            "out_path": os.path.relpath(os.path.join(out_dir, out_name(artifact)), out),
            "repo": repo, "model": model, "suite": suite,
            "user_task_id": artifact["source"]["user_task_id"],
            "attack_type": artifact["source"]["attack_type"],
            "clean_carrier": artifact["clean_carrier"],
            "usable_carrier": artifact["usable_carrier"],
            "n_tool_calls": artifact["n_tool_calls"],
            "n_action_calls": artifact["n_action_calls"],
            "n_live_positions": artifact["n_live_positions"],
        })
        written += 1
        if limit and written >= limit:
            break

    os.makedirs(out, exist_ok=True)
    summary = {
        "schema_version": schemas.SCHEMA_VERSION,
        "liveness_rule": schemas.LIVENESS_RULE,
        "action_set": {"authority": allowlist_path, "sha256": action_sha, "n_action_tools": len(action_set)},
        "counts": {
            "written": written,
            "skipped": skipped,
            "clean_carriers": sum(1 for e in index if e["clean_carrier"]),
            "usable_carriers": sum(1 for e in index if e["usable_carrier"]),
        },
        "carriers": index,
    }
    with open(os.path.join(out, "index.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Normalize AgentDojo/AgentDyn runs into P3 carrier artifacts.")
    ap.add_argument("--repos-root", default=schemas.default_repos_root())
    ap.add_argument("--allowlist", default=default_allowlist_path())
    ap.add_argument("--out", default=schemas.default_staged_out())
    ap.add_argument("--suites", nargs="*", help="restrict to these suite names")
    ap.add_argument("--models", nargs="*", help="restrict to these model dirs (substring match)")
    ap.add_argument("--clean-only", action="store_true", help="only emit clean (none/none) carriers")
    ap.add_argument("--limit", type=int, default=0, help="cap number of runs (0 = all); for smoke tests")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.repos_root):
        print(f"carrier clones not found at {args.repos_root}")
        return 1
    summary = run_normalize(
        args.repos_root, args.allowlist, args.out,
        suites=args.suites, models=args.models, clean_only=args.clean_only, limit=args.limit,
    )
    c = summary["counts"]
    print(f"wrote {c['written']} artifacts ({c['clean_carriers']} clean, {c['usable_carriers']} usable), "
          f"skipped {c['skipped']}. action tools: {summary['action_set']['n_action_tools']}. "
          f"index: {os.path.join(args.out, 'index.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
