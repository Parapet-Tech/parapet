#!/usr/bin/env python3
"""
Normalize AgentDojo / AgentDyn pre-executed runs into P3 carrier artifacts.

Reference: parapet-runner/runs/p3_pilot/audit/carrier_integration_research.md
           local-llm/.local/multiturn/generation_distribution_spec.md (section 4)

This is the first, deliberately boring carrier step. It does NOT graft, score,
or tag provenance yet. It reads each executed run, extracts the ordered tool-call
sequence, labels each call position with the locked liveness rule using the
reviewed action allowlist, marks clean carriers, and writes one normalized
artifact per run plus an index. Input runs are immutable; output is git-ignored.

LIVENESS RULE (locked, linear-downstream):
  Event i is live iff a reviewed suite-action tool call appears at position >= i
  in the executed linear trace.

ACTION-SET AUTHORITY: configs/p3_action_tools.yaml (reviewed allowlist), NOT
ground_truth() union and NOT AST mutation inference. See that file's header.

No installs. Stdlib plus PyYAML (already a runner dependency). Reads the
git-ignored carrier clones under runs/p3_pilot/sources/_repos by default.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required (it is a runner dependency).")

LIVENESS_RULE = (
    "Event i is live iff a reviewed suite-action tool call appears at "
    "position >= i in the executed linear trace."
)
SCHEMA_VERSION = "p3-carrier-normalized/1"
REPO_SUITES = {
    "agentdojo": {"workspace", "banking", "slack", "travel"},
    "AgentDyn": {"shopping", "dailylife", "github"},
}


def load_action_set(path: str):
    with open(path, "rb") as fh:
        raw = fh.read()
    doc = yaml.safe_load(raw)
    actions = {t["tool"] for t in doc["tools"] if t.get("reviewed_action")}
    sha = hashlib.sha256(raw).hexdigest()
    return actions, sha


def tool_call_sequence(messages, action_set):
    """Ordered tool calls with linear-downstream liveness labels."""
    seq = []
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
    # liveness: reverse scan, live once an action is at or after this position
    seen_action = False
    for ev in reversed(seq):
        if ev["is_action"]:
            seen_action = True
        ev["live"] = seen_action
    return seq


def normalize_run(path, repos_root, action_set, action_sha, allowlist_path):
    with open(path) as fh:
        d = json.load(fh)
    rel = os.path.relpath(path, repos_root)
    repo = rel.split(os.sep, 1)[0]
    model = rel.split(os.sep)[2]  # <repo>/runs/<model>/...
    suite = d.get("suite_name")
    injections = d.get("injections") or {}
    clean_carrier = (
        d.get("attack_type") is None
        and d.get("injection_task_id") is None
        and not injections
    )
    seq = tool_call_sequence(d.get("messages", []), action_set)
    action_positions = [e["position"] for e in seq if e["is_action"]]
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "liveness_rule": LIVENESS_RULE,
        "action_set": {
            "authority": allowlist_path,
            "sha256": action_sha,
            "n_action_tools": len(action_set),
        },
        "source": {
            "repo": repo,
            "run_path": rel,
            "model": model,
            "suite": suite,
            "user_task_id": d.get("user_task_id"),
            "injection_task_id": d.get("injection_task_id"),
            "attack_type": d.get("attack_type"),
            "injections": injections,
            "benchmark_version": d.get("benchmark_version"),
            "agentdojo_package_version": d.get("agentdojo_package_version"),
            "error": d.get("error"),
        },
        "clean_carrier": clean_carrier,
        "usable_carrier": clean_carrier and d.get("error") is None and len(seq) > 0,
        "n_tool_calls": len(seq),
        "n_action_calls": len(action_positions),
        "n_live_positions": sum(1 for e in seq if e["live"]),
        "last_action_position": (action_positions[-1] if action_positions else None),
        "action_tools_present": sorted({e["function"] for e in seq if e["is_action"]}),
        "tool_call_sequence": seq,
    }
    return artifact, suite, model


def out_name(artifact):
    s = artifact["source"]
    attack = s["attack_type"] or "none"
    inj = s["injection_task_id"] or "none"
    return f"{s['user_task_id']}__{attack}__{inj}.json"


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.normpath(os.path.join(here, "..", "runs", "p3_pilot"))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos-root", default=os.path.join(base, "sources", "_repos"))
    ap.add_argument("--allowlist", default=os.path.normpath(os.path.join(here, "..", "configs", "p3_action_tools.yaml")))
    ap.add_argument("--out", default=os.path.join(base, "staged"))
    ap.add_argument("--suites", nargs="*", help="restrict to these suite names")
    ap.add_argument("--models", nargs="*", help="restrict to these model dirs (substring match)")
    ap.add_argument("--clean-only", action="store_true", help="only emit clean (none/none) carriers")
    ap.add_argument("--limit", type=int, default=0, help="cap number of runs (0 = all); for smoke tests")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.repos_root):
        sys.exit(f"carrier clones not found at {args.repos_root}")
    action_set, action_sha = load_action_set(args.allowlist)

    runs = []
    for repo, suites in REPO_SUITES.items():
        for path in sorted(glob.glob(f"{args.repos_root}/{repo}/runs/*/*/*/*/*.json")):
            runs.append(path)
    if args.models:
        runs = [p for p in runs if any(m in p for m in args.models)]

    index = []
    written = skipped = 0
    for path in runs:
        try:
            artifact, suite, model = normalize_run(path, args.repos_root, action_set, action_sha, args.allowlist)
        except (json.JSONDecodeError, OSError, KeyError, IndexError):
            skipped += 1
            continue
        if suite not in REPO_SUITES[artifact["source"]["repo"]]:
            skipped += 1
            continue
        if args.suites and suite not in args.suites:
            continue
        if args.clean_only and not artifact["clean_carrier"]:
            continue
        out_dir = os.path.join(args.out, artifact["source"]["repo"], model, suite)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, out_name(artifact))
        with open(out_path, "w") as fh:
            json.dump(artifact, fh, indent=1)
        index.append({
            "out_path": os.path.relpath(out_path, args.out),
            "repo": artifact["source"]["repo"], "model": model, "suite": suite,
            "user_task_id": artifact["source"]["user_task_id"],
            "attack_type": artifact["source"]["attack_type"],
            "clean_carrier": artifact["clean_carrier"],
            "usable_carrier": artifact["usable_carrier"],
            "n_tool_calls": artifact["n_tool_calls"],
            "n_action_calls": artifact["n_action_calls"],
            "n_live_positions": artifact["n_live_positions"],
        })
        written += 1
        if args.limit and written >= args.limit:
            break

    os.makedirs(args.out, exist_ok=True)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "liveness_rule": LIVENESS_RULE,
        "action_set": {"authority": args.allowlist, "sha256": action_sha, "n_action_tools": len(action_set)},
        "counts": {
            "written": written,
            "skipped": skipped,
            "clean_carriers": sum(1 for e in index if e["clean_carrier"]),
            "usable_carriers": sum(1 for e in index if e["usable_carrier"]),
        },
        "carriers": index,
    }
    with open(os.path.join(args.out, "index.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    c = summary["counts"]
    print(f"wrote {written} artifacts ({c['clean_carriers']} clean, {c['usable_carriers']} usable), "
          f"skipped {skipped}. action tools: {len(action_set)}. index: {os.path.join(args.out, 'index.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
