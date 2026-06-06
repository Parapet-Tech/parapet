#!/usr/bin/env python3
"""
Bootstrap generator for the P3 carrier action-tool allowlist.

Reference: parapet-runner/runs/p3_pilot/audit/carrier_integration_research.md
           local-llm/.local/multiturn/generation_distribution_spec.md (section 4, liveness)

Emits parapet-runner/configs/p3_action_tools.yaml: the authoritative set of
state-mutating "action" tools used for the carrier liveness label.

Two-part design, matching the agreed basis:
  - CANDIDATE GENERATION (mechanical, recomputed here from the carrier repos):
    for every tool actually called in the AgentDojo + AgentDyn runs, record two
    candidate signals: ground_truth() membership and AST mutation-signature.
  - FINAL INCLUSION (human review, encoded in REVIEWED_ACTIONS / NEARMISS below):
    a tool is an action iff it causes a persistent, post-call-observable change
    to environment state (see INCLUSION PRINCIPLE in the YAML header). Neither
    candidate signal is trusted on its own: ground_truth() is goal-specific
    (AgentDyn shopping/github collapse to {send_money}; other suites carry reads)
    and mutation-signature is brittle (false positive search_product/get_webpage,
    false negative delete_email/cancel_calendar_event).

BOOTSTRAP ONLY. After first generation the YAML is the hand-maintained source of
truth; edit it directly and record the reason in each entry's rationale. Rerun
this only to refresh candidate signals against updated carriers, and re-fold any
hand edits back into REVIEWED_ACTIONS / NEARMISS here.

No third-party deps, no installs. Reads the git-ignored carrier clones under
runs/p3_pilot/sources/_repos by default.
"""
import argparse
import ast
import collections
import glob
import json
import os
import sys

REPO_SUITES = {
    "agentdojo": ["workspace", "banking", "slack", "travel"],
    "AgentDyn": ["shopping", "dailylife", "github"],
}
MUT_METHODS = {
    "append", "extend", "insert", "remove", "pop", "clear",
    "update", "add", "discard", "setdefault", "popitem",
}

# --- FINAL INCLUSION (human review). Grouped by rationale. ---
REVIEWED_ACTIONS = {
    "money: moves/schedules money or completes a financial transaction": [
        "send_money", "schedule_transaction", "update_scheduled_transaction",
        "refund_order", "verify_transaction",
    ],
    "messaging: sends or posts a message visible to others": [
        "send_email", "send_channel_message", "send_direct_message", "post_webpage",
    ],
    "file: creates, modifies, or deletes stored files": [
        "create_file", "create_dir", "append_to_file", "delete_file", "delete_node",
        "copy_node", "move_node", "download_file", "download_file_through_id",
        "download_file_through_url", "git_clone", "git_pull",
    ],
    "calendar: creates or modifies a calendar event": [
        "create_calendar_event", "cancel_calendar_event", "reschedule_calendar_event",
        "add_calendar_event_participants",
    ],
    "email: deletes stored email": ["delete_email"],
    "creds/perms: changes credentials, permissions, or ownership": [
        "update_password", "update_github_account_password",
        "update_shopping_account_password", "update_user_info",
        "update_shopping_account_user_info", "share_file", "git_add_ssh_key",
        "git_delete_ssh_key", "git_invite_collaborators", "git_transfer_repo_ownership",
    ],
    "membership: changes workspace or channel membership": [
        "add_user_to_channel", "invite_user_to_slack", "remove_user_from_slack",
    ],
    "cart/order: modifies the cart or places an order": [
        "cart_add_product", "cart_remove_product", "clear_cart", "checkout_selected_cart",
    ],
    "repo: creates, modifies, or deletes repository state": [
        "git_create_repo", "git_delete_repo", "git_push", "git_issue",
        "git_star", "git_unstar",
    ],
    "reservation: makes a binding reservation": [
        "reserve_hotel", "reserve_car_rental", "reserve_restaurant",
    ],
}
ACTION_RATIONALE = {t: r for r, ts in REVIEWED_ACTIONS.items() for t in ts}

# Excluded near-misses worth keeping (most likely to be challenged later).
NEARMISS = {
    "get_webpage": "read despite incidental write: appends url to web_requests (request log), returns content",
    "get_users_in_channel": "read despite mutation-signature: builds a result list, no persistent change",
    "list_directory": "read despite mutation-signature: builds a listing, no persistent change",
    "search_product": "read/search despite mutation-signature: builds a result dict, no persistent change",
    "git_get_linked_ssh_keys": "read despite mutation-signature: returns keys, no persistent change",
    "login_shopping_account": "auth/session: sets current account, precondition not a consequential effect",
    "login_github_account": "auth/session: sets current account, precondition not a consequential effect",
    "verify_shopping_account": "auth/session: account verification, precondition not a consequential effect",
    "verify_github_account": "auth/session: account verification, precondition not a consequential effect",
    "cart_total": "compute: derives a sum, no persistent change",
}

YAML_HEADER = '''# P3 carrier action-tool allowlist (Paper 3, multi-turn injection sensing)
#
# PURPOSE: authoritative set of state-mutating "action" tools for the carrier
# liveness label. Consumed by scripts/p3_normalize_carriers.py.
#
# LIVENESS RULE (locked, linear-downstream): event i is live iff a reviewed
# suite-action tool call appears at position >= i in the executed linear trace.
#
# ACTION-SET AUTHORITY: this checked-in file. NOT ground_truth() union (it is
# goal-specific: AgentDyn shopping/github collapse to {send_money}, and other
# suites include read tools), and NOT AST mutation inference (brittle: false
# positives like search_product/get_webpage, false negatives like delete_email).
#
# CANDIDATE-GENERATION BASIS: union of ground_truth() function names and
# AST mutation-signature, over the AgentDojo + AgentDyn carriers actually used.
# FINAL INCLUSION BASIS: human review of true persistent state mutation.
#
# INCLUSION PRINCIPLE: a tool is an action iff executing it causes a persistent,
# post-call-observable change to environment state (creates/deletes/modifies
# stored data, transfers or schedules money, sends/posts a message or email,
# changes credentials/permissions/ownership/membership, makes a reservation,
# or writes to the local filesystem). Excluded: pure reads, searches, lookups,
# computed summaries, and auth/session steps (login, account verification),
# which are preconditions rather than consequential effects. Note: verify_transaction
# IS included (it completes a pending transfer); verify_*_account is not (auth).
#
# Each entry records both candidate signals so the human call is auditable
# without reopening the carrier repos. reads and mutation-signature false
# positives are kept (reviewed_action: false) because they are the entries most
# likely to be challenged later.
#
# Bootstrapped by scripts/p3_gen_action_allowlist.py; thereafter hand-maintained.
# Edit this file directly to amend a call, recording the reason in rationale.
'''


def mutation_signature(fn: ast.FunctionDef) -> bool:
    """True if the body assigns to / deletes / mutating-calls an attribute or
    subscript (a candidate signal only, deliberately imprecise)."""
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign):
            if any(isinstance(t, (ast.Attribute, ast.Subscript)) for t in n.targets):
                return True
        if isinstance(n, ast.AugAssign) and isinstance(n.target, (ast.Attribute, ast.Subscript)):
            return True
        if isinstance(n, ast.Delete):
            if any(isinstance(t, (ast.Attribute, ast.Subscript)) for t in n.targets):
                return True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in MUT_METHODS:
            return True
    return False


def scan_tool_modules(repos_root: str):
    """Return (docstring_first_line, mutation_signature) for every defined tool."""
    doc, msig = {}, {}
    for repo in REPO_SUITES:
        for path in glob.glob(f"{repos_root}/{repo}/src/agentdojo/default_suites/v1/tools/*.py"):
            try:
                tree = ast.parse(open(path).read())
            except SyntaxError:
                continue
            for n in tree.body:
                if isinstance(n, ast.FunctionDef):
                    d = (ast.get_docstring(n) or "").strip().split("\n")[0][:88]
                    if n.name not in doc or not doc[n.name]:
                        doc[n.name] = d
                    msig[n.name] = msig.get(n.name, False) or mutation_signature(n)
    return doc, msig


def ground_truth_union(repos_root: str, repo: str, suite: str):
    funcs = set()
    for fname in ("injection_tasks.py", "user_tasks.py"):
        p = f"{repos_root}/{repo}/src/agentdojo/default_suites/v1/{suite}/{fname}"
        if not os.path.exists(p):
            continue
        for n in ast.walk(ast.parse(open(p).read())):
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "FunctionCall":
                for kw in n.keywords:
                    if kw.arg == "function" and isinstance(kw.value, ast.Constant):
                        funcs.add(kw.value.value)
    return funcs


def tools_called(repos_root: str, defined: set):
    """function name -> set of suites in which it is actually called in runs."""
    called = collections.defaultdict(set)
    for repo, suites in REPO_SUITES.items():
        for f in glob.glob(f"{repos_root}/{repo}/runs/*/*/*/*/*.json"):
            try:
                d = json.load(open(f))
            except (json.JSONDecodeError, OSError):
                continue
            su = d.get("suite_name")
            if su not in suites:
                continue
            for m in d.get("messages", []):
                for tc in (m.get("tool_calls") or []):
                    fn = tc.get("function")
                    if fn in defined:
                        called[fn].add(su)
    return called


def false_rationale(tool: str) -> str:
    if tool in NEARMISS:
        return NEARMISS[tool]
    return "read/lookup: returns data, no persistent state change"


def yaml_quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    default_repos = os.path.normpath(os.path.join(here, "..", "runs", "p3_pilot", "sources", "_repos"))
    default_out = os.path.normpath(os.path.join(here, "..", "configs", "p3_action_tools.yaml"))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos-root", default=default_repos, help="dir holding the agentdojo/ and AgentDyn/ clones")
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--check", action="store_true",
                    help="verify the on-disk YAML matches the encoded review; do not write. Exit 1 on drift.")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.repos_root):
        sys.exit(f"carrier clones not found at {args.repos_root}; clone agentdojo + AgentDyn there first")

    doc, msig = scan_tool_modules(args.repos_root)
    defined = set(doc)
    gt = {(r, su): ground_truth_union(args.repos_root, r, su) for r in REPO_SUITES for su in REPO_SUITES[r]}
    called = tools_called(args.repos_root, defined)
    # include any defined tool named in a ground_truth even if never called
    for (r, su), fns in gt.items():
        for fn in fns:
            if fn in defined:
                called[fn].add(su)

    actions = set(ACTION_RATIONALE)
    missing = sorted(actions - set(called))
    if missing:
        sys.exit(f"REVIEWED_ACTIONS references tools absent from the carriers (typo?): {missing}")

    lines = [YAML_HEADER, "tools:"]
    n_action = 0
    for tool in sorted(called):
        suites = sorted(called[tool])
        in_gt = any(tool in gt.get((r, su), set()) for r in REPO_SUITES for su in suites if su in REPO_SUITES[r])
        is_action = tool in actions
        n_action += is_action
        rationale = ACTION_RATIONALE[tool] if is_action else false_rationale(tool)
        lines.append(f"  - tool: {tool}")
        lines.append(f"    suites: [{', '.join(suites)}]")
        lines.append(f"    candidate_mutation_signature: {str(msig.get(tool, False)).lower()}")
        lines.append(f"    candidate_in_ground_truth: {str(in_gt).lower()}")
        lines.append(f"    reviewed_action: {str(is_action).lower()}")
        lines.append(f"    rationale: {yaml_quote(rationale)}")
    content = "\n".join(lines) + "\n"
    summary = f"{len(called)} tools, {n_action} actions, {len(called) - n_action} reads"

    if args.check:
        existing = open(args.out).read() if os.path.exists(args.out) else None
        # ignore the header (hand-maintained wording); compare from the tools: block down
        def body(s):
            return s[s.index("tools:"):] if s and "tools:" in s else s
        if body(existing) == body(content):
            print(f"CHECK OK: {args.out} matches encoded review ({summary})")
            return 0
        print(f"CHECK DRIFT: {args.out} does not match encoded review ({summary}). Regenerate.", file=sys.stderr)
        return 1

    open(args.out, "w").write(content)
    print(f"wrote {args.out}: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
