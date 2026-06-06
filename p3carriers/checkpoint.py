"""Verify staged P3 carrier artifacts faithfully preserve source carrier geometry.

A verification tool, NOT a transformer. For each staged artifact it re-reads the
carrier source run, re-derives the normalized artifact with the CURRENT allowlist
via p3carriers.normalize, and asserts the staged artifact still matches on the
geometry the liveness label depends on (tool-call sequence, action/live labels,
derived counts) plus its source metadata.

Why this exists: liveness ground-truth is a function of carrier control-flow. A
silent normalization bug or an allowlist drift would corrupt it with no downstream
error (verification_design.md P3: catch errors at origin, not five stages later at
the cross-tab). This is that origin check.

Two failure dispositions are kept distinct, because the fix differs:
- GEOMETRY mismatch (seq / scalar / source-meta): the staged artifact disagrees
  with its source. Real corruption -> re-normalize.
- SHA-ONLY mismatch: the artifact's recorded allowlist sha differs from the current
  allowlist but ALL geometry still matches -> the allowlist drifted cosmetically
  (e.g. a comment edit) with no membership change; liveness is intact and only the
  provenance pointer is stale -> re-stamp, do not re-normalize.

Never emits carrier payloads (injection text). Failure output is reason codes and
paths only. Logic lives here (tested); scripts/p3_checkpoint_staged.py is a thin CLI.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any

from p3carriers import schemas
from p3carriers.action_allowlist import default_allowlist_path, load_action_set
from p3carriers.normalize import _path_fields, normalize_run

# Per-event fields the liveness label is built from; equality of these is the core
# guarantee (a single wrong is_action/live silently mislabels a carrier).
SEQ_FIELDS = ("position", "message_index", "function", "call_id", "is_action", "live")

# Derived geometry that carrier selection (graft-readiness) depends on.
SCALAR_FIELDS = (
    "n_tool_calls", "n_action_calls", "n_live_positions",
    "last_action_position", "action_tools_present",
    "clean_carrier", "usable_carrier",
)

# Source metadata tying the artifact to its carrier. injections is compared but its
# value is NEVER emitted (it holds attack payload text).
SOURCE_META_FIELDS = (
    "repo", "model", "suite", "user_task_id",
    "injection_task_id", "attack_type", "error", "injections",
)

SHA_REASON = "action_sha_mismatch"


@dataclass
class CheckResult:
    status: str  # "ok" | "failed" | "skipped"
    reasons: list[str] = field(default_factory=list)

    @property
    def geometry_failed(self) -> bool:
        """True if any reason is a real geometry/metadata mismatch (not just stale sha)."""
        return any(r != SHA_REASON for r in self.reasons)


def _compare(staged: dict[str, Any], recomputed: dict[str, Any]) -> list[str]:
    """Reason codes for every disagreement. Structural only; no payload values."""
    reasons: list[str] = []

    if staged.get("liveness_rule") != schemas.LIVENESS_RULE:
        reasons.append("liveness_rule_mismatch")
    if staged.get("action_set", {}).get("sha256") != recomputed["action_set"]["sha256"]:
        reasons.append(SHA_REASON)

    s_seq = staged.get("tool_call_sequence", [])
    r_seq = recomputed["tool_call_sequence"]
    if len(s_seq) != len(r_seq):
        reasons.append(f"seq_length_mismatch:staged={len(s_seq)},recomputed={len(r_seq)}")
    else:
        for i, (a, b) in enumerate(zip(s_seq, r_seq)):
            for f in SEQ_FIELDS:
                if a.get(f) != b.get(f):
                    reasons.append(f"seq_field_mismatch:{f}@{i}")

    for f in SCALAR_FIELDS:
        if staged.get(f) != recomputed[f]:
            reasons.append(f"scalar_mismatch:{f}")

    s_src = staged.get("source", {})
    r_src = recomputed["source"]
    for f in SOURCE_META_FIELDS:
        if s_src.get(f) != r_src.get(f):
            reasons.append(f"source_meta_mismatch:{f}")  # value withheld (may be payload)

    return reasons


def check_artifact(
    staged: dict[str, Any],
    *,
    repos_root: str,
    action_set: set[str],
    action_sha: str,
    allowlist_path: str,
) -> CheckResult:
    """Re-derive one staged artifact from its source carrier and compare."""
    if staged.get("schema_version") != schemas.SCHEMA_VERSION:
        # A different schema is not comparable with current logic; surface, do not guess.
        return CheckResult("skipped", ["schema_version_skip"])

    run_path = staged.get("source", {}).get("run_path")
    if not run_path:
        return CheckResult("failed", ["source_run_path_absent"])

    abs_path = os.path.join(repos_root, run_path)
    if not os.path.isfile(abs_path):
        # Explicit choice: an unverifiable artifact is a FAILURE, never a silent skip.
        return CheckResult("failed", ["source_missing"])
    try:
        with open(abs_path) as fh:
            run = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return CheckResult("failed", ["source_unreadable"])

    try:
        rel, repo, model = _path_fields(abs_path, repos_root)
        recomputed = normalize_run(
            run, rel_path=rel, repo=repo, model=model,
            action_set=action_set, action_sha=action_sha, allowlist_path=allowlist_path,
        )
    except (KeyError, IndexError):
        return CheckResult("failed", ["recompute_error"])

    reasons = _compare(staged, recomputed)
    return CheckResult("failed" if reasons else "ok", reasons)


def iter_staged_paths(staged_dir: str):
    """Yield every staged artifact path (skips the index)."""
    for root, _, files in os.walk(staged_dir):
        for fn in sorted(files):
            if fn.endswith(".json") and fn != "index.json":
                yield os.path.join(root, fn)


def run_checkpoint(
    staged_dir: str,
    repos_root: str,
    allowlist_path: str,
    *,
    limit: int = 0,
) -> dict[str, Any]:
    """Check staged artifacts against their source carriers; return a summary dict."""
    action_set, action_sha = load_action_set(allowlist_path)

    checked = failed = skipped = geometry_failed = 0
    failures: list[tuple[str, list[str]]] = []
    staging_shas: set[str] = set()

    for path in iter_staged_paths(staged_dir):
        try:
            with open(path) as fh:
                staged = json.load(fh)
        except (json.JSONDecodeError, OSError):
            checked += 1
            failed += 1
            geometry_failed += 1
            failures.append((path, ["staged_unreadable"]))
            if limit and checked >= limit:
                break
            continue

        sha = staged.get("action_set", {}).get("sha256")
        if sha:
            staging_shas.add(sha)

        res = check_artifact(
            staged, repos_root=repos_root, action_set=action_set,
            action_sha=action_sha, allowlist_path=allowlist_path,
        )
        if res.status == "skipped":
            skipped += 1
            continue

        checked += 1
        if res.status == "failed":
            failed += 1
            if res.geometry_failed:
                geometry_failed += 1
            failures.append((path, res.reasons))
        if limit and checked >= limit:
            break

    return {
        "checked": checked,
        "ok": checked - failed,
        "failed": failed,
        "geometry_failed": geometry_failed,
        "sha_only_failed": failed - geometry_failed,
        "skipped": skipped,
        "current_sha": action_sha,
        "staging_shas": sorted(staging_shas),
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Verify staged P3 carrier artifacts faithfully preserve source carrier geometry."
    )
    ap.add_argument("--staged", default=schemas.default_staged_out())
    ap.add_argument("--repos-root", default=schemas.default_repos_root())
    ap.add_argument("--allowlist", default=default_allowlist_path())
    ap.add_argument("--limit", type=int, default=0, help="cap artifacts checked (0 = all)")
    ap.add_argument("--show", type=int, default=10, help="how many failing artifacts to list")
    ap.add_argument("--strict-sha", action="store_true",
                    help="also fail (nonzero exit) on sha-only staleness, not just geometry")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.staged):
        print(f"staged dir not found at {args.staged}")
        return 2
    if not os.path.isdir(args.repos_root):
        print(f"carrier clones not found at {args.repos_root}")
        return 2

    s = run_checkpoint(args.staged, args.repos_root, args.allowlist, limit=args.limit)

    print(
        f"checked {s['checked']} ({s['ok']} ok, {s['failed']} failed: "
        f"{s['geometry_failed']} geometry, {s['sha_only_failed']} sha-only), skipped {s['skipped']}."
    )
    cur = s["current_sha"]
    staged_shas = s["staging_shas"]
    note = "match" if staged_shas == [cur] else "MISMATCH"
    print(f"allowlist: current {cur[:12]}; staged pool {[h[:12] for h in staged_shas]} ({note})")

    if s["sha_only_failed"] and not args.strict_sha:
        print(f"  note: {s['sha_only_failed']} artifacts are sha-stale only (geometry intact); "
              f"re-stamp advised. Pass --strict-sha to gate on this.")

    for path, reasons in s["failures"][:args.show]:
        rel = os.path.relpath(path, args.staged)
        print(f"  FAIL {rel}: {','.join(reasons)}")
    extra = len(s["failures"]) - args.show
    if extra > 0:
        print(f"  ... and {extra} more failing artifacts")

    bad = s["failed"] if args.strict_sha else s["geometry_failed"]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
