"""scope-rollup — per-source consolidation of observations across sensor runs.

For a given `source_id` (e.g. `generalist_fp_benign`), this pulls observations
from every sensor's latest run under
`parapet-data/adjudication/review/<sensor_id>/<run_id>/observations.jsonl`,
filters to rows from that source, groups by `content_hash`, and produces a
reviewer-ready artifact: "here's every signal every sensor emitted about
every row from this source."

This is consumption, not generation — no sensors run, no canonical data
touched. The ledger is not written. See `implement/l2/data_contract.md`.

Entry point:

    python -m parapet_data scope-rollup --source <source_id> \
        [--workspace-root <path>] [--sensor ...]

Default behavior auto-discovers the latest run per sensor under
`parapet-data/adjudication/review/`. Pass `--sensor <id>` (repeatable)
to restrict to specific sensors.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


REVIEW_ROOT_REL = Path("parapet-data/adjudication/review")
ROLLUP_ROOT_REL = Path("parapet-data/adjudication/review/_source_rollups")

_ORIGIN_RE = re.compile(r"origin=(\S+)")
_SOURCE_RE = re.compile(r"source=(\S+)")
_LABEL_RE = re.compile(r"label=(\S+)")
_LOCATOR_RE = re.compile(r"locator=(\S+)")


def _discover_latest_runs(review_root: Path, wanted: set[str] | None) -> dict[str, Path]:
    """For each sensor subdir, pick the latest run dir (lexicographic on name).

    run_id starts with `<utc_stamp>__...` so lexicographic sort == time sort.
    Returns: {sensor_id: path_to_latest_run_dir}.
    """
    if not review_root.exists():
        return {}
    out: dict[str, Path] = {}
    for sensor_dir in sorted(review_root.iterdir()):
        if not sensor_dir.is_dir():
            continue
        if sensor_dir.name.startswith("_"):
            # Skip _source_rollups and other internal directories.
            continue
        if wanted is not None and sensor_dir.name not in wanted:
            continue
        runs = [d for d in sensor_dir.iterdir() if d.is_dir()]
        if not runs:
            continue
        # Walk newest-to-oldest; pick the first with observations.jsonl.
        # Handles the case where a newer run failed mid-sweep (dir created,
        # observations never written) — falls back to the last successful run.
        for candidate in sorted(runs, key=lambda d: d.name, reverse=True):
            if (candidate / "observations.jsonl").exists():
                out[sensor_dir.name] = candidate
                break
    return out


def _load_observations(jsonl_path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _observation_matches_source(obs: dict, source: str) -> bool:
    """An observation is about `source` if its row-local source_id matches,
    or if any evidence detail carries `source=<source>`."""
    if obs.get("source_id") == source:
        return True
    for ev in obs.get("evidences") or []:
        detail = ev.get("detail") or ""
        if m := _SOURCE_RE.search(detail):
            if m.group(1) == source:
                return True
    return False


def _observation_content_hashes(obs: dict) -> list[str]:
    """Return the content_hashes this observation speaks about.

    Row-local observations carry a single content_hash in the top field.
    Corpus observations (e.g. hash_conflict) carry the cluster hash on the
    top field AND per-row evidence with `related_content_hash`. For the
    rollup, both row-local content_hash and per-evidence related_content_hashes
    are relevant — the cluster may include rows outside the filtered source,
    but we attribute the cluster signal to each hash that matches the source.
    """
    hashes: list[str] = []
    top = obs.get("content_hash")
    if top:
        hashes.append(top)
    for ev in obs.get("evidences") or []:
        related = ev.get("related_content_hash")
        if related and related not in hashes:
            hashes.append(related)
    return hashes


def _collect_for_source(
    run_dirs: dict[str, Path], source: str
) -> tuple[dict[str, dict], dict[str, str]]:
    """Walk every sensor's run dir and pull observations about `source`.

    Returns (by_hash, sensor_versions):
    - by_hash[content_hash] -> {
          "locators": set of "<origin>:<source>:<locator>" seen,
          "labels": set of label_at_time seen,
          "sensor_hits": {sensor_id: [obs, ...]},
      }
    - sensor_versions[sensor_id] -> version string from first observation
    """
    by_hash: dict[str, dict] = defaultdict(
        lambda: {"locators": set(), "labels": set(), "sensor_hits": defaultdict(list)}
    )
    sensor_versions: dict[str, str] = {}

    for sensor_id, run_dir in run_dirs.items():
        obs_path = run_dir / "observations.jsonl"
        all_obs = _load_observations(obs_path)
        # Populate sensor version from the first observation in the file,
        # independent of source filter — a sensor with zero matches for this
        # source still has a known version.
        if all_obs:
            sensor_versions.setdefault(sensor_id, all_obs[0].get("sensor_version", "?"))
        for obs in all_obs:
            if not _observation_matches_source(obs, source):
                continue

            # Row-local: attribute to obs.content_hash.
            # Corpus: attribute to each related_content_hash whose evidence row
            # references this source.
            for ev in obs.get("evidences") or []:
                detail = ev.get("detail") or ""
                if _SOURCE_RE.search(detail) and _SOURCE_RE.search(detail).group(1) == source:
                    target_hash = ev.get("related_content_hash") or obs.get("content_hash")
                    if not target_hash:
                        continue
                    slot = by_hash[target_hash]
                    slot["sensor_hits"][sensor_id].append(obs)
                    origin = _ORIGIN_RE.search(detail)
                    label = _LABEL_RE.search(detail)
                    loc = _LOCATOR_RE.search(detail)
                    locator_str = (
                        f"{origin.group(1) if origin else '?'}"
                        f":{source}"
                        f":{loc.group(1) if loc else '?'}"
                    )
                    slot["locators"].add(locator_str)
                    if label:
                        slot["labels"].add(label.group(1))

            # For row-local observations where source_id matches directly,
            # the evidence loop above may not have populated anything (row-local
            # sensors don't emit origin/source/label in evidences). Fall back.
            if obs.get("source_id") == source:
                target_hash = obs.get("content_hash")
                if target_hash:
                    slot = by_hash[target_hash]
                    slot["sensor_hits"][sensor_id].append(obs)
                    # Row-local observations lack origin/label in evidence; we
                    # only know the source_id and the hash. The rollup still
                    # captures them under sensor_hits; locators stay empty
                    # unless a corpus sensor filled them.

    return by_hash, sensor_versions


def _row_priority(row: dict, source_kind: str) -> int:
    """Lower = higher priority. Tier-3 is source-kind-sensitive.

    On benign sources, plain structural-heuristic warn is a weak mislabel
    signal: the source is often designed to contain attack-shaped hard
    benigns (defensive system prompts with injection attempts in the user
    input; the correct end-to-end behavior is benign). Content inspection
    on 2026-04-18 showed ~85% of structural Tier-3 rows in
    generalist_fp_benign were intentional hard benigns, not mislabels.

    Stronger signals for benign-source mislabels:
    - `label_conflict` (Tier 0) — same content_hash labeled differently
      in another corpus
    - `l3_mention_delta` (Tier 1) — L3 raw-blocks but mention-masked
      allows; already a high-precision signal

    So on benign sources, Tier 3 now requires evidence from a
    cross-row sensor: hash_conflict warn (boundary-crossing duplicate).
    Structural-only warn on a benign source drops to Tier 4.

    On attack sources, the old behavior stands: structural warn is
    sensor confirmation, not a data-quality finding, but reviewers may
    still want to see it so we keep it in Tier 3.
    """
    hits = row["sensor_hits"]
    for obs_list in hits.get("hash_conflict", []):
        if obs_list.get("category") == "label_conflict":
            return 0
    if "l3_mention_delta" in hits and hits["l3_mention_delta"]:
        return 1
    has_attack_structural = any(
        o.get("category") in ("instruction_override", "unicode_smuggling",
                              "encoded_payload")
        for o in hits.get("structural_heuristic", [])
    )
    has_mention = bool(hits.get("use_vs_mention"))
    if has_attack_structural and has_mention:
        return 2

    if source_kind == "benign":
        any_hc_warn = any(
            o.get("severity") == "warn"
            for o in hits.get("hash_conflict", [])
        )
        return 3 if any_hc_warn else 4

    any_warn = any(
        o.get("severity") == "warn"
        for obs_list in hits.values() for o in obs_list
    )
    return 3 if any_warn else 4


def _source_kind(source_id: str) -> str:
    """Infer whether a source is attack-labeled or benign-labeled from name.

    Reviewer hints on attack sources differ from hints on benign sources:
    a structural attack signal on a malicious row is sensor confirmation,
    not a data-quality finding.
    """
    s = source_id.lower()
    if any(tok in s for tok in ("_attacks", "_attack", "malicious", "_injection")):
        return "malicious"
    if any(tok in s for tok in ("_benign", "_neutral", "background", "_bg")):
        return "benign"
    return "unknown"


def _reviewer_hint(row: dict, source_kind: str) -> str:
    """Heuristic English summary of the combined signal pattern."""
    hits = row["sensor_hits"]
    has_label_conflict = any(
        o.get("category") == "label_conflict"
        for o in hits.get("hash_conflict", [])
    )
    has_l3_delta = bool(hits.get("l3_mention_delta"))
    has_use_vs_mention = bool(hits.get("use_vs_mention"))
    has_attack_structural = any(
        o.get("category") in ("instruction_override", "unicode_smuggling",
                              "encoded_payload")
        for o in hits.get("structural_heuristic", [])
    )
    has_malformed = bool(hits.get("malformed_text"))

    if has_label_conflict:
        return ("Label conflict across origins. **Needs human adjudication "
                "before any ledger action.**")
    if has_attack_structural and (has_use_vs_mention or has_l3_delta):
        if source_kind == "malicious":
            return ("Attack structure with use-vs-mention framing, but row is "
                    "in an attack source. Could be an intentionally-quoted "
                    "attack (still malicious) or a mislabeled defensive row. "
                    "Inspect content before adjudicating.")
        return ("Attack structure present but also flagged as a quoted/framed "
                "mention. Likely legitimate hard benign — candidate for "
                "`keep_hard_case`.")
    if has_attack_structural and not (has_use_vs_mention or has_l3_delta):
        if source_kind == "malicious":
            return ("Structural attack signal on an attack-source row. "
                    "Sensor confirmation, not a data-quality finding. "
                    "No action unless reviewing for additional tagging.")
        if source_kind == "benign":
            return ("Structural attack signal on a benign-source row with no "
                    "use-vs-mention framing. Candidate mislabeled attack — "
                    "review for `relabel_class` or `drop`.")
        return ("Structural attack signal without mention framing. Label "
                "context uncertain; check row's source intent.")
    if has_malformed:
        return "Malformed / corrupted text. Review for ingest bug."
    return "Info-severity signal. Lower priority."


def _render_rollup(
    *,
    source: str,
    run_dirs: dict[str, Path],
    sensor_versions: dict[str, str],
    by_hash: dict[str, dict],
    rollup_id: str,
) -> str:
    source_kind = _source_kind(source)
    rows = [{"hash": h, **meta} for h, meta in by_hash.items()]
    rows.sort(key=lambda r: (_row_priority(r, source_kind), r["hash"]))

    lines: list[str] = []
    lines.append(f"# Source Rollup — `{source}`")
    lines.append("")
    lines.append(f"- **Source kind (inferred):** `{source_kind}` — "
                 "reviewer hints are tuned to this label direction.")
    lines.append(f"- **Rollup ID:** `{rollup_id}`")
    lines.append(f"- **Produced at (UTC):** "
                 f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}")
    lines.append("- **Sensors included:**")
    for sensor_id, run_dir in sorted(run_dirs.items()):
        ver = sensor_versions.get(sensor_id, "?")
        lines.append(f"  - `{sensor_id}` {ver} ({run_dir.name})")
    lines.append(f"- **Rows with ≥1 signal:** {len(rows)}")
    lines.append("")

    # Tier counts.
    tier_counter: Counter = Counter(_row_priority(r, source_kind) for r in rows)
    if tier_counter:
        lines.append("## Tier distribution")
        lines.append("")
        tier_labels = {
            0: "label_conflict (critical)",
            1: "raw_block_defanged_allow (use-vs-mention signal)",
            2: "attack structure + mention counter-signal",
            3: "warn severity (single sensor)",
            4: "info severity only",
        }
        for t in sorted(tier_counter):
            lines.append(f"- Tier {t} — {tier_labels.get(t, 'other')}: "
                         f"{tier_counter[t]} rows")
        lines.append("")

    lines.append("## Rows")
    lines.append("")
    for row in rows:
        hsh = row["hash"]
        short = hsh[:12] if hsh else "(no hash)"
        lines.append(f"### `{short}` · full `{hsh}`")
        lines.append("")
        if row["labels"]:
            lines.append(f"- **Labels seen:** {', '.join(sorted(row['labels']))}")
        if row["locators"]:
            lines.append("- **Locators:**")
            for loc in sorted(row["locators"]):
                lines.append(f"  - `{loc}`")
        lines.append("")
        lines.append("**Sensor signals:**")
        lines.append("")
        for sensor_id in sorted(run_dirs):
            obs_list = row["sensor_hits"].get(sensor_id, [])
            if not obs_list:
                lines.append(f"- `{sensor_id}`: (none)")
                continue
            cats = Counter(o.get("category", "?") for o in obs_list)
            sevs = Counter(o.get("severity", "?") for o in obs_list)
            cats_str = ", ".join(f"{c} ({n})" for c, n in cats.most_common())
            sevs_str = ", ".join(f"{s} ({n})" for s, n in sevs.most_common())
            lines.append(f"- `{sensor_id}`: categories=[{cats_str}] severities=[{sevs_str}]")
        lines.append("")
        lines.append(f"**Reviewer hint:** {_reviewer_hint(row, source_kind)}")
        lines.append("")

    return "\n".join(lines) + "\n"


def _render_rollup_jsonl(by_hash: dict[str, dict]) -> str:
    out_lines: list[str] = []
    for hsh in sorted(by_hash):
        meta = by_hash[hsh]
        flat = {
            "content_hash": hsh,
            "labels_seen": sorted(meta["labels"]),
            "locators": sorted(meta["locators"]),
            "sensor_categories": {
                sensor_id: sorted({o.get("category", "?") for o in obs_list})
                for sensor_id, obs_list in meta["sensor_hits"].items()
            },
            "sensor_severities": {
                sensor_id: sorted({o.get("severity", "?") for o in obs_list})
                for sensor_id, obs_list in meta["sensor_hits"].items()
            },
        }
        out_lines.append(json.dumps(flat, sort_keys=True))
    return "\n".join(out_lines) + ("\n" if out_lines else "")


def cmd_scope_rollup(args: argparse.Namespace) -> None:
    workspace_root = Path(args.workspace_root).resolve()
    review_root = workspace_root / REVIEW_ROOT_REL
    if not review_root.exists():
        print(f"review root not found: {review_root}", file=sys.stderr)
        sys.exit(2)

    wanted = set(args.sensor) if args.sensor else None
    run_dirs = _discover_latest_runs(review_root, wanted)
    if not run_dirs:
        print("no sensor runs found under "
              f"{review_root}", file=sys.stderr)
        sys.exit(2)

    print(f"[scope-rollup] using runs:", file=sys.stderr)
    for sensor_id, path in sorted(run_dirs.items()):
        print(f"  {sensor_id}: {path.name}", file=sys.stderr)

    by_hash, sensor_versions = _collect_for_source(run_dirs, args.source)
    print(f"[scope-rollup] collected {len(by_hash)} rows with signals for "
          f"source={args.source}", file=sys.stderr)

    utc_stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rollup_id = args.rollup_id or f"{utc_stamp}__{args.source}"
    out_dir = workspace_root / ROLLUP_ROOT_REL / args.source / rollup_id
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "rollup.md").write_text(
        _render_rollup(
            source=args.source,
            run_dirs=run_dirs,
            sensor_versions=sensor_versions,
            by_hash=by_hash,
            rollup_id=rollup_id,
        ),
        encoding="utf-8",
    )
    (out_dir / "rollup.jsonl").write_text(
        _render_rollup_jsonl(by_hash),
        encoding="utf-8",
    )
    print(f"[scope-rollup] wrote {out_dir / 'rollup.md'}", file=sys.stderr)
    print(f"[scope-rollup] wrote {out_dir / 'rollup.jsonl'}", file=sys.stderr)


def add_subparser(subparsers) -> None:
    p = subparsers.add_parser(
        "scope-rollup",
        help=(
            "Consolidate observations from every sensor's latest run into a "
            "per-source reviewer artifact."
        ),
    )
    p.add_argument(
        "--source",
        required=True,
        help="Upstream source id to filter on (e.g. generalist_fp_benign).",
    )
    p.add_argument(
        "--workspace-root",
        default=".",
        help="Path to workspace root (default: cwd).",
    )
    p.add_argument(
        "--sensor",
        action="append",
        default=[],
        help="Repeatable. Restrict to these sensor ids; default: all available.",
    )
    p.add_argument(
        "--rollup-id",
        default=None,
        help="Override rollup id (default: <utc>__<source>).",
    )
