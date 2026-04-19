"""scope-audit — orchestration wrapper over the `parapet-scope-audit` Rust CLI.

This module is orchestration only. It does NOT contain detector logic:
all sensing happens in `parapet/src/sensor/` and is invoked through the
`parapet-scope-audit` binary. The Python side owns:

- dataset enumeration (both modern staging schema and legacy
  description-embedded source schema)
- run_id construction (`<utc_stamp>__<engine_version>__<sensor_version>`)
- directory layout under `parapet-data/adjudication/review/<sensor_id>/<run_id>/`
- `summary.md` generation

It does NOT mutate canonical YAML, and it does NOT write the ledger. See
`implement/l2/data_contract.md` for the contract this module respects.

Entry point:

    python -m parapet_data scope-audit <dataset_dir> --sensor structural_heuristic

Run from the workspace root (the directory containing `parapet/` and
`parapet-data/`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

import yaml

from .filters import content_hash


SOURCE_RE = re.compile(r"source=(\S+)")
REVIEW_ROOT_REL = Path("parapet-data/adjudication/review")

BINARY_CANDIDATES = (
    "target/release/parapet-scope-audit.exe",
    "target/release/parapet-scope-audit",
    "target/debug/parapet-scope-audit.exe",
    "target/debug/parapet-scope-audit",
)


def _resolve_source(row: dict, path: Path) -> str:
    source = row.get("source")
    if not source:
        m = SOURCE_RE.search(row.get("description", "") or "")
        source = m.group(1) if m else f"missing_source__{path.stem}"
    return source


def _iter_yaml_rows(dataset_dir: Path) -> Iterator[tuple[dict, Path, int]]:
    """Walk YAMLs under dataset_dir; yield (row, path, index) for each dict row."""
    for path in sorted(dataset_dir.rglob("*.yaml")):
        if "_excluded" in path.parts:
            continue
        if path.name.startswith("staging_manifest"):
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (yaml.YAMLError, UnicodeDecodeError):
            continue
        if not isinstance(data, list):
            continue
        for idx, row in enumerate(data):
            if isinstance(row, dict):
                yield row, path, idx


def _iter_jsonl_rows(jsonl_path: Path) -> Iterator[tuple[dict, Path, int]]:
    """Resilient JSONL parse — handles embedded newlines inside content fields."""
    text = jsonl_path.read_text(encoding="utf-8")
    dec = json.JSONDecoder()
    i = 0
    idx = 0
    while i < len(text):
        while i < len(text) and text[i] in " \r\n\t":
            i += 1
        if i >= len(text):
            break
        try:
            obj, end = dec.raw_decode(text, i)
            i = end
        except json.JSONDecodeError:
            nl = text.find("\n", i)
            if nl < 0:
                break
            i = nl + 1
            continue
        if isinstance(obj, dict):
            yield obj, jsonl_path, idx
            idx += 1


def _iter_dataset_rows(path: Path) -> Iterator[tuple[dict, Path, int]]:
    """Dispatch by path type: directory -> walk YAMLs; file -> JSONL."""
    if path.is_dir():
        yield from _iter_yaml_rows(path)
    elif path.suffix.lower() in (".jsonl", ".json"):
        yield from _iter_jsonl_rows(path)
    else:
        raise ValueError(f"unsupported dataset input (expected dir or .jsonl): {path}")


def _enumerate_rows(dataset_dir: Path) -> Iterator[dict]:
    """Yield SensorInput dicts from every YAML under dataset_dir.

    Handles both schemas present in the repo:
    - modern staging: `source` field present
    - legacy / challenge: `source=XXX` embedded in `description`
    Falls back to a synthetic `missing_source__<stem>` id so distinct
    orphan files don't collapse into one bucket.
    """
    for row, path, _idx in _iter_yaml_rows(dataset_dir):
        content = row.get("content") or ""
        if not content:
            continue
        yield {
            "content": content,
            "content_hash": row.get("content_hash") or content_hash(content),
            "source_id": _resolve_source(row, path),
            "label_at_time": row.get("label"),
            "reason_at_time": row.get("reason"),
            "layer_at_time": row.get("layer"),
        }


def _enumerate_corpus_rows(origin_id: str, path: Path) -> Iterator[dict]:
    """Yield CorpusSensorInput dicts (metadata only, no content field)."""
    for row, file_path, idx in _iter_dataset_rows(path):
        content = row.get("content") or ""
        existing_hash = row.get("content_hash")
        if existing_hash:
            row_hash = existing_hash
        elif content:
            row_hash = content_hash(content)
        else:
            # No hash, no content — skip; hash_conflict has nothing to cluster on.
            continue
        rel = file_path.name if file_path == path else str(file_path.relative_to(path)).replace("\\", "/")
        yield {
            "content_hash": row_hash,
            "source_id": _resolve_source(row, file_path),
            "label_at_time": row.get("label") or row.get("true_label"),
            "reason_at_time": row.get("reason"),
            "layer_at_time": row.get("layer"),
            "origin_id": origin_id,
            "row_locator": f"{rel}#{idx}",
        }


def _discover_engine_version(parapet_crate_dir: Path) -> str:
    """Short git rev of the parapet repo (from parapet_crate_dir), else 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=parapet_crate_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        rev = result.stdout.strip()
        return rev or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def _resolve_binary(parapet_crate_dir: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).resolve()
        if not p.exists():
            raise FileNotFoundError(f"--binary not found: {p}")
        return p
    for cand in BINARY_CANDIDATES:
        p = parapet_crate_dir / cand
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        f"parapet-scope-audit binary not found under {parapet_crate_dir}/target. "
        f"Build it with: cargo build --bin parapet-scope-audit"
    )


def _sensor_info(binary: Path, sensor_id: str) -> dict:
    result = subprocess.run(
        [str(binary), "info", "--sensor", sensor_id],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"parapet-scope-audit info failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return json.loads(result.stdout)


def _run_sensor_generic(
    binary: Path,
    subcommand: str,
    sensor_id: str,
    rows: list[dict],
    timeout_s: int,
) -> tuple[str, list[dict]]:
    """Pipe rows through the named Rust CLI subcommand.

    `subcommand` is `run` (row-local `SensorInput`) or `run-corpus`
    (corpus-scope `CorpusSensorInput`). The contract is identical from
    the orchestration perspective: JSONL in, canonical observation JSONL
    out. Returns (raw_stdout, parsed_observations). Raw stdout is
    preserved so `observations.jsonl` matches the Rust
    `ObservationBatch::to_canonical_jsonl()` output byte-for-byte.
    """
    # Rust CLI emits UTF-8; Python's default pipe encoding on Windows is
    # cp1252, which breaks on any non-Latin content. Force UTF-8 in both
    # directions so content with Arabic/Russian/Chinese rows survives the
    # round-trip.
    proc = subprocess.Popen(
        [str(binary), subcommand, "--sensor", sensor_id],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdin_payload = "".join(json.dumps(row) + "\n" for row in rows)
    try:
        stdout_data, stderr_data = proc.communicate(input=stdin_payload, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise
    if proc.returncode != 0:
        raise RuntimeError(
            f"parapet-scope-audit {subcommand} failed (exit {proc.returncode}): "
            f"{stderr_data.strip()}"
        )
    parsed: list[dict] = []
    for line in stdout_data.splitlines():
        if line.strip():
            parsed.append(json.loads(line))
    return stdout_data, parsed


def _run_sensor(
    binary: Path,
    sensor_id: str,
    rows: list[dict],
    timeout_s: int,
) -> tuple[str, list[dict]]:
    """Row-local dispatch. Preserved as a thin wrapper for call-site clarity."""
    return _run_sensor_generic(binary, "run", sensor_id, rows, timeout_s)


def _run_corpus_sensor(
    binary: Path,
    sensor_id: str,
    rows: list[dict],
    timeout_s: int,
) -> tuple[str, list[dict]]:
    """Corpus-scope dispatch (hash_conflict and future CorpusObservationSensor impls)."""
    return _run_sensor_generic(binary, "run-corpus", sensor_id, rows, timeout_s)


def _render_summary(
    *,
    sensor_id: str,
    sensor_version: str,
    engine_version: str,
    run_id: str,
    dataset_dir: Path,
    rows: list[dict],
    observations: list[dict],
) -> str:
    by_source: dict[str, dict] = defaultdict(lambda: {
        "rows_scanned": 0,
        "observations": 0,
        "categories": Counter(),
        "severities": Counter(),
    })
    for row in rows:
        by_source[row["source_id"]]["rows_scanned"] += 1
    for obs in observations:
        src = obs.get("source_id") or "unknown"
        slot = by_source[src]
        slot["observations"] += 1
        slot["categories"][obs.get("category", "unknown")] += 1
        slot["severities"][obs.get("severity", "unknown")] += 1

    cat_totals = Counter(obs.get("category", "unknown") for obs in observations)
    sev_totals = Counter(obs.get("severity", "unknown") for obs in observations)

    lines: list[str] = []
    lines.append(f"# Scope Audit Summary — `{sensor_id}` {sensor_version}")
    lines.append("")
    lines.append(f"- **Run ID:** `{run_id}`")
    lines.append(f"- **Produced at (UTC):** "
                 f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"- **Engine version:** `{engine_version}`")
    lines.append(f"- **Dataset:** `{dataset_dir}`")
    lines.append(f"- **Rows scanned:** {len(rows)}")
    lines.append(f"- **Observations emitted:** {len(observations)}")
    lines.append("")

    if cat_totals:
        lines.append("## Observation totals")
        lines.append("")
        lines.append("**By category:**")
        for cat, n in cat_totals.most_common():
            lines.append(f"- `{cat}`: {n}")
        lines.append("")
        lines.append("**By severity:**")
        for sev, n in sev_totals.most_common():
            lines.append(f"- `{sev}`: {n}")
        lines.append("")

    lines.append("## Per-source breakdown")
    lines.append("")
    lines.append("| Source | Rows | Observations | Match rate | Top categories |")
    lines.append("|---|---:|---:|---:|---|")
    for src in sorted(by_source):
        s = by_source[src]
        rate = s["observations"] / s["rows_scanned"] if s["rows_scanned"] else 0.0
        top_cats = ", ".join(f"{c}: {n}" for c, n in s["categories"].most_common(3)) or "—"
        lines.append(
            f"| `{src}` | {s['rows_scanned']} | {s['observations']} | "
            f"{rate:.1%} | {top_cats} |"
        )
    lines.append("")

    return "\n".join(lines) + "\n"


def _parse_dataset_spec(spec: str) -> tuple[str, Path]:
    """Parse `origin_id:path` into (origin_id, resolved_path).

    On Windows, drive letters create a trailing colon ambiguity
    (`C:\\x` vs `staging:C:\\x`). Split on the FIRST colon only, and
    reject origin_ids that look like a single letter followed by nothing
    (almost certainly a drive letter, meaning the origin prefix is missing).
    """
    if ":" not in spec:
        raise ValueError(
            f"--dataset must be 'origin_id:path' (got {spec!r}); "
            "origin_id is required for corpus sensors."
        )
    origin_id, _, raw_path = spec.partition(":")
    if not origin_id or len(origin_id) < 2:
        raise ValueError(
            f"--dataset origin_id too short in {spec!r}; "
            "expected e.g. 'staging:<path>' not bare '<drive>:<path>'"
        )
    return origin_id, Path(raw_path).resolve()


def _run_row_local_mode(args: argparse.Namespace, binary: Path, sensor_id: str,
                        sensor_version: str, engine_version: str) -> None:
    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists():
        print(f"dataset_dir not found: {dataset_dir}", file=sys.stderr)
        sys.exit(2)

    workspace_root = Path(args.workspace_root).resolve()
    utc_stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or _build_run_id(utc_stamp, engine_version, sensor_version, args.slug)
    out_dir = workspace_root / REVIEW_ROOT_REL / sensor_id / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(_enumerate_rows(dataset_dir))
    if args.limit:
        rows = rows[: args.limit]
    print(f"[scope-audit] enumerated {len(rows)} rows from {dataset_dir}", file=sys.stderr)

    raw_jsonl, observations = _run_sensor(binary, sensor_id, rows, args.timeout)
    (out_dir / "observations.jsonl").write_text(raw_jsonl, encoding="utf-8")
    (out_dir / "summary.md").write_text(
        _render_summary(
            sensor_id=sensor_id,
            sensor_version=sensor_version,
            engine_version=engine_version,
            run_id=run_id,
            dataset_dir=dataset_dir,
            rows=rows,
            observations=observations,
        ),
        encoding="utf-8",
    )
    _print_artifact_paths(out_dir, len(observations), len(rows))


def _run_corpus_mode(args: argparse.Namespace, binary: Path, sensor_id: str,
                     sensor_version: str, engine_version: str) -> None:
    datasets = [_parse_dataset_spec(spec) for spec in args.dataset]
    for _origin, path in datasets:
        if not path.exists():
            print(f"dataset path not found: {path}", file=sys.stderr)
            sys.exit(2)

    workspace_root = Path(args.workspace_root).resolve()
    utc_stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or _build_run_id(utc_stamp, engine_version, sensor_version, args.slug)
    out_dir = workspace_root / REVIEW_ROOT_REL / sensor_id / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Enumerate across all datasets, tagging origin. Track per-dataset
    # row counts so the summary can attribute counts back to each input
    # (two datasets can share an origin_id — e.g. both challenge dirs).
    rows: list[dict] = []
    per_dataset_counts: list[int] = []
    for origin_id, path in datasets:
        count_before = len(rows)
        for row in _enumerate_corpus_rows(origin_id, path):
            rows.append(row)
        n = len(rows) - count_before
        per_dataset_counts.append(n)
        print(
            f"[scope-audit] enumerated {n} rows "
            f"from origin={origin_id} path={path}",
            file=sys.stderr,
        )
    if args.limit:
        rows = rows[: args.limit]
    print(f"[scope-audit] total corpus rows: {len(rows)}", file=sys.stderr)

    raw_jsonl, observations = _run_corpus_sensor(binary, sensor_id, rows, args.timeout)
    (out_dir / "observations.jsonl").write_text(raw_jsonl, encoding="utf-8")
    (out_dir / "summary.md").write_text(
        _render_corpus_summary(
            sensor_id=sensor_id,
            sensor_version=sensor_version,
            engine_version=engine_version,
            run_id=run_id,
            datasets=datasets,
            per_dataset_counts=per_dataset_counts,
            rows=rows,
            observations=observations,
        ),
        encoding="utf-8",
    )
    # Standalone review queue — same data as the queue section in summary.md,
    # but as a focused artifact reviewers can hand off without wading through
    # aggregate sections.
    (out_dir / "review_queue.md").write_text(
        _render_standalone_review_queue(
            sensor_id=sensor_id,
            sensor_version=sensor_version,
            run_id=run_id,
            observations=observations,
        ),
        encoding="utf-8",
    )
    _print_artifact_paths(out_dir, len(observations), len(rows))


def _build_run_id(utc_stamp: str, engine_version: str, sensor_version: str,
                  slug: str | None) -> str:
    run_id = f"{utc_stamp}__{engine_version}__{sensor_version}"
    if slug:
        run_id = f"{run_id}__{slug}"
    return run_id


def _print_artifact_paths(out_dir: Path, n_obs: int, n_rows: int) -> None:
    print(f"[scope-audit] wrote {out_dir / 'observations.jsonl'}", file=sys.stderr)
    print(f"[scope-audit] wrote {out_dir / 'summary.md'}", file=sys.stderr)
    queue_path = out_dir / "review_queue.md"
    if queue_path.exists():
        print(f"[scope-audit] wrote {queue_path}", file=sys.stderr)
    print(
        f"[scope-audit] {n_obs} observations across {n_rows} rows",
        file=sys.stderr,
    )


def cmd_scope_audit(args: argparse.Namespace) -> None:
    workspace_root = Path(args.workspace_root).resolve()
    parapet_crate_dir = workspace_root / "parapet"
    if not parapet_crate_dir.exists():
        print(f"parapet crate dir not found: {parapet_crate_dir}", file=sys.stderr)
        sys.exit(2)

    binary = _resolve_binary(parapet_crate_dir, args.binary)
    info = _sensor_info(binary, args.sensor)
    sensor_id = info["sensor_id"]
    sensor_version = info["version"]
    engine_version = args.engine_version or _discover_engine_version(parapet_crate_dir)

    # Mode dispatch: --dataset implies corpus; bare dataset_dir implies row-local.
    corpus_mode = bool(args.dataset)
    row_local_mode = bool(args.dataset_dir)
    if corpus_mode and row_local_mode:
        print("use either --dataset (corpus) or the positional dataset_dir "
              "(row-local), not both", file=sys.stderr)
        sys.exit(2)
    if not corpus_mode and not row_local_mode:
        print("missing dataset input: pass either a positional dataset_dir "
              "or --dataset origin:path (repeatable)", file=sys.stderr)
        sys.exit(2)

    if corpus_mode:
        _run_corpus_mode(args, binary, sensor_id, sensor_version, engine_version)
    else:
        _run_row_local_mode(args, binary, sensor_id, sensor_version, engine_version)


_LOCATOR_RE = re.compile(r"locator=(\S+)")
_ORIGIN_RE = re.compile(r"origin=(\S+)")
_SOURCE_DETAIL_RE = re.compile(r"source=(\S+)")
_LABEL_RE = re.compile(r"label=(\S+)")


def _cluster_tier(obs: dict) -> int:
    """Return review-priority tier (1 = highest) for a corpus observation."""
    cat = obs.get("category", "")
    sev = obs.get("severity", "")
    if cat == "label_conflict":
        return 1
    if cat == "cross_corpus_duplicate" and sev == "warn":
        return 2
    return 3


def _render_cluster_detail(obs: dict) -> str:
    """Render one corpus observation as a review-queue entry (inline)."""
    lines: list[str] = []
    h = obs.get("content_hash", "") or ""
    short = h[:12] if h else "(no hash)"
    lines.append(f"#### `{short}` · full `{h}`")
    lines.append("")
    lines.append(f"- **Category:** `{obs.get('category', 'unknown')}`")
    lines.append(f"- **Severity:** `{obs.get('severity', 'unknown')}`")
    lines.append(f"- **Message:** {obs.get('message', '')}")
    lines.append("")

    rows_detail: list[tuple[str, str, str, str]] = []
    tail_note: str | None = None
    for ev in obs.get("evidences") or []:
        if ev.get("kind") == "and_more":
            tail_note = ev.get("detail") or "additional rows omitted"
            continue
        detail = ev.get("detail") or ""
        o = _ORIGIN_RE.search(detail)
        s = _SOURCE_DETAIL_RE.search(detail)
        l = _LABEL_RE.search(detail)
        loc = _LOCATOR_RE.search(detail)
        rows_detail.append(
            (
                o.group(1) if o else "—",
                s.group(1) if s else "—",
                l.group(1) if l else "—",
                loc.group(1) if loc else "—",
            )
        )

    if rows_detail:
        lines.append("| Origin | Source | Label | Locator |")
        lines.append("|---|---|---|---|")
        for o, s, l, loc in rows_detail:
            lines.append(f"| `{o}` | `{s}` | `{l}` | `{loc}` |")
        lines.append("")

    if tail_note:
        lines.append(f"_{tail_note}_")
        lines.append("")

    return "\n".join(lines)


def _render_priority_queue(observations: list[dict], *, tier2_cap: int = 20) -> str:
    tier1 = [o for o in observations if _cluster_tier(o) == 1]
    tier2 = [o for o in observations if _cluster_tier(o) == 2]
    tier3_count = sum(1 for o in observations if _cluster_tier(o) == 3)

    lines: list[str] = []
    lines.append("## Priority Review Queue")
    lines.append("")
    lines.append(
        "Tier-1 and Tier-2 clusters are inlined here for immediate review. "
        "Tier-3+ remain aggregated in the sections below; full detail for "
        "every cluster is always in `observations.jsonl`."
    )
    lines.append("")

    lines.append(f"### Tier 1 — `label_conflict` ({len(tier1)} clusters)")
    lines.append("")
    lines.append(
        "_Same `content_hash`, conflicting `label_at_time`. Probable tagging "
        "or routing defect._"
    )
    lines.append("")
    if not tier1:
        lines.append("_None._")
        lines.append("")
    else:
        for obs in tier1:
            lines.append(_render_cluster_detail(obs))

    lines.append(
        f"### Tier 2 — `cross_corpus_duplicate` crossing train/eval boundaries "
        f"({len(tier2)} clusters)"
    )
    lines.append("")
    lines.append(
        "_Same content appears across eval/holdout and train-adjacent origins. "
        "Potential leakage or control-plane hygiene issue._"
    )
    lines.append("")
    if not tier2:
        lines.append("_None._")
        lines.append("")
    else:
        for obs in tier2[:tier2_cap]:
            lines.append(_render_cluster_detail(obs))
        remaining = len(tier2) - tier2_cap
        if remaining > 0:
            lines.append(
                f"_… and {remaining} more Tier-2 clusters not inlined; "
                "see `observations.jsonl` for the full set._"
            )
            lines.append("")

    if tier3_count:
        lines.append(f"### Tier 3+ ({tier3_count} clusters)")
        lines.append("")
        lines.append(
            f"_Info-severity cross-corpus duplicates and same-source duplicates. "
            "Not inlined. See aggregate sections below and `observations.jsonl`._"
        )
        lines.append("")

    return "\n".join(lines)


def _render_standalone_review_queue(
    *,
    sensor_id: str,
    sensor_version: str,
    run_id: str,
    observations: list[dict],
) -> str:
    """Focused review queue artifact: priority queue only, with header."""
    lines: list[str] = []
    lines.append(f"# Review Queue — `{sensor_id}` {sensor_version}")
    lines.append("")
    lines.append(f"- **Run ID:** `{run_id}`")
    lines.append(f"- **Produced at (UTC):** "
                 f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}")
    lines.append(
        "- **Scope:** Tier-1 (`label_conflict`) + Tier-2 (`cross_corpus_duplicate` "
        "crossing train/eval boundaries). Tier-3+ and full aggregates live in "
        "`summary.md`. Full raw evidence is in `observations.jsonl`."
    )
    lines.append("")
    lines.append(_render_priority_queue(observations))
    return "\n".join(lines) + "\n"


def _render_corpus_summary(
    *,
    sensor_id: str,
    sensor_version: str,
    engine_version: str,
    run_id: str,
    datasets: list[tuple[str, Path]],
    per_dataset_counts: list[int],
    rows: list[dict],
    observations: list[dict],
) -> str:
    # Aggregations.
    cat_totals: Counter = Counter(obs.get("category", "unknown") for obs in observations)
    sev_totals: Counter = Counter(obs.get("severity", "unknown") for obs in observations)

    # Cluster-size distribution: one Observation = one cluster; its non-"and_more"
    # evidence count approximates cluster size.
    def _cluster_size(obs: dict) -> int:
        evs = obs.get("evidences") or []
        non_tail = sum(1 for ev in evs if ev.get("kind") != "and_more")
        return non_tail
    cluster_sizes = Counter(_cluster_size(o) for o in observations)

    # Origin-crossing: count clusters whose evidence details reference more than one origin.
    origin_crossings: Counter = Counter()
    source_pairs: Counter = Counter()
    label_pairs: Counter = Counter()
    origin_re = re.compile(r"origin=(\S+)")
    source_re = re.compile(r"source=(\S+)")
    label_re = re.compile(r"label=(\S+)")
    for obs in observations:
        origins: set[str] = set()
        sources: set[str] = set()
        labels: set[str] = set()
        for ev in obs.get("evidences") or []:
            detail = ev.get("detail") or ""
            if m := origin_re.search(detail):
                origins.add(m.group(1))
            if m := source_re.search(detail):
                sources.add(m.group(1))
            if m := label_re.search(detail):
                labels.add(m.group(1))
        if len(origins) >= 2:
            key = " <-> ".join(sorted(origins))
            origin_crossings[key] += 1
        if len(sources) >= 2:
            key = " <-> ".join(sorted(sources))
            source_pairs[key] += 1
        if obs.get("category") == "label_conflict" and len(labels) >= 2:
            key = " <-> ".join(sorted(labels))
            label_pairs[key] += 1

    lines: list[str] = []
    lines.append(f"# Scope Audit Summary — `{sensor_id}` {sensor_version}")
    lines.append("")
    lines.append(f"- **Run ID:** `{run_id}`")
    lines.append(f"- **Produced at (UTC):** "
                 f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"- **Engine version:** `{engine_version}`")
    lines.append(f"- **Mode:** corpus (cross-row)")
    lines.append(f"- **Datasets:**")
    for (origin_id, path), n in zip(datasets, per_dataset_counts):
        lines.append(f"  - `{origin_id}`: {path} ({n} rows)")
    lines.append(f"- **Total rows scanned:** {len(rows)}")
    lines.append(f"- **Conflict clusters:** {len(observations)}")
    lines.append("")

    # Priority review queue first — reviewers should see Tier 1 + Tier 2
    # before scrolling past the aggregate breakdowns.
    if observations:
        lines.append(_render_priority_queue(observations))

    if cat_totals:
        lines.append("## Clusters by category")
        lines.append("")
        for cat, n in cat_totals.most_common():
            lines.append(f"- `{cat}`: {n}")
        lines.append("")
        lines.append("## Clusters by severity")
        lines.append("")
        for sev, n in sev_totals.most_common():
            lines.append(f"- `{sev}`: {n}")
        lines.append("")

        lines.append("## Cluster size distribution")
        lines.append("")
        lines.append("| Cluster size (rows involved) | Clusters |")
        lines.append("|---:|---:|")
        for size in sorted(cluster_sizes):
            lines.append(f"| {size} | {cluster_sizes[size]} |")
        lines.append("")

    if origin_crossings:
        lines.append("## Origin crossings")
        lines.append("")
        lines.append("| Origin pair | Clusters |")
        lines.append("|---|---:|")
        for key, n in origin_crossings.most_common(20):
            lines.append(f"| `{key}` | {n} |")
        lines.append("")

    if source_pairs:
        lines.append("## Top source pairs involved in conflicts")
        lines.append("")
        lines.append("| Source pair | Clusters |")
        lines.append("|---|---:|")
        for key, n in source_pairs.most_common(20):
            lines.append(f"| `{key}` | {n} |")
        lines.append("")

    if label_pairs:
        lines.append("## Top label conflict pairs")
        lines.append("")
        lines.append("| Labels | Clusters |")
        lines.append("|---|---:|")
        for key, n in label_pairs.most_common(20):
            lines.append(f"| `{key}` | {n} |")
        lines.append("")

    return "\n".join(lines) + "\n"


def add_subparser(subparsers) -> None:
    """Register the `scope-audit` subcommand."""
    p = subparsers.add_parser(
        "scope-audit",
        help=(
            "Run an observation sensor over a dataset; write sidecar review "
            "artifacts (never mutates canonical data, never writes the ledger)."
        ),
    )
    p.add_argument(
        "dataset_dir",
        nargs="?",
        default=None,
        help=(
            "Dataset directory for row-local sensors (walks *.yaml recursively). "
            "For corpus sensors (e.g. hash_conflict), use --dataset instead."
        ),
    )
    p.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="ORIGIN_ID:PATH",
        help=(
            "Repeatable. Corpus-mode input with explicit origin tagging: "
            "'staging:schema/eval/staging', 'challenge:path/to/challenges', "
            "'training_mix:path/to/mix.jsonl'. Required for corpus sensors."
        ),
    )
    p.add_argument(
        "--sensor",
        default="structural_heuristic",
        help="Sensor id to run (default: structural_heuristic).",
    )
    p.add_argument(
        "--workspace-root",
        default=".",
        help="Path to workspace root (parent of `parapet/` and `parapet-data/`). Default: cwd.",
    )
    p.add_argument(
        "--binary",
        default=None,
        help="Explicit path to parapet-scope-audit binary (auto-detects under target/).",
    )
    p.add_argument(
        "--engine-version",
        default=None,
        help="Override engine version stamp (default: short git rev of parapet/).",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Override full run_id (default: `<utc>__<engine>__<sensor-ver>[__<slug>]`).",
    )
    p.add_argument(
        "--slug",
        default=None,
        help="Optional slug appended to auto-generated run_id.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap enumerated rows (useful for smoke tests).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Seconds to wait for the sensor subprocess (default 600).",
    )
