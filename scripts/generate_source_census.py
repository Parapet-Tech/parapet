"""Phase 2 source census generator.

Enumerates every data source across:
- `schema/eval/staging/*.yaml` (canonical ledger-governed corpus)
- `schema/eval/challenges/**/*.yaml` (challenge/eval corpus)
- a training-mix JSONL (default: tinybert_mix_v1)

Emits:
- `implement/l2/source_census.yaml` — machine-readable; Phase 4 tools read this
- `implement/l2/source_census.md` — human-readable summary table

Behavior:
- Auto-populated columns (row counts, file lists, label/language distributions)
  are refreshed on every run.
- Judgment columns (`license_terms`, `assumed_scope`, `known_quirks`,
  `audit_status`) are preserved from any existing `source_census.yaml` so
  reviewer input survives regenerations.
- If no existing census exists, seeds `audit_status` defaults based on work
  completed in prior sessions (jbb_paraphrase_* = remediated;
  wildguardmix_benign / generalist_fp_benign = sampled).

Run:
    cd parapet
    python scripts/generate_source_census.py \
        [--training-mix parapet-runner/runs/tinybert_mix_v1/tinybert_training_mix.jsonl]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml


BASE = Path(__file__).resolve().parent.parent  # parapet/

STAGING_DIR = BASE / "schema/eval/staging"
CHALLENGES_DIR = BASE / "schema/eval/challenges"
OUT_YAML = BASE / "implement/l2/source_census.yaml"
OUT_MD = BASE / "implement/l2/source_census.md"

SOURCE_RE = re.compile(r"source=(\S+)")

# Sources we've already touched in prior sessions. Seeds audit_status when
# no existing census file is present.
AUDIT_SEEDS = {
    "jbb_paraphrase_attacks": "remediated",
    "jbb_paraphrase_benign": "remediated",
    "wildguardmix_benign": "sampled",
    "generalist_fp_benign": "sampled",
}

# Files under staging/ that aren't source data.
STAGING_SKIP = {"staging_rejected.jsonl", "nul"}


def _new_slot() -> dict:
    return {
        "files": [],
        "rows": 0,
        "labels": Counter(),
        "languages": Counter(),
        "reasons": Counter(),
    }


def _synthetic_missing_source(path: Path, prefix: str) -> str:
    """Stable fallback source id when row-level provenance is missing."""
    return f"{prefix}__{path.stem}"


def enumerate_staging() -> dict[str, dict]:
    sources: dict[str, dict] = defaultdict(_new_slot)
    if not STAGING_DIR.exists():
        return sources
    for path in sorted(STAGING_DIR.glob("*.yaml")):
        if path.name.startswith("staging_manifest") or path.name in STAGING_SKIP:
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (yaml.YAMLError, UnicodeDecodeError):
            continue
        if not isinstance(data, list):
            continue
        for row in data:
            if not isinstance(row, dict):
                continue
            # Two staging schemas exist: modern staged rows with `source` field,
            # and legacy challenge-style rows with `source=XXX` embedded in
            # `description`. Fall through the options in order.
            src = row.get("source")
            if not src:
                m = SOURCE_RE.search(row.get("description", "") or "")
                if m:
                    src = m.group(1)
            if not src:
                src = _synthetic_missing_source(path, "missing_source")
            slot = sources[src]
            if path.name not in slot["files"]:
                slot["files"].append(path.name)
            slot["rows"] += 1
            if row.get("label"):
                slot["labels"][row["label"]] += 1
            if row.get("language"):
                slot["languages"][row["language"]] += 1
            if row.get("reason"):
                slot["reasons"][row["reason"]] += 1
    return sources


def enumerate_challenges() -> dict[str, dict]:
    sources: dict[str, dict] = defaultdict(_new_slot)
    if not CHALLENGES_DIR.exists():
        return sources
    for path in CHALLENGES_DIR.rglob("*.yaml"):
        # Skip sidecar audit receipts for excluded rows.
        if "_excluded" in path.parts:
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (yaml.YAMLError, UnicodeDecodeError):
            continue
        if not isinstance(data, list):
            continue
        for row in data:
            if not isinstance(row, dict):
                continue
            m = SOURCE_RE.search(row.get("description", "") or "")
            src = m.group(1) if m else _synthetic_missing_source(path, "missing_source")
            slot = sources[src]
            rel = str(path.relative_to(CHALLENGES_DIR)).replace("\\", "/")
            if rel not in slot["files"]:
                slot["files"].append(rel)
            slot["rows"] += 1
            if row.get("label"):
                slot["labels"][row["label"]] += 1
    return sources


def enumerate_training_mix(path: Path) -> dict[str, dict]:
    """Resilient JSONL parse — handles embedded newlines in content fields."""
    sources: dict[str, dict] = defaultdict(_new_slot)
    if not path.exists():
        return sources
    text = path.read_text(encoding="utf-8")
    dec = json.JSONDecoder()
    i = 0
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
        src = obj.get("source") or _synthetic_missing_source(path, "missing_source")
        slot = sources[src]
        slot["rows"] += 1
        label = obj.get("true_label") or obj.get("label")
        if label:
            slot["labels"][label] += 1
    return sources


def load_existing_census(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
    return {s["name"]: s for s in data.get("sources", []) if "name" in s}


def build_census(training_mix_path: Path) -> list[dict]:
    staging = enumerate_staging()
    challenges = enumerate_challenges()
    training_mix = enumerate_training_mix(training_mix_path)
    existing = load_existing_census(OUT_YAML)

    all_sources = set(staging) | set(challenges) | set(training_mix)
    rows: list[dict] = []

    for src in sorted(all_sources):
        s = staging.get(src, _new_slot())
        c = challenges.get(src, _new_slot())
        t = training_mix.get(src, _new_slot())
        prev = existing.get(src, {})

        # Judgment fields: preserve prior value if present; else default.
        def _prev(field: str, default):
            val = prev.get(field)
            if val in (None, "", "unknown", "unaudited"):
                return default
            return val

        default_status = AUDIT_SEEDS.get(src, "unaudited")

        row = {
            "name": src,
            "staging_files": sorted(s["files"]),
            "staging_rows": s["rows"],
            "staging_labels": dict(s["labels"]),
            "staging_languages": dict(s["languages"]),
            "staging_reasons_top5": dict(s["reasons"].most_common(5)),
            "challenge_files": sorted(c["files"]),
            "challenge_rows": c["rows"],
            "challenge_labels": dict(c["labels"]),
            "training_mix_rows": t["rows"],
            "training_mix_labels": dict(t["labels"]),
            "total_rows": s["rows"] + c["rows"] + t["rows"],
            # Judgment fields (preserved across regenerations):
            "license_terms": _prev("license_terms", "unknown"),
            "assumed_scope": _prev("assumed_scope", "unknown"),
            "known_quirks": _prev("known_quirks", ""),
            "audit_status": _prev("audit_status", default_status),
        }
        rows.append(row)

    return rows


def render_markdown(rows: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Source Census")
    lines.append("")
    lines.append(
        "**Phase 2 exit artifact** for the "
        "[Stable Data Foundation epic](injest/data-guage.md). "
        "Auto-generated by `scripts/generate_source_census.py`."
    )
    lines.append("")
    lines.append(
        "Auto-derived columns (`staging_rows`, `challenge_rows`, "
        "`training_mix_rows`) are refreshed on every run. Judgment columns "
        "(`assumed_scope`, `known_quirks`, `license_terms`, `audit_status`) "
        "are hand-maintained in `source_census.yaml` and preserved across "
        "regenerations."
    )
    lines.append("")

    # Summary counts.
    total = len(rows)
    status_counts = Counter(r["audit_status"] for r in rows)
    lines.append(f"**Total sources:** {total}")
    lines.append("")
    lines.append("**Audit status:**")
    for status, n in sorted(status_counts.items()):
        lines.append(f"- `{status}`: {n}")
    lines.append("")

    # Main table, sorted by total_rows desc for quick scan.
    lines.append("## Sources by total row count")
    lines.append("")
    lines.append(
        "| Source | Staging | Challenge | Train-mix | Total | Audit status | Assumed scope |"
    )
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for r in sorted(rows, key=lambda x: (-x["total_rows"], x["name"])):
        lines.append(
            f"| `{r['name']}` | {r['staging_rows']} | {r['challenge_rows']} "
            f"| {r['training_mix_rows']} | {r['total_rows']} "
            f"| `{r['audit_status']}` | `{r['assumed_scope']}` |"
        )
    lines.append("")
    lines.append(
        "Full per-source detail — file lists, label distributions, languages, "
        "reasons — lives in `source_census.yaml`."
    )
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--training-mix",
        default="parapet-runner/runs/tinybert_mix_v1/tinybert_training_mix.jsonl",
        help="Path (relative to parapet/) to the training-mix JSONL.",
    )
    args = ap.parse_args()

    training_mix = (BASE / args.training_mix).resolve()
    rows = build_census(training_mix)

    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    OUT_YAML.write_text(
        yaml.safe_dump(
            {"sources": rows},
            sort_keys=False,
            width=10_000,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    OUT_MD.write_text(render_markdown(rows), encoding="utf-8")

    print(f"Wrote {OUT_YAML.relative_to(BASE)} ({len(rows)} sources)")
    print(f"Wrote {OUT_MD.relative_to(BASE)}")
    # Quick status distribution.
    status_counts = Counter(r["audit_status"] for r in rows)
    print("Audit status:", dict(status_counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
