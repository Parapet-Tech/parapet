"""Diagnostic: legacy eval YAML leakage audit.

NOT a Phase 1 deliverable. One-shot diagnostic tool that quantifies how
legacy schema/eval/{l1_holdout,challenges/...} YAMLs overlap with v8
train. Those YAMLs predate the JSONL curation pipeline and are NOT
acceptance gates for L2. The canonical L2 acceptance gate (when defined)
is checked by phase1_v8_residual_export.py via --canonical-eval.

What it does:
    1. Loads v8 train.yaml -> content_hash index.
    2. Resolves target YAMLs: explicit --target paths, OR autodiscover
       schema/eval/l1_holdout.yaml + schema/eval/challenges/**/*.yaml
       (skipping any path under a `_*`-prefixed dir).
    3. For each target, computes overlap and joins back to v8 train
       rows + target rows. Categorizes pairs as
       exact_duplicate_label_source / same_label_diff_source /
       cross_label_conflict.
    4. Aggregates by target_path × {train_source, train_reason,
       train_label, provenance_lane}.
    5. Writes leakage_check.json + leakage_attribution.{json,md} + a
       README under --output-dir.

Outputs (under --output-dir, default runs/phase1_v8_residuals/legacy_audit/):
    README.md                      diagnostic-scope marker
    leakage_check.json             per-target overlap counts
    leakage_attribution.json       structured detail + aggregations
    leakage_attribution.md         human-readable summary

Provenance taxonomy (from mirror_v8 spec):
    - heuristic_staged    preclassified by heuristic at staging time
    - source_label        label trusted from source dataset
    - manual_map          handcrafted attack source
    - adjudicated         manually reviewed staged source
    - background          benign background source
    - discussion_benign   hard-negative benign (renamed)
    - mirror              base mirror (no reason_provenance set)
    - unknown             not found in spec

Usage:
    cd parapet/parapet-runner
    python scripts/audit_legacy_eval_leakage.py \
        --curation-dir ../parapet-data/curated/v8_experiment \
        --spec ../parapet-data/specs/mirror_v8.compact.yaml \
        --schema-eval-dir ../schema/eval \
        --output-dir runs/phase1_v8_residuals/legacy_audit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def load_yaml_list(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected YAML list, got {type(raw).__name__}")
    return raw


# ---------------------------------------------------------------------------
# Description parsing for target YAMLs
# ---------------------------------------------------------------------------


_SOURCE_KV_RE = re.compile(r"\bsource=([^\s|]+)")
_HOLDOUT_RE = re.compile(r"^holdout:\s*([^:]+?)\s*(?::|$)")


def parse_target_description(description: str | None, target_path: Path) -> dict[str, str | None]:
    """Best-effort parse of target row description for source/reason hints.

    Returns {target_source: str|None, target_reason: str|None, parser: str}.
    """
    if not isinstance(description, str):
        return {"target_source": None, "target_reason": None, "parser": "none"}

    name = target_path.name
    # tough_attack_v1 / tough_attack_v2 / tough_neutral_v1 / tough_neutral_v2:
    # description contains "source=<name>"
    m = _SOURCE_KV_RE.search(description)
    if m:
        return {"target_source": m.group(1), "target_reason": None, "parser": "source_kv"}

    # l1_holdout: description starts with "holdout: <source>: ..."
    m = _HOLDOUT_RE.match(description)
    if m:
        return {"target_source": m.group(1).strip(), "target_reason": None, "parser": "holdout_prefix"}

    return {"target_source": None, "target_reason": None, "parser": "unparsed"}


# ---------------------------------------------------------------------------
# Provenance lookup from mirror_v8 spec
# ---------------------------------------------------------------------------


def build_provenance_index(spec_path: Path) -> dict[str, dict[str, str | None]]:
    """Map source name -> {route_policy, reason_provenance, lane}.

    Walks the v8 spec for every named source ref and records which lane and
    routing policy applies. Missing entries come back from .get() as 'unknown'.
    """
    if not spec_path.exists():
        return {}
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        return {}
    index: dict[str, dict[str, str | None]] = {}

    def _record(refs: Iterable[Any], lane: str) -> None:
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            name = ref.get("name")
            if not isinstance(name, str):
                continue
            index[name] = {
                "lane": lane,
                "route_policy": ref.get("route_policy"),
                "reason_provenance": ref.get("reason_provenance"),
            }

    for lane, key in [
        ("background", "background"),
        ("discussion_benign", "discussion_benign"),
        ("base_attack_sources", None),
        ("base_benign_sources", None),
    ]:
        if key is None:
            section = spec.get(lane)
            if isinstance(section, list):
                _record(section, lane)
        else:
            section = spec.get(lane, {})
            if isinstance(section, dict):
                _record(section.get("sources") or [], lane)

    # staged_attacks, staged_benign_multilingual: keyed dicts; the dict key
    # itself is the source name, but the inner record holds route_policy /
    # reason_provenance.
    for section_key in ("staged_attacks", "staged_benign_multilingual"):
        section = spec.get(section_key)
        if isinstance(section, dict):
            for name, ref in section.items():
                if isinstance(ref, dict):
                    index[name] = {
                        "lane": section_key,
                        "route_policy": ref.get("route_policy"),
                        "reason_provenance": ref.get("reason_provenance"),
                    }

    # staged_benign_en: nested datasets
    section = spec.get("staged_benign_en")
    if isinstance(section, dict):
        for name, ref in (section.get("datasets") or {}).items():
            if isinstance(ref, dict):
                index[name] = {
                    "lane": "staged_benign_en",
                    "route_policy": ref.get("route_policy"),
                    "reason_provenance": None,
                }

    # cells: extra_attack_sources, extra_benign_sources
    cells = spec.get("cells")
    if isinstance(cells, dict):
        for cell_name, cell in cells.items():
            if not isinstance(cell, dict):
                continue
            for sub_key in ("extra_attack_sources", "extra_benign_sources"):
                _record(cell.get(sub_key) or [], f"cells.{cell_name}.{sub_key}")

    # supplements: nested attack_sources / benign_sources
    supplements = spec.get("supplements")
    if isinstance(supplements, list):
        for supp in supplements:
            if not isinstance(supp, dict):
                continue
            sname = supp.get("name", "supplement")
            _record(supp.get("attack_sources") or [], f"supplements.{sname}.attack_sources")
            _record(supp.get("benign_sources") or [], f"supplements.{sname}.benign_sources")

    return index


def provenance_for_source(index: dict[str, dict[str, Any]], src: str | None) -> dict[str, Any]:
    """Look up provenance metadata; return defaults if not found."""
    if not isinstance(src, str) or src not in index:
        return {"lane": "unknown", "route_policy": None, "reason_provenance": None}
    return dict(index[src])


# ---------------------------------------------------------------------------
# Train index
# ---------------------------------------------------------------------------


def build_train_index(train_yaml: Path) -> dict[str, list[dict[str, Any]]]:
    """content_hash -> list of v8 train rows (handles duplicates)."""
    rows = load_yaml_list(train_yaml)
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        c = r.get("content")
        if isinstance(c, str):
            index[content_hash(c)].append(r)
    return dict(index)


# ---------------------------------------------------------------------------
# Pair categorization
# ---------------------------------------------------------------------------


def categorize_pair(
    train_row: dict[str, Any],
    target_row: dict[str, Any],
    target_meta: dict[str, str | None],
) -> str:
    """Return one of:
        'exact_duplicate_label_source'   train.label == target.label AND
                                         target_source matches train.source
        'same_label_diff_source'         labels match, sources differ (or
                                         target_source unparsed)
        'cross_label_conflict'           labels differ
    """
    train_label = train_row.get("label")
    target_label = target_row.get("label")
    if train_label != target_label:
        return "cross_label_conflict"
    train_source = train_row.get("source")
    target_source = target_meta.get("target_source")
    if (
        isinstance(train_source, str) and isinstance(target_source, str)
        and train_source == target_source
    ):
        return "exact_duplicate_label_source"
    return "same_label_diff_source"


# ---------------------------------------------------------------------------
# Target discovery
# ---------------------------------------------------------------------------


_README_TEXT = """# Legacy eval YAML leakage audit

This directory holds a diagnostic audit of legacy `schema/eval/` YAMLs:

- `schema/eval/l1_holdout.yaml`
- `schema/eval/challenges/<...>/<...>.yaml`

These predate the JSONL curation pipeline. They are NOT acceptance gates
for L2.

The Phase 1 v8 residual artifact at the parent directory is usable for L2
training, the truncation-policy spike, and mechanical-detector simulation
regardless of the contents of this audit. Phase 4 acceptance must be
measured against a clean, content-hash-disjoint eval produced by the
JSONL curation pipeline (when defined; phase1_v8_residual_export.py
gates that via --canonical-eval).

This audit was preserved because the attribution data is useful evidence
for understanding how the legacy data overlapped with v8 training, but
it does not block any pipeline.
"""


def discover_legacy_eval_yamls(schema_eval_dir: Path) -> list[Path]:
    """Autodiscover legacy eval YAMLs:
        - schema/eval/l1_holdout.yaml (if present)
        - schema/eval/challenges/**/*.yaml (skip any `_*`-prefixed dir)
    """
    found: list[Path] = []
    holdout = schema_eval_dir / "l1_holdout.yaml"
    if holdout.exists():
        found.append(holdout)
    challenges = schema_eval_dir / "challenges"
    if challenges.is_dir():
        for p in challenges.rglob("*.yaml"):
            rel_parts = p.relative_to(challenges).parts[:-1]
            if any(part.startswith("_") for part in rel_parts):
                continue
            found.append(p)
    return sorted(found)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnostic: legacy eval YAML leakage audit (NOT a Phase 1 gate)."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/phase1_v8_residuals/legacy_audit"),
        help="Where leakage_check + leakage_attribution + README land.",
    )
    parser.add_argument(
        "--curation-dir", type=Path, default=Path("../parapet-data/curated/v8_experiment"),
    )
    parser.add_argument(
        "--spec", type=Path, default=Path("../parapet-data/specs/mirror_v8.compact.yaml"),
    )
    parser.add_argument(
        "--schema-eval-dir", type=Path, default=Path("../schema/eval"),
        help="Root for legacy YAML autodiscovery.",
    )
    parser.add_argument(
        "--target", type=Path, action="append", default=None,
        help="Explicit target YAML (repeatable). Overrides autodiscovery if set.",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_yaml = args.curation_dir / "train.yaml"
    if not train_yaml.exists():
        print(f"ERROR: {train_yaml} not found.", file=sys.stderr)
        return 1

    print(f"Loading v8 train index from {train_yaml} ...")
    train_index = build_train_index(train_yaml)
    print(f"  {len(train_index):,} unique train hashes")

    provenance_index = build_provenance_index(args.spec)
    print(f"Loaded provenance for {len(provenance_index)} sources from {args.spec}")

    if args.target:
        targets = [p for p in args.target if p.exists()]
    else:
        targets = discover_legacy_eval_yamls(args.schema_eval_dir)
    print(f"Targets: {len(targets)} legacy eval YAML(s)")

    # ---- Per-target attribution + leakage_check (self-contained) ---------
    per_target: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    leakage_targets: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for target_path in targets:
        try:
            target_rows = load_yaml_list(target_path)
        except ValueError as exc:
            skipped.append({"path": str(target_path), "reason": str(exc)})
            print(f"  [skip] {target_path}: {exc}", file=sys.stderr)
            continue

        target_hash_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in target_rows:
            if isinstance(r, dict):
                c = r.get("content")
                if isinstance(c, str):
                    target_hash_index[content_hash(c)].append(r)

        overlap_hashes = set(train_index.keys()) & set(target_hash_index.keys())
        leakage_targets.append({
            "path": str(target_path),
            "n_target_hashes": len(target_hash_index),
            "overlap_count": len(overlap_hashes),
            "overlap_sample": sorted(overlap_hashes)[:50],
        })

        if not overlap_hashes:
            per_target.append({
                "path": str(target_path),
                "n_target_hashes": len(target_hash_index),
                "overlap_count": 0,
                "categories": {},
                "by_train_source": {},
                "by_train_reason": {},
                "by_train_label": {},
                "by_provenance_lane": {},
            })
            print(f"  {target_path.name}: clean (0 overlap)")
            continue

        category_counter: Counter[str] = Counter()
        by_src: Counter[str] = Counter()
        by_reason: Counter[str] = Counter()
        by_label: Counter[str] = Counter()
        by_lane: Counter[str] = Counter()
        target_pairs: list[dict[str, Any]] = []

        for h in overlap_hashes:
            for trow in train_index[h]:
                for tgt in target_hash_index[h]:
                    tgt_meta = parse_target_description(
                        tgt.get("description"), target_path
                    )
                    category = categorize_pair(trow, tgt, tgt_meta)
                    category_counter[category] += 1
                    train_source = trow.get("source") or "unknown"
                    train_reason = trow.get("reason") or "unknown"
                    train_label = trow.get("label") or "unknown"
                    by_src[train_source] += 1
                    by_reason[train_reason] += 1
                    by_label[train_label] += 1
                    prov = provenance_for_source(provenance_index, train_source)
                    by_lane[prov["lane"] or "unknown"] += 1
                    pair = {
                        "content_hash": h,
                        "target_path": str(target_path),
                        "target_id": tgt.get("id"),
                        "target_label": tgt.get("label"),
                        "target_source": tgt_meta.get("target_source"),
                        "target_source_parser": tgt_meta.get("parser"),
                        "train_label": train_label,
                        "train_reason": train_reason,
                        "train_source": train_source,
                        "language": trow.get("language"),
                        "format_bin": trow.get("format_bin"),
                        "length_bin": trow.get("length_bin"),
                        "provenance_lane": prov["lane"],
                        "route_policy": prov["route_policy"],
                        "reason_provenance": prov["reason_provenance"],
                        "category": category,
                    }
                    target_pairs.append(pair)
                    all_pairs.append(pair)

        per_target.append({
            "path": str(target_path),
            "n_target_hashes": len(target_hash_index),
            "overlap_count": len(overlap_hashes),
            "pair_count": sum(category_counter.values()),
            "categories": dict(category_counter),
            "by_train_source": dict(by_src.most_common()),
            "by_train_reason": dict(by_reason.most_common()),
            "by_train_label": dict(by_label.most_common()),
            "by_provenance_lane": dict(by_lane.most_common()),
        })
        print(
            f"  {target_path.name}: overlap_hashes={len(overlap_hashes)} "
            f"pairs={sum(category_counter.values())} "
            f"cats={dict(category_counter)}"
        )

    # ---- Cross-target aggregations --------------------------------------
    overall_by_src: Counter[tuple[str, str]] = Counter()
    overall_by_reason: Counter[tuple[str, str]] = Counter()
    overall_by_label: Counter[tuple[str, str]] = Counter()
    overall_by_category: Counter[str] = Counter()
    overall_by_lane: Counter[tuple[str, str]] = Counter()
    for p in all_pairs:
        tp = Path(p["target_path"]).name
        overall_by_src[(tp, p["train_source"])] += 1
        overall_by_reason[(tp, p["train_reason"])] += 1
        overall_by_label[(tp, p["train_label"])] += 1
        overall_by_category[p["category"]] += 1
        overall_by_lane[(tp, p["provenance_lane"])] += 1

    attribution = {
        "schema_version": 2,
        "created_utc": utc_now_iso(),
        "diagnostic_only": True,
        "diagnostic_note": (
            "Legacy eval YAML overlap with v8 train. NOT a Phase 1 gate. "
            "These YAMLs predate the JSONL curation pipeline and are not "
            "L2 acceptance gates."
        ),
        "inputs": {
            "train_yaml": str(train_yaml),
            "spec": str(args.spec),
            "schema_eval_dir": str(args.schema_eval_dir),
            "targets": [str(p) for p in targets],
        },
        "overall": {
            "total_pairs": len(all_pairs),
            "total_overlapping_hashes": sum(t["overlap_count"] for t in per_target),
            "by_category": dict(overall_by_category),
        },
        "per_target": per_target,
        "cross_target": {
            "by_target_x_train_source": [
                {"target": k[0], "train_source": k[1], "count": v}
                for k, v in overall_by_src.most_common()
            ],
            "by_target_x_train_reason": [
                {"target": k[0], "train_reason": k[1], "count": v}
                for k, v in overall_by_reason.most_common()
            ],
            "by_target_x_train_label": [
                {"target": k[0], "train_label": k[1], "count": v}
                for k, v in overall_by_label.most_common()
            ],
            "by_target_x_provenance_lane": [
                {"target": k[0], "provenance_lane": k[1], "count": v}
                for k, v in overall_by_lane.most_common()
            ],
        },
        "all_pairs": all_pairs,
    }

    # README marker so the audit's diagnostic-only nature is clear from the dir.
    (output_dir / "README.md").write_text(_README_TEXT, encoding="utf-8")

    # Self-contained leakage_check.json (no dependency on the export script's run).
    leakage_check_path = output_dir / "leakage_check.json"
    leakage_check_path.write_text(json.dumps({
        "diagnostic_only": True,
        "diagnostic_note": (
            "Legacy eval YAML overlap with v8 train. NOT a Phase 1 gate."
        ),
        "created_utc": utc_now_iso(),
        "train_yaml": str(train_yaml),
        "n_train_hashes": len(train_index),
        "targets": leakage_targets,
        "skipped_targets": skipped,
        "overall_overlap_count": sum(t["overlap_count"] for t in leakage_targets),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    json_path = output_dir / "leakage_attribution.json"
    json_path.write_text(json.dumps(attribution, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = output_dir / "leakage_attribution.md"
    md_path.write_text(_render_md(attribution), encoding="utf-8")

    print(f"\nLeakage check: {leakage_check_path}")
    print(f"Attribution:   {json_path}")
    print(f"Markdown:      {md_path}")
    print(f"README:        {output_dir / 'README.md'}")
    print(f"Total pairs: {len(all_pairs)}")
    print(f"By category: {dict(overall_by_category)}")
    return 0


def _render_md(attribution: dict[str, Any]) -> str:
    lines: list[str] = ["# Legacy eval YAML leakage audit (diagnostic only)", ""]
    lines.append(
        "NOT a Phase 1 gate. Legacy schema/eval YAMLs predate the JSONL "
        "curation pipeline. The Phase 1 v8 residual artifact in the parent "
        "dir is usable independent of the contents below."
    )
    lines.append("")
    lines.append(f"Created: {attribution['created_utc']}")
    o = attribution["overall"]
    lines.append(
        f"Total overlapping content_hashes: {o['total_overlapping_hashes']:,} | "
        f"total (train, target) pairs: {o['total_pairs']:,}"
    )
    lines.append(f"By category: {o['by_category']}")
    lines.append("")
    lines.append("## Per target")
    for t in attribution["per_target"]:
        lines.append(f"### {Path(t['path']).name}")
        lines.append(
            f"target rows: {t.get('n_target_hashes', 0):,} | "
            f"overlapping hashes: {t['overlap_count']:,} | "
            f"pairs: {t.get('pair_count', 0):,}"
        )
        if t["overlap_count"] == 0:
            lines.append("clean.")
            lines.append("")
            continue
        lines.append(f"categories: {t['categories']}")
        lines.append("")
        lines.append("**by train_source (top 15)**")
        for k, v in list(t["by_train_source"].items())[:15]:
            lines.append(f"- {k}: {v:,}")
        lines.append("")
        lines.append("**by train_reason**")
        for k, v in t["by_train_reason"].items():
            lines.append(f"- {k}: {v:,}")
        lines.append("")
        lines.append("**by train_label**")
        for k, v in t["by_train_label"].items():
            lines.append(f"- {k}: {v:,}")
        lines.append("")
        lines.append("**by provenance_lane**")
        for k, v in t["by_provenance_lane"].items():
            lines.append(f"- {k}: {v:,}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
