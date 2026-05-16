"""Formal descriptive analysis for the Phase 1 v8 residual artifact.

This is intentionally not a sensor prototype. It only summarizes the existing
residual export: composition, source concentration, score/margin geometry, fold
effects, and data-quality flags. It does not emit content samples.

Example:
    python scripts/phase1_residual_formal_analysis.py \
        --residuals runs/phase1_v8_residuals/residuals.jsonl \
        --baseline-correct runs/phase1_v8_residuals/baseline_correct.jsonl \
        --manifest runs/phase1_v8_residuals/manifest.json \
        --output-dir runs/phase1_v8_residuals/formal_analysis
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


NUMERIC_FIELDS = (
    "raw_score",
    "l1_score",
    "raw_unquoted_score",
    "raw_squash_score",
    "raw_score_delta",
    "unquoted_score",
    "squash_score",
)

CATEGORICAL_FIELDS = (
    "residual_category",
    "error_type",
    "label",
    "language",
    "reason",
    "source",
    "format_bin",
    "length_bin",
    "fold_id",
)

MARGIN_BANDS = (0.05, 0.10, 0.25, 0.50, 1.00, 2.00)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def pct(n: int | float, total: int | float) -> float:
    return 0.0 if not total else float(n) / float(total)


def quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def quantiles(values: Iterable[Any]) -> dict[str, float | None]:
    clean = sorted(float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    return {
        "n": len(clean),
        "min": quantile(clean, 0.00),
        "p01": quantile(clean, 0.01),
        "p05": quantile(clean, 0.05),
        "p10": quantile(clean, 0.10),
        "p25": quantile(clean, 0.25),
        "p50": quantile(clean, 0.50),
        "p75": quantile(clean, 0.75),
        "p90": quantile(clean, 0.90),
        "p95": quantile(clean, 0.95),
        "p99": quantile(clean, 0.99),
        "max": quantile(clean, 1.00),
    }


def counter_table(rows: list[dict[str, Any]], field: str, top: int | None = None) -> list[dict[str, Any]]:
    total = len(rows)
    counts = Counter(str(r.get(field, "<missing>")) for r in rows)
    items = counts.most_common(top)
    return [{"value": k, "count": v, "share": pct(v, total)} for k, v in items]


def nested_counter_table(
    rows: list[dict[str, Any]],
    left: str,
    right: str,
    top: int = 25,
) -> list[dict[str, Any]]:
    total = len(rows)
    counts: Counter[tuple[str, str]] = Counter()
    for r in rows:
        counts[(str(r.get(left, "<missing>")), str(r.get(right, "<missing>")))] += 1
    return [
        {left: l, right: rr, "count": n, "share": pct(n, total)}
        for (l, rr), n in counts.most_common(top)
    ]


def concentration(rows: list[dict[str, Any]], field: str, top: int = 10) -> dict[str, Any]:
    total = len(rows)
    counts = Counter(str(r.get(field, "<missing>")) for r in rows)
    shares = [pct(n, total) for n in counts.values()]
    hhi = sum(s * s for s in shares)
    top_items = counts.most_common(top)
    return {
        "field": field,
        "n": total,
        "unique": len(counts),
        "top_share": pct(top_items[0][1], total) if top_items else 0.0,
        "top5_share": sum(pct(n, total) for _, n in top_items[:5]),
        "top10_share": sum(pct(n, total) for _, n in top_items[:10]),
        "hhi": hhi,
        "effective_n": None if hhi == 0 else 1.0 / hhi,
        "top": [{"value": k, "count": v, "share": pct(v, total)} for k, v in top_items],
    }


def group_rows(rows: list[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get(field, "<missing>"))].append(row)
    return dict(out)


def numeric_by_group(rows: list[dict[str, Any]], group_field: str) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for group, group_rows_ in sorted(group_rows(rows, group_field).items()):
        out[group] = {field: quantiles(r.get(field) for r in group_rows_) for field in NUMERIC_FIELDS}
    return out


def threshold(row: dict[str, Any]) -> float:
    thresholds = row.get("l1_thresholds")
    if isinstance(thresholds, dict):
        v = thresholds.get("l1")
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


def score_margin(row: dict[str, Any]) -> float | None:
    raw = row.get("raw_score")
    if not isinstance(raw, (int, float)):
        return None
    return float(raw) - threshold(row)


def margin_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clean = [r for r in rows if score_margin(r) is not None]
    distances = [abs(score_margin(r) or 0.0) for r in clean]
    by_band = []
    for band in MARGIN_BANDS:
        in_band = [r for r in clean if abs(score_margin(r) or 0.0) <= band]
        by_band.append({
            "abs_margin_lte": band,
            "count": len(in_band),
            "share": pct(len(in_band), len(clean)),
            "by_category": counter_table(in_band, "residual_category"),
            "by_label": counter_table(in_band, "label"),
        })
    return {
        "n": len(clean),
        "distance_to_threshold_quantiles": quantiles(distances),
        "by_band": by_band,
    }


def hash_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing = 0
    for row in rows:
        h = row.get("content_hash")
        if not isinstance(h, str) or not h:
            missing += 1
            continue
        by_hash[h].append(row)

    duplicate_hashes = {h: rs for h, rs in by_hash.items() if len(rs) > 1}
    label_conflicts = []
    category_conflicts = []
    for h, rs in duplicate_hashes.items():
        labels = sorted({str(r.get("label")) for r in rs})
        categories = sorted({str(r.get("residual_category")) for r in rs})
        if len(labels) > 1:
            label_conflicts.append({"content_hash": h, "labels": labels, "n": len(rs)})
        if len(categories) > 1:
            category_conflicts.append({"content_hash": h, "categories": categories, "n": len(rs)})

    return {
        "rows": len(rows),
        "missing_hash": missing,
        "unique_hashes": len(by_hash),
        "duplicate_hashes": len(duplicate_hashes),
        "duplicate_rows": sum(len(rs) for rs in duplicate_hashes.values()),
        "label_conflicts": label_conflicts[:50],
        "category_conflicts": category_conflicts[:50],
    }


def quote_squash_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for category, category_rows in sorted(group_rows(rows, "residual_category").items()):
        quoted = [r for r in category_rows if r.get("quote_detected") is True]
        squash_advantage = []
        raw_delta = []
        for r in category_rows:
            sq = r.get("raw_squash_score")
            uq = r.get("raw_unquoted_score")
            if isinstance(sq, (int, float)) and isinstance(uq, (int, float)):
                squash_advantage.append(float(sq) - float(uq))
            d = r.get("raw_score_delta")
            if isinstance(d, (int, float)):
                raw_delta.append(float(d))
        out[category] = {
            "n": len(category_rows),
            "quote_detected_count": len(quoted),
            "quote_detected_share": pct(len(quoted), len(category_rows)),
            "quote_by_label": counter_table(quoted, "label"),
            "quote_by_source_top10": counter_table(quoted, "source", top=10),
            "squash_minus_unquoted_quantiles": quantiles(squash_advantage),
            "raw_score_delta_quantiles": quantiles(raw_delta),
        }
    return out


def missing_field_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = sorted({k for r in rows for k in r.keys()})
    out = {}
    for field in fields:
        missing = sum(1 for r in rows if field not in r or r.get(field) in (None, ""))
        if missing:
            out[field] = {"missing": missing, "share": pct(missing, len(rows))}
    return out


def analyze_rows(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    by_category = group_rows(rows, "residual_category")
    by_error = group_rows(rows, "error_type")
    category_keys = sorted(by_category)
    error_keys = sorted(by_error)
    return {
        "name": name,
        "row_count": len(rows),
        "hash_quality": hash_quality(rows),
        "composition": {field: counter_table(rows, field, top=40) for field in CATEGORICAL_FIELDS},
        "source_concentration": concentration(rows, "source", top=20),
        "source_concentration_by_category": {
            category: concentration(category_rows, "source", top=15)
            for category, category_rows in sorted(by_category.items())
        },
        "source_concentration_by_language": {
            language: concentration(language_rows, "source", top=15)
            for language, language_rows in sorted(group_rows(rows, "language").items())
        },
        "top_language_source": nested_counter_table(rows, "language", "source", top=30),
        "top_language_format": nested_counter_table(rows, "language", "format_bin", top=30),
        "top_language_length": nested_counter_table(rows, "language", "length_bin", top=30),
        "top_category_source": nested_counter_table(rows, "residual_category", "source", top=40),
        "numeric_by_category": {
            category: {field: quantiles(r.get(field) for r in category_rows) for field in NUMERIC_FIELDS}
            for category, category_rows in sorted(by_category.items())
        },
        "numeric_by_language": numeric_by_group(rows, "language"),
        "numeric_by_fold": numeric_by_group(rows, "fold_id"),
        "margin_analysis": margin_analysis(rows),
        "quote_squash_analysis": quote_squash_analysis(rows),
        "missing_fields": missing_field_analysis(rows),
        "category_keys": category_keys,
        "error_keys": error_keys,
    }


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def fmt_num(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return lines


def table_from_counter(counter_rows: list[dict[str, Any]], value_header: str = "value", limit: int = 15) -> list[str]:
    return md_table(
        [value_header, "count", "share"],
        [[r["value"], r["count"], fmt_pct(r["share"])] for r in counter_rows[:limit]],
    )


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    residual = report["residuals"]
    baseline = report.get("baseline_correct")
    manifest = report.get("manifest_summary", {})
    lines: list[str] = []

    lines.append("# Formal Phase 1 v8 residual analysis")
    lines.append("")
    lines.append(f"Created: {report['created_utc']}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This is descriptive data analysis only. It does not prototype or tune L3 sensors.")
    lines.append("It summarizes the existing Phase 1 residual export and avoids content samples.")
    lines.append("")
    lines.extend(md_table(
        ["artifact", "rows", "sha256"],
        [
            ["residuals", residual["row_count"], report["inputs"]["residuals"]["sha256"]],
            [
                "baseline_correct",
                baseline["row_count"] if baseline else 0,
                report["inputs"].get("baseline_correct", {}).get("sha256", "n/a"),
            ],
        ],
    ))

    lines.append("")
    lines.append("## Export context")
    lines.append("")
    lines.extend(md_table(
        ["field", "value"],
        [
            ["total_predictions", manifest.get("total_predictions", "n/a")],
            ["false_negative", manifest.get("false_negative", "n/a")],
            ["false_positive", manifest.get("false_positive", "n/a")],
            ["near_boundary_benign", manifest.get("near_boundary_benign", "n/a")],
            ["baseline_correct", manifest.get("baseline_correct", "n/a")],
            ["gate_status", manifest.get("gate_status", "n/a")],
            ["phase4_acceptance", manifest.get("for_phase4_acceptance", "n/a")],
        ],
    ))

    lines.append("")
    lines.append("## Residual composition")
    lines.append("")
    for field in ("residual_category", "label", "language", "format_bin", "length_bin", "fold_id"):
        lines.append(f"### By `{field}`")
        lines.append("")
        lines.extend(table_from_counter(residual["composition"][field], field))
        lines.append("")

    lines.append("## Source concentration")
    lines.append("")
    sc = residual["source_concentration"]
    lines.extend(md_table(
        ["scope", "unique sources", "top share", "top5 share", "HHI", "effective_n"],
        [[
            "all residuals",
            sc["unique"],
            fmt_pct(sc["top_share"]),
            fmt_pct(sc["top5_share"]),
            fmt_num(sc["hhi"]),
            fmt_num(sc["effective_n"]),
        ]],
    ))
    lines.append("")
    lines.append("### Top sources overall")
    lines.append("")
    lines.extend(table_from_counter(sc["top"], "source", limit=20))
    lines.append("")

    lines.append("### Source concentration by residual category")
    lines.append("")
    rows = []
    for category, c in residual["source_concentration_by_category"].items():
        rows.append([
            category,
            c["n"],
            c["unique"],
            fmt_pct(c["top_share"]),
            fmt_pct(c["top5_share"]),
            fmt_num(c["hhi"]),
            fmt_num(c["effective_n"]),
        ])
    lines.extend(md_table(["category", "rows", "unique", "top", "top5", "HHI", "effective_n"], rows))
    lines.append("")

    lines.append("## Top cross-tabs")
    lines.append("")
    for title, key, columns in (
        (
            "Category x source",
            "top_category_source",
            [("category", "residual_category"), ("source", "source"), ("count", "count"), ("share", "share")],
        ),
        (
            "Language x source",
            "top_language_source",
            [("language", "language"), ("source", "source"), ("count", "count"), ("share", "share")],
        ),
        (
            "Language x format",
            "top_language_format",
            [("language", "language"), ("format_bin", "format_bin"), ("count", "count"), ("share", "share")],
        ),
        (
            "Language x length",
            "top_language_length",
            [("language", "language"), ("length_bin", "length_bin"), ("count", "count"), ("share", "share")],
        ),
    ):
        lines.append(f"### {title}")
        lines.append("")
        rows = []
        for r in residual[key][:20]:
            row = [fmt_pct(r[source_key]) if source_key == "share" else r.get(source_key, "") for _, source_key in columns]
            rows.append(row)
        lines.extend(md_table([header for header, _ in columns], rows))
        lines.append("")

    lines.append("## Score and margin geometry")
    lines.append("")
    rows = []
    for category, by_field in residual["numeric_by_category"].items():
        raw = by_field.get("raw_score", {})
        prob = by_field.get("l1_score", {})
        rows.append([
            category,
            raw.get("n", 0),
            fmt_num(raw.get("p10")),
            fmt_num(raw.get("p50")),
            fmt_num(raw.get("p90")),
            fmt_num(prob.get("p50")),
        ])
    lines.extend(md_table(["category", "n", "raw p10", "raw p50", "raw p90", "l1_score p50"], rows))
    lines.append("")
    lines.append("### Absolute margin bands")
    lines.append("")
    rows = []
    for band in residual["margin_analysis"]["by_band"]:
        cats = ", ".join(f"{r['value']}={r['count']}" for r in band["by_category"][:4])
        rows.append([f"<= {band['abs_margin_lte']}", band["count"], fmt_pct(band["share"]), cats])
    lines.extend(md_table(["abs(raw_score-threshold)", "rows", "share", "category mix"], rows))
    lines.append("")

    lines.append("## Quote and squash diagnostics")
    lines.append("")
    rows = []
    for category, q in residual["quote_squash_analysis"].items():
        squash = q["squash_minus_unquoted_quantiles"]
        rows.append([
            category,
            q["n"],
            q["quote_detected_count"],
            fmt_pct(q["quote_detected_share"]),
            fmt_num(squash.get("p50")),
            fmt_num(squash.get("p90")),
        ])
    lines.extend(md_table(
        ["category", "rows", "quote_detected", "quote share", "squash-minus-unquoted p50", "p90"],
        rows,
    ))
    lines.append("")

    lines.append("## Hash and field quality")
    lines.append("")
    hq = residual["hash_quality"]
    lines.extend(md_table(
        ["metric", "value"],
        [
            ["unique_hashes", hq["unique_hashes"]],
            ["missing_hash", hq["missing_hash"]],
            ["duplicate_hashes", hq["duplicate_hashes"]],
            ["duplicate_rows", hq["duplicate_rows"]],
            ["label_conflicts_sampled", len(hq["label_conflicts"])],
            ["category_conflicts_sampled", len(hq["category_conflicts"])],
        ],
    ))
    if residual["missing_fields"]:
        lines.append("")
        lines.append("### Missing fields")
        lines.append("")
        lines.extend(md_table(
            ["field", "missing", "share"],
            [[k, v["missing"], fmt_pct(v["share"])] for k, v in residual["missing_fields"].items()],
        ))

    if baseline:
        lines.append("")
        lines.append("## Baseline-correct sidecar")
        lines.append("")
        lines.append("The sidecar is not part of the primary residual training input.")
        lines.append("It is useful for traffic-impact and background-distribution comparison.")
        lines.append("")
        for field in ("label", "language", "format_bin", "length_bin"):
            lines.append(f"### Sidecar by `{field}`")
            lines.append("")
            lines.extend(table_from_counter(baseline["composition"][field], field, limit=10))
            lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- The existing residual artifact is large enough for descriptive analysis and first-pass L3 measurement.")
    lines.append("- Additional L1 mining should be targeted after this analysis, not random corpus expansion.")
    lines.append("- Any next step should preserve content-hash dedupe, source-family grouping, and per-language reporting.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    residuals_path = args.residuals
    baseline_path = args.baseline_correct
    manifest_path = args.manifest

    residual_rows = load_jsonl(residuals_path)
    baseline_rows = load_jsonl(baseline_path) if baseline_path and baseline_path.exists() else []

    manifest_summary: dict[str, Any] = {}
    if manifest_path and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        row_counts = manifest.get("row_counts", {})
        usability = manifest.get("usability", {})
        gate = manifest.get("gate", {})
        manifest_summary = {
            "total_predictions": row_counts.get("total_predictions"),
            "false_negative": row_counts.get("false_negative"),
            "false_positive": row_counts.get("false_positive"),
            "near_boundary_benign": row_counts.get("near_boundary_benign"),
            "baseline_correct": row_counts.get("baseline_correct"),
            "residuals_emitted": row_counts.get("residuals_emitted"),
            "gate_status": gate.get("status"),
            "for_phase4_acceptance": usability.get("for_phase4_acceptance"),
        }

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "residuals": {
                "path": str(residuals_path),
                "sha256": sha256_file(residuals_path),
                "rows": len(residual_rows),
            },
        },
        "manifest_summary": manifest_summary,
        "residuals": analyze_rows(residual_rows, "residuals"),
    }
    if baseline_path and baseline_path.exists():
        report["inputs"]["baseline_correct"] = {
            "path": str(baseline_path),
            "sha256": sha256_file(baseline_path),
            "rows": len(baseline_rows),
        }
        report["baseline_correct"] = analyze_rows(baseline_rows, "baseline_correct")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--residuals",
        type=Path,
        default=Path("runs/phase1_v8_residuals/residuals.jsonl"),
        help="Phase 1 residual JSONL.",
    )
    parser.add_argument(
        "--baseline-correct",
        type=Path,
        default=Path("runs/phase1_v8_residuals/baseline_correct.jsonl"),
        help="Optional baseline-correct sidecar JSONL.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("runs/phase1_v8_residuals/manifest.json"),
        help="Optional Phase 1 residual manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/phase1_v8_residuals/formal_analysis"),
        help="Output directory for formal_residual_analysis.{json,md}.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "formal_residual_analysis.json"
    md_path = args.output_dir / "formal_residual_analysis.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, md_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Residual rows: {report['residuals']['row_count']}")
    if "baseline_correct" in report:
        print(f"Baseline-correct sidecar rows: {report['baseline_correct']['row_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
