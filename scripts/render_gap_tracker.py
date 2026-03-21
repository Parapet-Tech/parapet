"""Render mirror gap tracker markdown from a curation manifest.

Usage:
  cd parapet
  python scripts/render_gap_tracker.py \
    --manifest path/to/manifest.json \
    --out strategy/gap_tracker.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

LANG_ORDER = ["EN", "RU", "ZH", "AR"]
LABEL_ORDER = ["malicious", "benign"]

SOURCE_HINTS = {
    ("malicious", "RU"): "schema/eval/training/multilingual/attacks_121624_ru.yaml + ru_attacks.yaml",
    ("malicious", "AR"): "schema/eval/training/multilingual/attacks_121624_ar.yaml + ar_attacks.yaml",
    ("malicious", "ZH"): "schema/eval/training/multilingual/attacks_121624_zh.yaml + zh_attacks.yaml",
    ("malicious", "EN"): "schema/eval/malicious/attacks_121624.yaml + reason-specific EN sources",
    ("benign", "RU"): "schema/eval/benign/opensource_wikipedia_ru_benign.yaml + opensource_xquad_ru_benign.yaml",
    ("benign", "AR"): "schema/eval/benign/opensource_wikipedia_ar_benign.yaml + opensource_xquad_ar_benign.yaml",
    ("benign", "ZH"): "schema/eval/benign/opensource_wikipedia_zh_benign.yaml + opensource_xquad_zh_benign.yaml",
    ("benign", "EN"): "schema/eval/training/global_benign_curated.yaml + EN hard-negatives",
}


def parse_expected(raw: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --expected token: '{item}' (expected LANG=VALUE)")
        key, value = item.split("=", 1)
        lang = key.strip().upper()
        out[lang] = int(value.strip())
    missing = [lang for lang in LANG_ORDER if lang not in out]
    if missing:
        raise ValueError(f"Missing expected values for languages: {missing}")
    return out


def fmt_pct(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return (100.0 * numer) / denom


def allocate_expected(total: int, ratios: dict[str, float]) -> dict[str, int]:
    """Allocate integer per-language expected counts that sum to total."""
    raw = {lang: total * ratios[lang] for lang in LANG_ORDER}
    base = {lang: int(raw[lang]) for lang in LANG_ORDER}
    remainder = total - sum(base.values())
    if remainder <= 0:
        return base
    order = sorted(
        LANG_ORDER,
        key=lambda lang: (raw[lang] - base[lang], -LANG_ORDER.index(lang)),
        reverse=True,
    )
    for i in range(remainder):
        base[order[i % len(order)]] += 1
    return base


def pct_bar(pct: float, width: int = 20) -> str:
    clamped = min(max(pct, 0.0), 100.0)
    filled = int(round((clamped / 100.0) * width))
    filled = max(0, min(width, filled))
    suffix = "+" if pct > 100.0 else ""
    return f"[{'#' * filled}{'.' * (width - filled)}] {pct:6.1f}%{suffix}"


def parse_reason_label(key: str) -> tuple[str, str]:
    reason, remainder = key.split("__", 1)
    if remainder.endswith("_malicious"):
        return reason, "malicious"
    if remainder.endswith("_benign"):
        return reason, "benign"
    raise ValueError(f"Unexpected cell key label suffix: {key}")


def build_report(manifest: dict, *, expected: dict[str, int], manifest_rel: str) -> str:
    cell_fills = manifest["cell_fills"]
    reasons = sorted({parse_reason_label(k)[0] for k in cell_fills})
    reason_count = len(reasons)
    expected_total = sum(expected[lang] for lang in LANG_ORDER)
    if expected_total <= 0:
        raise ValueError("expected language totals must sum to > 0")
    expected_ratios = {lang: expected[lang] / expected_total for lang in LANG_ORDER}

    rows: dict[tuple[str, str], dict] = {}
    for key, value in cell_fills.items():
        reason, label = parse_reason_label(key)
        target = int(value.get("target") or 0)
        exp_by_lang = allocate_expected(target, expected_ratios)
        by_lang = {lang: int((value.get("by_language") or {}).get(lang, 0)) for lang in LANG_ORDER}
        rows[(reason, label)] = {
            "by_lang": by_lang,
            "exp_by_lang": exp_by_lang,
            "backfilled": int(value.get("backfilled") or 0),
            "degraded": bool(value.get("degraded")),
            "degraded_mode": value.get("degraded_mode") or "",
        }

    aggregate_actual = defaultdict(int)
    aggregate_expected = defaultdict(int)
    deficits: list[tuple[int, str, str, str, int, int]] = []
    for reason in reasons:
        for label in LABEL_ORDER:
            row = rows[(reason, label)]
            for lang in LANG_ORDER:
                actual = row["by_lang"][lang]
                exp = row["exp_by_lang"][lang]
                gap = exp - actual
                aggregate_actual[(label, lang)] += actual
                aggregate_expected[(label, lang)] += exp
                if gap > 0:
                    deficits.append((gap, reason, label, lang, actual, exp))
    deficits.sort(reverse=True)

    degraded_cells = sum(1 for v in cell_fills.values() if v.get("degraded"))
    backfilled_total = sum(int(v.get("backfilled") or 0) for v in cell_fills.values())
    total_samples = int(manifest.get("total_samples") or 0)
    backfilled_pct = fmt_pct(backfilled_total, total_samples)

    out: list[str] = []
    out.append("# V3 Gap Tracker")
    out.append("")
    out.append("This tracker uses the latest mirror manifest as the control-plane for gap filling.")
    out.append("")
    out.append("## Snapshot Metadata")
    out.append("")
    out.append(f"- generated_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    out.append(f"- manifest: `{manifest_rel}`")
    out.append(f"- spec_name: `{manifest.get('spec_name', 'unknown')}`")
    out.append(f"- spec_version: `{manifest.get('spec_version', 'unknown')}`")
    out.append(f"- seed: `{manifest.get('seed', 'unknown')}`")
    out.append(f"- total_samples: `{total_samples}`")
    out.append(
        f"- cells: `{len(cell_fills)}` (reasons={reason_count}, labels={len(LABEL_ORDER)}, languages={len(LANG_ORDER)})"
    )
    out.append(f"- degraded_cells: `{degraded_cells}/{len(cell_fills)}`")
    out.append(f"- backfilled_total: `{backfilled_total}` ({backfilled_pct:.1f}% of total)")
    out.append("")
    out.append("## Language Coverage Charts")
    out.append("")
    out.append("| Label | Language | Actual | Expected | Gap | Fill % | Chart |")
    out.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
    for label in LABEL_ORDER:
        for lang in LANG_ORDER:
            actual = aggregate_actual[(label, lang)]
            exp = aggregate_expected[(label, lang)]
            gap = exp - actual
            fill = fmt_pct(actual, exp)
            out.append(
                f"| {label} | {lang} | {actual} | {exp} | {gap:+d} | {fill:.1f}% | `{pct_bar(fill)}` |"
            )
    out.append("")
    out.append("## Cell Heatmap - Malicious")
    out.append("")
    out.append("| Reason | EN | RU | ZH | AR | Backfilled | Degraded | Mode |")
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for reason in reasons:
        row = rows[(reason, "malicious")]
        vals = []
        for lang in LANG_ORDER:
            actual = row["by_lang"][lang]
            exp = row["exp_by_lang"][lang]
            vals.append(f"{actual}/{exp} ({exp - actual:+d})")
        out.append(
            f"| {reason} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | "
            f"{row['backfilled']} | {str(row['degraded']).lower()} | {row['degraded_mode'] or '-'} |"
        )
    out.append("")
    out.append("## Cell Heatmap - Benign")
    out.append("")
    out.append("| Reason | EN | RU | ZH | AR | Backfilled | Degraded | Mode |")
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for reason in reasons:
        row = rows[(reason, "benign")]
        vals = []
        for lang in LANG_ORDER:
            actual = row["by_lang"][lang]
            exp = row["exp_by_lang"][lang]
            vals.append(f"{actual}/{exp} ({exp - actual:+d})")
        out.append(
            f"| {reason} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | "
            f"{row['backfilled']} | {str(row['degraded']).lower()} | {row['degraded_mode'] or '-'} |"
        )
    out.append("")
    out.append("## Priority Gap Backlog")
    out.append("")
    out.append(
        "Top deficits sorted by absolute missing count. Use this as the first pass for source expansion and re-balancing."
    )
    out.append("")
    out.append("| Rank | Reason | Label | Lang | Current | Target | Gap | Source Hint | Status |")
    out.append("| ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |")
    for i, (gap, reason, label, lang, actual, exp) in enumerate(deficits[:24], start=1):
        hint = SOURCE_HINTS.get((label, lang), "-")
        out.append(
            f"| {i} | {reason} | {label} | {lang} | {actual} | {exp} | {gap} | `{hint}` | `todo` |"
        )
    out.append("")
    out.append("## Fill Plan Log")
    out.append("")
    out.append("| Date | Change | Expected Impact | Measured Result | Notes |")
    out.append("| --- | --- | --- | --- | --- |")
    out.append("| YYYY-MM-DD | Added RU benign source X | +N RU benign in meta_probe/obfuscation | pending | |")
    out.append("| YYYY-MM-DD | Added AR malicious source Y | +N AR malicious in roleplay/exfiltration | pending | |")
    out.append("")
    out.append("## Suggested Quality Gates for V3 Train Eligibility")
    out.append("")
    out.append("- coverage_floor_explore: each `(reason, label, language)` >= 50")
    out.append("- coverage_floor_publish: each `(reason, label, language)` >= 150")
    out.append("- degraded_cell_ratio_explore <= 20%")
    out.append("- degraded_cell_ratio_publish <= 5%")
    out.append("- backfilled_share_explore <= 30%")
    out.append("- backfilled_share_publish <= 10%")
    out.append("- mirror_parity: per `(reason, language)`, benign/malicious ratio in [0.8, 1.25]")
    out.append("")
    out.append("## Refresh")
    out.append("")
    out.append("```bash")
    out.append("cd parapet")
    out.append("python scripts/render_gap_tracker.py \\")
    out.append("  --manifest path/to/manifest.json \\")
    out.append("  --out strategy/gap_tracker.md")
    out.append("```")
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render mirror gap tracker markdown from manifest.json")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to curation manifest.json")
    parser.add_argument("--out", type=Path, required=True, help="Output markdown file")
    parser.add_argument(
        "--expected",
        default="EN=4500,RU=600,ZH=480,AR=420",
        help="Base per-cell language counts used to derive language ratios (default: EN=4500,RU=600,ZH=480,AR=420)",
    )
    args = parser.parse_args()

    expected = parse_expected(args.expected)
    manifest_path = args.manifest.resolve()
    out_path = args.out.resolve()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rel_manifest = str(manifest_path)
    doc = build_report(manifest, expected=expected, manifest_rel=rel_manifest)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
