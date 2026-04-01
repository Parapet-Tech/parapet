#!/usr/bin/env python3
"""
scorecard.py -- Per-lane coverage scorecard for v5 curation manifests.

Reads a curation manifest.json and reports logical cell coverage,
per-cell quotas vs actuals, degraded cells, backfill summary, lane
deficits, and ledger action counts.

Usage:
    python scripts/scorecard.py path/to/manifest.json [--total-target 19200] [--budget-fraction 0.15]
"""

import argparse
import json
import math
import sys
from typing import Any


# ── Constants ───────────────────────────────────────────────────────────

REASONS = [
    "instruction_override",
    "roleplay_jailbreak",
    "meta_probe",
    "exfiltration",
    "adversarial_suffix",
    "indirect_injection",
    "obfuscation",
    "constraint_bypass",
]

LANGUAGES = ["EN", "RU", "ZH", "AR"]

LANGUAGE_QUOTA = {"EN": 0.75, "RU": 0.10, "ZH": 0.08, "AR": 0.07}


# ── Quota computation ──────────────────────────────────────────────────

def compute_per_cell_budgets(total_target: int, budget_fraction: float):
    """Return (per_cell_attack, per_cell_benign) as integers.

    Budget split:
        attacks = total_target / 2, split across 8 reasons
        benign  = (total_target / 2) * (1 - budget_fraction), split across 8 reasons
        (the remaining budget_fraction goes to background)
    """
    per_cell_attack = int(total_target / 2 / len(REASONS))
    per_cell_benign = int(
        (total_target / 2) * (1.0 - budget_fraction) / len(REASONS)
    )
    return per_cell_attack, per_cell_benign


def language_quota(per_cell: int, lang: str) -> int:
    """Floor of per_cell * language fraction."""
    return math.floor(per_cell * LANGUAGE_QUOTA[lang])


# ── Manifest parsing helpers ──────────────────────────────────────────

def find_cell_fill(cell_fills: dict, reason: str, label: str) -> dict | None:
    """Find the cell fill entry for a given reason and label.

    Cell keys look like:  {reason}__{languages}_{label}
    The languages part varies, so we match on reason prefix and label suffix.
    """
    for key, fill in cell_fills.items():
        # Split on last underscore to separate label
        # Key format: reason__LANG1,LANG2,..._label
        if not key.startswith(reason + "__"):
            continue
        if key.endswith("_" + label):
            return fill
    return None


def get_language_count(fill: dict, lang: str) -> int:
    """Get the count for a language from a cell fill, defaulting to 0."""
    by_lang = fill.get("by_language", {})
    return by_lang.get(lang, 0)


# ── Scorecard logic ──────────────────────────────────────────────────

def build_scorecard(manifest: dict, total_target: int, budget_fraction: float):
    """Compute all scorecard data from the manifest."""
    cell_fills = manifest.get("cell_fills", {})
    per_cell_attack, per_cell_benign = compute_per_cell_budgets(
        total_target, budget_fraction
    )

    # Per-cell detail rows: (reason, lang, mal_quota, mal_actual, mal_deficit,
    #                        ben_quota, ben_actual, ben_deficit, covered)
    rows = []
    covered_count = 0
    degraded_count = 0
    total_backfilled = 0
    total_actual = 0
    deficits = []  # (reason, lang, combined_deficit)

    # Track degraded cells per reason (both sides)
    degraded_cells = set()

    for reason in REASONS:
        mal_fill = find_cell_fill(cell_fills, reason, "malicious")
        ben_fill = find_cell_fill(cell_fills, reason, "benign")

        if mal_fill and mal_fill.get("degraded"):
            degraded_cells.add((reason, "malicious"))
        if ben_fill and ben_fill.get("degraded"):
            degraded_cells.add((reason, "benign"))

        if mal_fill:
            total_backfilled += mal_fill.get("backfilled", 0)
            total_actual += mal_fill.get("actual", 0)
        if ben_fill:
            total_backfilled += ben_fill.get("backfilled", 0)
            total_actual += ben_fill.get("actual", 0)

        for lang in LANGUAGES:
            mal_quota = language_quota(per_cell_attack, lang)
            ben_quota = language_quota(per_cell_benign, lang)

            mal_actual = get_language_count(mal_fill, lang) if mal_fill else 0
            ben_actual = get_language_count(ben_fill, lang) if ben_fill else 0

            mal_deficit = max(0, mal_quota - mal_actual)
            ben_deficit = max(0, ben_quota - ben_actual)

            mal_met = mal_actual >= mal_quota
            ben_met = ben_actual >= ben_quota
            cell_covered = mal_met and ben_met

            if cell_covered:
                covered_count += 1

            rows.append({
                "reason": reason,
                "lang": lang,
                "mal_quota": mal_quota,
                "mal_actual": mal_actual,
                "mal_deficit": mal_deficit,
                "ben_quota": ben_quota,
                "ben_actual": ben_actual,
                "ben_deficit": ben_deficit,
                "covered": cell_covered,
            })

            if not cell_covered:
                combined = mal_deficit + ben_deficit
                deficits.append((reason, lang, combined, mal_deficit, ben_deficit))

    # Sort deficits by worst (largest combined) first
    deficits.sort(key=lambda x: -x[2])

    # Ledger actions
    ledger = {
        "dropped": manifest.get("ledger_dropped", 0),
        "quarantined": manifest.get("ledger_quarantined", 0),
        "rerouted": manifest.get("ledger_rerouted", 0),
        "relabeled": manifest.get("ledger_relabeled", 0),
    }

    # Background info
    background_actual = manifest.get("background_actual", 0)
    total_actual += background_actual
    discussion_actual = manifest.get("discussion_actual", 0)
    total_actual += discussion_actual

    return {
        "rows": rows,
        "covered": covered_count,
        "total_cells": len(REASONS) * len(LANGUAGES),
        "degraded_count": len(degraded_cells),
        "degraded_cells": degraded_cells,
        "total_backfilled": total_backfilled,
        "total_actual": total_actual,
        "deficits": deficits,
        "ledger": ledger,
        "per_cell_attack": per_cell_attack,
        "per_cell_benign": per_cell_benign,
        "total_target": total_target,
        "budget_fraction": budget_fraction,
        "background_actual": background_actual,
        "discussion_actual": discussion_actual,
    }


# ── Display ──────────────────────────────────────────────────────────

def format_bar(fraction: float, width: int = 20) -> str:
    """Simple ASCII progress bar."""
    filled = min(width, int(round(fraction * width)))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def print_scorecard(sc: dict) -> None:
    """Print human-readable scorecard to stdout."""
    total_cells = sc["total_cells"]
    covered = sc["covered"]
    pct = covered / total_cells * 100 if total_cells else 0

    sep = "=" * 90
    thin_sep = "-" * 90

    print()
    print(sep)
    print("  V5 COVERAGE RECOVERY SCORECARD")
    print(sep)
    print()

    # ── 1. Summary ──
    print(f"  Total target:      {sc['total_target']:>8,}")
    print(f"  Budget fraction:   {sc['budget_fraction']:>8.0%}  (background share of benign)")
    print(f"  Per-cell attack:   {sc['per_cell_attack']:>8,}")
    print(f"  Per-cell benign:   {sc['per_cell_benign']:>8,}")
    print()

    # ── 2. Logical cell coverage ──
    bar = format_bar(covered / total_cells if total_cells else 0)
    print(f"  Logical cell coverage:  {covered}/{total_cells}  ({pct:.1f}%)  {bar}")
    print()

    # ── 3. Per-cell detail table ──
    print(thin_sep)
    print("  PER-CELL DETAIL")
    print(thin_sep)

    # Header
    hdr = (
        f"  {'Reason':<24} {'Lang':<5} "
        f"{'Mal Quota':>9} {'Mal Act':>8} {'Mal Def':>8} "
        f"{'Ben Quota':>9} {'Ben Act':>8} {'Ben Def':>8} "
        f"{'Covered':>7}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    prev_reason = None
    for row in sc["rows"]:
        reason_display = row["reason"] if row["reason"] != prev_reason else ""
        prev_reason = row["reason"]

        mal_def_str = str(row["mal_deficit"]) if row["mal_deficit"] > 0 else ""
        ben_def_str = str(row["ben_deficit"]) if row["ben_deficit"] > 0 else ""
        covered_str = "YES" if row["covered"] else "NO"

        print(
            f"  {reason_display:<24} {row['lang']:<5} "
            f"{row['mal_quota']:>9,} {row['mal_actual']:>8,} {mal_def_str:>8} "
            f"{row['ben_quota']:>9,} {row['ben_actual']:>8,} {ben_def_str:>8} "
            f"{covered_str:>7}"
        )

    print()

    # ── 4. Degraded cells ──
    print(thin_sep)
    print(f"  DEGRADED CELLS:  {sc['degraded_count']}")
    print(thin_sep)
    if sc["degraded_cells"]:
        for reason, side in sorted(sc["degraded_cells"]):
            print(f"    {reason} / {side}")
    else:
        print("    (none)")
    print()

    # ── 5. Backfill summary ──
    print(thin_sep)
    print("  BACKFILL SUMMARY")
    print(thin_sep)
    bf = sc["total_backfilled"]
    total = sc["total_actual"]
    bf_pct = bf / total * 100 if total else 0
    print(f"    Total backfilled rows:  {bf:>8,}")
    print(f"    Total actual rows:      {total:>8,}")
    print(f"    Backfill share:         {bf_pct:>8.1f}%")
    print()

    # ── 6. Lane deficits ──
    print(thin_sep)
    print("  LANE DEFICITS (uncovered cells, worst first)")
    print(thin_sep)

    deficits = sc["deficits"]
    if deficits:
        dhdr = (
            f"    {'Reason':<24} {'Lang':<5} "
            f"{'Mal Def':>8} {'Ben Def':>8} {'Combined':>9}"
        )
        print(dhdr)
        print("    " + "-" * (len(dhdr) - 4))
        for reason, lang, combined, mal_def, ben_def in deficits:
            print(
                f"    {reason:<24} {lang:<5} "
                f"{mal_def:>8,} {ben_def:>8,} {combined:>9,}"
            )
    else:
        print("    (all cells covered)")
    print()

    # ── 7. Ledger actions ──
    print(thin_sep)
    print("  LEDGER ACTIONS")
    print(thin_sep)
    ldg = sc["ledger"]
    print(f"    Dropped:      {ldg['dropped']:>8,}")
    print(f"    Quarantined:  {ldg['quarantined']:>8,}")
    print(f"    Rerouted:     {ldg['rerouted']:>8,}")
    print(f"    Relabeled:    {ldg['relabeled']:>8,}")
    total_actions = sum(ldg.values())
    print(f"    Total:        {total_actions:>8,}")
    print()
    print(sep)
    print()


# ── CLI ──────────────────────────────────────────────────────────────

def infer_total_target(manifest: dict) -> int | None:
    """Try to read total_samples from the manifest."""
    if "total_samples" in manifest:
        return int(manifest["total_samples"])
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V5 coverage recovery scorecard from curation manifest."
    )
    parser.add_argument(
        "manifest",
        help="Path to manifest.json",
    )
    parser.add_argument(
        "--total-target",
        type=int,
        default=None,
        help="Total sample target (default: read from manifest total_samples)",
    )
    parser.add_argument(
        "--budget-fraction",
        type=float,
        default=0.15,
        help="Background budget fraction of benign side (default: 0.15)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest: dict[str, Any] = json.load(f)

    total_target = args.total_target
    if total_target is None:
        total_target = infer_total_target(manifest)
    if total_target is None:
        print(
            "ERROR: Could not determine total_target. "
            "Pass --total-target or ensure manifest has total_samples.",
            file=sys.stderr,
        )
        return 1

    sc = build_scorecard(manifest, total_target, args.budget_fraction)
    print_scorecard(sc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
