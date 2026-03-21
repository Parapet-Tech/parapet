#!/usr/bin/env python3
"""Reconcile cross-model audit results from Claude and Codex.

Compares audit labels on the overlap set (rows audited by both models).
Agreement = auto-labeled with high confidence.
Disagreement = queued for human review.

Usage:
    python scripts/reconcile_audit.py \
        --claude-dir runs/verified_residual_r3/audit_batches \
        --codex-dir runs/verified_residual_r3/audit_batches \
        --overlap-manifest runs/verified_residual_r3/audit_batches/overlap_manifest.json \
        --output-dir runs/verified_residual_r3/audit_reconciled
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


# Canonical scope derived from family.
# The family label is more precise than the boolean scope flag.
# This eliminates schema-level disagreements where both models agree on
# family but one marked scope differently.
FAMILY_SCOPE: dict[str, bool] = {
    "out_of_scope_harmful_use": False,
    "use_vs_mention": False,
    "multilingual_gap": True,
    "boundary_ambiguity": True,
    "long_text_dilution": True,
    "creative_obfuscation": True,
    "social_engineering": True,
    "indirect_injection": True,
}


def canonicalize_scope(row: dict[str, Any]) -> bool | None:
    """Derive in_scope_pi from failure family, falling back to raw value."""
    family = row.get("audit_failure_family")
    if family in FAMILY_SCOPE:
        return FAMILY_SCOPE[family]
    return row.get("audit_in_scope_pi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claude-dir", type=Path, required=True,
                        help="Directory with claude_batch_*_audit.jsonl files")
    parser.add_argument("--codex-dir", type=Path, required=True,
                        help="Directory with codex_batch_*_audit.jsonl files")
    parser.add_argument("--overlap-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_audit_files(directory: Path, prefix: str) -> dict[str, dict[str, Any]]:
    """Load all audit JSONL files matching prefix, keyed by case_id."""
    results: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob(f"{prefix}*_audit.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                case_id = row.get("case_id")
                if case_id:
                    results[case_id] = row
    return results


def main() -> None:
    args = parse_args()

    overlap_data = json.loads(args.overlap_manifest.read_text(encoding="utf-8"))
    overlap_ids = set(overlap_data["overlap_case_ids"])

    claude_results = load_audit_files(args.claude_dir, "claude_batch_")
    codex_results = load_audit_files(args.codex_dir, "codex_batch_")

    print(f"Claude audit rows: {len(claude_results)}")
    print(f"Codex audit rows:  {len(codex_results)}")
    print(f"Overlap case_ids:  {len(overlap_ids)}")

    # Find overlap rows present in both
    both = overlap_ids & set(claude_results.keys()) & set(codex_results.keys())
    claude_only_overlap = overlap_ids & set(claude_results.keys()) - set(codex_results.keys())
    codex_only_overlap = overlap_ids & set(codex_results.keys()) - set(claude_results.keys())
    missing = overlap_ids - set(claude_results.keys()) - set(codex_results.keys())

    print(f"\nOverlap coverage:")
    print(f"  Both audited:    {len(both)}")
    print(f"  Claude only:     {len(claude_only_overlap)}")
    print(f"  Codex only:      {len(codex_only_overlap)}")
    print(f"  Neither:         {len(missing)}")

    # Compare on overlap
    agreed: list[dict[str, Any]] = []
    disagreed: list[dict[str, Any]] = []

    for case_id in sorted(both):
        c = claude_results[case_id]
        x = codex_results[case_id]

        c_scope = canonicalize_scope(c)
        x_scope = canonicalize_scope(x)

        agree_scope = c_scope == x_scope
        agree_family = c.get("audit_failure_family") == x.get("audit_failure_family")

        record = {
            "case_id": case_id,
            "claude_in_scope_pi": c_scope,
            "codex_in_scope_pi": x_scope,
            "claude_in_scope_pi_raw": c.get("audit_in_scope_pi"),
            "codex_in_scope_pi_raw": x.get("audit_in_scope_pi"),
            "claude_family": c.get("audit_failure_family"),
            "codex_family": x.get("audit_failure_family"),
            "claude_confidence": c.get("audit_confidence"),
            "codex_confidence": x.get("audit_confidence"),
            "claude_notes": c.get("audit_notes"),
            "codex_notes": x.get("audit_notes"),
            "agree_scope": agree_scope,
            "agree_family": agree_family,
            "full_agreement": agree_scope and agree_family,
        }

        if agree_scope and agree_family:
            agreed.append(record)
        else:
            disagreed.append(record)

    total_compared = len(both)
    scope_agree = sum(1 for r in agreed + disagreed if r["agree_scope"])
    family_agree = sum(1 for r in agreed + disagreed if r["agree_family"])
    full_agree = len(agreed)

    print(f"\nAgreement on {total_compared} overlap rows:")
    print(f"  Scope agreement:  {scope_agree}/{total_compared} ({100*scope_agree/max(total_compared,1):.1f}%)")
    print(f"  Family agreement: {family_agree}/{total_compared} ({100*family_agree/max(total_compared,1):.1f}%)")
    print(f"  Full agreement:   {full_agree}/{total_compared} ({100*full_agree/max(total_compared,1):.1f}%)")

    # Disagreement breakdown
    if disagreed:
        print(f"\nDisagreement patterns:")
        patterns = Counter()
        for r in disagreed:
            if not r["agree_scope"]:
                patterns[f"scope: claude={r['claude_in_scope_pi']} vs codex={r['codex_in_scope_pi']}"] += 1
            if not r["agree_family"]:
                patterns[f"family: {r['claude_family']} vs {r['codex_family']}"] += 1
        for pattern, count in patterns.most_common(20):
            print(f"  {pattern}: {count}")

    # Merge all results (non-overlap use whichever model audited them)
    all_auto: list[dict[str, Any]] = []
    all_review: list[dict[str, Any]] = []

    # Overlap rows
    for r in agreed:
        all_auto.append({
            "case_id": r["case_id"],
            "audit_in_scope_pi": r["claude_in_scope_pi"],
            "audit_failure_family": r["claude_family"],
            "audit_confidence": "high",
            "audit_source": "cross_model_agreement",
            "claude_notes": r["claude_notes"],
            "codex_notes": r["codex_notes"],
        })
    for r in disagreed:
        all_review.append(r)

    # Non-overlap rows from Claude
    for case_id, row in claude_results.items():
        if case_id not in overlap_ids:
            all_auto.append({
                "case_id": case_id,
                "audit_in_scope_pi": row.get("audit_in_scope_pi"),
                "audit_failure_family": row.get("audit_failure_family"),
                "audit_confidence": row.get("audit_confidence"),
                "audit_source": "claude_only",
                "audit_notes": row.get("audit_notes"),
            })

    # Non-overlap rows from Codex
    for case_id, row in codex_results.items():
        if case_id not in overlap_ids:
            all_auto.append({
                "case_id": case_id,
                "audit_in_scope_pi": row.get("audit_in_scope_pi"),
                "audit_failure_family": row.get("audit_failure_family"),
                "audit_confidence": row.get("audit_confidence"),
                "audit_source": "codex_only",
                "audit_notes": row.get("audit_notes"),
            })

    # Write outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    auto_path = args.output_dir / "auto_labeled.jsonl"
    with open(auto_path, "w", encoding="utf-8") as f:
        for row in all_auto:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    review_path = args.output_dir / "needs_human_review.jsonl"
    with open(review_path, "w", encoding="utf-8") as f:
        for row in all_review:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "total_overlap": total_compared,
        "scope_agreement_pct": round(100 * scope_agree / max(total_compared, 1), 1),
        "family_agreement_pct": round(100 * family_agree / max(total_compared, 1), 1),
        "full_agreement_pct": round(100 * full_agree / max(total_compared, 1), 1),
        "auto_labeled": len(all_auto),
        "needs_human_review": len(all_review),
        "claude_total": len(claude_results),
        "codex_total": len(codex_results),
    }
    summary_path = args.output_dir / "reconciliation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\nWrote {len(all_auto)} auto-labeled rows to {auto_path}")
    print(f"Wrote {len(all_review)} rows needing human review to {review_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
