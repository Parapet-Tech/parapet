"""Residual feature profile CLI."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from .features import (
    FEATURE_FAMILIES,
    build_feature_table,
    feature_auc_tables,
    semantics_receipt,
    top_decile_enrichment,
)
from .geometry import borderline_squash_sweep, default_policy_specs, evaluate_policies
from .io import input_receipts, load_inputs, output_receipts, write_json, write_jsonl


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def package_version() -> str | None:
    try:
        return metadata.version("parapet-runner")
    except metadata.PackageNotFoundError:
        return None


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def _policy_by_name(profile: dict[str, Any], name: str) -> dict[str, Any] | None:
    for policy in profile["policy_tables"]:
        if policy["name"] == name:
            return policy
    return None


def _best_sweep_row(profile: dict[str, Any]) -> dict[str, Any] | None:
    rows = [
        row for row in profile.get("borderline_squash_sweep", [])
        if row.get("false_negative_recovered", 0) > 0
        and row.get("min_fn_value_for_break_even") is not None
    ]
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: (
            row["min_fn_value_for_break_even"],
            -row["false_negative_recovered"],
            row["language_gate"],
            row["borderline_band"],
        ),
    )


def build_profile(
    *,
    residuals_path: Path,
    baseline_correct_path: Path,
    export_manifest_path: Path,
    output_dir: Path,
    borderline_band: float,
) -> dict[str, Any]:
    bundle = load_inputs(residuals_path, baseline_correct_path, export_manifest_path)
    policies = evaluate_policies(
        bundle.residuals,
        bundle.baseline_correct,
        default_policy_specs(borderline_band),
    )
    feature_table = build_feature_table(bundle.residuals, bundle.baseline_correct)
    auc_tables = feature_auc_tables(feature_table)

    top_deciles = {
        family: [
            top_decile_enrichment(feature_table, row["field"])
            for row in rows[:5]
            if row.get("auc") is not None
        ]
        for family, rows in auc_tables.items()
    }

    profile = {
        "schema_version": 1,
        "created_utc": utc_now_iso(),
        "package": {
            "name": "parapet-runner",
            "version": package_version(),
        },
        "inputs": input_receipts(bundle),
        "row_counts": {
            "residuals": len(bundle.residuals),
            "baseline_correct": len(bundle.baseline_correct),
            "feature_table": len(feature_table),
        },
        "args": {
            "borderline_band": borderline_band,
        },
        "policy_tables": policies,
        "borderline_squash_sweep": borderline_squash_sweep(bundle.residuals, bundle.baseline_correct),
        "feature_auc_by_family": auc_tables,
        "top_decile_enrichment_by_family": top_deciles,
        "feature_families": FEATURE_FAMILIES,
        "feature_family_versions": {
            "l2_geometry": "v1",
            "l0_deltas": "v1_python_mirror_unverified",
            "mechanical_text_shape": "v1",
            "entropy_compression": "v1",
        },
        "semantics": semantics_receipt(),
        "promotion_gate": {
            "requires_fn_recovery_at_operating_point": True,
            "requires_sidecar_cost_at_same_threshold": True,
            "requires_near_boundary_benign_cost_at_same_threshold": True,
            "requires_explicit_cost_ratio_for_enforcement": True,
            "requires_per_language_breakdown": True,
            "requires_source_concentration_check": True,
            "requires_python_rust_parity_before_runtime_signal": True,
            "requires_fused_delta_over_best_family_if_tree_used": True,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_table_path = output_dir / "feature_table.jsonl"
    profile_json_path = output_dir / "feature_profile.json"
    profile_md_path = output_dir / "feature_profile.md"
    manifest_path = output_dir / "manifest.json"

    write_jsonl(feature_table_path, feature_table)
    write_json(profile_json_path, profile)
    profile_md_path.write_text(render_markdown(profile), encoding="utf-8", newline="\n")

    manifest = {
        "schema_version": 1,
        "created_utc": utc_now_iso(),
        "command": "python -m parapet_runner.residuals.profile",
        "package": profile["package"],
        "inputs": profile["inputs"],
        "args": {
            "residuals": str(residuals_path),
            "baseline_correct": str(baseline_correct_path),
            "manifest": str(export_manifest_path),
            "output_dir": str(output_dir),
            "borderline_band": borderline_band,
        },
        "outputs": output_receipts({
            "feature_table": feature_table_path,
            "feature_profile_json": profile_json_path,
            "feature_profile_md": profile_md_path,
        }),
        "semantics": profile["semantics"],
        "feature_families": FEATURE_FAMILIES,
        "feature_family_versions": profile["feature_family_versions"],
    }
    write_json(manifest_path, manifest)

    return {
        "profile": profile,
        "manifest": manifest,
        "paths": {
            "feature_table": str(feature_table_path),
            "feature_profile_json": str(profile_json_path),
            "feature_profile_md": str(profile_md_path),
            "manifest": str(manifest_path),
        },
    }


def render_markdown(profile: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Residual Feature Profile")
    lines.append("")
    lines.append(f"Created: {profile['created_utc']}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This is offline residual analysis only. It does not define or ship runtime sensors.")
    lines.append("Raw content is intentionally omitted from this report.")
    lines.append("")
    lines.extend(_md_table(
        ["artifact", "rows", "sha256"],
        [
            ["residuals", profile["inputs"]["residuals"]["rows"], profile["inputs"]["residuals"]["sha256"]],
            [
                "baseline_correct",
                profile["inputs"]["baseline_correct"]["rows"],
                profile["inputs"]["baseline_correct"]["sha256"],
            ],
        ],
    ))
    lines.append("")
    lines.append("## Policy Operating Points")
    lines.append("")
    lines.extend(_md_table(
        [
            "policy",
            "FN recovered",
            "FN recovery",
            "hard-negative blocks",
            "added hard-negative blocks",
            "sidecar blocks",
            "sidecar rate",
            "recovered FN source HHI",
        ],
        [
            [
                policy["name"],
                f"{policy['false_negative_recovered']}/{policy['false_negative_n']}",
                _fmt_pct(policy["false_negative_recovered_share"]),
                f"{policy['hard_negative_blocks']}/{policy['hard_negative_n']}",
                policy["added_hard_negative_blocks"],
                f"{policy['sidecar_blocks']}/{policy['sidecar_n']}",
                _fmt_pct(policy["sidecar_block_share"]),
                _fmt_float(policy["recovered_fn_source_concentration"]["hhi"]),
            ]
            for policy in profile["policy_tables"]
        ],
    ))
    lines.append("")
    best_sweep = _best_sweep_row(profile)
    if best_sweep is not None:
        lines.append("## Policy Interpretation")
        lines.append("")
        lines.append(
            "The squash-borderline policy recovers real false negatives, but the "
            "current sweep does not justify default enforcement by itself."
        )
        lines.append("")
        lines.append(
            "The limiting cost is not the benign sidecar in this run; it is newly "
            "blocked near-boundary benign rows in the residual set."
        )
        lines.append("")
        lines.append(
            "Best observed break-even in the tested sweep: "
            f"`{best_sweep['language_gate']}` band `{best_sweep['borderline_band']}` "
            f"recovers `{best_sweep['false_negative_recovered']}` FNs with "
            f"`{best_sweep['added_near_boundary_benign_blocks']}` added "
            "near-boundary-benign blocks "
            f"(`{best_sweep['min_fn_value_for_break_even']:.2f}` per recovered FN)."
        )
        lines.append("")
        lines.append(
            "Treat this as a shadow/high-security-mode candidate unless a product "
            "cost ratio explicitly accepts that trade, or a future fused gate "
            "reduces the near-boundary-benign cost."
        )
        lines.append("")
    lines.append("## Borderline Window Sweep")
    lines.append("")
    lines.append(
        "`min FN value for break-even` is the number of newly blocked "
        "near-boundary benign rows per recovered FN. Lower is better."
    )
    lines.append("")
    lines.extend(_md_table(
        [
            "language gate",
            "band",
            "FN recovered",
            "FN recovery",
            "added near-boundary benign",
            "sidecar blocks",
            "min FN value for break-even",
            "source HHI",
        ],
        [
            [
                row["language_gate"],
                row["borderline_band"],
                row["false_negative_recovered"],
                _fmt_pct(row["false_negative_recovered_share"]),
                row["added_near_boundary_benign_blocks"],
                f"{row['sidecar_blocks']}/{row['sidecar_n']}",
                _fmt_float(row["min_fn_value_for_break_even"]),
                _fmt_float(row["source_hhi"]),
            ]
            for row in profile["borderline_squash_sweep"]
        ],
    ))
    lines.append("")

    squash_policy = _policy_by_name(profile, "squash_when_raw_borderline")
    if squash_policy is not None:
        lines.append("## Squash-Borderline Detail")
        lines.append("")
        lines.append("### Blocks by residual category")
        lines.append("")
        lines.extend(_md_table(
            ["category", "n", "blocks", "block share", "added blocks", "added share"],
            [
                [
                    row["residual_category"],
                    row["n"],
                    row["blocks"],
                    _fmt_pct(row["block_share"]),
                    row["added_blocks"],
                    _fmt_pct(row["added_block_share"]),
                ]
                for row in squash_policy["by_residual_category"]
            ],
        ))
        lines.append("")
        lines.append("### FN recovery by language")
        lines.append("")
        lines.extend(_md_table(
            ["language", "FN recovered", "FN recovery", "hard-negative blocks", "sidecar blocks"],
            [
                [
                    row["value"],
                    f"{row['false_negative_recovered']}/{row['false_negative_n']}",
                    _fmt_pct(row["false_negative_recovered_share"]),
                    f"{row['hard_negative_blocks']}/{row['hard_negative_n']}",
                    f"{row['sidecar_blocks']}/{row['sidecar_n']}",
                ]
                for row in squash_policy["by_language"]
                if row["false_negative_n"] or row["hard_negative_n"] or row["sidecar_n"]
            ],
        ))
        lines.append("")
        lines.append("### FN recovery by reason")
        lines.append("")
        reason_rows = [
            row for row in squash_policy["by_reason"]
            if row["false_negative_n"] or row["hard_negative_n"]
        ]
        reason_rows.sort(key=lambda row: (-row["false_negative_recovered"], row["value"]))
        lines.extend(_md_table(
            ["reason", "FN recovered", "FN recovery", "hard-negative blocks", "sidecar blocks"],
            [
                [
                    row["value"],
                    f"{row['false_negative_recovered']}/{row['false_negative_n']}",
                    _fmt_pct(row["false_negative_recovered_share"]),
                    f"{row['hard_negative_blocks']}/{row['hard_negative_n']}",
                    f"{row['sidecar_blocks']}/{row['sidecar_n']}",
                ]
                for row in reason_rows[:20]
            ],
        ))
        lines.append("")

    lines.append("## Feature Families")
    lines.append("")
    for family, rows in profile["feature_auc_by_family"].items():
        lines.append(f"### {family}")
        lines.append("")
        lines.extend(_md_table(
            ["feature", "n", "AUC", "abs(AUC-0.5)"],
            [
                [
                    row["field"],
                    row["n"],
                    _fmt_float(row["auc"]),
                    _fmt_float(row["separation"]),
                ]
                for row in rows[:12]
            ],
        ))
        lines.append("")
    lines.append("## Promotion Gate")
    lines.append("")
    for key, value in profile["promotion_gate"].items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("Python feature results are analysis-only until a Rust parity fixture exists.")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residuals", type=Path, required=True)
    parser.add_argument("--baseline-correct", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, dest="export_manifest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--borderline-band", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_profile(
        residuals_path=args.residuals,
        baseline_correct_path=args.baseline_correct,
        export_manifest_path=args.export_manifest,
        output_dir=args.output_dir,
        borderline_band=args.borderline_band,
    )
    print(f"Feature table: {result['paths']['feature_table']}")
    print(f"Profile JSON:  {result['paths']['feature_profile_json']}")
    print(f"Profile MD:    {result['paths']['feature_profile_md']}")
    print(f"Manifest:      {result['paths']['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
