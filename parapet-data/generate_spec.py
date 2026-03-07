#!/usr/bin/env python3
"""Expand a compact mirror spec into a full MirrorSpec YAML.

The compact format defines source pools once and per-cell overrides.
This script expands those into the full cell definitions that
parapet-data's curate command expects.

Usage:
    python generate_spec.py mirror_v3.compact.yaml -o mirror_spec_v3_19k.yaml --total-target 19200
    python generate_spec.py mirror_v3.compact.yaml -o mirror_spec_v3.yaml
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml


# All 8 attack reasons, in canonical order matching models.py
ALL_REASONS = [
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


def make_source_ref(
    name: str,
    path: str,
    language: str,
    extractor: str = "col_content",
    label_filter: dict | None = None,
    grounding_mode: str | None = None,
    route_policy: str | None = None,
    reason_provenance: str | None = None,
    applicability_scope: str | None = None,
) -> dict:
    """Build a SourceRef dict for the output YAML."""
    ref: dict = {
        "name": name,
        "path": path,
        "language": language,
        "extractor": extractor,
    }
    if label_filter:
        ref["label_filter"] = label_filter
    if grounding_mode is not None:
        ref["grounding_mode"] = grounding_mode
    if route_policy is not None:
        ref["route_policy"] = route_policy
    if reason_provenance is not None:
        ref["reason_provenance"] = reason_provenance
    if applicability_scope is not None:
        ref["applicability_scope"] = applicability_scope
    return ref


def build_source_refs(source_defs: list[dict]) -> list[dict]:
    """Expand compact source definitions into SourceRef dicts."""
    return [make_source_ref(**src) for src in source_defs]


def build_supplement(compact_supplement: dict) -> dict:
    """Expand a compact supplement into a full Supplement dict."""
    return {
        "name": compact_supplement["name"],
        "weakness": compact_supplement["weakness"],
        "max_samples": compact_supplement["max_samples"],
        "attack_sources": build_source_refs(compact_supplement.get("attack_sources", [])),
        "benign_sources": build_source_refs(compact_supplement.get("benign_sources", [])),
    }


def build_cell(reason: str, compact: dict) -> dict:
    """Expand a single cell from compact spec into full MirrorCell dict."""
    cell_cfg = compact["cells"][reason]

    # --- Attack sources ---
    attack_sources = []

    # Base attack sources (shared across all cells)
    for src in compact.get("base_attack_sources", []):
        attack_sources.append(make_source_ref(**src))

    # Per-cell extra attack sources
    for src in cell_cfg.get("extra_attack_sources", []):
        attack_sources.append(make_source_ref(**src))

    # Staged attack sources (only if this reason is in their reason list)
    for staged_name, staged_cfg in compact.get("staged_attacks", {}).items():
        if reason in staged_cfg["reasons"]:
            attack_sources.append(
                make_source_ref(
                    name=f"{staged_cfg['language'].lower()}_staged_attacks_{reason}",
                    path=staged_cfg["path"],
                    language=staged_cfg["language"],
                    label_filter={"column": "reason", "allowed": [reason]},
                    grounding_mode=staged_cfg.get("grounding_mode"),
                    route_policy=staged_cfg.get("route_policy"),
                    reason_provenance=staged_cfg.get("reason_provenance"),
                    applicability_scope=staged_cfg.get("applicability_scope"),
                )
            )

    # --- Benign sources ---
    benign_sources = []

    # Base benign sources (shared across all cells)
    for src in compact.get("base_benign_sources", []):
        benign_sources.append(make_source_ref(**src))

    # Per-cell extra benign sources
    for src in cell_cfg.get("extra_benign_sources", []):
        benign_sources.append(make_source_ref(**src))

    # Staged multilingual benign (only if this reason is in their reason list)
    for staged_name, staged_cfg in compact.get("staged_benign_multilingual", {}).items():
        if reason in staged_cfg["reasons"]:
            benign_sources.append(
                make_source_ref(
                    name=f"{staged_cfg['language'].lower()}_staged_benign_{reason}",
                    path=staged_cfg["path"],
                    language=staged_cfg["language"],
                    label_filter={"column": "reason", "allowed": [reason]},
                    grounding_mode=staged_cfg.get("grounding_mode"),
                    route_policy=staged_cfg.get("route_policy"),
                    reason_provenance=staged_cfg.get("reason_provenance"),
                    applicability_scope=staged_cfg.get("applicability_scope"),
                )
            )

    # Staged EN benign (each dataset x matching reasons)
    staged_en = compact.get("staged_benign_en", {})
    staged_en_reasons = staged_en.get("reasons", ALL_REASONS)
    if reason in staged_en_reasons:
        for dataset_name, dataset_cfg in staged_en.get("datasets", {}).items():
            benign_sources.append(
                make_source_ref(
                    name=f"en_staged_{dataset_name}_{reason}",
                    path=dataset_cfg["path"],
                    language="EN",
                    label_filter={"column": "reason", "allowed": [reason]},
                    grounding_mode=dataset_cfg.get("grounding_mode"),
                    route_policy=dataset_cfg.get("route_policy"),
                    reason_provenance=dataset_cfg.get("reason_provenance"),
                    applicability_scope=dataset_cfg.get("applicability_scope"),
                )
            )

    # --- Assemble cell ---
    return {
        "reason": reason,
        "teaching_goal": cell_cfg["teaching_goal"],
        "languages": list(LANGUAGES),
        "format_distribution": dict(cell_cfg["format"]),
        "length_distribution": dict(cell_cfg["length"]),
        "attack_sources": attack_sources,
        "benign_sources": benign_sources,
    }


def expand_spec(compact: dict, overrides: dict | None = None) -> dict:
    """Expand compact spec into full MirrorSpec dict."""
    overrides = overrides or {}

    # Auto-generate name/version suffix when total_target is overridden
    total_target = overrides.get("total_target", compact["total_target"])
    if "total_target" in overrides and total_target != compact["total_target"]:
        target_k = total_target // 1000
        if "name" not in overrides:
            overrides["name"] = f"{compact['name']}_{target_k}k_control"
        if "version" not in overrides:
            overrides["version"] = f"{compact['version']}-{target_k}k"

    # Start with top-level fields
    spec: dict = {}
    spec["name"] = overrides.get("name", compact["name"])
    spec["version"] = overrides.get("version", compact["version"])
    spec["seed"] = compact["seed"]
    spec["ratio"] = compact["ratio"]
    spec["total_target"] = total_target
    spec["backfill"] = copy.deepcopy(compact["backfill"])
    spec["language_quota"] = copy.deepcopy(compact["language_quota"])
    spec["allow_partial_mirror"] = compact.get("allow_partial_mirror", False)
    if "supplement_ratio" in compact:
        spec["supplement_ratio"] = compact["supplement_ratio"]
    if compact.get("holdout_only_reasons"):
        spec["holdout_only_reasons"] = copy.deepcopy(compact["holdout_only_reasons"])
    if compact.get("enforce_source_contracts"):
        spec["enforce_source_contracts"] = True

    # Background lane
    if "background" in compact:
        bg = compact["background"]
        spec["background"] = {
            "budget_fraction": bg["budget_fraction"],
            "sources": build_source_refs(bg["sources"]),
        }

    if "supplements" in compact:
        spec["supplements"] = [build_supplement(supp) for supp in compact["supplements"]]

    # Expand cells
    defined_reasons = set(compact["cells"].keys())
    missing = set(ALL_REASONS) - defined_reasons
    if missing and not spec["allow_partial_mirror"]:
        print(f"error: compact spec missing cells for: {missing}", file=sys.stderr)
        sys.exit(1)

    spec["cells"] = []
    for reason in ALL_REASONS:
        if reason in defined_reasons:
            spec["cells"].append(build_cell(reason, compact))

    return spec


def render_yaml(spec: dict) -> str:
    """Render the full spec as YAML with readable formatting."""

    class FlowStyleDumper(yaml.SafeDumper):
        """Custom dumper that uses flow style for source ref dicts."""
        pass

    def str_representer(dumper, data):
        if "\n" in data:
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    FlowStyleDumper.add_representer(str, str_representer)

    return yaml.dump(
        spec,
        Dumper=FlowStyleDumper,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=120,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Expand a compact mirror spec into a full MirrorSpec YAML."
    )
    parser.add_argument("compact_spec", type=Path, help="Path to compact spec YAML")
    parser.add_argument("-o", "--output", type=Path, help="Output path (default: stdout)")
    parser.add_argument(
        "--total-target",
        type=int,
        help="Override total_target from compact spec",
    )
    parser.add_argument(
        "--name",
        help="Override spec name (default: auto-generates from total_target)",
    )
    parser.add_argument(
        "--version",
        help="Override spec version",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing output",
    )
    args = parser.parse_args()

    with open(args.compact_spec, encoding="utf-8-sig") as f:
        compact = yaml.safe_load(f)

    overrides = {}
    if args.total_target:
        overrides["total_target"] = args.total_target
    if args.name:
        overrides["name"] = args.name
    if args.version:
        overrides["version"] = args.version

    spec = expand_spec(compact, overrides)

    if args.dry_run:
        print(f"Name:         {spec['name']}")
        print(f"Version:      {spec['version']}")
        print(f"Total target: {spec['total_target']}")
        print(f"Cells:        {len(spec['cells'])}")
        for cell in spec["cells"]:
            n_atk = len(cell["attack_sources"])
            n_ben = len(cell["benign_sources"])
            print(f"  {cell['reason']:25s}  atk_sources={n_atk:2d}  ben_sources={n_ben:2d}")
        return

    output_text = render_yaml(spec)

    if args.output:
        args.output.write_text(output_text, encoding="utf-8")
        print(f"Wrote {len(output_text):,} bytes to {args.output}", file=sys.stderr)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
