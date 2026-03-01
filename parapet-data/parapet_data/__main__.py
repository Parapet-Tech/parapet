"""
CLI entry point for parapet-data.

Usage:
    python -m parapet_data curate --spec mirror_v1.json --output ./curated/
    python -m parapet_data curate --spec mirror_v1.json --output ./curated/ --base-dir /data
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .compositor import OutputFormat, compose, composition_report
from .models import MirrorSpec
from .sampler import sample_spec


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def cmd_curate(args: argparse.Namespace) -> None:
    """Run the full curation pipeline: spec -> sample -> compose -> manifest."""
    spec_path = Path(args.spec)
    output_dir = Path(args.output)
    base_dir = Path(args.base_dir) if args.base_dir else spec_path.parent

    # Load spec
    spec_text = spec_path.read_text(encoding="utf-8")
    if spec_path.suffix == ".json":
        spec = MirrorSpec.model_validate_json(spec_text)
    else:
        import yaml
        spec = MirrorSpec.model_validate(yaml.safe_load(spec_text))

    print(f"Spec: {spec.name} v{spec.version}", file=sys.stderr)
    print(f"Cells: {len(spec.cells)}, seed: {spec.seed}", file=sys.stderr)
    if spec.total_target:
        print(f"Target: {spec.total_target:,} samples", file=sys.stderr)
    print(f"Output: {output_dir}", file=sys.stderr)

    # Sample
    print("\nSampling...", file=sys.stderr)
    sampling_result = sample_spec(spec, base_dir=base_dir)
    n_attack = len(sampling_result.attack_samples)
    n_benign = len(sampling_result.benign_samples)
    print(
        f"Sampled: {n_attack + n_benign:,} "
        f"({n_attack:,} attack, {n_benign:,} benign)",
        file=sys.stderr,
    )

    if sampling_result.gaps:
        print(f"\nGaps ({len(sampling_result.gaps)}):", file=sys.stderr)
        for gap in sampling_result.gaps:
            print(f"  - {gap}", file=sys.stderr)

    if sampling_result.cross_contamination_dropped:
        print(
            f"\nCross-contamination dropped: "
            f"{sampling_result.cross_contamination_dropped}",
            file=sys.stderr,
        )

    # Compose
    fmt: OutputFormat = args.format
    print(f"\nComposing splits ({fmt})...", file=sys.stderr)
    manifest = compose(
        spec=spec,
        sampling_result=sampling_result,
        output_dir=output_dir,
        base_dir=base_dir,
        fmt=fmt,
    )

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )

    # Write composition report
    all_samples = sampling_result.attack_samples + sampling_result.benign_samples
    report = composition_report(all_samples)
    report_path = output_dir / "composition.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Summary
    print(f"\nDone.", file=sys.stderr)
    print(f"  Samples: {manifest.total_samples:,}", file=sys.stderr)
    print(f"  Splits:  {', '.join(f'{k}={v.sample_count}' for k, v in manifest.splits.items())}", file=sys.stderr)
    print(f"  Manifest: {manifest_path}", file=sys.stderr)
    print(f"  Semantic hash: {manifest.semantic_hash[:16]}...", file=sys.stderr)
    print(f"  Output hash:   {manifest.output_hash[:16]}...", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="parapet-data",
        description="Mirror-based corpus curation for prompt injection classifiers",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    curate_parser = subparsers.add_parser("curate", help="Run curation pipeline")
    curate_parser.add_argument(
        "--spec", required=True, help="Path to MirrorSpec JSON or YAML file"
    )
    curate_parser.add_argument(
        "--output", required=True, help="Output directory for curated data"
    )
    curate_parser.add_argument(
        "--base-dir",
        default=None,
        help="Base directory for resolving relative source paths (default: spec parent dir)",
    )
    curate_parser.add_argument(
        "--format",
        choices=["yaml", "jsonl"],
        default="yaml",
        help="Output format for split files (default: yaml)",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.command == "curate":
        cmd_curate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
