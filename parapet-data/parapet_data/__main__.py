"""
CLI entry point for parapet-data.

Usage:
    python -m parapet_data curate --spec mirror_v1.json --output ./curated/
    python -m parapet_data curate --spec mirror_v1.json --output ./curated/ --base-dir /data
    python -m parapet_data stage --index ../TheWall/INDEX.yaml --output schema/eval/staging/ \
        --holdout-sets schema/eval/l1_holdout.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .compositor import OutputFormat, compose, composition_report
from .models import MirrorSpec, VerifiedSyncManifest
from .sampler import sample_spec


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _write_verified_sync_stats(verified_dir: Path, stats: object) -> Path:
    """Write verified-sync stats JSON sidecar and return its path."""
    stats_path = verified_dir / "sync_stats.json"
    stats_path.write_text(
        json.dumps({
            "files_processed": stats.files_processed,
            "total_input": stats.total_input,
            "passed": stats.passed,
            "dropped": stats.dropped,
            "quarantined": stats.quarantined,
            "rerouted": stats.rerouted,
            "relabeled": stats.relabeled,
        }, indent=2),
        encoding="utf-8",
    )
    return stats_path


def cmd_curate(args: argparse.Namespace) -> None:
    """Run the full curation pipeline: spec -> sample -> compose -> manifest."""
    spec_path = Path(args.spec)
    output_dir = Path(args.output)
    base_dir = Path(args.base_dir) if args.base_dir else spec_path.parent
    ledger = None
    verified_sync_manifest: VerifiedSyncManifest | None = None

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
    if args.ledger:
        from .ledger import Ledger

        ledger_path = Path(args.ledger)
        ledger = Ledger.load(ledger_path)
        print(f"Ledger: {ledger_path} ({len(ledger)} entries)", file=sys.stderr)
        if args.materialize_verified_dir:
            from .verified import sync_verified

            staging_dir = Path(args.verified_staging_dir) if args.verified_staging_dir else (
                base_dir / "schema" / "eval" / "staging"
            )
            verified_dir = Path(args.materialize_verified_dir)
            if not staging_dir.is_absolute():
                staging_dir = (base_dir / staging_dir).resolve()
            if not verified_dir.is_absolute():
                verified_dir = (base_dir / verified_dir).resolve()
            if not staging_dir.exists():
                raise FileNotFoundError(
                    f"verified preflight staging dir not found: {staging_dir}"
                )

            print("\nVerified preflight...", file=sys.stderr)
            print(f"  Staging:  {staging_dir}", file=sys.stderr)
            print(f"  Verified: {verified_dir}", file=sys.stderr)
            stats = sync_verified(staging_dir, verified_dir, ledger)
            stats_path = _write_verified_sync_stats(verified_dir, stats)
            verified_sync_manifest = VerifiedSyncManifest(
                staging_dir=staging_dir,
                verified_dir=verified_dir,
                files_processed=stats.files_processed,
                total_input=stats.total_input,
                passed=stats.passed,
                dropped=stats.dropped,
                quarantined=stats.quarantined,
                rerouted=stats.rerouted,
                relabeled=stats.relabeled,
            )
            print(
                "  Verified sync: "
                f"{stats.files_processed} files, {stats.total_input} rows, "
                f"dropped={stats.dropped}, quarantined={stats.quarantined}, "
                f"rerouted={stats.rerouted}, relabeled={stats.relabeled}",
                file=sys.stderr,
            )
            print(f"  Stats:    {stats_path}", file=sys.stderr)
    elif args.materialize_verified_dir:
        raise ValueError("--materialize-verified-dir requires --ledger")

    # Sample
    print("\nSampling...", file=sys.stderr)
    sampling_result = sample_spec(spec, base_dir=base_dir, ledger=ledger)
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
    if sampling_result.duplicates_dropped:
        print(
            f"Duplicate rows dropped: {sampling_result.duplicates_dropped}",
            file=sys.stderr,
        )
    if (
        sampling_result.ledger_dropped
        or sampling_result.ledger_quarantined
        or sampling_result.ledger_rerouted
        or sampling_result.ledger_relabeled
    ):
        print(
            "Ledger actions: "
            f"dropped={sampling_result.ledger_dropped}, "
            f"quarantined={sampling_result.ledger_quarantined}, "
            f"rerouted={sampling_result.ledger_rerouted}, "
            f"relabeled={sampling_result.ledger_relabeled}",
            file=sys.stderr,
        )

    if (args.min_df is None) != (args.max_features is None):
        raise ValueError("--min-df and --max-features must be provided together")

    # Parse split ratios
    split_ratios = None
    if args.split_ratios:
        parts = args.split_ratios.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"--split-ratios must be 3 colon-separated floats (got {args.split_ratios!r})"
            )
        train_r, val_r, holdout_r = float(parts[0]), float(parts[1]), float(parts[2])
        total = train_r + val_r + holdout_r
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"--split-ratios must sum to 1.0 (got {total:.4f})"
            )
        split_ratios = {"train": train_r, "val": val_r, "holdout": holdout_r}
        print(
            f"Split ratios: train={train_r:.2f}, val={val_r:.2f}, holdout={holdout_r:.2f}",
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
        stratified_split=args.stratified,
        min_df=args.min_df,
        max_features=args.max_features,
        verified_sync=verified_sync_manifest,
        split_ratios=split_ratios,
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
    if manifest.feature_coverage_warnings:
        print(
            f"  Feature coverage warnings: {len(manifest.feature_coverage_warnings)}",
            file=sys.stderr,
        )
        for warning in manifest.feature_coverage_warnings:
            print(f"    - {warning}", file=sys.stderr)
    if manifest.source_alias_warnings:
        print(
            f"  Source alias warnings: {len(manifest.source_alias_warnings)}",
            file=sys.stderr,
        )
        for warning in manifest.source_alias_warnings:
            print(f"    - {warning}", file=sys.stderr)


def cmd_stage(args: argparse.Namespace) -> None:
    """Run the TheWall staging pipeline."""
    from .staging import stage_all

    index_path = Path(args.index)
    output_dir = Path(args.output)
    holdout_paths = [Path(p) for p in args.holdout_sets]

    print(f"Index: {index_path}", file=sys.stderr)
    print(f"Output: {output_dir}", file=sys.stderr)
    print(f"Holdout sets: {len(holdout_paths)}", file=sys.stderr)
    if args.datasets:
        print(f"Filter: {args.datasets}", file=sys.stderr)
    if args.max_rows_per_dataset is not None:
        print(f"Row limit per dataset: {args.max_rows_per_dataset}", file=sys.stderr)
    if args.checkpoint_every_rows:
        print(f"Checkpoint every rows: {args.checkpoint_every_rows}", file=sys.stderr)
    if args.checkpoint_dir:
        print(f"Checkpoint dir: {args.checkpoint_dir}", file=sys.stderr)

    manifest = stage_all(
        index_path=index_path,
        output_dir=output_dir,
        holdout_paths=holdout_paths,
        dataset_filter=args.datasets or None,
        max_rows_per_dataset=args.max_rows_per_dataset,
        checkpoint_every_rows=args.checkpoint_every_rows,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
    )

    print(f"\nStaged: {manifest['total_staged']:,}", file=sys.stderr)
    print(f"Rejected: {manifest['total_rejected']:,}", file=sys.stderr)
    for ds in manifest["datasets_processed"]:
        print(
            f"  {ds['name']}: {ds['rows_staged']}/{ds['rows_read']} staged",
            file=sys.stderr,
        )
        if ds["rejection_reasons"]:
            for gate, count in sorted(ds["rejection_reasons"].items()):
                print(f"    {gate}: {count}", file=sys.stderr)


def cmd_verified_sync(args: argparse.Namespace) -> None:
    """Sync staging through the adjudication ledger to produce verified output."""
    from .ledger import Ledger
    from .verified import sync_verified

    staging_dir = Path(args.staging_dir)
    verified_dir = Path(args.verified_dir)
    ledger_path = Path(args.ledger)

    print(f"Staging: {staging_dir}", file=sys.stderr)
    print(f"Verified: {verified_dir}", file=sys.stderr)
    print(f"Ledger: {ledger_path}", file=sys.stderr)

    ledger = Ledger.load(ledger_path)
    print(f"Ledger entries: {len(ledger)}", file=sys.stderr)

    stats = sync_verified(staging_dir, verified_dir, ledger)

    print(f"\nVerified sync complete.", file=sys.stderr)
    print(f"  Files processed: {stats.files_processed}", file=sys.stderr)
    print(f"  Input rows:      {stats.total_input:,}", file=sys.stderr)
    print(f"  Passed:          {stats.passed:,}", file=sys.stderr)
    print(f"  Dropped:         {stats.dropped:,}", file=sys.stderr)
    print(f"  Quarantined:     {stats.quarantined:,}", file=sys.stderr)
    print(f"  Rerouted:        {stats.rerouted:,}", file=sys.stderr)
    print(f"  Relabeled:       {stats.relabeled:,}", file=sys.stderr)

    stats_path = _write_verified_sync_stats(verified_dir, stats)
    print(f"  Stats: {stats_path}", file=sys.stderr)


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
        "--ledger",
        default=None,
        help="Optional adjudication ledger YAML to apply during curation",
    )
    curate_parser.add_argument(
        "--materialize-verified-dir",
        default=None,
        help=(
            "Optional verified-sync output directory for staged-source preflight "
            "(requires --ledger)"
        ),
    )
    curate_parser.add_argument(
        "--verified-staging-dir",
        default=None,
        help=(
            "Optional staging directory for verified preflight "
            "(default: <base-dir>/schema/eval/staging)"
        ),
    )
    curate_parser.add_argument(
        "--format",
        choices=["yaml", "jsonl"],
        default="yaml",
        help="Output format for split files (default: yaml)",
    )
    curate_parser.add_argument(
        "--stratified",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use stratified split logic (default: true)",
    )
    curate_parser.add_argument(
        "--split-ratios",
        default=None,
        help=(
            "Custom train:val:holdout split ratios as colon-separated floats "
            "(e.g. '0.70:0.12:0.18'). Must sum to 1.0. "
            "Default: 0.8:0.1:0.1"
        ),
    )
    curate_parser.add_argument(
        "--min-df",
        type=int,
        default=None,
        help="Vectorizer min_df for feature-coverage guardrails",
    )
    curate_parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Vectorizer max_features for feature-coverage guardrails",
    )

    stage_parser = subparsers.add_parser(
        "stage", help="Stage TheWall datasets through quality gates"
    )
    stage_parser.add_argument(
        "--index",
        required=True,
        help="Path to TheWall INDEX.yaml",
    )
    stage_parser.add_argument(
        "--output",
        required=True,
        help="Output directory for staged YAMLs and manifest",
    )
    stage_parser.add_argument(
        "--holdout-sets",
        nargs="+",
        required=True,
        help="Eval/tough YAML files to exclude (holdout-leakage protection)",
    )
    stage_parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Only process these dataset names (for pilot runs)",
    )
    stage_parser.add_argument(
        "--max-rows-per-dataset",
        type=int,
        default=None,
        help="Optional hard cap on rows processed per dataset (for fast pilots)",
    )
    stage_parser.add_argument(
        "--checkpoint-every-rows",
        type=int,
        default=5000,
        help="Write progress checkpoints every N rows (default: 5000, 0 disables periodic updates)",
    )
    stage_parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Optional directory for partial checkpoint files (default: --output)",
    )

    vsync_parser = subparsers.add_parser(
        "verified-sync",
        help="Sync staging through adjudication ledger to verified output",
    )
    vsync_parser.add_argument(
        "--staging-dir",
        required=True,
        help="Directory containing staged YAML files",
    )
    vsync_parser.add_argument(
        "--verified-dir",
        required=True,
        help="Output directory for verified YAML files",
    )
    vsync_parser.add_argument(
        "--ledger",
        required=True,
        help="Path to adjudication ledger YAML file",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.command == "curate":
        cmd_curate(args)
    elif args.command == "stage":
        cmd_stage(args)
    elif args.command == "verified-sync":
        cmd_verified_sync(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
