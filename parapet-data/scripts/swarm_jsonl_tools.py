"""Utilities for bounded swarm-classification JSONL workflows.

This script supports three operations:

1. re-shard existing JSONL batches into smaller fixed-size batches
2. audit a classified output against its input and emit a repair batch for
   missing rows
3. merge a primary output with a repair output back into input order

Run from the parapet/ directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def key_counts(rows: list[dict], key: str) -> Counter:
    return Counter(row.get(key) for row in rows)


def ensure_keys(rows: list[dict], key: str, path: Path) -> None:
    missing = [index + 1 for index, row in enumerate(rows) if key not in row]
    if missing:
        preview = ", ".join(str(index) for index in missing[:5])
        raise ValueError(f"{path}: missing '{key}' in rows {preview}")


def command_reshard(args: argparse.Namespace) -> int:
    input_dir = args.input_dir
    input_files = sorted(input_dir.glob(args.glob))
    if not input_files:
        raise ValueError(f"no files matched {args.glob!r} in {input_dir}")

    all_rows: list[dict] = []
    for path in input_files:
        all_rows.extend(load_jsonl(path))

    batch_size = args.batch_size
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    output_dir = args.output_dir
    batch_paths: list[str] = []
    batch_number = args.start_index
    for offset in range(0, len(all_rows), batch_size):
        batch_rows = all_rows[offset : offset + batch_size]
        batch_path = output_dir / f"{args.prefix}_{batch_number:04d}.jsonl"
        write_jsonl(batch_path, batch_rows)
        batch_paths.append(str(batch_path))
        batch_number += 1

    summary = {
        "mode": "reshard",
        "input_dir": str(input_dir),
        "glob": args.glob,
        "input_file_count": len(input_files),
        "input_row_count": len(all_rows),
        "output_dir": str(output_dir),
        "batch_size": batch_size,
        "batch_count": len(batch_paths),
        "batch_paths": batch_paths,
    }
    print(json.dumps(summary, indent=2))
    return 0


def command_audit(args: argparse.Namespace) -> int:
    input_rows = load_jsonl(args.input_file)
    output_rows = load_jsonl(args.output_file)
    ensure_keys(input_rows, args.key, args.input_file)
    ensure_keys(output_rows, args.key, args.output_file)

    input_keys = [row[args.key] for row in input_rows]
    output_keys = [row[args.key] for row in output_rows]
    output_seen = set(output_keys)

    duplicates = {key: count for key, count in key_counts(output_rows, args.key).items() if count > 1}
    missing_rows = [row for row in input_rows if row[args.key] not in output_seen]

    summary = {
        "mode": "audit",
        "input_file": str(args.input_file),
        "output_file": str(args.output_file),
        "input_rows": len(input_rows),
        "output_rows": len(output_rows),
        "input_unique_keys": len(set(input_keys)),
        "output_unique_keys": len(set(output_keys)),
        "duplicate_keys": duplicates,
        "missing_count": len(missing_rows),
        "missing_keys": [row[args.key] for row in missing_rows],
        "ok": not duplicates and not missing_rows and len(input_rows) == len(output_rows),
    }

    if args.repair_file and missing_rows:
        write_jsonl(args.repair_file, missing_rows)
        summary["repair_file"] = str(args.repair_file)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def command_merge(args: argparse.Namespace) -> int:
    input_rows = load_jsonl(args.input_file)
    primary_rows = load_jsonl(args.primary_output)
    repair_rows = load_jsonl(args.repair_output)
    ensure_keys(input_rows, args.key, args.input_file)
    ensure_keys(primary_rows, args.key, args.primary_output)
    ensure_keys(repair_rows, args.key, args.repair_output)

    merged_by_key: dict[str, dict] = {}
    for row in primary_rows + repair_rows:
        key = row[args.key]
        merged_by_key.setdefault(key, row)

    missing_after_merge = [row[args.key] for row in input_rows if row[args.key] not in merged_by_key]
    if missing_after_merge:
        raise ValueError(f"cannot merge: still missing keys {missing_after_merge[:5]}")

    merged_rows = [merged_by_key[row[args.key]] for row in input_rows]
    write_jsonl(args.merged_output, merged_rows)

    summary = {
        "mode": "merge",
        "input_file": str(args.input_file),
        "primary_output": str(args.primary_output),
        "repair_output": str(args.repair_output),
        "merged_output": str(args.merged_output),
        "merged_rows": len(merged_rows),
        "unique_keys": len({row[args.key] for row in merged_rows}),
    }
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Utilities for swarm JSONL workflows")
    subparsers = parser.add_subparsers(dest="command", required=True)

    reshard = subparsers.add_parser("reshard", help="re-shard JSONL batches into smaller fixed-size batches")
    reshard.add_argument("--input-dir", type=Path, required=True)
    reshard.add_argument("--glob", default="batch_*.jsonl")
    reshard.add_argument("--output-dir", type=Path, required=True)
    reshard.add_argument("--batch-size", type=int, default=25)
    reshard.add_argument("--prefix", default="batch")
    reshard.add_argument("--start-index", type=int, default=1)
    reshard.set_defaults(func=command_reshard)

    audit = subparsers.add_parser("audit", help="audit classified output and optionally emit a repair batch")
    audit.add_argument("--input-file", type=Path, required=True)
    audit.add_argument("--output-file", type=Path, required=True)
    audit.add_argument("--repair-file", type=Path)
    audit.add_argument("--key", default="row_id")
    audit.set_defaults(func=command_audit)

    merge = subparsers.add_parser("merge", help="merge primary and repair outputs back into input order")
    merge.add_argument("--input-file", type=Path, required=True)
    merge.add_argument("--primary-output", type=Path, required=True)
    merge.add_argument("--repair-output", type=Path, required=True)
    merge.add_argument("--merged-output", type=Path, required=True)
    merge.add_argument("--key", default="row_id")
    merge.set_defaults(func=command_merge)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
