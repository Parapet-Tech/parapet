"""Export candidate attack rows from local corpora into review-ready JSONL.

Run from the parapet/ directory.

Examples:
    python parapet-data/scripts/export_attack_candidates.py \
        --input schema/eval/staging/en_llmail-inject-challenge_attacks_staged.yaml \
        --text-field content \
        --id-field content_hash \
        --where reason=indirect_injection \
        --dataset-name llmail_indirect_seed \
        --language-hint EN \
        --shuffle \
        --limit 400 \
        --output parapet-data/adjudication/review/exports/gap_mining/v5_5k/en_indirect_injection/llmail_seed.jsonl \
        --manifest-output parapet-data/adjudication/review/exports/gap_mining/v5_5k/en_indirect_injection/llmail_seed_manifest.json
"""

from __future__ import annotations

import argparse
import bz2
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml


def content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _open_text(path: Path):
    if path.suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _load_yaml(path: Path) -> list[dict[str, Any]]:
    with _open_text(path) as handle:
        rows = yaml.safe_load(handle) or []
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected top-level YAML list")
    return _ensure_dict_rows(rows, path)


def _load_json(path: Path) -> list[dict[str, Any]]:
    with _open_text(path) as handle:
        obj = json.load(handle)
    if isinstance(obj, list):
        return _ensure_dict_rows(obj, path)
    if isinstance(obj, dict):
        if all(isinstance(value, dict) for value in obj.values()):
            rows: list[dict[str, Any]] = []
            for key, value in obj.items():
                row = dict(value)
                row.setdefault("_json_key", key)
                rows.append(row)
            return rows
        for key in ("rows", "data", "items"):
            value = obj.get(key)
            if isinstance(value, list):
                return _ensure_dict_rows(value, path)
    raise ValueError(f"{path}: unsupported JSON structure")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _open_text(path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def _load_csv_like(path: Path, delimiter: str) -> list[dict[str, Any]]:
    with _open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = [dict(row) for row in reader]
    return rows


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _ensure_dict_rows(rows: list[Any], path: Path) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{index}: expected mapping row")
        normalized.append(row)
    return normalized


def load_rows(path: Path) -> list[dict[str, Any]]:
    suffixes = path.suffixes
    if suffixes[-2:] in ([".jsonl", ".bz2"], [".json", ".bz2"], [".yaml", ".bz2"], [".yml", ".bz2"]):
        logical_suffix = suffixes[-2]
    else:
        logical_suffix = suffixes[-1] if suffixes else ""

    if logical_suffix in {".yaml", ".yml"}:
        return _load_yaml(path)
    if logical_suffix == ".jsonl":
        return _load_jsonl(path)
    if logical_suffix == ".json":
        return _load_json(path)
    if logical_suffix == ".csv":
        return _load_csv_like(path, ",")
    if logical_suffix == ".tsv":
        return _load_csv_like(path, "\t")
    if logical_suffix == ".parquet":
        return _load_parquet(path)
    raise ValueError(f"{path}: unsupported format")


def get_field(row: Any, field: str) -> Any:
    current = row
    for part in field.split("."):
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError) as exc:
                raise KeyError(field) from exc
            continue
        if not isinstance(current, dict) or part not in current:
            raise KeyError(field)
        current = current[part]
    return current


def parse_key_value(expr: str) -> tuple[str, str]:
    if "=" not in expr:
        raise ValueError(f"expected field=value, got: {expr}")
    key, value = expr.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"missing field name in expression: {expr}")
    return key, value


def row_matches(
    row: dict[str, Any],
    where_filters: list[tuple[str, str]],
    contains_filters: list[tuple[str, str]],
    exists_filters: list[str],
) -> bool:
    for field, expected in where_filters:
        try:
            value = get_field(row, field)
        except KeyError:
            return False
        if str(value) != expected:
            return False

    for field, needle in contains_filters:
        try:
            value = get_field(row, field)
        except KeyError:
            return False
        haystack = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        if needle not in haystack:
            return False

    for field in exists_filters:
        try:
            get_field(row, field)
        except KeyError:
            return False

    return True


def normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def copy_fields(row: dict[str, Any], fields: list[tuple[str, str]]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for source_field, alias in fields:
        try:
            copied[alias] = get_field(row, source_field)
        except KeyError:
            continue
    return copied


def export_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    where_filters = [parse_key_value(expr) for expr in args.where]
    contains_filters = [parse_key_value(expr) for expr in args.contains]
    copy_specs = []
    for expr in args.copy_field:
        if ":" in expr:
            source_field, alias = expr.split(":", 1)
        else:
            source_field = expr
            alias = expr.split(".")[-1]
        copy_specs.append((source_field, alias))

    emitted: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    input_summaries: list[dict[str, Any]] = []

    for input_path in args.input:
        dataset_name = args.dataset_name if len(args.input) == 1 and args.dataset_name else input_path.stem
        rows = load_rows(input_path)
        matched = 0
        emitted_before = len(emitted)

        for index, row in enumerate(rows):
            if not row_matches(row, where_filters, contains_filters, args.exists):
                continue

            try:
                raw_text = get_field(row, args.text_field)
            except KeyError:
                continue

            text = normalize_text(raw_text).strip()
            if not text:
                continue

            matched += 1
            output_row: dict[str, Any] = {
                "row_id": None,
                "content_hash": content_hash(text),
                "text": text,
                "dataset": dataset_name,
                "language_hint": args.language_hint,
                "source_path": str(input_path),
            }

            if args.id_field:
                try:
                    output_row["row_id"] = str(get_field(row, args.id_field))
                except KeyError:
                    output_row["row_id"] = f"{dataset_name}-{index:06d}"
            else:
                output_row["row_id"] = f"{dataset_name}-{index:06d}"

            output_row.update(copy_fields(row, copy_specs))

            dedupe_value = output_row["content_hash"] if args.dedupe_key == "content_hash" else output_row["row_id"]
            if dedupe_value in seen_keys:
                continue
            seen_keys.add(dedupe_value)
            emitted.append(output_row)

        input_summaries.append(
            {
                "path": str(input_path),
                "dataset": dataset_name,
                "rows_read": len(rows),
                "rows_matched": matched,
                "rows_emitted": len(emitted) - emitted_before,
            }
        )

    if args.shuffle:
        random.Random(args.seed).shuffle(emitted)

    if args.limit > 0:
        emitted = emitted[: args.limit]

    manifest = {
        "inputs": input_summaries,
        "output": str(args.output),
        "output_rows": len(emitted),
        "text_field": args.text_field,
        "id_field": args.id_field,
        "where": args.where,
        "contains": args.contains,
        "exists": args.exists,
        "copy_field": args.copy_field,
        "language_hint": args.language_hint,
        "dedupe_key": args.dedupe_key,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "limit": args.limit,
    }
    return emitted, manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--text-field", required=True, help="Dotted field path for attack text")
    parser.add_argument("--id-field", help="Optional dotted field path for stable row id")
    parser.add_argument("--dataset-name", help="Dataset label to use when a single input is provided")
    parser.add_argument("--language-hint", default="")
    parser.add_argument("--where", action="append", default=[], help="Exact-match filter: field=value")
    parser.add_argument("--contains", action="append", default=[], help="Substring filter: field=value")
    parser.add_argument("--exists", action="append", default=[], help="Require dotted field to exist")
    parser.add_argument(
        "--copy-field",
        action="append",
        default=[],
        help="Copy a field into the output row. Use field or field:alias.",
    )
    parser.add_argument(
        "--dedupe-key",
        choices=("content_hash", "row_id"),
        default="content_hash",
        help="Deduplicate exported rows by content hash or row id",
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    rows, manifest = export_rows(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
