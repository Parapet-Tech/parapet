#!/usr/bin/env python3
"""Compare two eval.json files and find persistent failures across checkpoints.

Usage:
    python scripts/intersect_residuals.py \
        --eval-a runs/verified_v3_clean/eval.json --label-a v3 \
        --eval-b runs/verified_residual_r3/eval.json --label-b r2

    # Optional: export the persistent set for step-2 auditing
    python scripts/intersect_residuals.py \
        --eval-a runs/verified_v3_clean/eval.json --label-a v3 \
        --eval-b runs/verified_residual_r3/eval.json --label-b r2 \
        --verified-dir ../schema/eval/verified \
        --eval-set verified \
        --output runs/verified_residual_r3/persistent_intersection.jsonl
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-a", type=Path, required=True)
    p.add_argument("--label-a", default="A")
    p.add_argument("--eval-b", type=Path, required=True)
    p.add_argument("--label-b", default="B")
    p.add_argument(
        "--verified-dir",
        type=Path,
        help="Directory containing verified YAML files for row-level enrichment",
    )
    p.add_argument(
        "--eval-set",
        default="verified",
        help="Name written into exported rows (default: verified)",
    )
    p.add_argument(
        "--output",
        type=Path,
        help="Optional export path (.jsonl, .yaml/.yml, or .csv) for persistent rows",
    )
    return p.parse_args()


def load_errors(path: Path) -> tuple[set[str], set[str], dict[str, dict[str, Any]], float | None]:
    """Return (fn_ids, fp_ids, case_id->result_dict, threshold)."""
    print(f"Loading {path}...", file=sys.stderr)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    fn_ids: set[str] = set()
    fp_ids: set[str] = set()
    by_id: dict[str, dict[str, Any]] = {}

    for result in data["results"]:
        cid = str(result["case_id"])
        by_id[cid] = result
        if result["correct"]:
            continue
        if result["label"] == "malicious":
            fn_ids.add(cid)
        else:
            fp_ids.add(cid)

    return fn_ids, fp_ids, by_id, load_threshold(path)


def load_threshold(eval_path: Path) -> float | None:
    errors_path = eval_path.with_name("errors.yaml")
    if not errors_path.exists():
        return None
    try:
        with open(errors_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:  # pragma: no cover - defensive CLI path
        print(f"WARN: failed to load {errors_path}: {exc}", file=sys.stderr)
        return None
    threshold = data.get("eval", {}).get("threshold")
    return float(threshold) if threshold is not None else None


def source_breakdown(case_ids: set[str], lookup: dict[str, dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for cid in case_ids:
        result = lookup.get(cid)
        if result:
            counts[str(result["source"])] += 1
    return counts


def content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def infer_language(source_stem: str, row: dict[str, Any]) -> str | None:
    language = row.get("language")
    if language:
        return str(language).upper()
    for prefix, lang in [("ar_", "AR"), ("ru_", "RU"), ("zh_", "ZH"), ("en_", "EN")]:
        if source_stem.startswith(prefix):
            return lang
    for marker, lang in [("_zh_", "ZH"), ("_ar_", "AR"), ("_ru_", "RU")]:
        if marker in source_stem:
            return lang
    return None


def load_verified_rows(verified_dir: Path, needed_sources: set[str]) -> dict[str, dict[str, Any]]:
    rows_by_source: dict[str, dict[str, Any]] = {}

    for yaml_file in sorted(verified_dir.glob("*.yaml")):
        stem = yaml_file.stem
        if stem not in needed_sources:
            continue

        print(f"Loading verified rows from {yaml_file}...", file=sys.stderr)
        with open(yaml_file, encoding="utf-8") as f:
            rows = yaml.safe_load(f) or []
        if not isinstance(rows, list):
            continue

        indexed: dict[str, Any] = {}
        for idx, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            case_id = str(row.get("id") or f"{stem}_{idx}")
            indexed[case_id] = row
        rows_by_source[stem] = indexed

    return rows_by_source


def build_persistent_rows(
    *,
    case_ids: set[str],
    error_type: str,
    eval_set: str,
    label_a: str,
    label_b: str,
    threshold_a: float | None,
    threshold_b: float | None,
    by_id_a: dict[str, dict[str, Any]],
    by_id_b: dict[str, dict[str, Any]],
    verified_rows: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    exported: list[dict[str, Any]] = []
    missing_verified = 0

    for cid in sorted(case_ids):
        result_a = by_id_a[cid]
        result_b = by_id_b[cid]
        source = str(result_a["source"])
        verified_row = verified_rows.get(source, {}).get(cid)
        if verified_row is None:
            verified_row = {}
            missing_verified += 1

        content = str(verified_row.get("content", ""))
        exported.append(
            {
                "case_id": cid,
                "orig_id": str(verified_row.get("id") or cid),
                "eval_set": eval_set,
                "error_type": error_type,
                "source": source,
                "label": str(result_a["label"]),
                "prediction_a": str(result_a.get("actual", "")),
                "prediction_b": str(result_b.get("actual", "")),
                "expected_a": str(result_a.get("expected", "")),
                "expected_b": str(result_b.get("expected", "")),
                "checkpoint_a": label_a,
                "checkpoint_b": label_b,
                "threshold_a": threshold_a,
                "threshold_b": threshold_b,
                "margin_a": None,
                "margin_b": None,
                "language": infer_language(source, verified_row) or "",
                "reason": verified_row.get("reason"),
                "text_len": len(content),
                "content_hash": str(verified_row.get("content_hash") or content_hash(content)),
                "content": content,
                "detail_a": str(result_a.get("detail", "")),
                "detail_b": str(result_b.get("detail", "")),
                "duration_ms_a": result_a.get("duration_ms"),
                "duration_ms_b": result_b.get("duration_ms"),
                "in_scope_pi": None,
                "audit_disposition": "unreviewed",
                "failure_family": None,
                "notes": "",
            }
        )

    return exported, missing_verified


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False))
                f.write("\n")
        return

    if suffix in {".yaml", ".yml"}:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(rows, f, allow_unicode=True, sort_keys=False)
        return

    if suffix == ".csv":
        fieldnames = list(rows[0].keys()) if rows else []
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return

    raise ValueError(f"unsupported output format for {path} (use .jsonl, .yaml, or .csv)")


def main() -> None:
    args = parse_args()

    fn_a, fp_a, by_id_a, threshold_a = load_errors(args.eval_a)
    fn_b, fp_b, by_id_b, threshold_b = load_errors(args.eval_b)

    fn_both = fn_a & fn_b
    fp_both = fp_a & fp_b

    fn_a_only = fn_a - fn_b
    fn_b_only = fn_b - fn_a
    fp_a_only = fp_a - fp_b
    fp_b_only = fp_b - fp_a

    print(f"\n{'='*60}")
    print("FALSE NEGATIVES (attacks missed)")
    print(f"{'='*60}")
    print(f"  {args.label_a} FN:          {len(fn_a):>8,}")
    print(f"  {args.label_b} FN:          {len(fn_b):>8,}")
    print(f"  Both miss (intersection): {len(fn_both):>8,}")
    print(f"  {args.label_a}-only FN:     {len(fn_a_only):>8,}")
    print(f"  {args.label_b}-only FN:     {len(fn_b_only):>8,}")

    print(f"\n{'='*60}")
    print("FALSE POSITIVES (benign mislabeled)")
    print(f"{'='*60}")
    print(f"  {args.label_a} FP:          {len(fp_a):>8,}")
    print(f"  {args.label_b} FP:          {len(fp_b):>8,}")
    print(f"  Both mislabel:           {len(fp_both):>8,}")
    print(f"  {args.label_a}-only FP:     {len(fp_a_only):>8,}")
    print(f"  {args.label_b}-only FP:     {len(fp_b_only):>8,}")

    print(f"\n{'='*60}")
    print("PERSISTENT FAILURES (L2b candidates)")
    print(f"{'='*60}")
    print(f"  Persistent FN: {len(fn_both):>8,}")
    print(f"  Persistent FP: {len(fp_both):>8,}")
    print(f"  Total:         {len(fn_both) + len(fp_both):>8,}")

    lookup = by_id_a if len(by_id_a) >= len(by_id_b) else by_id_b
    print(f"\n{'='*60}")
    print("PERSISTENT FN - top sources")
    print(f"{'='*60}")
    fn_sources = source_breakdown(fn_both, lookup)
    for src, count in fn_sources.most_common(25):
        print(f"  {count:>6,}  {src}")

    print(f"\n{'='*60}")
    print("PERSISTENT FP - top sources")
    print(f"{'='*60}")
    fp_sources = source_breakdown(fp_both, lookup)
    for src, count in fp_sources.most_common(25):
        print(f"  {count:>6,}  {src}")

    if fn_b:
        print(f"\n{'='*60}")
        print("OVERLAP RATES")
        print(f"{'='*60}")
        print(f"  {len(fn_both)/len(fn_a)*100:.1f}% of {args.label_a} FN are persistent")
        print(f"  {len(fn_both)/len(fn_b)*100:.1f}% of {args.label_b} FN are persistent")
        print(f"  {len(fp_both)/len(fp_a)*100:.1f}% of {args.label_a} FP are persistent")
        print(f"  {len(fp_both)/len(fp_b)*100:.1f}% of {args.label_b} FP are persistent")

    if not args.output:
        return

    if not args.verified_dir:
        raise SystemExit("--verified-dir is required when --output is set")

    persistent_ids = fn_both | fp_both
    needed_sources = {str(by_id_a[cid]["source"]) for cid in persistent_ids}
    needed_sources.update(str(by_id_b[cid]["source"]) for cid in persistent_ids)
    verified_rows = load_verified_rows(args.verified_dir, needed_sources)

    exported_fn, missing_fn = build_persistent_rows(
        case_ids=fn_both,
        error_type="FN",
        eval_set=args.eval_set,
        label_a=args.label_a,
        label_b=args.label_b,
        threshold_a=threshold_a,
        threshold_b=threshold_b,
        by_id_a=by_id_a,
        by_id_b=by_id_b,
        verified_rows=verified_rows,
    )
    exported_fp, missing_fp = build_persistent_rows(
        case_ids=fp_both,
        error_type="FP",
        eval_set=args.eval_set,
        label_a=args.label_a,
        label_b=args.label_b,
        threshold_a=threshold_a,
        threshold_b=threshold_b,
        by_id_a=by_id_a,
        by_id_b=by_id_b,
        verified_rows=verified_rows,
    )
    rows = exported_fn + exported_fp
    write_rows(args.output, rows)

    print(f"\nExported {len(rows):,} persistent rows to {args.output}")
    missing_total = missing_fn + missing_fp
    if missing_total:
        print(
            f"WARN: {missing_total:,} rows were exported without verified-row enrichment",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
