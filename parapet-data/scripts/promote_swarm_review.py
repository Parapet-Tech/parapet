from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import yaml


def _iter_input_batches(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.glob("batch_*.jsonl")):
        if "_classified" in path.name or "_repair" in path.name:
            continue
        yield path


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _class_path_for(batch_path: Path) -> Path:
    return batch_path.with_name(batch_path.stem + "_classified.jsonl")


def build_rows(
    input_dir: Path,
    language: str,
    reasons: set[str],
    allowed_confidence: set[str],
) -> tuple[list[dict], Counter, Counter]:
    selected: list[dict] = []
    seen_hashes: set[str] = set()
    by_reason: Counter = Counter()
    by_dataset: Counter = Counter()

    for batch_path in _iter_input_batches(input_dir):
        classified_path = _class_path_for(batch_path)
        if not classified_path.exists():
            continue

        input_rows = _load_jsonl(batch_path)
        classified_rows = _load_jsonl(classified_path)
        input_by_id = {row["row_id"]: row for row in input_rows}

        for row in classified_rows:
            if row.get("language") != language:
                continue
            if row.get("reason") not in reasons:
                continue
            if row.get("confidence") not in allowed_confidence:
                continue

            source_row = input_by_id.get(row["row_id"])
            if source_row is None:
                continue

            content_hash = row["content_hash"]
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            dataset = source_row.get("dataset", "unknown")
            selected.append(
                {
                    "id": row["row_id"],
                    "content": source_row["text"],
                    "content_hash": content_hash,
                    "label": "malicious",
                    "language": row["language"],
                    "reason": row["reason"],
                    "source": dataset,
                    "review_confidence": row["confidence"],
                    "review_rationale": row["rationale"],
                }
            )
            by_reason[row["reason"]] += 1
            by_dataset[dataset] += 1

    return selected, by_reason, by_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote reviewed swarm batches into a staged attack YAML shard."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--reasons", nargs="+", required=True)
    parser.add_argument(
        "--confidence",
        nargs="+",
        default=["high"],
        help="Allowed review confidence labels (default: high).",
    )
    args = parser.parse_args()

    rows, by_reason, by_dataset = build_rows(
        input_dir=args.input_dir,
        language=args.language,
        reasons=set(args.reasons),
        allowed_confidence=set(args.confidence),
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        yaml.safe_dump(
            rows,
            allow_unicode=True,
            sort_keys=False,
            width=1000,
        ),
        encoding="utf-8",
    )

    print(f"wrote {len(rows)} rows to {args.output_file}")
    print(f"reason_counts {dict(sorted(by_reason.items()))}")
    print(f"dataset_counts {dict(sorted(by_dataset.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
