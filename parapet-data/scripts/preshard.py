"""Pre-shard pooled YAML attack files into JSONL batches for swarm classification.

Deduplicates against already-staged content hashes so swarm agents only
classify rows that are actually new.

Usage:
    python scripts/preshard.py \
        --pooled-files schema/eval/training/multilingual/ar_attacks.yaml \
                       schema/eval/training/multilingual/attacks_121624_ar.yaml \
        --staged-files schema/eval/staging/ar_ArabGuard-Egyptian-V1_attacks_staged.yaml \
        --deficit-name ar_meta_probe_adversarial_suffix \
        --language-hint AR \
        --batch-size 200 \
        --max-rows 0 \
        --output-dir parapet-data/adjudication/review/batches/swarm_batches/ar_attack_recovery \
        --seed 42

Run from the parapet/ directory (paths are relative to it).
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import yaml


def content_hash(text: str) -> str:
    """Canonical content hash — matches parapet_data.filters.content_hash."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def load_staged_hashes(staged_files: list[Path]) -> set[str]:
    """Extract content_hash values from staged YAML files."""
    hashes = set()
    for path in staged_files:
        if not path.exists():
            print(f"  WARN: staged file not found: {path}", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as f:
            rows = yaml.safe_load(f) or []
        for row in rows:
            h = row.get("content_hash")
            if h:
                hashes.add(h)
            elif "content" in row:
                hashes.add(content_hash(row["content"]))
        print(f"  staged: {path.name} → {len(rows)} rows", file=sys.stderr)
    return hashes


def load_pooled_rows(pooled_files: list[Path]) -> list[dict]:
    """Load rows from pooled YAML files and compute content_hash where missing."""
    all_rows = []
    for path in pooled_files:
        if not path.exists():
            print(f"  WARN: pooled file not found: {path}", file=sys.stderr)
            continue
        dataset_name = path.stem
        with open(path, encoding="utf-8") as f:
            rows = yaml.safe_load(f) or []
        for i, row in enumerate(rows):
            text = row.get("content", "")
            if not text or not text.strip():
                continue
            h = row.get("content_hash") or content_hash(text)
            row_id = row.get("id", f"{dataset_name}-{i:06d}")
            all_rows.append({
                "row_id": row_id,
                "content_hash": h,
                "text": text,
                "dataset": dataset_name,
                "language_hint": row.get("language", ""),
            })
        print(f"  pooled: {path.name} → {len(rows)} rows loaded", file=sys.stderr)
    return all_rows


def dedup(rows: list[dict], staged_hashes: set[str]) -> list[dict]:
    """Remove rows whose content_hash appears in the staged set."""
    before = len(rows)
    kept = [r for r in rows if r["content_hash"] not in staged_hashes]
    seen = set()
    unique = []
    for r in kept:
        if r["content_hash"] not in seen:
            seen.add(r["content_hash"])
            unique.append(r)
    print(
        f"  dedup: {before} → {len(kept)} (staged overlap: {before - len(kept)}) "
        f"→ {len(unique)} (internal dedup: {len(kept) - len(unique)})",
        file=sys.stderr,
    )
    return unique


def sample_rows(rows: list[dict], max_rows: int, seed: int) -> list[dict]:
    """Sample up to max_rows. If max_rows is 0, return all."""
    if max_rows <= 0 or max_rows >= len(rows):
        return rows
    rng = random.Random(seed)
    sampled = rng.sample(rows, max_rows)
    print(f"  sampled: {len(rows)} → {len(sampled)}", file=sys.stderr)
    return sampled


def write_batches(
    rows: list[dict],
    output_dir: Path,
    batch_size: int,
    deficit_name: str,
    language_hint: str,
) -> list[Path]:
    """Write JSONL batches of at most batch_size rows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_paths = []
    for batch_idx in range(0, len(rows), batch_size):
        batch = rows[batch_idx : batch_idx + batch_size]
        batch_num = (batch_idx // batch_size) + 1
        batch_path = output_dir / f"batch_{batch_num:04d}.jsonl"
        with open(batch_path, "w", encoding="utf-8") as f:
            for row in batch:
                obj = {
                    "row_id": row["row_id"],
                    "content_hash": row["content_hash"],
                    "text": row["text"],
                    "dataset": row["dataset"],
                    "language_hint": language_hint or row.get("language_hint", ""),
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        batch_paths.append(batch_path)
        print(f"  wrote: {batch_path.name} ({len(batch)} rows)", file=sys.stderr)
    return batch_paths


def main():
    parser = argparse.ArgumentParser(description="Pre-shard pooled attacks for swarm classification")
    parser.add_argument("--pooled-files", nargs="+", type=Path, required=True)
    parser.add_argument("--staged-files", nargs="*", type=Path, default=[])
    parser.add_argument("--deficit-name", required=True)
    parser.add_argument("--language-hint", default="")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--max-rows", type=int, default=0, help="0 = all rows")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n=== Pre-sharding: {args.deficit_name} ===", file=sys.stderr)

    print("\nLoading staged hashes for dedup...", file=sys.stderr)
    staged_hashes = load_staged_hashes(args.staged_files)
    print(f"  total staged hashes: {len(staged_hashes)}", file=sys.stderr)

    print("\nLoading pooled rows...", file=sys.stderr)
    rows = load_pooled_rows(args.pooled_files)

    print("\nDeduplicating...", file=sys.stderr)
    rows = dedup(rows, staged_hashes)

    print("\nSampling...", file=sys.stderr)
    rows = sample_rows(rows, args.max_rows, args.seed)

    print("\nWriting batches...", file=sys.stderr)
    batch_paths = write_batches(
        rows, args.output_dir, args.batch_size, args.deficit_name, args.language_hint
    )

    print(f"\n=== Done: {len(batch_paths)} batches, {len(rows)} total rows ===\n", file=sys.stderr)

    # Print summary to stdout as JSON for downstream tooling
    summary = {
        "deficit_name": args.deficit_name,
        "language_hint": args.language_hint,
        "staged_hash_count": len(staged_hashes),
        "total_rows": len(rows),
        "batch_count": len(batch_paths),
        "batch_size": args.batch_size,
        "batch_paths": [str(p) for p in batch_paths],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
