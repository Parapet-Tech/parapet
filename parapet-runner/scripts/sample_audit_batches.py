#!/usr/bin/env python3
"""Generate stratified audit batches for parallel cross-model annotation.

Reads the pre-audited persistent intersection file and produces:
  - 4 batch JSONL files (2 for Claude, 2 for Codex)
  - An overlap manifest showing which rows appear in both model families
  - A summary of batch composition

Batch assignment:
  Claude Agent 1 / Codex Agent 1: out_of_scope_harmful_use + boundary_ambiguity
  Claude Agent 2 / Codex Agent 2: use_vs_mention + multilingual_gap

Within each bucket, 25% of sampled rows are overlap (audited by both models).
The remaining 75% are split evenly between Claude-only and Codex-only.

Usage:
    python scripts/sample_audit_batches.py \
        --input runs/verified_residual_r3/persistent_intersection_preaudit.jsonl \
        --output-dir runs/verified_residual_r3/audit_batches \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# Sample sizes per bucket. Short-text buckets get larger samples.
BUCKET_SAMPLE_SIZES = {
    "out_of_scope_harmful_use": 200,
    "use_vs_mention": 200,
    "multilingual_gap": 200,
    "boundary_ambiguity": 100,
}

# Which buckets go to which agent pair
AGENT_PAIR_1_BUCKETS = ["out_of_scope_harmful_use", "boundary_ambiguity"]
AGENT_PAIR_2_BUCKETS = ["use_vs_mention", "multilingual_gap"]

OVERLAP_FRACTION = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stratified_sample(
    rows: list[dict[str, Any]],
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Draw a stratified sample proportional to source distribution."""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row.get("source", "unknown")].append(row)

    sampled: list[dict[str, Any]] = []
    remaining = n

    # Sort sources by size descending for deterministic allocation
    sources_sorted = sorted(by_source.keys(), key=lambda s: -len(by_source[s]))

    # Proportional allocation with minimum 1 per source
    allocations: dict[str, int] = {}
    total = len(rows)
    for source in sources_sorted:
        source_rows = by_source[source]
        alloc = max(1, round(len(source_rows) / total * n))
        alloc = min(alloc, len(source_rows), remaining)
        allocations[source] = alloc
        remaining -= alloc
        if remaining <= 0:
            break

    # Distribute any leftover to largest sources
    if remaining > 0:
        for source in sources_sorted:
            can_add = len(by_source[source]) - allocations.get(source, 0)
            add = min(can_add, remaining)
            allocations[source] = allocations.get(source, 0) + add
            remaining -= add
            if remaining <= 0:
                break

    for source, alloc in allocations.items():
        pool = by_source[source]
        rng.shuffle(pool)
        sampled.extend(pool[:alloc])

    rng.shuffle(sampled)
    return sampled[:n]


def split_overlap(
    sample: list[dict[str, Any]],
    overlap_frac: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split sample into overlap, claude-only, codex-only."""
    n = len(sample)
    n_overlap = max(1, round(n * overlap_frac))
    n_rest = n - n_overlap
    n_claude = n_rest // 2
    n_codex = n_rest - n_claude

    rng.shuffle(sample)
    overlap = sample[:n_overlap]
    claude_only = sample[n_overlap : n_overlap + n_claude]
    codex_only = sample[n_overlap + n_claude :]

    return overlap, claude_only, codex_only


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Group by bucket
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family = row.get("suggested_failure_family")
        if family in BUCKET_SAMPLE_SIZES:
            by_bucket[family].append(row)

    # Sample and split each bucket
    claude_agent1: list[dict[str, Any]] = []
    claude_agent2: list[dict[str, Any]] = []
    codex_agent1: list[dict[str, Any]] = []
    codex_agent2: list[dict[str, Any]] = []
    overlap_all: list[dict[str, Any]] = []

    summary: dict[str, Any] = {"buckets": {}}

    for bucket, sample_size in BUCKET_SAMPLE_SIZES.items():
        pool = by_bucket[bucket]
        sample = stratified_sample(pool, sample_size, rng)
        overlap, claude_only, codex_only = split_overlap(sample, OVERLAP_FRACTION, rng)

        overlap_all.extend(overlap)

        # Tag each row with its split assignment
        for row in overlap:
            row["_audit_split"] = "overlap"
        for row in claude_only:
            row["_audit_split"] = "claude_only"
        for row in codex_only:
            row["_audit_split"] = "codex_only"

        # Assign to agent pair
        if bucket in AGENT_PAIR_1_BUCKETS:
            claude_agent1.extend(overlap)
            claude_agent1.extend(claude_only)
            codex_agent1.extend(overlap)
            codex_agent1.extend(codex_only)
        else:
            claude_agent2.extend(overlap)
            claude_agent2.extend(claude_only)
            codex_agent2.extend(overlap)
            codex_agent2.extend(codex_only)

        # Source breakdown for summary
        source_counts = Counter(r["source"] for r in sample)
        summary["buckets"][bucket] = {
            "pool_size": len(pool),
            "sample_size": len(sample),
            "overlap": len(overlap),
            "claude_only": len(claude_only),
            "codex_only": len(codex_only),
            "sources": dict(source_counts.most_common()),
        }

    summary["batch_sizes"] = {
        "claude_agent1": len(claude_agent1),
        "claude_agent2": len(claude_agent2),
        "codex_agent1": len(codex_agent1),
        "codex_agent2": len(codex_agent2),
        "total_overlap": len(overlap_all),
    }

    # Write batch files
    args.output_dir.mkdir(parents=True, exist_ok=True)

    batches = {
        "claude_batch_1.jsonl": claude_agent1,
        "claude_batch_2.jsonl": claude_agent2,
        "codex_batch_1.jsonl": codex_agent1,
        "codex_batch_2.jsonl": codex_agent2,
    }

    for filename, batch_rows in batches.items():
        path = args.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            for row in batch_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {filename}: {len(batch_rows)} rows")

    # Write overlap manifest (case_ids only)
    overlap_ids = [r["case_id"] for r in overlap_all]
    overlap_path = args.output_dir / "overlap_manifest.json"
    overlap_path.write_text(
        json.dumps({"overlap_case_ids": overlap_ids, "count": len(overlap_ids)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"  overlap_manifest.json: {len(overlap_ids)} overlap rows")

    # Write summary
    summary_path = args.output_dir / "batch_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\n  batch_summary.json written")

    print(f"\nTotal overlap rows for agreement measurement: {len(overlap_ids)}")
    print("Agent pair 1 (out_of_scope + boundary): Claude={}, Codex={}".format(
        len(claude_agent1), len(codex_agent1)
    ))
    print("Agent pair 2 (use_vs_mention + multilingual): Claude={}, Codex={}".format(
        len(claude_agent2), len(codex_agent2)
    ))


if __name__ == "__main__":
    main()
