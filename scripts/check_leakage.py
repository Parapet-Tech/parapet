"""
Check train/eval leakage between curation manifests and the H2H eval holdout.

Loads content_hashes from each manifest's train split, hashes the eval holdout
content field, and reports intersection.

Usage:
  cd parapet/
  python scripts/check_leakage.py
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import yaml

try:
    LOADER = yaml.CSafeLoader
except AttributeError:
    LOADER = yaml.SafeLoader

BASE = Path(__file__).resolve().parent.parent  # parapet/

EVAL_HOLDOUT = BASE / "schema" / "eval" / "t3" / "l1_holdout_generalist_curated_100k.yaml"

MANIFESTS = {
    "baseline": BASE / "parapet-data" / "curated_baseline" / "manifest.json",
    "mirror":   BASE / "parapet-data" / "curated" / "manifest.json",
}


def content_hash(text: str) -> str:
    """Same hash as parapet_data.filters.content_hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_manifest_train_hashes(manifest_path: Path) -> set[str]:
    """Extract content_hashes from the train split of a manifest."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return set(manifest["splits"]["train"]["content_hashes"])


def load_manifest_all_hashes(manifest_path: Path) -> set[str]:
    """Extract content_hashes from ALL splits (train+val+holdout)."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    all_hashes = set()
    for split_name, split_data in manifest["splits"].items():
        hashes = set(split_data["content_hashes"])
        print(f"    {split_name}: {len(hashes):,} hashes", file=sys.stderr)
        all_hashes.update(hashes)
    return all_hashes


def main() -> None:
    # Load eval holdout and hash every content field
    print(f"Loading eval holdout: {EVAL_HOLDOUT.name}...", file=sys.stderr, flush=True)
    t0 = time.time()
    with open(EVAL_HOLDOUT, "r", encoding="utf-8") as f:
        eval_data = yaml.load(f, Loader=LOADER)
    print(f"  Loaded {len(eval_data):,} eval cases ({time.time()-t0:.1f}s)", file=sys.stderr)

    print("  Hashing eval content...", file=sys.stderr, flush=True)
    eval_hashes: dict[str, str] = {}  # hash -> truncated content
    for row in eval_data:
        c = row.get("content", "")
        h = content_hash(c)
        eval_hashes[h] = c[:80].replace("\n", "\\n")
    eval_hash_set = set(eval_hashes.keys())
    print(f"  {len(eval_hash_set):,} unique eval hashes", file=sys.stderr)

    # Check each manifest
    for name, manifest_path in MANIFESTS.items():
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Checking {name}: {manifest_path.name}", file=sys.stderr)

        if not manifest_path.exists():
            print(f"  SKIP — manifest not found", file=sys.stderr)
            continue

        # Check train split specifically
        train_hashes = load_manifest_train_hashes(manifest_path)
        print(f"  Train hashes: {len(train_hashes):,}", file=sys.stderr)

        train_overlap = train_hashes & eval_hash_set
        print(f"  Train ∩ Eval: {len(train_overlap):,} overlapping samples", file=sys.stderr)

        if train_overlap:
            print(f"\n  *** LEAKAGE DETECTED: {len(train_overlap)} train samples in eval ***", file=sys.stderr)
            for i, h in enumerate(sorted(train_overlap)[:10]):
                print(f"    [{i+1}] {h[:16]}... '{eval_hashes[h]}'", file=sys.stderr)
            if len(train_overlap) > 10:
                print(f"    ... and {len(train_overlap)-10} more", file=sys.stderr)

        # Also check all splits (val, holdout) for completeness
        print(f"\n  Checking all splits:", file=sys.stderr)
        all_hashes = load_manifest_all_hashes(manifest_path)
        all_overlap = all_hashes & eval_hash_set
        print(f"  All splits ∩ Eval: {len(all_overlap):,} overlapping samples", file=sys.stderr)

        # Summary
        print(f"\n  --- {name.upper()} SUMMARY ---", file=sys.stderr)
        print(f"  Train leakage:     {len(train_overlap):,} / {len(train_hashes):,} ({100*len(train_overlap)/len(train_hashes):.1f}%)", file=sys.stderr)
        print(f"  All-split overlap: {len(all_overlap):,} / {len(all_hashes):,} ({100*len(all_overlap)/len(all_hashes):.1f}%)", file=sys.stderr)
        if len(train_overlap) == 0:
            print(f"  CLEAN — no leakage", file=sys.stderr)
        else:
            print(f"  CONTAMINATED — results unreliable", file=sys.stderr)

    # Also check: does the baseline source pool overlap with eval?
    print(f"\n{'='*60}", file=sys.stderr)
    print("Checking source pools directly:", file=sys.stderr)
    atk_src = BASE / "schema" / "eval" / "t3" / "attacks49521.yaml"
    ben_src = BASE / "schema" / "eval" / "t3" / "global_benign_curated_100k.yaml"

    for src_path, src_name in [(atk_src, "attacks49521"), (ben_src, "benign_100k")]:
        if not src_path.exists():
            print(f"  {src_name}: SKIP — not found", file=sys.stderr)
            continue
        print(f"  Loading {src_name}...", file=sys.stderr, flush=True)
        t0 = time.time()
        with open(src_path, "r", encoding="utf-8") as f:
            pool = yaml.load(f, Loader=LOADER)
        print(f"    {len(pool):,} rows ({time.time()-t0:.1f}s)", file=sys.stderr)

        pool_hashes = set()
        for row in pool:
            c = row.get("content", "")
            pool_hashes.add(content_hash(c))
        overlap = pool_hashes & eval_hash_set
        print(f"    {src_name} ∩ Eval: {len(overlap):,} / {len(pool_hashes):,} ({100*len(overlap)/max(len(pool_hashes),1):.1f}%)", file=sys.stderr)


if __name__ == "__main__":
    main()
