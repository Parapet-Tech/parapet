"""
Create a pruned eval holdout that excludes any content seen in training.

Loads content hashes from both mirror and baseline curation manifests,
then writes a new holdout file containing only unseen eval cases.

Usage:
  cd parapet/
  python scripts/prune_holdout.py
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

# Both curation manifests — collect ALL training content hashes
MANIFESTS = [
    BASE / "parapet-data" / "curated" / "manifest.json",          # mirror
    BASE / "parapet-data" / "curated_baseline" / "manifest.json",  # baseline
]

OUT_PATH = BASE / "schema" / "eval" / "t3" / "l1_holdout_pruned.yaml"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect_train_hashes(manifest_paths: list[Path]) -> set[str]:
    """Collect content hashes from ALL splits of ALL manifests."""
    all_hashes: set[str] = set()
    for path in manifest_paths:
        if not path.exists():
            print(f"  SKIP {path.name} — not found", file=sys.stderr)
            continue
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        name = manifest.get("spec_name", path.parent.name)
        for split_name, split_data in manifest["splits"].items():
            hashes = set(split_data["content_hashes"])
            all_hashes.update(hashes)
            print(f"  {name}/{split_name}: {len(hashes):,} hashes", file=sys.stderr)
    return all_hashes


def main() -> None:
    print("Collecting training hashes from all manifests...", file=sys.stderr)
    train_hashes = collect_train_hashes(MANIFESTS)
    print(f"  Total unique training hashes: {len(train_hashes):,}\n", file=sys.stderr)

    print(f"Loading eval holdout: {EVAL_HOLDOUT.name}...", file=sys.stderr, flush=True)
    t0 = time.time()
    with open(EVAL_HOLDOUT, "r", encoding="utf-8") as f:
        eval_data = yaml.load(f, Loader=LOADER)
    print(f"  {len(eval_data):,} cases ({time.time()-t0:.1f}s)", file=sys.stderr)

    kept = []
    dropped = 0
    for row in eval_data:
        h = content_hash(row.get("content", ""))
        if h in train_hashes:
            dropped += 1
        else:
            kept.append(row)

    malicious = sum(1 for r in kept if r.get("label") == "malicious")
    benign = len(kept) - malicious

    print(f"\nPruned: {dropped:,} overlapping cases removed", file=sys.stderr)
    print(f"Kept:   {len(kept):,} cases ({malicious:,} malicious, {benign:,} benign)", file=sys.stderr)

    print(f"\nWriting {OUT_PATH.name}...", file=sys.stderr, flush=True)
    header = (
        f"# L1 pruned holdout — eval cases not seen in any training set\n"
        f"# Pruned from: {EVAL_HOLDOUT.name} ({len(eval_data):,} cases)\n"
        f"# Removed: {dropped:,} cases overlapping with mirror or baseline training\n"
        f"# Remaining: {len(kept):,} cases ({malicious:,} malicious, {benign:,} benign)\n\n"
    )
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.dump(kept, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=200)

    print(f"Done: {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
