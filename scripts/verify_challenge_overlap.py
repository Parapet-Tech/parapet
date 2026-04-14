"""Verify zero content-hash overlap between v6 corpus and challenge sets.

Usage:
    cd parapet
    python scripts/verify_challenge_overlap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_manifest_hashes(manifest_path: Path) -> set[str]:
    """Extract all content hashes from all splits in a curation manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes: set[str] = set()
    for split_name, split_data in manifest.get("splits", {}).items():
        for h in split_data.get("content_hashes", []):
            hashes.add(h)
    return hashes


def load_sidecar_hashes(hash_path: Path) -> set[str]:
    """Load hashes from a .yaml.hashes sidecar file (one hash per line)."""
    return {
        line.strip()
        for line in hash_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main() -> int:
    base = Path(__file__).resolve().parent.parent

    default_manifest = base / "parapet-data" / "curated" / "v7_35k_experiment" / "manifest.json"
    if not default_manifest.exists():
        default_manifest = base / "parapet-data" / "curated" / "v6_25k_experiment" / "manifest.json"
    if not default_manifest.exists():
        default_manifest = base / "parapet-data" / "curated" / "v6_25k" / "manifest.json"
    manifest_path = default_manifest
    challenge_hash_files = [
        (
            "tough_attack_v2",
            base / "schema" / "eval" / "challenges" / "tough_attack_v2"
            / "tough_attack_v6_novel.yaml.hashes",
        ),
        (
            "tough_neutral_v2",
            base / "schema" / "eval" / "challenges" / "tough_neutral_v2"
            / "tough_neutral_v6_novel.yaml.hashes",
        ),
    ]

    print(f"Loading v6 manifest hashes from {manifest_path}...")
    v6_hashes = load_manifest_hashes(manifest_path)
    print(f"  v6 corpus: {len(v6_hashes):,} unique content hashes")

    all_ok = True
    for name, hash_path in challenge_hash_files:
        if not hash_path.exists():
            print(f"  {name}: MISSING hash file {hash_path}", file=sys.stderr)
            all_ok = False
            continue

        challenge_hashes = load_sidecar_hashes(hash_path)
        overlap = v6_hashes & challenge_hashes
        print(f"  {name}: {len(challenge_hashes):,} hashes, overlap={len(overlap)}")

        if overlap:
            print(f"    FAIL: {len(overlap)} overlapping hashes!", file=sys.stderr)
            for h in sorted(overlap)[:10]:
                print(f"      {h}", file=sys.stderr)
            if len(overlap) > 10:
                print(f"      ... and {len(overlap) - 10} more", file=sys.stderr)
            all_ok = False

    if all_ok:
        print("\nPASS: Zero overlap between v6 corpus and all challenge sets.")
        return 0
    else:
        print("\nFAIL: Overlap detected — challenge sets are not clean holdouts.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
