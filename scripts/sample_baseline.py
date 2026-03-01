"""
Create baseline training files by evenly sampling from unstructured pools.

Shuffle, then take every Nth sample for a deterministic subsample.

Usage:
  cd parapet/
  python scripts/sample_baseline.py
"""

import random
import sys
import time
from pathlib import Path

import yaml

try:
    LOADER = yaml.CSafeLoader
except AttributeError:
    LOADER = yaml.SafeLoader

SEED = 42
ATK_EVERY = 33   # ~49521/33 ≈ 1500
BEN_EVERY = 67   # ~100000/67 ≈ 1500

BASE = Path(__file__).resolve().parent.parent  # parapet/

ATK_SRC = BASE / "schema" / "eval" / "t3" / "attacks49521.yaml"
BEN_SRC = BASE / "schema" / "eval" / "t3" / "global_benign_curated_100k.yaml"

OUT_DIR = BASE / "schema" / "eval" / "baseline"


def load_yaml(path: Path) -> list[dict]:
    print(f"  Loading {path.name}...", file=sys.stderr, flush=True)
    t0 = time.time()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=LOADER)
    print(f"  Loaded {len(data):,} rows ({time.time()-t0:.1f}s)", file=sys.stderr)
    return data


def subsample(rows: list[dict], every_n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return shuffled[::every_n]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading attack pool...", file=sys.stderr)
    atk_rows = load_yaml(ATK_SRC)

    print("Loading benign pool...", file=sys.stderr)
    ben_rows = load_yaml(BEN_SRC)

    print(f"\nSubsampling (seed={SEED}):", file=sys.stderr)
    atk_sample = subsample(atk_rows, ATK_EVERY, SEED)
    ben_sample = subsample(ben_rows, BEN_EVERY, SEED)
    print(f"  Attacks: {len(atk_rows):,} → every {ATK_EVERY}th → {len(atk_sample):,}", file=sys.stderr)
    print(f"  Benign:  {len(ben_rows):,} → every {BEN_EVERY}th → {len(ben_sample):,}", file=sys.stderr)

    atk_out = OUT_DIR / "baseline_attacks.yaml"
    ben_out = OUT_DIR / "baseline_benign.yaml"

    print(f"\nWriting {atk_out.name}...", file=sys.stderr)
    with open(atk_out, "w", encoding="utf-8") as f:
        yaml.dump(atk_sample, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=200)

    print(f"Writing {ben_out.name}...", file=sys.stderr)
    with open(ben_out, "w", encoding="utf-8") as f:
        yaml.dump(ben_sample, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=200)

    print(f"\nDone: {len(atk_sample)} attack + {len(ben_sample)} benign = {len(atk_sample)+len(ben_sample)} total", file=sys.stderr)


if __name__ == "__main__":
    main()
