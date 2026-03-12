"""Diagnose backfill drift between old and new samplers."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parapet_data.models import MirrorSpec
from parapet_data.ledger import Ledger
from parapet_data import sampler as new_sampler
from parapet_data import sampler_old as old_sampler


def summarize(result, name: str) -> dict[str, dict]:
    """Extract per-cell by_language and global source counts."""
    info = {"cell_langs": {}, "sources": Counter()}
    for key, fill in result.cell_fills.items():
        info["cell_langs"][key] = dict(fill.by_language)
    for s in result.attack_samples:
        info["sources"][s.source_name] += 1
    print(f"[{name}] attacks={len(result.attack_samples)} benign={len(result.benign_samples)}")
    print(f"  en_attacks_merged={info['sources'].get('en_attacks_merged', 0)}")
    print(f"  zh_attacks_novel={info['sources'].get('zh_attacks_novel', 0)}")
    return info


def diff_cells(name_a, name_b, info_a, info_b):
    cells_a = info_a["cell_langs"]
    cells_b = info_b["cell_langs"]
    diffs = 0
    for key in sorted(set(cells_a) | set(cells_b)):
        a = cells_a.get(key, {})
        b = cells_b.get(key, {})
        if a == b:
            continue
        diffs += 1
        print(f"  {key}:")
        for lang in sorted(set(a) | set(b)):
            av, bv = a.get(lang, 0), b.get(lang, 0)
            d = bv - av
            if d != 0:
                print(f"    {lang}: {name_a}={av} {name_b}={bv} delta={d:+d}")
    if diffs == 0:
        print("  (identical)")


def main() -> int:
    base_dir = ROOT.parent
    spec_path = ROOT / "mirror_spec_v4_19k.yaml"
    ledger_path = ROOT / "adjudication" / "ledger.yaml"

    spec = MirrorSpec.model_validate(yaml.safe_load(spec_path.read_text("utf-8")))
    ledger = Ledger.load(ledger_path)
    print(f"Spec: {spec.name} seed={spec.seed} total={spec.total_target} cells={len(spec.cells)}")
    print(f"Ledger: {len(ledger)} entries\n")

    # Run A: old sampler
    print("=" * 50)
    old_sampler._source_cache.clear()
    new_sampler._source_cache.clear()
    result_a = old_sampler.sample_spec(spec, base_dir=base_dir)
    info_a = summarize(result_a, "A: old sampler")

    # Run B: new sampler, no ledger
    print("=" * 50)
    old_sampler._source_cache.clear()
    new_sampler._source_cache.clear()
    result_b = new_sampler.sample_spec(spec, base_dir=base_dir, ledger=None)
    info_b = summarize(result_b, "B: new sampler, no ledger")

    # Run C: new sampler, with ledger
    print("=" * 50)
    old_sampler._source_cache.clear()
    new_sampler._source_cache.clear()
    result_c = new_sampler.sample_spec(spec, base_dir=base_dir, ledger=ledger)
    info_c = summarize(result_c, "C: new sampler, WITH ledger")

    # Diffs
    print("\n" + "=" * 50)
    print("DIFF A vs B (old vs new, no ledger):")
    diff_cells("old", "new_no", info_a, info_b)

    print("\nDIFF A vs C (old vs new+ledger):")
    diff_cells("old", "new_led", info_a, info_c)

    print("\nDIFF B vs C (new no-ledger vs new+ledger):")
    diff_cells("new_no", "new_led", info_b, info_c)

    # Source summary
    print("\n" + "=" * 50)
    print("SOURCE DRIFT (malicious):")
    all_src = sorted(set(info_a["sources"]) | set(info_b["sources"]) | set(info_c["sources"]))
    for s in all_src:
        a, b, c = info_a["sources"].get(s, 0), info_b["sources"].get(s, 0), info_c["sources"].get(s, 0)
        if a != b or a != c:
            print(f"  {s}: old={a} new_no={b}({b-a:+d}) new_led={c}({c-a:+d})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
