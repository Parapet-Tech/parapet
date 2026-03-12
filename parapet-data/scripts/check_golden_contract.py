"""Check a curated artifact against a golden source/language contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parapet_data.golden_contract import build_golden_contract, compare_contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, help="Golden contract JSON")
    parser.add_argument("--curated-dir", required=True, help="Curated artifact directory to check")
    parser.add_argument("--runner-root", help="Optional runner/runs root to annotate matching runs")
    parser.add_argument("--project-root", help="Optional project root for relative path serialization")
    parser.add_argument("--max-source-share-delta", type=float, default=0.05)
    parser.add_argument("--max-language-share-delta", type=float, default=0.03)
    parser.add_argument("--min-monitored-source-share", type=float, default=0.005)
    parser.add_argument("--check-source-hashes", action="store_true")
    args = parser.parse_args()

    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    project_root = Path(args.project_root).resolve() if args.project_root else None
    observed = build_golden_contract(
        Path(args.curated_dir),
        runner_root=Path(args.runner_root).resolve() if args.runner_root else None,
        project_root=project_root,
    )
    violations = compare_contract(
        contract,
        observed,
        max_source_share_delta=args.max_source_share_delta,
        max_language_share_delta=args.max_language_share_delta,
        min_monitored_source_share=args.min_monitored_source_share,
        check_source_hashes=args.check_source_hashes,
    )

    print(f"Baseline: {contract['curated_dir']} ({contract['semantic_hash']})")
    print(f"Observed: {observed['curated_dir']} ({observed['semantic_hash']})")
    print(
        "Thresholds: "
        f"source={args.max_source_share_delta:.3f} "
        f"language={args.max_language_share_delta:.3f} "
        f"min_source_share={args.min_monitored_source_share:.3f}"
    )

    if not violations:
        print("PASS: no contract drift detected")
        return 0

    print("FAIL: contract drift detected")
    for violation in violations:
        print(f"  - {violation}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
