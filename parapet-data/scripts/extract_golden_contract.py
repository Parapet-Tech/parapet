"""Extract a golden contract from a trusted curated artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parapet_data.golden_contract import build_golden_contract, write_contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated-dir", required=True, help="Curated artifact directory")
    parser.add_argument("--output", required=True, help="Output contract JSON path")
    parser.add_argument("--run-manifest", help="Optional linked run_manifest.json")
    parser.add_argument("--runner-root", help="Optional runner/runs root to auto-discover matching runs")
    parser.add_argument("--project-root", help="Optional project root for relative path serialization")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve() if args.project_root else None
    contract = build_golden_contract(
        Path(args.curated_dir),
        run_manifest_path=Path(args.run_manifest).resolve() if args.run_manifest else None,
        runner_root=Path(args.runner_root).resolve() if args.runner_root else None,
        project_root=project_root,
    )
    output_path = Path(args.output).resolve()
    write_contract(contract, output_path)
    print(f"Wrote {output_path}")
    print(f"  spec={contract['spec_name']} semantic_hash={contract['semantic_hash']}")
    print(f"  matching_runs={len(contract['matching_runs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
