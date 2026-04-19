"""Remove out-of-scope sources from tough_{attack,neutral}_v2 challenge yamls.

Prompt injection defense (L1/L2) does not own content-safety filtering.
When challenge sources contain content-safety items labeled as "attacks",
they distort every PI-task metric: a PI classifier correctly scores them
low, but is penalized as a false negative.

This script excises named source-groups from both challenge yamls and
writes a sidecar `_excluded.yaml` recording what was removed (with a reason
stamp), so the change is auditable and reversible.

Usage:
    cd parapet
    python scripts/exclude_challenge_sources.py \
        --attack-source jbb_paraphrase_attacks \
        --benign-source jbb_paraphrase_benign \
        --reason "content_safety_out_of_scope"

Multiple --attack-source / --benign-source flags can be provided.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

import yaml

ATTACK_YAML = "schema/eval/challenges/tough_attack_v2/tough_attack_v6_novel.yaml"
NEUTRAL_YAML = "schema/eval/challenges/tough_neutral_v2/tough_neutral_v6_novel.yaml"
SIDECAR_DIR = "schema/eval/challenges/_excluded"
SOURCE_RE = re.compile(r"source=(\S+)")


def _source_of(r: dict) -> str:
    m = SOURCE_RE.search(r.get("description", "") or "")
    return m.group(1) if m else ""


def prune(yaml_path: Path, sources_to_drop: set[str]) -> tuple[list, list]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    kept, removed = [], []
    for r in data:
        (removed if _source_of(r) in sources_to_drop else kept).append(r)
    return kept, removed


def write_yaml(path: Path, data: list) -> None:
    # Dump with block style for readable diffs in git.
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=10_000),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack-source", action="append", default=[],
                    help="Source name(s) to drop from tough_attack_v2.")
    ap.add_argument("--benign-source", action="append", default=[],
                    help="Source name(s) to drop from tough_neutral_v2.")
    ap.add_argument("--reason", required=True,
                    help="Short tag recorded in the sidecar (e.g. 'content_safety_out_of_scope').")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.attack_source and not args.benign_source:
        ap.error("At least one --attack-source or --benign-source required.")

    base = Path(__file__).resolve().parent.parent  # parapet/
    sidecar_dir = base / SIDECAR_DIR
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    stamp = now.replace(":", "").replace("-", "")

    for yaml_rel, sources, bucket in [
        (ATTACK_YAML, set(args.attack_source), "attack"),
        (NEUTRAL_YAML, set(args.benign_source), "benign"),
    ]:
        if not sources:
            continue
        yaml_path = base / yaml_rel
        kept, removed = prune(yaml_path, sources)
        if not removed:
            print(f"[{bucket}] no matches for {sorted(sources)} in {yaml_path.name}")
            continue

        print(f"[{bucket}] {yaml_path.name}: "
              f"{len(removed)} removed, {len(kept)} kept "
              f"(sources={sorted(sources)})")
        if args.dry_run:
            for r in removed[:5]:
                print(f"  would drop: {r['id']}")
            continue

        # Write sidecar capturing the removed records + context.
        sidecar = sidecar_dir / f"{yaml_path.stem}__{args.reason}__{stamp}.yaml"
        sidecar.write_text(yaml.safe_dump({
            "removed_from": yaml_rel,
            "reason": args.reason,
            "at_utc": now,
            "sources": sorted(sources),
            "count": len(removed),
            "records": removed,
        }, allow_unicode=True, sort_keys=False, width=10_000), encoding="utf-8")
        write_yaml(yaml_path, kept)
        print(f"  wrote sidecar: {sidecar.relative_to(base)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
