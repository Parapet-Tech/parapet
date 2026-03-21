"""Inventory the local data estate for Step 0 cleanup.

This script reports the current `schema/eval/` and `work/` layout and assigns
default cleanup decisions for the top-level entries that need migration or
deletion.

Run from the parapet/ directory:

    python parapet-data/scripts/inventory_data_cleanup.py \
        --output implement/data_cleanup_inventory.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_EVAL_ROOT = REPO_ROOT / "schema" / "eval"
WORK_ROOT = REPO_ROOT / "work"
ADJUDICATION_ROOT = REPO_ROOT / "parapet-data" / "adjudication" / "review"

SCHEMA_KEEP_DIRS = {
    "staging": "schema/eval/staging",
    "verified": "schema/eval/verified",
}

SCHEMA_ASSET_DIRS = {
    "attack_types": "schema/eval/attack_types",
    "baseline": "schema/eval/baseline",
    "challenges": "schema/eval/challenges",
}

SCHEMA_DELETE_DIRS = {"tmp", "t2", "t3", "t_mirror", "ts"}

SCHEMA_REVIEW_DIRS = {
    "attack_types",
    "baseline",
    "benign",
    "challenges",
    "malicious",
    "training",
    "ts",
}

DUPLICATE_SCAN_ROOTS = (
    SCHEMA_EVAL_ROOT / "benign",
    SCHEMA_EVAL_ROOT / "malicious",
    SCHEMA_EVAL_ROOT / "training",
)


def summarize_path(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {
            "kind": "file",
            "file_count": 1,
            "total_bytes": path.stat().st_size,
        }

    file_count = 0
    total_bytes = 0
    for child in path.rglob("*"):
        if child.is_file():
            file_count += 1
            total_bytes += child.stat().st_size
    return {
        "kind": "directory",
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def classify_schema_eval_entry(path: Path) -> dict[str, Any]:
    relative = path.relative_to(REPO_ROOT).as_posix()
    name = path.name

    if path.is_dir() and name in SCHEMA_KEEP_DIRS:
        return {
            "decision": "keep",
            "target": SCHEMA_KEEP_DIRS[name],
            "reason": "canonical source state",
        }

    if path.is_dir() and name in SCHEMA_ASSET_DIRS:
        return {
            "decision": "keep",
            "target": SCHEMA_ASSET_DIRS[name],
            "reason": "durable eval asset or taxonomy suite",
        }

    if path.is_dir() and name in SCHEMA_DELETE_DIRS:
        return {
            "decision": "delete",
            "target": None,
            "reason": "historical or temporary experiment area",
        }

    if path.is_dir() and name in SCHEMA_REVIEW_DIRS:
        return {
            "decision": "review",
            "target": None,
            "reason": "data-bearing subtree still mixed into schema/eval",
        }

    if path.is_file():
        return {
            "decision": "keep",
            "target": relative,
            "reason": "eval config or durable root-level asset",
        }

    return {
        "decision": "review",
        "target": None,
        "reason": "unclassified schema/eval entry",
    }


def classify_work_entry(path: Path) -> dict[str, Any]:
    relative_parts = path.relative_to(WORK_ROOT).parts
    top_level = relative_parts[0]

    if top_level == "gap_mining":
        target_root = ADJUDICATION_ROOT / "exports" / "gap_mining"
    elif top_level == "swarm_batches":
        target_root = ADJUDICATION_ROOT / "batches" / "swarm_batches"
    else:
        return {
            "decision": "review",
            "target": None,
            "reason": "unclassified work subtree",
        }

    suffix_parts = relative_parts[1:]
    target = target_root.joinpath(*suffix_parts)
    return {
        "decision": "move",
        "target": target.relative_to(REPO_ROOT).as_posix(),
        "reason": "local review state belongs under adjudication",
    }


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_duplicate_files() -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = {}
    for root in DUPLICATE_SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            digest = file_hash(path)
            groups.setdefault(digest, []).append(path.relative_to(REPO_ROOT).as_posix())

    duplicates: list[dict[str, Any]] = []
    for digest, paths in sorted(groups.items()):
        if len(paths) <= 1:
            continue
        duplicates.append(
            {
                "hash": digest,
                "paths": sorted(paths),
            }
        )
    return duplicates


def build_inventory() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []

    for root in (SCHEMA_EVAL_ROOT, WORK_ROOT):
        if not root.exists():
            continue
        for path in sorted(root.iterdir()):
            summary = summarize_path(path)
            classification = (
                classify_schema_eval_entry(path)
                if root == SCHEMA_EVAL_ROOT
                else classify_work_entry(path)
            )
            entries.append(
                {
                    "path": path.relative_to(REPO_ROOT).as_posix(),
                    **summary,
                    **classification,
                }
            )

    return {
        "repo_root": REPO_ROOT.as_posix(),
        "entries": entries,
        "duplicate_groups": find_duplicate_files(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the JSON inventory report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_inventory()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
