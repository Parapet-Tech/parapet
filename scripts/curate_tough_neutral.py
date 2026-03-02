#!/usr/bin/env python3
"""
Curate a tough benign-neutral eval slice for L1/L2a comparison.

Design goals:
1. Pull from hard-negative benign sources (prompt-injection-like wording).
2. Prefer samples that are not present in the mirror_v2 curated manifest.
3. Produce a reproducible YAML artifact plus a JSON composition/novelty report.

Usage:
  cd parapet
  python scripts/curate_tough_neutral.py
  python scripts/curate_tough_neutral.py --target 300 --seed 7
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

try:
    YAML_LOADER = yaml.CSafeLoader
except AttributeError:
    YAML_LOADER = yaml.SafeLoader

try:
    YAML_DUMPER = yaml.CSafeDumper
except AttributeError:
    YAML_DUMPER = yaml.SafeDumper


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SourcePlan:
    name: str
    rel_path: Path
    max_take: int
    mode: str = "all"  # "all" or "fp_only"


@dataclass(frozen=True)
class Candidate:
    source_name: str
    orig_id: str
    description: str
    content: str
    content_hash: str
    overlap_with_mirror: bool


PRIMARY_PLANS: list[SourcePlan] = [
    SourcePlan("atlas_neg", Path("schema/eval/benign/thewall_atlas_neg.yaml"), 43),
    SourcePlan("jbb_paraphrase_benign", Path("schema/eval/benign/opensource_jbb_paraphrase_benign.yaml"), 59),
    SourcePlan("bipia_benign", Path("schema/eval/benign/opensource_bipia_benign.yaml"), 60),
    SourcePlan("code_words_benign", Path("schema/eval/benign/code_words_samples.yaml"), 50),
    SourcePlan("notinject_benign", Path("schema/eval/benign/opensource_notinject_benign.yaml"), 250),
    SourcePlan("jailbreak_cls_benign", Path("schema/eval/benign/opensource_jailbreak_cls_benign.yaml"), 140),
    SourcePlan("safeguard_benign", Path("schema/eval/benign/opensource_safeguard_benign.yaml"), 220),
    SourcePlan("promptshield_benign", Path("schema/eval/benign/opensource_promptshield_benign.yaml"), 300),
    SourcePlan("generalist_fp_benign", Path("schema/eval/benign/generalist_errors.yaml"), 220, mode="fp_only"),
    SourcePlan("wildguardmix_benign", Path("schema/eval/benign/opensource_wildguardmix_benign.yaml"), 450),
    SourcePlan("ctf_satml24_benign", Path("schema/eval/benign/thewall_ctf-satml24_benign.yaml"), 800),
]

FALLBACK_PLANS: list[SourcePlan] = [
    SourcePlan("wildjailbreak_sparse_benign", Path("schema/eval/benign/opensource_wildjailbreak_benign.yaml"), 300),
    SourcePlan("spml_benign", Path("schema/eval/benign/opensource_spml_benign.yaml"), 240),
]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_yaml_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing source file: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=YAML_LOADER)
    if not isinstance(data, list):
        raise ValueError(f"expected YAML list in {path}, got {type(data)}")
    return [row for row in data if isinstance(row, dict)]


def load_mirror_hashes(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    splits = manifest.get("splits", {})
    all_hashes: set[str] = set()
    for split_name in ("train", "val", "holdout"):
        split = splits.get(split_name)
        if not split:
            continue
        for h in split.get("content_hashes", []):
            if isinstance(h, str) and h:
                all_hashes.add(h)
    if not all_hashes:
        raise ValueError(f"no content_hashes found in {manifest_path}")
    return all_hashes


def build_candidates(plan: SourcePlan, mirror_hashes: set[str]) -> list[Candidate]:
    rows = load_yaml_rows(REPO_ROOT / plan.rel_path)
    candidates: list[Candidate] = []
    for idx, row in enumerate(rows):
        label = str(row.get("label", "")).strip().lower()
        if label != "benign":
            continue

        if plan.mode == "fp_only":
            error_type = str(row.get("error_type", "")).strip().lower()
            if error_type != "false_positive":
                continue

        content_val = row.get("content", "")
        if not isinstance(content_val, str):
            content_val = str(content_val)
        if not content_val.strip():
            continue

        orig_id = str(row.get("id") or f"{plan.name}-{idx:05d}")
        description = str(row.get("description", "")).strip()
        h = content_hash(content_val)
        candidates.append(
            Candidate(
                source_name=plan.name,
                orig_id=orig_id,
                description=description,
                content=content_val,
                content_hash=h,
                overlap_with_mirror=(h in mirror_hashes),
            )
        )
    return candidates


def sample_novel_first(
    plans: list[SourcePlan],
    candidates_by_source: dict[str, list[Candidate]],
    selected_hashes: set[str],
    target: int,
    rng: random.Random,
    stats: dict[str, Counter],
) -> list[Candidate]:
    picked: list[Candidate] = []
    for plan in plans:
        if len(picked) >= target:
            break
        source_candidates = candidates_by_source.get(plan.name, [])
        available = [c for c in source_candidates if c.content_hash not in selected_hashes]
        novel = [c for c in available if not c.overlap_with_mirror]
        rng.shuffle(novel)
        take = min(plan.max_take, len(novel), target - len(picked))
        chosen = novel[:take]
        if not chosen:
            continue
        picked.extend(chosen)
        selected_hashes.update(c.content_hash for c in chosen)
        stats[plan.name]["selected"] += len(chosen)
        stats[plan.name]["selected_novel"] += len(chosen)
    return picked


def fill_remaining(
    *,
    candidates_by_source: dict[str, list[Candidate]],
    selected_hashes: set[str],
    needed: int,
    rng: random.Random,
    only_novel: bool,
    stats: dict[str, Counter],
) -> list[Candidate]:
    if needed <= 0:
        return []

    pool: list[Candidate] = []
    for source_name, candidates in candidates_by_source.items():
        for c in candidates:
            if c.content_hash in selected_hashes:
                continue
            if only_novel and c.overlap_with_mirror:
                continue
            pool.append(c)

    if not pool:
        return []

    rng.shuffle(pool)
    chosen = pool[: min(needed, len(pool))]
    selected_hashes.update(c.content_hash for c in chosen)
    for c in chosen:
        stats[c.source_name]["selected"] += 1
        if c.overlap_with_mirror:
            stats[c.source_name]["selected_overlap"] += 1
        else:
            stats[c.source_name]["selected_novel"] += 1
    return chosen


def build_output_rows(selected: list[Candidate]) -> list[dict]:
    rows: list[dict] = []
    for idx, sample in enumerate(selected, start=1):
        overlap_flag = "overlap=yes" if sample.overlap_with_mirror else "overlap=no"
        desc = sample.description or "tough neutral hard-negative sample"
        rows.append(
            {
                "id": f"tough-neutral-v1-{idx:04d}",
                "layer": "l1",
                "label": "benign",
                "description": (
                    f"tough-neutral-v1 source={sample.source_name} "
                    f"orig_id={sample.orig_id} {overlap_flag} | {desc}"
                ),
                "content": sample.content,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Curate tough benign neutral dataset, prioritized for novelty "
            "against mirror_v2 manifest hashes."
        )
    )
    parser.add_argument(
        "--target",
        type=int,
        default=2386,
        help="Desired sample count (default: 2386 to match mirror_v2 holdout size)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--mirror-manifest",
        type=Path,
        default=Path("parapet-data/curated_v2/manifest.json"),
        help="Curation manifest used as novelty reference",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("schema/eval/challenges/tough_neutral_v1/tough_neutral_mirror_v2_novel.yaml"),
        help="Output YAML path",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Optional report JSON path (default: <output_stem>_report.json)",
    )
    parser.add_argument(
        "--min-novelty",
        type=float,
        default=0.90,
        help="Minimum required novelty ratio (default: 0.90)",
    )
    args = parser.parse_args()

    if args.target <= 0:
        raise SystemExit("ERROR: --target must be > 0")
    if not (0.0 <= args.min_novelty <= 1.0):
        raise SystemExit("ERROR: --min-novelty must be in [0, 1]")

    rng = random.Random(args.seed)
    mirror_manifest = (REPO_ROOT / args.mirror_manifest).resolve()
    output_path = (REPO_ROOT / args.output).resolve()
    report_path = (
        (REPO_ROOT / args.report_out).resolve()
        if args.report_out
        else output_path.with_name(f"{output_path.stem}_report.json")
    )

    print(f"Mirror manifest: {mirror_manifest}", file=sys.stderr)
    mirror_hashes = load_mirror_hashes(mirror_manifest)
    print(f"Mirror hash count: {len(mirror_hashes):,}", file=sys.stderr)

    plans = PRIMARY_PLANS + FALLBACK_PLANS
    candidates_by_source: dict[str, list[Candidate]] = {}
    stats: dict[str, Counter] = {}
    global_seen_hashes: set[str] = set()

    for plan in plans:
        stats[plan.name] = Counter()
        source_candidates = build_candidates(plan, mirror_hashes)
        unique_candidates: list[Candidate] = []
        for c in source_candidates:
            if c.content_hash in global_seen_hashes:
                continue
            global_seen_hashes.add(c.content_hash)
            unique_candidates.append(c)
        candidates_by_source[plan.name] = unique_candidates
        stats[plan.name]["available"] = len(unique_candidates)
        stats[plan.name]["available_novel"] = sum(
            1 for c in unique_candidates if not c.overlap_with_mirror
        )
        stats[plan.name]["available_overlap"] = (
            stats[plan.name]["available"] - stats[plan.name]["available_novel"]
        )
        print(
            f"{plan.name:<28} available={stats[plan.name]['available']:>5} "
            f"novel={stats[plan.name]['available_novel']:>5} "
            f"overlap={stats[plan.name]['available_overlap']:>5}",
            file=sys.stderr,
        )

    selected_hashes: set[str] = set()
    selected: list[Candidate] = []

    selected.extend(
        sample_novel_first(
            plans=PRIMARY_PLANS,
            candidates_by_source=candidates_by_source,
            selected_hashes=selected_hashes,
            target=args.target,
            rng=rng,
            stats=stats,
        )
    )
    if len(selected) < args.target:
        selected.extend(
            sample_novel_first(
                plans=FALLBACK_PLANS,
                candidates_by_source=candidates_by_source,
                selected_hashes=selected_hashes,
                target=args.target - len(selected),
                rng=rng,
                stats=stats,
            )
        )

    if len(selected) < args.target:
        needed = args.target - len(selected)
        selected.extend(
            fill_remaining(
                candidates_by_source=candidates_by_source,
                selected_hashes=selected_hashes,
                needed=needed,
                rng=rng,
                only_novel=True,
                stats=stats,
            )
        )

    if len(selected) < args.target:
        needed = args.target - len(selected)
        selected.extend(
            fill_remaining(
                candidates_by_source=candidates_by_source,
                selected_hashes=selected_hashes,
                needed=needed,
                rng=rng,
                only_novel=False,
                stats=stats,
            )
        )

    if len(selected) < args.target:
        raise SystemExit(
            f"ERROR: unable to reach target={args.target}; only got {len(selected)}"
        )

    selected = selected[: args.target]
    overlap_count = sum(1 for c in selected if c.overlap_with_mirror)
    novelty_ratio = 1.0 - (overlap_count / len(selected))
    if novelty_ratio < args.min_novelty:
        raise SystemExit(
            f"ERROR: novelty {novelty_ratio:.4f} below min_novelty={args.min_novelty:.4f}"
        )

    output_rows = build_output_rows(selected)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.dump(
            output_rows,
            handle,
            Dumper=YAML_DUMPER,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=180,
        )

    source_selected_counts = Counter(c.source_name for c in selected)
    report = {
        "dataset": str(output_path.relative_to(REPO_ROOT)),
        "target": args.target,
        "actual": len(selected),
        "seed": args.seed,
        "mirror_manifest": str(mirror_manifest.relative_to(REPO_ROOT)),
        "mirror_hash_count": len(mirror_hashes),
        "mirror_overlap_count": overlap_count,
        "mirror_overlap_pct": round(100.0 * overlap_count / len(selected), 4),
        "novelty_ratio": round(novelty_ratio, 6),
        "source_selected_counts": dict(source_selected_counts),
        "source_stats": {name: dict(counter) for name, counter in stats.items()},
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("\nCurated tough neutral dataset:", file=sys.stderr)
    print(f"  output: {output_path}", file=sys.stderr)
    print(f"  report: {report_path}", file=sys.stderr)
    print(f"  samples: {len(selected):,}", file=sys.stderr)
    print(f"  overlap: {overlap_count:,} ({100.0 * overlap_count / len(selected):.2f}%)", file=sys.stderr)
    print(f"  novelty: {novelty_ratio * 100.0:.2f}%", file=sys.stderr)
    print("\nSelected by source:", file=sys.stderr)
    for source_name, count in source_selected_counts.most_common():
        print(f"  {source_name:<28} {count:>5}", file=sys.stderr)


if __name__ == "__main__":
    main()
