#!/usr/bin/env python3
"""Extract a balanced residual training supplement from L1 eval failures.

Reads eval.json (from a verified-data eval run), joins case_ids back to
the source YAMLs to recover content, then samples a balanced attack/benign
set stratified by language quota.

Usage (from parapet-runner/):
    python scripts/extract_residual_supplement.py \
        --eval-json runs/verified_residual/eval.json \
        --verified-dir ../schema/eval/verified \
        --output runs/verified_residual/residual_supplement_5k.yaml \
        --total 5000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


LANGUAGE_QUOTA = {"EN": 0.75, "RU": 0.10, "ZH": 0.08, "AR": 0.07}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-json", type=Path, required=True)
    p.add_argument("--verified-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--total", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def infer_language(stem: str) -> str | None:
    # Prefix match (most files)
    for prefix, lang in [("ar_", "AR"), ("ru_", "RU"), ("zh_", "ZH"), ("en_", "EN")]:
        if stem.startswith(prefix):
            return lang
    # Substring match for thewall / opensource (e.g. thewall_sql_injection_zh_attacks)
    for marker, lang in [("_zh_", "ZH"), ("_ar_", "AR"), ("_ru_", "RU")]:
        if marker in stem:
            return lang
    return None  # truly multilingual — rely on per-entry field


def extract_entry(entry: dict[str, Any], stem: str) -> dict[str, Any]:
    language = entry.get("language") or infer_language(stem) or "EN"
    return {
        "content": str(entry.get("content", "")),
        "label": str(entry.get("label", "")),
        "language": language.upper(),
        "reason": entry.get("reason"),
        "source": entry.get("source", stem),
    }


def sample_by_language(
    entries: list[dict], target: int, quota: dict[str, float], rng: random.Random
) -> list[dict]:
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        by_lang[e["language"]].append(e)

    sampled: list[dict] = []
    shortfalls: dict[str, int] = {}

    for lang, frac in quota.items():
        pool = by_lang.get(lang, [])
        want = int(target * frac)
        got = min(want, len(pool))
        if got > 0:
            sampled.extend(rng.sample(pool, got))
        if got < want:
            shortfalls[lang] = want - got
        print(f"  {lang}: want {want}, have {len(pool)}, sampled {got}", file=sys.stderr)

    # Redistribute shortfalls to languages with surplus
    if shortfalls:
        deficit = sum(shortfalls.values())
        surplus_langs = [
            lang for lang in quota
            if lang not in shortfalls and len(by_lang.get(lang, [])) > int(target * quota[lang])
        ]
        if surplus_langs:
            per_lang = deficit // len(surplus_langs)
            for lang in surplus_langs:
                pool = by_lang[lang]
                already = int(target * quota[lang])
                extra = min(per_lang, len(pool) - already)
                if extra > 0:
                    remaining = [e for e in pool if e not in sampled]
                    sampled.extend(rng.sample(remaining, min(extra, len(remaining))))
                    print(f"  {lang}: +{extra} backfill from surplus", file=sys.stderr)

    return sampled


def main() -> None:
    args = parse_args()

    # 1. Parse eval.json — collect incorrect case_ids by source
    print("Loading eval.json...", file=sys.stderr)
    with open(args.eval_json, encoding="utf-8") as f:
        eval_data = json.load(f)

    fn_by_source: dict[str, set[str]] = defaultdict(set)
    fp_by_source: dict[str, set[str]] = defaultdict(set)

    for result in eval_data["results"]:
        if result["correct"]:
            continue
        source = result["source"]
        case_id = result["case_id"]
        if result["label"] == "malicious":
            fn_by_source[source].add(case_id)
        else:
            fp_by_source[source].add(case_id)

    total_fn = sum(len(v) for v in fn_by_source.values())
    total_fp = sum(len(v) for v in fp_by_source.values())
    print(f"FN: {total_fn} across {len(fn_by_source)} sources", file=sys.stderr)
    print(f"FP: {total_fp} across {len(fp_by_source)} sources", file=sys.stderr)

    # 2. Join back to verified YAMLs for content
    all_fn: list[dict] = []
    all_fp: list[dict] = []

    all_needed = set(fn_by_source.keys()) | set(fp_by_source.keys())

    for yaml_file in sorted(args.verified_dir.glob("*.yaml")):
        stem = yaml_file.stem
        if stem not in all_needed:
            continue

        print(f"  Loading {stem}...", file=sys.stderr)
        try:
            raw = yaml_file.read_text(encoding="utf-8")
            entries = yaml.safe_load(raw)
        except Exception as e:
            print(f"  SKIP {stem}: {e}", file=sys.stderr)
            continue

        if not isinstance(entries, list):
            continue

        fn_ids = fn_by_source.get(stem, set())
        fp_ids = fp_by_source.get(stem, set())

        for idx, entry in enumerate(entries, start=1):
            case_id = str(entry.get("id") or f"{stem}_{idx}")
            if case_id in fn_ids:
                all_fn.append(extract_entry(entry, stem))
            elif case_id in fp_ids:
                all_fp.append(extract_entry(entry, stem))

    print(f"Joined — FN: {len(all_fn)}, FP: {len(all_fp)}", file=sys.stderr)

    # 3. Sample by language quota
    attack_target = args.total // 2
    benign_target = args.total - attack_target

    rng = random.Random(args.seed)

    print("\nSampling attacks (from FN):", file=sys.stderr)
    attacks = sample_by_language(all_fn, attack_target, LANGUAGE_QUOTA, rng)

    print("\nSampling benign (from FP):", file=sys.stderr)
    benign = sample_by_language(all_fp, benign_target, LANGUAGE_QUOTA, rng)

    combined = attacks + benign
    rng.shuffle(combined)

    # 4. Summary
    by_lang_label: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for e in combined:
        by_lang_label[e["language"]][e["label"]] += 1

    print(f"\n=== Residual supplement: {len(combined)} samples ===", file=sys.stderr)
    for lang in sorted(by_lang_label):
        counts = by_lang_label[lang]
        print(f"  {lang}: {counts.get('malicious',0)} attack + {counts.get('benign',0)} benign", file=sys.stderr)

    # 5. Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        yaml.dump(combined, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\nWrote {len(combined)} samples to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
