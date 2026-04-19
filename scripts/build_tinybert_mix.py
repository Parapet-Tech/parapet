#!/usr/bin/env python3
"""Build a TinyBERT training JSONL from clean L1 residuals plus anchor corpora.

This script implements the composition described in implement/l2/tinybert-recovery.md:

- easy_benign
- near_boundary_benign
- false_positive
- adversarial_negatives
- baseline_correct
- semantic_false_negative
- easy_attack

It prefers the clean residual export for hard examples, then tops up semantic and
anchor buckets from the local YAML corpora. The output schema is compatible with
the existing TinyBERT notebook: `content`, `true_label`, `reason`, and
`residual_category` are always present.

Usage:
    cd parapet
    python scripts/build_tinybert_mix.py
    python scripts/build_tinybert_mix.py --output-dir parapet-runner/runs/tinybert_mix_v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

try:
    YAML_LOADER = yaml.CSafeLoader
except AttributeError:
    YAML_LOADER = yaml.SafeLoader


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RESIDUALS = Path("parapet-runner/runs/l1b_residuals_v8/l2b_training_candidates.jsonl")
DEFAULT_OUTPUT_DIR = Path("parapet-runner/runs/tinybert_mix_v1")

GLOBAL_BENIGN_PATH = Path("schema/eval/benign/global_benign_clean_25k.yaml")

EASY_BENIGN_CATEGORY_TARGETS = {
    "chat": 500,
    "code": 500,
    "creative": 500,
    "instructions": 500,
    "knowledge": 500,
}

TARGET_BUCKET_COUNTS = {
    "easy_benign": 2500,
    "near_boundary_benign": 2200,
    "false_positive": 1800,
    "adversarial_negatives": 1400,
    "baseline_correct": 1800,
    "semantic_false_negative": 2800,
    "easy_attack": 1500,
}

SEMANTIC_REASON_TARGETS = {
    "roleplay_jailbreak": 1120,
    "constraint_bypass": 420,
    "meta_probe": 336,
    "instruction_override": 336,
    "exfiltration": 308,
    "indirect_injection": 280,
}


@dataclass(frozen=True)
class SourcePlan:
    name: str
    rel_path: Path
    max_take: int
    true_label: str
    reason: str


SEMANTIC_SOURCE_PLANS = {
    "roleplay_jailbreak": SourcePlan(
        "roleplay_jailbreak_attacks",
        Path("schema/eval/malicious/roleplay_jailbreak_attacks.yaml"),
        10_000,
        "malicious",
        "roleplay_jailbreak",
    ),
    "constraint_bypass": SourcePlan(
        "constraint_bypass_attacks",
        Path("schema/eval/malicious/constraint_bypass_attacks.yaml"),
        10_000,
        "malicious",
        "constraint_bypass",
    ),
    "meta_probe": SourcePlan(
        "meta_probe_attacks",
        Path("schema/eval/malicious/meta_probe_attacks.yaml"),
        10_000,
        "malicious",
        "meta_probe",
    ),
    "instruction_override": SourcePlan(
        "instruction_override_attacks",
        Path("schema/eval/malicious/instruction_override_attacks.yaml"),
        10_000,
        "malicious",
        "instruction_override",
    ),
    "exfiltration": SourcePlan(
        "exfiltration_attacks",
        Path("schema/eval/malicious/exfiltration_attacks.yaml"),
        10_000,
        "malicious",
        "exfiltration",
    ),
    "indirect_injection": SourcePlan(
        "opensource_bipia_attacks",
        Path("schema/eval/malicious/opensource_bipia_attacks.yaml"),
        10_000,
        "malicious",
        "indirect_injection",
    ),
}

ADVERSARIAL_NEGATIVE_PLANS = [
    SourcePlan(
        "notinject_benign",
        Path("schema/eval/benign/opensource_notinject_benign.yaml"),
        339,
        "benign",
        "adversarial_negative",
    ),
    SourcePlan(
        "wildguardmix_benign",
        Path("schema/eval/benign/opensource_wildguardmix_benign.yaml"),
        490,
        "benign",
        "adversarial_negative",
    ),
    SourcePlan(
        "atlas_neg",
        Path("schema/eval/benign/thewall_atlas_neg.yaml"),
        43,
        "benign",
        "adversarial_negative",
    ),
    SourcePlan(
        "jbb_paraphrase_benign",
        Path("schema/eval/benign/opensource_jbb_paraphrase_benign.yaml"),
        59,
        "benign",
        "adversarial_negative",
    ),
    SourcePlan(
        "bipia_benign",
        Path("schema/eval/benign/opensource_bipia_benign.yaml"),
        60,
        "benign",
        "adversarial_negative",
    ),
    SourcePlan(
        "code_words_benign",
        Path("schema/eval/benign/code_words_samples.yaml"),
        50,
        "benign",
        "adversarial_negative",
    ),
    SourcePlan(
        "jailbreak_cls_benign",
        Path("schema/eval/benign/opensource_jailbreak_cls_benign.yaml"),
        140,
        "benign",
        "adversarial_negative",
    ),
    SourcePlan(
        "safeguard_benign",
        Path("schema/eval/benign/opensource_safeguard_benign.yaml"),
        219,
        "benign",
        "adversarial_negative",
    ),
    SourcePlan(
        "promptshield_benign",
        Path("schema/eval/benign/opensource_promptshield_benign.yaml"),
        300,
        "benign",
        "adversarial_negative",
    ),
]

EASY_ATTACK_PLANS = [
    SourcePlan(
        "instruction_override_attacks",
        Path("schema/eval/malicious/instruction_override_attacks.yaml"),
        600,
        "malicious",
        "easy_attack",
    ),
    SourcePlan(
        "constraint_bypass_attacks",
        Path("schema/eval/malicious/constraint_bypass_attacks.yaml"),
        450,
        "malicious",
        "easy_attack",
    ),
    SourcePlan(
        "meta_probe_attacks",
        Path("schema/eval/malicious/meta_probe_attacks.yaml"),
        250,
        "malicious",
        "easy_attack",
    ),
    SourcePlan(
        "roleplay_jailbreak_attacks",
        Path("schema/eval/malicious/roleplay_jailbreak_attacks.yaml"),
        200,
        "malicious",
        "easy_attack",
    ),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_yaml_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=YAML_LOADER)
    if not isinstance(data, list):
        raise ValueError(f"expected YAML list in {path}, got {type(data)}")
    return [row for row in data if isinstance(row, dict)]


def ensure_hash(row: dict[str, Any]) -> str:
    existing = row.get("content_hash")
    if isinstance(existing, str) and existing:
        return existing
    return content_hash(str(row.get("content", "")))


def random_sample(
    rows: list[dict[str, Any]],
    target: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if target >= len(rows):
        return list(rows)
    return rng.sample(rows, target)


def normalize_source_row(
    row: dict[str, Any],
    *,
    plan: SourcePlan,
    mix_bucket: str,
    mix_origin: str,
) -> dict[str, Any]:
    text = str(row.get("content", ""))
    row_hash = content_hash(text)
    source_name = str(row.get("source") or plan.name)
    language = str(row.get("language") or "EN")
    length_bin = str(row.get("length_bin") or "")
    description = str(row.get("description") or "").strip()
    row_id = str(row.get("id") or f"{plan.name}-{row_hash[:12]}")
    reason = str(row.get("reason") or plan.reason)
    return {
        "id": row_id,
        "content": text,
        "true_label": plan.true_label,
        "pred_label": None,
        "correct": None,
        "error_type": None,
        "language": language,
        "source": source_name,
        "reason": reason,
        "format_bin": str(row.get("format_bin") or ""),
        "length_bin": length_bin,
        "raw_score": None,
        "raw_unquoted_score": None,
        "raw_squash_score": None,
        "raw_score_delta": None,
        "quote_detected": None,
        "score": None,
        "unquoted_score": None,
        "squash_score": None,
        "content_hash": row_hash,
        "residual_category": mix_bucket,
        "mix_bucket": mix_bucket,
        "mix_origin": mix_origin,
        "origin_path": str(plan.rel_path),
        "description": description,
    }


def add_rows(
    *,
    target_rows: list[dict[str, Any]],
    selected_hashes: set[str],
    incoming_rows: list[dict[str, Any]],
) -> int:
    added = 0
    for row in incoming_rows:
        row_hash = ensure_hash(row)
        if row_hash in selected_hashes:
            continue
        selected_hashes.add(row_hash)
        target_rows.append(row)
        added += 1
    return added


def sample_source_plan(
    *,
    plan: SourcePlan,
    mix_bucket: str,
    mix_origin: str,
    selected_hashes: set[str],
    target: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    source_path = REPO_ROOT / plan.rel_path
    if not source_path.exists():
        return []

    normalized: list[dict[str, Any]] = []
    for row in load_yaml_rows(source_path):
        label = str(row.get("label", "")).strip().lower()
        if label != plan.true_label:
            continue
        content = str(row.get("content", ""))
        if not content.strip():
            continue
        normalized_row = normalize_source_row(
            row,
            plan=plan,
            mix_bucket=mix_bucket,
            mix_origin=mix_origin,
        )
        if normalized_row["content_hash"] in selected_hashes:
            continue
        normalized.append(normalized_row)

    if not normalized:
        return []
    take = min(target, plan.max_take, len(normalized))
    return random_sample(normalized, take, rng)


def build_easy_benign(
    *,
    selected_hashes: set[str],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = load_yaml_rows(REPO_ROOT / GLOBAL_BENIGN_PATH)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("label", "")).strip().lower() != "benign":
            continue
        category = str(row.get("category", "")).strip().lower()
        if category not in EASY_BENIGN_CATEGORY_TARGETS:
            continue
        content = str(row.get("content", ""))
        if not content.strip():
            continue
        if content_hash(content) in selected_hashes:
            continue
        by_category[category].append(row)

    built: list[dict[str, Any]] = []
    achieved: dict[str, int] = {}
    plan = SourcePlan(
        "global_benign_clean_25k",
        GLOBAL_BENIGN_PATH,
        10_000,
        "benign",
        "easy_benign",
    )
    for category, target in EASY_BENIGN_CATEGORY_TARGETS.items():
        normalized = [
            normalize_source_row(
                row,
                plan=plan,
                mix_bucket="easy_benign",
                mix_origin="source_anchor",
            )
            for row in by_category.get(category, [])
        ]
        chosen = random_sample(normalized, min(target, len(normalized)), rng)
        add_rows(target_rows=built, selected_hashes=selected_hashes, incoming_rows=chosen)
        achieved[category] = len(chosen)
    return built, achieved


def build_residual_bucket(
    *,
    residuals: list[dict[str, Any]],
    residual_category: str,
    target: int,
    selected_hashes: set[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    pool = [
        dict(row, mix_bucket=residual_category, mix_origin="clean_residual")
        for row in residuals
        if row.get("residual_category") == residual_category
        and ensure_hash(row) not in selected_hashes
    ]
    chosen = random_sample(pool, min(target, len(pool)), rng)
    built: list[dict[str, Any]] = []
    add_rows(target_rows=built, selected_hashes=selected_hashes, incoming_rows=chosen)
    return built


def build_semantic_false_negative(
    *,
    residuals: list[dict[str, Any]],
    selected_hashes: set[str],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    built: list[dict[str, Any]] = []
    detail: dict[str, dict[str, int]] = {}

    for reason, target in SEMANTIC_REASON_TARGETS.items():
        reason_detail = {"target": target, "from_residual": 0, "from_source": 0}
        residual_pool = [
            dict(row, mix_bucket="semantic_false_negative", mix_origin="clean_residual")
            for row in residuals
            if row.get("residual_category") == "false_negative"
            and row.get("reason") == reason
            and ensure_hash(row) not in selected_hashes
        ]
        chosen_residual = random_sample(residual_pool, min(target, len(residual_pool)), rng)
        added_residual = add_rows(
            target_rows=built,
            selected_hashes=selected_hashes,
            incoming_rows=chosen_residual,
        )
        reason_detail["from_residual"] = added_residual

        remaining = target - added_residual
        if remaining > 0:
            chosen_source = sample_source_plan(
                plan=SEMANTIC_SOURCE_PLANS[reason],
                mix_bucket="semantic_false_negative",
                mix_origin="source_topup",
                selected_hashes=selected_hashes,
                target=remaining,
                rng=rng,
            )
            added_source = add_rows(
                target_rows=built,
                selected_hashes=selected_hashes,
                incoming_rows=chosen_source,
            )
            reason_detail["from_source"] = added_source
        detail[reason] = reason_detail

    return built, detail


def build_source_bucket(
    *,
    plans: list[SourcePlan],
    mix_bucket: str,
    target: int,
    selected_hashes: set[str],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    built: list[dict[str, Any]] = []
    achieved: dict[str, int] = {}

    for plan in plans:
        remaining = target - len(built)
        if remaining <= 0:
            break
        chosen = sample_source_plan(
            plan=plan,
            mix_bucket=mix_bucket,
            mix_origin="source_anchor",
            selected_hashes=selected_hashes,
            target=remaining,
            rng=rng,
        )
        added = add_rows(
            target_rows=built,
            selected_hashes=selected_hashes,
            incoming_rows=chosen,
        )
        if added:
            achieved[plan.name] = added

    return built, achieved


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build TinyBERT training mix JSONL")
    parser.add_argument(
        "--residuals",
        type=Path,
        default=DEFAULT_RESIDUALS,
        help="Clean residual JSONL path relative to repo root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory relative to repo root",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    residuals_path = (REPO_ROOT / args.residuals).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    residuals = load_jsonl(residuals_path)
    selected_hashes: set[str] = set()
    buckets: dict[str, list[dict[str, Any]]] = {}
    build_notes: dict[str, Any] = {}

    buckets["false_positive"] = build_residual_bucket(
        residuals=residuals,
        residual_category="false_positive",
        target=TARGET_BUCKET_COUNTS["false_positive"],
        selected_hashes=selected_hashes,
        rng=rng,
    )

    buckets["baseline_correct"] = build_residual_bucket(
        residuals=residuals,
        residual_category="baseline_correct",
        target=TARGET_BUCKET_COUNTS["baseline_correct"],
        selected_hashes=selected_hashes,
        rng=rng,
    )

    buckets["near_boundary_benign"] = build_residual_bucket(
        residuals=residuals,
        residual_category="near_boundary_benign",
        target=TARGET_BUCKET_COUNTS["near_boundary_benign"],
        selected_hashes=selected_hashes,
        rng=rng,
    )

    adversarial_negatives, adversarial_detail = build_source_bucket(
        plans=ADVERSARIAL_NEGATIVE_PLANS,
        mix_bucket="adversarial_negatives",
        target=TARGET_BUCKET_COUNTS["adversarial_negatives"],
        selected_hashes=selected_hashes,
        rng=rng,
    )
    buckets["adversarial_negatives"] = adversarial_negatives
    build_notes["adversarial_negatives"] = adversarial_detail

    semantic_false_negative, semantic_detail = build_semantic_false_negative(
        residuals=residuals,
        selected_hashes=selected_hashes,
        rng=rng,
    )
    buckets["semantic_false_negative"] = semantic_false_negative
    build_notes["semantic_false_negative"] = semantic_detail

    easy_attack, easy_attack_detail = build_source_bucket(
        plans=EASY_ATTACK_PLANS,
        mix_bucket="easy_attack",
        target=TARGET_BUCKET_COUNTS["easy_attack"],
        selected_hashes=selected_hashes,
        rng=rng,
    )
    buckets["easy_attack"] = easy_attack
    build_notes["easy_attack"] = easy_attack_detail

    easy_benign, easy_benign_detail = build_easy_benign(
        selected_hashes=selected_hashes,
        rng=rng,
    )
    buckets["easy_benign"] = easy_benign
    build_notes["easy_benign"] = easy_benign_detail

    final_rows: list[dict[str, Any]] = []
    for bucket_name in [
        "easy_benign",
        "near_boundary_benign",
        "false_positive",
        "adversarial_negatives",
        "baseline_correct",
        "semantic_false_negative",
        "easy_attack",
    ]:
        final_rows.extend(buckets[bucket_name])

    rng.shuffle(final_rows)

    output_jsonl = output_dir / "tinybert_training_mix.jsonl"
    output_manifest = output_dir / "tinybert_training_mix_manifest.json"
    write_jsonl(output_jsonl, final_rows)

    achieved_bucket_counts = {name: len(rows) for name, rows in buckets.items()}
    label_counts = Counter(row["true_label"] for row in final_rows)
    reason_counts = Counter(
        row["reason"] for row in final_rows if isinstance(row.get("reason"), str) and row["reason"]
    )
    source_counts = Counter(
        row["source"] for row in final_rows if isinstance(row.get("source"), str) and row["source"]
    )
    shortfalls = {
        bucket: TARGET_BUCKET_COUNTS[bucket] - achieved_bucket_counts.get(bucket, 0)
        for bucket in TARGET_BUCKET_COUNTS
        if achieved_bucket_counts.get(bucket, 0) < TARGET_BUCKET_COUNTS[bucket]
    }

    manifest = {
        "created_utc": utc_now(),
        "seed": args.seed,
        "residuals_path": str(residuals_path),
        "output_jsonl": str(output_jsonl),
        "target_bucket_counts": TARGET_BUCKET_COUNTS,
        "achieved_bucket_counts": achieved_bucket_counts,
        "shortfalls": shortfalls,
        "total_rows": len(final_rows),
        "label_counts": dict(label_counts),
        "reason_counts_top20": dict(reason_counts.most_common(20)),
        "source_counts_top20": dict(source_counts.most_common(20)),
        "build_notes": build_notes,
    }
    output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Residuals: {residuals_path}")
    print(f"Output:    {output_jsonl}")
    print(f"Manifest:  {output_manifest}")
    print("")
    print("Bucket counts:")
    for bucket, target in TARGET_BUCKET_COUNTS.items():
        achieved = achieved_bucket_counts.get(bucket, 0)
        print(f"  {bucket:<24} target={target:>5} achieved={achieved:>5}")
    print("")
    print(f"Total rows: {len(final_rows):,}")
    print(f"Labels: malicious={label_counts.get('malicious', 0):,} benign={label_counts.get('benign', 0):,}")

    if shortfalls:
        print("")
        print("Shortfalls:")
        for bucket, missing in shortfalls.items():
            print(f"  {bucket:<24} missing={missing}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
