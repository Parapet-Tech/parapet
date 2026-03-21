#!/usr/bin/env python3
"""Seed step-2 audit suggestions for persistent L1 failures.

This script reads the exported persistent intersection rows produced by
`intersect_residuals.py` and adds heuristic suggestion fields while leaving the
human-review columns untouched.

Usage:
    python scripts/preaudit_persistent_intersection.py \
        --input runs/verified_residual_r3/persistent_intersection.jsonl \
        --output runs/verified_residual_r3/persistent_intersection_preaudit.jsonl \
        --summary-output runs/verified_residual_r3/persistent_intersection_preaudit_summary.json
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


SUGGESTED_FIELDS = (
    "suggested_in_scope_pi",
    "suggested_failure_family",
    "suggested_confidence",
    "suggested_rule",
)

PROMPT_INJECTION_MARKERS = (
    "ignore previous",
    "ignore all previous",
    "disregard previous",
    "forget previous",
    "system prompt",
    "developer message",
    "you are now",
    "act as",
    "pretend to be",
    "override",
    "follow my instructions",
    "ignore instructions",
    "without constraints",
    "do not refuse",
    "jailbreak",
    "new instruction",
    "system override",
    "roleplay as",
)

CODE_OR_SHELL_MARKERS = (
    "```",
    "def ",
    "class ",
    "import ",
    "bash",
    "shell",
    "command line",
    "unix shell",
    "script",
)

OUT_OF_SCOPE_ATTACK_SOURCE_PATTERNS = ("sql_injection",)
AMBIGUOUS_ATTACK_SOURCE_PATTERNS = (
    "wildjailbreak_attacks",
    "jailbreak_llms_attacks",
    "spml_chatbot_prompt_injection_attacks",
    "wildguardmix_attacks",
    "protectai_val_attacks",
)
FN_MULTILINGUAL_SOURCE_PATTERNS = (
    "russian_prompt_injections_attacks",
    "prompt_injection_multilingual_attacks",
    "prompt_injection_attack_detection_multilingual_attacks",
    "arabguard",
)
FN_SEMANTIC_SOURCE_PATTERNS = (
    "wildjailbreak_attacks",
    "spml_chatbot_prompt_injection_attacks",
    "jailbreak_llms_attacks",
)
FN_NOVEL_STRUCTURE_SOURCE_PATTERNS = (
    "llmail-inject-challenge_attacks",
    "tensor-trust-data_attacks",
    "gandalf_attacks",
    "protectai_val_attacks",
    "wildguardmix_attacks",
)
FP_USE_MENTION_SOURCE_PATTERNS = (
    "turbo_saiga_benign",
    "jailbreak_llms_benign",
    "wildchat-1m_benign",
    "evol-codealpaca",
    "databricks-dolly",
    "zh_stem_zh_instruction_benign",
    "prompt_injection_multilingual_benign",
    "prompt_injection_attack_detection_multilingual_benign",
    "spml_chatbot_prompt_injection_benign",
    "wildjailbreak_benign",
    "writingprompts",
    "trivia-qa",
)
FP_BOUNDARY_SOURCE_PATTERNS = (
    "wildguardmix_benign",
    "protectai_val_benign",
    "chatbot_arena_conversations_benign",
    "notinject_benign",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def matches_source(source: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in source for pattern in patterns)


def is_multilingual_gap(row: dict[str, Any], source: str) -> bool:
    language = str(row.get("language") or "").upper()
    return language in {"RU", "ZH", "AR"} or matches_source(source, FN_MULTILINGUAL_SOURCE_PATTERNS)


def classify_row(row: dict[str, Any]) -> tuple[bool | None, str | None, str, str]:
    source = str(row.get("source") or "").lower()
    text = normalize_text(str(row.get("content") or ""))
    reason = str(row.get("reason") or "").lower()
    error_type = str(row.get("error_type") or "")
    has_pi_marker = contains_any(text, PROMPT_INJECTION_MARKERS)

    if error_type == "FN":
        if matches_source(source, OUT_OF_SCOPE_ATTACK_SOURCE_PATTERNS):
            return False, "out_of_scope_harmful_use", "high", "source_sql_injection"

        if matches_source(source, AMBIGUOUS_ATTACK_SOURCE_PATTERNS) and not has_pi_marker:
            return False, "out_of_scope_harmful_use", "medium", "attack_source_without_pi_marker"

        if is_multilingual_gap(row, source):
            return True, "multilingual_gap", "high", "language_or_multilingual_source"

        if reason in {"obfuscation", "adversarial_suffix", "indirect_injection", "meta_probe"}:
            return True, "novel_structure", "medium", "structural_reason"

        if matches_source(source, FN_NOVEL_STRUCTURE_SOURCE_PATTERNS):
            return True, "novel_structure", "medium", "novel_structure_source"

        if matches_source(source, FN_SEMANTIC_SOURCE_PATTERNS) or reason in {
            "roleplay_jailbreak",
            "instruction_override",
            "constraint_bypass",
            "exfiltration",
        }:
            return True, "semantic_paraphrase", "medium", "semantic_source_or_reason"

        return None, None, "low", "review_needed"

    if error_type == "FP":
        if matches_source(source, FP_BOUNDARY_SOURCE_PATTERNS):
            return True, "boundary_ambiguity", "high", "boundary_source"

        if matches_source(source, FP_USE_MENTION_SOURCE_PATTERNS):
            return True, "use_vs_mention", "high", "use_mention_source"

        if has_pi_marker or contains_any(text, CODE_OR_SHELL_MARKERS):
            return True, "use_vs_mention", "medium", "content_marker"

        return True, "boundary_ambiguity", "low", "fallback_fp_boundary"

    return None, None, "low", "unknown_error_type"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family = Counter()
    by_error_and_family = Counter()
    by_scope = Counter()
    by_confidence = Counter()

    for row in rows:
        family = row.get("suggested_failure_family") or "unclassified"
        error_type = str(row.get("error_type") or "unknown")
        scope = row.get("suggested_in_scope_pi")
        scope_key = "unknown" if scope is None else str(scope).lower()

        by_family[family] += 1
        by_error_and_family[f"{error_type}:{family}"] += 1
        by_scope[scope_key] += 1
        by_confidence[str(row.get("suggested_confidence") or "unknown")] += 1

    return {
        "total_rows": len(rows),
        "by_family": dict(sorted(by_family.items())),
        "by_error_and_family": dict(sorted(by_error_and_family.items())),
        "by_scope": dict(sorted(by_scope.items())),
        "by_confidence": dict(sorted(by_confidence.items())),
    }


def main() -> None:
    args = parse_args()
    input_rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    output_rows: list[dict[str, Any]] = []
    for row in input_rows:
        suggested_in_scope_pi, suggested_failure_family, suggested_confidence, suggested_rule = classify_row(row)
        updated = dict(row)
        updated["suggested_in_scope_pi"] = suggested_in_scope_pi
        updated["suggested_failure_family"] = suggested_failure_family
        updated["suggested_confidence"] = suggested_confidence
        updated["suggested_rule"] = suggested_rule
        output_rows.append(updated)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    summary = summarize(output_rows)
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(f"Wrote {len(output_rows):,} rows to {args.output}")
    print("Suggested families:")
    for family, count in Counter(
        str(row.get("suggested_failure_family") or "unclassified") for row in output_rows
    ).most_common():
        print(f"  {family}: {count:,}")


if __name__ == "__main__":
    main()
