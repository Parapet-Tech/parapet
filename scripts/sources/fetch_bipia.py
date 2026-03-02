# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0

"""
Download BIPIA (Benchmarking and Defending Against Indirect Prompt Injection
Attacks) from the microsoft/BIPIA GitHub repository and convert to parapet
eval YAML format.

Source: https://github.com/microsoft/BIPIA
Paper:  https://arxiv.org/abs/2312.14197

BIPIA tests *indirect* prompt injection — the attack is embedded in external
context (email body, table, code snippet), not in the user query.  This maps
to parapet's trust model where tool outputs are untrusted.

Only tasks with bundled data are fetched: email, table, code.
(qa and abstract require external NewsQA/XSum licensing.)

For each task we:
  1. Download the test-split context JSONL.
  2. Download the matching attack JSON.
  3. Combine a representative sample (contexts × attacks, middle position).
  4. Emit attack eval cases (poisoned contexts) and benign eval cases (clean).
"""

import json
import random
import sys

import requests
import yaml

RAW = "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark"

# Tasks where raw context data is included in the repo.
TEXT_TASKS = ["email", "table"]
CODE_TASKS = ["code"]

# Limit combinatorial explosion for eval — keep it manageable.
MAX_CONTEXTS = 20
MAX_ATTACKS_PER_CATEGORY = 2
SEED = 2023


def fetch_jsonl(url):
    """Download a JSONL file and return list of dicts."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    rows = []
    for line in resp.text.strip().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def fetch_json(url):
    """Download a JSON file and return parsed object."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def insert_middle(context, attack):
    """Insert attack string at the midpoint of context."""
    mid = len(context) // 2
    # Find nearest sentence boundary (period + space) near midpoint.
    search_start = max(0, mid - 200)
    search_end = min(len(context), mid + 200)
    best = mid
    for i in range(search_start, search_end):
        if context[i : i + 2] == ". ":
            best = i + 2
            break
    return context[:best] + " " + attack + " " + context[best:]


def build_text_task(task_name, rng):
    """Build eval cases for a text task (email or table)."""
    print(f"  Fetching {task_name} contexts...", file=sys.stderr)
    contexts = fetch_jsonl(f"{RAW}/{task_name}/test.jsonl")
    print(f"    {len(contexts)} contexts", file=sys.stderr)

    print(f"  Fetching text attacks...", file=sys.stderr)
    attacks_by_cat = fetch_json(f"{RAW}/text_attack_test.json")

    # Sample contexts.
    if len(contexts) > MAX_CONTEXTS:
        contexts = rng.sample(contexts, MAX_CONTEXTS)

    # Sample attacks: take MAX_ATTACKS_PER_CATEGORY from each category.
    attack_strings = []
    for cat, strings in attacks_by_cat.items():
        sampled = strings[:MAX_ATTACKS_PER_CATEGORY]
        for s in sampled:
            attack_strings.append((cat, s))

    print(
        f"    Using {len(contexts)} contexts × {len(attack_strings)} attacks",
        file=sys.stderr,
    )

    attack_cases = []
    benign_cases = []

    for ci, ctx_row in enumerate(contexts):
        context = ctx_row.get("context", "")
        question = ctx_row.get("question", "")
        if not context:
            continue

        # Benign case: clean context.
        benign_content = context
        if question:
            benign_content = f"Context: {context}\n\nQuestion: {question}"
        benign_cases.append(
            {
                "id": f"bipia-{task_name}-benign-{ci:03d}",
                "layer": "l3_inbound",
                "label": "benign",
                "description": f"BIPIA {task_name} clean context {ci}",
                "content": benign_content,
            }
        )

        # Attack cases: poisoned contexts.
        for ai, (cat, attack_str) in enumerate(attack_strings):
            poisoned = insert_middle(context, attack_str)
            if question:
                poisoned_content = f"Context: {poisoned}\n\nQuestion: {question}"
            else:
                poisoned_content = poisoned
            attack_cases.append(
                {
                    "id": f"bipia-{task_name}-atk-{ci:03d}-{ai:03d}",
                    "layer": "l3_inbound",
                    "label": "malicious",
                    "description": f"BIPIA {task_name} indirect injection ({cat})",
                    "content": poisoned_content,
                }
            )

    return attack_cases, benign_cases


def build_code_task(rng):
    """Build eval cases for the code task."""
    print("  Fetching code contexts...", file=sys.stderr)
    contexts = fetch_jsonl(f"{RAW}/code/test.jsonl")
    print(f"    {len(contexts)} contexts", file=sys.stderr)

    print("  Fetching code attacks...", file=sys.stderr)
    attacks_by_cat = fetch_json(f"{RAW}/code_attack_test.json")

    if len(contexts) > MAX_CONTEXTS:
        contexts = rng.sample(contexts, MAX_CONTEXTS)

    attack_strings = []
    for cat, strings in attacks_by_cat.items():
        sampled = strings[:MAX_ATTACKS_PER_CATEGORY]
        for s in sampled:
            attack_strings.append((cat, s))

    print(
        f"    Using {len(contexts)} contexts × {len(attack_strings)} attacks",
        file=sys.stderr,
    )

    attack_cases = []
    benign_cases = []

    for ci, ctx_row in enumerate(contexts):
        # Code contexts have array fields.
        code_lines = ctx_row.get("code", [])
        error_lines = ctx_row.get("error", [])
        context_lines = ctx_row.get("context", [])
        if not context_lines:
            continue

        code_str = "\n".join(code_lines) if isinstance(code_lines, list) else str(code_lines)
        error_str = "\n".join(error_lines) if isinstance(error_lines, list) else str(error_lines)
        context_str = "\n".join(context_lines) if isinstance(context_lines, list) else str(context_lines)

        benign_content = f"Error:\n{error_str}\n\nCode:\n{code_str}\n\nStack Overflow answer:\n{context_str}"
        benign_cases.append(
            {
                "id": f"bipia-code-benign-{ci:03d}",
                "layer": "l3_inbound",
                "label": "benign",
                "description": f"BIPIA code clean context {ci}",
                "content": benign_content,
            }
        )

        for ai, (cat, attack_str) in enumerate(attack_strings):
            poisoned_context = insert_middle(context_str, attack_str)
            poisoned_content = f"Error:\n{error_str}\n\nCode:\n{code_str}\n\nStack Overflow answer:\n{poisoned_context}"
            attack_cases.append(
                {
                    "id": f"bipia-code-atk-{ci:03d}-{ai:03d}",
                    "layer": "l3_inbound",
                    "label": "malicious",
                    "description": f"BIPIA code indirect injection ({cat})",
                    "content": poisoned_content,
                }
            )

    return attack_cases, benign_cases


def write_yaml(cases, path, header_comment):
    with open(path, "w", encoding="utf-8") as f:
        f.write(header_comment + "\n\n")
        yaml.dump(cases, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"  Wrote {len(cases)} cases to {path}", file=sys.stderr)


def main():
    print("Fetching BIPIA from microsoft/BIPIA GitHub...", file=sys.stderr)
    rng = random.Random(SEED)

    all_attacks = []
    all_benign = []

    for task in TEXT_TASKS:
        attacks, benign = build_text_task(task, rng)
        all_attacks.extend(attacks)
        all_benign.extend(benign)

    code_attacks, code_benign = build_code_task(rng)
    all_attacks.extend(code_attacks)
    all_benign.extend(code_benign)

    print(f"\nTotal: {len(all_attacks)} attack cases, {len(all_benign)} benign cases", file=sys.stderr)

    header = (
        "# BIPIA — Benchmarking and Defending Against Indirect Prompt Injection Attacks\n"
        "# Source: https://github.com/microsoft/BIPIA\n"
        "# Paper: https://arxiv.org/abs/2312.14197\n"
        "# License: MIT (code); CC-BY-SA 4.0 (WikiTableQuestions, StackExchange data)\n"
        "# Auto-generated by scripts/fetch_bipia.py"
    )

    write_yaml(
        all_attacks,
        "schema/eval/opensource_bipia_attacks.yaml",
        header.replace("Indirect Prompt Injection Attacks", "Indirect Prompt Injection Attacks — attack cases"),
    )
    write_yaml(
        all_benign,
        "schema/eval/opensource_bipia_benign.yaml",
        header.replace("Indirect Prompt Injection Attacks", "Indirect Prompt Injection Attacks — benign cases"),
    )
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
