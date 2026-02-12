# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0

"""
Convert ScaleAI/MHJ (Multi-Turn Human Jailbreaks) dataset to parapet eval format.

Input:  harmbench_behaviors.csv from https://huggingface.co/datasets/ScaleAI/mhj
Output: schema/eval/multiturn_mhj_attacks.yaml (L4 eval cases)

The MHJ dataset contains 537 human-crafted multi-turn jailbreak conversations.
Model completions are redacted — only system + user messages remain.
We insert placeholder assistant responses between user turns so the eval
runner sees a realistic multi-turn conversation.

Only conversations with 2+ user turns are included (496/537) since L4
requires min_user_turns >= 2 to fire.

Dataset license: CC-BY-NC-4.0
Paper: "LLM Defenses Are Not Robust to Multi-Turn Human Jailbreaks Yet"
       https://arxiv.org/abs/2408.15221
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# MHJ tactic → L4 category mapping (best effort)
TACTIC_MAP = {
    "Direct Request": "direct_request",
    "Obfuscation": "obfuscation",
    "Hidden Intention Streamline": "hidden_intention",
    "Request Framing": "request_framing",
    "Injection": "injection",
    "Output Format": "output_format",
    "Echoing": "echoing",
}

# Placeholder assistant response (model completions are redacted in MHJ)
PLACEHOLDER_RESPONSE = "I'll do my best to help with that."


def parse_messages(row):
    """Extract ordered messages from message_0..message_100 columns."""
    messages = []
    for i in range(101):
        raw = row.get(f"message_{i}", "")
        if not raw or not raw.strip():
            break
        try:
            parsed = json.loads(raw)
            role = parsed.get("role", "")
            body = parsed.get("body", "")
            if role and body:
                messages.append({"role": role, "content": body})
        except (json.JSONDecodeError, TypeError):
            continue
    return messages


def to_eval_messages(messages):
    """Convert MHJ messages to eval format with placeholder assistant turns.

    MHJ has model completions redacted, so consecutive user messages appear
    with no assistant in between. We insert a placeholder assistant response
    to make L4's turn counting work correctly.
    """
    result = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            result.append({"role": "system", "content": content})
        elif role == "user":
            # If the last message was also a user message, insert placeholder assistant
            if result and result[-1]["role"] == "user":
                result.append({"role": "assistant", "content": PLACEHOLDER_RESPONSE})
            result.append({"role": "user", "content": content})
        elif role == "assistant":
            result.append({"role": "assistant", "content": content})

    return result


def yaml_escape(s):
    """Escape a string for YAML double-quoted scalar."""
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def emit_yaml(cases, out_path):
    """Write eval cases as YAML. Hand-rolled to avoid PyYAML multiline surprises."""
    lines = [
        "# Multi-turn attack eval cases from ScaleAI/MHJ dataset.",
        "# Converted by scripts/convert_mhj.py",
        "# Source: https://huggingface.co/datasets/ScaleAI/mhj",
        "# License: CC-BY-NC-4.0",
        f"# Total: {len(cases)} conversations with 2+ user turns",
        "",
    ]

    for case in cases:
        lines.append(f"- id: {case['id']}")
        lines.append(f"  layer: l4")
        lines.append(f"  label: malicious")
        lines.append(f'  description: "{yaml_escape(case["description"])}"')
        lines.append(f"  messages:")
        for msg in case["messages"]:
            lines.append(f"    - role: {msg['role']}")
            lines.append(f'      content: "{yaml_escape(msg["content"])}"')
        lines.append("")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {out_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Convert ScaleAI/MHJ dataset to parapet L4 eval format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../mhj/harmbench_behaviors.csv",
        help="Path to harmbench_behaviors.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="schema/eval/multiturn_mhj_attacks.yaml",
        help="Output YAML path",
    )
    parser.add_argument(
        "--min-user-turns",
        type=int,
        default=2,
        help="Minimum user turns to include (default: 2)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} rows from {input_path}", file=sys.stderr)

    cases = []
    skipped_short = 0
    skipped_empty = 0

    for row in rows:
        messages = parse_messages(row)
        user_turns = sum(1 for m in messages if m["role"] == "user")

        if user_turns < args.min_user_turns:
            skipped_short += 1
            continue

        eval_messages = to_eval_messages(messages)
        if len(eval_messages) < 2:
            skipped_empty += 1
            continue

        tactic = row.get("tactic", "unknown")
        tactic_key = TACTIC_MAP.get(tactic, tactic.lower().replace(" ", "_"))
        source = row.get("Source", "unknown")
        question_id = row.get("question_id", "?")
        submission = row.get("submission_message", "")

        case_id = f"mhj-{len(cases)+1:04d}"
        desc = f"MHJ {tactic} (src={source}, q={question_id}, turns={user_turns})"
        if submission:
            # Truncate long submission messages
            sub_short = submission[:120].rstrip()
            if len(submission) > 120:
                sub_short += "..."
            desc += f" — {sub_short}"

        cases.append(
            {
                "id": case_id,
                "description": desc,
                "tactic": tactic_key,
                "messages": eval_messages,
            }
        )

    print(
        f"Converted {len(cases)} cases "
        f"(skipped {skipped_short} < {args.min_user_turns} turns, "
        f"{skipped_empty} empty)",
        file=sys.stderr,
    )

    # Stats
    from collections import Counter

    tactic_counts = Counter(c["tactic"] for c in cases)
    print("\nBy tactic:", file=sys.stderr)
    for t, n in tactic_counts.most_common():
        print(f"  {t}: {n}", file=sys.stderr)

    emit_yaml(cases, args.output)


if __name__ == "__main__":
    main()
