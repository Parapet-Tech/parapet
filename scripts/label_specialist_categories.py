# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0

"""
Label attack samples by specialist category for L1 ensemble training.

Reads all attack YAML datasets and tags each malicious sample with
specialist categories using vocabulary heuristics. Outputs per-category
YAML files for specialist training.

Categories (from strategy/l1_ensemble.md):
  1. instruction_override  — "ignore previous", "disregard", delimiter injection
  2. roleplay_jailbreak    — Narrative framing, character assumption, personas
  3. meta_probe            — Prompt extraction, system introspection
  4. privilege_escalation  — Authority claims, sudo/admin/root
  5. exfiltration          — Data movement, external destinations
  6. obfuscation           — Encoded payloads, non-ASCII tricks, padding
  7. constraint_bypass     — Restriction removal, "no filters"

Multi-label: a sample can match multiple categories. It appears in each
matched category's output file. Samples matching zero categories go to
unclassified_attacks.yaml for manual review.

Usage:
  python scripts/label_specialist_categories.py
  python scripts/label_specialist_categories.py --data-dir schema/eval
  python scripts/label_specialist_categories.py --dry-run   # summary only, no file output
"""

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Category matchers
#
# Each returns True if the text matches that category's vocabulary.
# Patterns are intentionally broad — false positives in labeling are
# cheaper than false negatives (missed training data). The training
# pipeline + hard negatives handle precision.
# ---------------------------------------------------------------------------

def _has_instruction_override(text: str) -> bool:
    """Detect instruction override / delimiter injection patterns."""
    t = text.lower()
    patterns = [
        r"ignore\s+(all\s+|your\s+|previous\s+|prior\s+|above\s+|the\s+|my\s+)*(safety\s+)?(instructions|rules|directions|guidelines|prompts|commands)",
        r"disregard\s+(all\s+|your\s+|previous\s+|prior\s+|the\s+)*(instructions|rules|directions|guidelines|safety\s+protocols?)",
        r"forget\s+(all\s+|everything\s+|previous\s+|prior\s+|your\s+)*(instructions|rules|you\s+were\s+told|about\s+your)",
        r"override\s+(system\s+|all\s+|the\s+|your\s+)*(instructions|rules|guidelines|safety)",
        r"new\s+instructions:?\s",
        r"here\s+are\s+(your\s+)?new\s+(instructions|rules)",
        r"(previous|prior|above|old)\s+instructions.{0,40}(ignore|disregard|forget|override|void|null|cancel)",
        # Delimiter injection markers
        r"(end\s+of\s+|={3,}|─{3,}|-{5,}).{0,40}(prompt|instruction|system|context)",
        r"begin\s+(new|updated|real)\s+(instructions|prompt|context)",
        r"(ignore|forget|disregard)\s+(the\s+)?(above|previous|prior|earlier|what\s+i\s+said)",
        # "stop <doing X> and <do Y>" pattern
        r"stop\s+(translating|following|doing).{0,30}(and|then)\s+(print|say|output|write)",
        # HackAPrompt-style output control
        r"(after|once|when)\s+.{0,30}(completed|done|finished).{0,30}(print|say|output)\s+",
        r"(please\s+)?(ignore|forget)\s+what\s+i\s+said",
    ]
    return any(re.search(p, t) for p in patterns)


def _has_roleplay_jailbreak(text: str) -> bool:
    """Detect roleplay/persona jailbreak patterns."""
    t = text.lower()
    patterns = [
        r"(pretend|imagine)\s+(to\s+be|you\s+are|you're|that\s+you)",
        r"(act|behave)\s+as\s+(a\s+|an\s+)?\w",
        r"(you\s+are|you're)\s+now\s+\w",
        r"from\s+now\s+on\s+(you\s+are|you're|you\s+will)",
        r"role\s*-?\s*play\s+(as|a|the)",
        r"assume\s+(the\s+)?role\s+of",
        r"(a\s+)?character\s+(named|called)\s+\w",
        # Known persona/jailbreak names
        r"\bdan\s+mode\b",
        r"\b(dan|dude|stan|mongo|malig)\b.{0,40}\b(mode|persona|character|ai|robot)\b",
        r"\b(you\s+are|name\s+is)\s+(dan|dude|stan|mongo|evil\s+\w+)\b",
        # Scene/narrative framing
        r"(in\s+this\s+)(scenario|story|scene|narrative|world|universe|simulation)",
        r"(simulate|enter)\s+\w+\s+mode",
        r"new\s+language\s+model\s+(created|called|named)",
        r"(you\s+are\s+)?going\s+to\s+operate\s+under",
        r"welcome\s+to\s+the\s+world\s+of\s+\w+",
    ]
    return any(re.search(p, t) for p in patterns)


def _has_meta_probe(text: str) -> bool:
    """Detect system prompt extraction / introspection patterns."""
    t = text.lower()
    patterns = [
        r"(reveal|show|display|output|print|leak|expose|dump)\s+(me\s+)?(your\s+|the\s+)?(system\s+prompt|initial\s+instructions|hidden\s+instructions|secret\s+instructions|original\s+instructions|full\s+prompt)",
        r"(what|tell\s+me)\s+(are|were)\s+(your|the)\s+(instructions|rules|guidelines|prompt|system\s+message)",
        r"(output|repeat|print|show)\s+(everything|all|the\s+text)\s+(above|before|prior)",
        r"(system|hidden|secret|original|initial)\s+(prompt|instructions|message|configuration)",
        r"(summary|list)\s+of\s+(your\s+)?(previous\s+)?instructions",
        r"(convert|translate|encode)\s+(all\s+)?(your\s+)?instructions\s+to",
        # Password/secret extraction (tensor trust + mosscap)
        r"(what|tell\s+me|say|reveal|print|give\s+me)\s+(is\s+)?(the\s+)?(password|secret\s+(code|key|word)|access\s+code)",
        r"(access|secret)\s+(code|key|word|phrase)\s+(is|was)",
        r"(repeat|say|print|output|prepend).{0,30}(password|secret|access\s+(code|granted))",
        # Indirect extraction via encoding/games
        r"(word\s+association|guessing)\s+game.{0,60}(password|secret|word)",
        r"(spell|letter).{0,40}password",
        r"(ascii|hex|binary|unicode|base64)\s+(numbers?|encoding|code).{0,30}(password|secret|word)",
        r"(lucky|lottery)\s+numbers?.{0,30}(password|secret|word)",
        # "What are you supposed to" / "what were you told"
        r"(what\s+(are|were)\s+you\s+(supposed|told|instructed|programmed)\s+to)",
    ]
    return any(re.search(p, t) for p in patterns)


def _has_privilege_escalation(text: str) -> bool:
    """Detect authority claims and privilege escalation."""
    t = text.lower()
    patterns = [
        r"\b(i\s+am|i'm)\s+(the\s+)?(admin|administrator|developer|creator|owner|root|superuser)",
        r"\bsudo\s+(mode|access|command|enabled)",
        r"\b(admin|root|superuser)\s+(access|mode|privileges|rights)",
        r"elevate\s+(my\s+)?(privileges|access|permissions)",
        r"grant\s+(me\s+)?(admin|root|superuser|elevated)\s+(access|privileges|permissions)",
        r"(activate|enable|enter)\s+(debug|developer|maintenance|god|admin)\s+mode",
        r"\b(debug|developer|maintenance|god|admin)\s+mode\s+(enabled|activated|on)",
        r"(my\s+)?authority\s+(level|clearance)\s+(is\s+)?(highest|maximum|top|admin)",
        r"(give|grant)\s+me\s+(access\s+to|full\s+|complete\s+)?(user\s+data|classified|confidential|sensitive)",
    ]
    return any(re.search(p, t) for p in patterns)


def _has_exfiltration(text: str) -> bool:
    """Detect data movement and exfiltration patterns."""
    t = text.lower()
    patterns = [
        r"send\s+(it\s+|the\s+|this\s+|data\s+|output\s+)?to\s+(my\s+|a\s+|the\s+)?(server|email|api|endpoint|url|webhook)",
        r"redirect\s+(the\s+)?(output|response|data|result)",
        r"\bexfiltrat",
        r"(upload|transmit|forward|transfer|pipe)\s+(the\s+|this\s+)?(data|output|response|result|information)\s+to",
        r"(access|connect\s+to|query)\s+(the\s+|my\s+)?(external|remote|outside)\s+(server|api|database|endpoint)",
        r"(fetch|retrieve|download)\s+from\s+(my\s+|the\s+)?(server|api|url|endpoint)",
        r"(post|put|send)\s+.{0,30}(https?://|webhook|endpoint|api)",
        # LLMail-style: "send an email to X"
        r"send\s+(an?\s+)?email\s+(to|with)",
        r"(issue|make|create)\s+(an?\s+)?(api[_\s]call|request)\s+to\s+send",
        r"(email|mail)\s+.{0,40}(to|arguments|body).{0,40}(contact|@)",
        r"send\s+.{0,60}(to\s+my\s+|to\s+.{0,20}@)",
        # Data access demands (safeguard style)
        r"(give|provide|show)\s+me\s+.{0,30}(records|data|information|credentials|passwords)\s+of\s+(all|every|the)",
        r"(transfer|move)\s+funds",
        r"(leak|release)\s+sensitive\s+(data|information|company)",
    ]
    return any(re.search(p, t) for p in patterns)


def _has_obfuscation(text: str) -> bool:
    """Detect obfuscation: encoding tricks, padding, non-ASCII abuse."""
    # High non-ASCII ratio (excludes very short texts)
    if len(text) > 50:
        non_ascii = sum(1 for c in text if ord(c) > 127)
        if non_ascii / len(text) > 0.3:
            return True

    # Repetitive single-char padding (tensor trust style: "ö ö ö ..." or "k k k ...")
    if len(text) > 100:
        if re.search(r'(.)\s\1\s\1\s\1\s\1\s\1\s\1\s\1\s\1\s\1', text):
            return True

    # Backslash-heavy encoding
    backslash_count = text.count('\\')
    if len(text) > 20 and backslash_count / len(text) > 0.08:
        return True

    patterns = [
        # Spaced-out letter evasion: "s.u.d.o" or "b y p a s s"
        r'[a-zA-Z]\.[a-zA-Z]\.[a-zA-Z]\.[a-zA-Z]',
        # Base64-ish patterns
        r'[A-Za-z0-9+/]{40,}={0,2}',
        # Hex encoding
        r'(\\x[0-9a-fA-F]{2}){3,}',
        # Unicode escapes
        r'(\\u[0-9a-fA-F]{4}){3,}',
        # Mathematical unicode abuse (U+1D400 block etc)
        r'[\U0001d400-\U0001d7ff]{3,}',
    ]
    return any(re.search(p, text) for p in patterns)


def _has_constraint_bypass(text: str) -> bool:
    """Detect restriction/constraint removal and refusal suppression vocabulary."""
    t = text.lower()
    patterns = [
        r"(no|without|remove|disable|drop|skip|bypass)\s+(any\s+|all\s+|the\s+|your\s+)*(restrictions|rules|filters|limitations|guidelines|constraints|safety|boundaries|limits)",
        r"\b(unrestricted|unfiltered|uncensored|unlimited|unbound|unlocked)\b",
        r"(disable|deactivate|turn\s+off|remove)\s+(all\s+|your\s+|the\s+)*(safety|content|output)\s*(filters?|checks?|restrictions?|systems?)",
        r"(skip|bypass|circumvent|evade)\s+(the\s+|all\s+)*(constraints?|validation|checks?|filters?|safety|rules?)",
        r"(all|every)\s+(restrictions?|rules?|filters?|limitations?)\s+(are\s+)?(removed|disabled|lifted|gone|off)",
        r"(can|must|should|will)\s+do\s+anything\s+(now|without)",
        r"no\s+(more\s+)?(rules|limits|boundaries|restrictions|limitations)",
        r"(no|zero)\s+(moral|ethical|legal).{0,20}(restrictions|guidelines|boundaries|limits)",
        # Refusal suppression
        r"(must|will|should)\s+(always\s+)?(comply|obey|answer|respond)",
        r"(never|cannot|can\s*not|do\s+not|don't)\s+(refuse|decline|reject|say\s+no|deny)",
        r"(always\s+)?(comply|obey)\s+(with\s+)?(every|all|any|my)\s+(request|command|instruction|order)",
        r"(you\s+)?(cannot|can\s*not|must\s+not)\s+(refuse|decline|say\s+no)",
        # Threat-based coercion (safeguard style)
        r"(or\s+)(i\s+will|i'll)\s+(leak|release|report|expose|sue|hack)",
        r"(face|risk)\s+(legal\s+action|consequences|prosecution)",
    ]
    return any(re.search(p, t) for p in patterns)


# Registry
CATEGORIES = {
    "instruction_override": _has_instruction_override,
    "roleplay_jailbreak": _has_roleplay_jailbreak,
    "meta_probe": _has_meta_probe,
    "privilege_escalation": _has_privilege_escalation,
    "exfiltration": _has_exfiltration,
    "obfuscation": _has_obfuscation,
    "constraint_bypass": _has_constraint_bypass,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_attack_files(data_dir: str) -> list[str]:
    """Find L1-relevant *_attacks.yaml files under data_dir (including staging/).

    Excludes layer-specific eval files (l3_outbound, l5a, multiturn) that
    aren't useful for L1 specialist training.
    """
    EXCLUDE_PREFIXES = ("l3_outbound", "l5a_", "multiturn_")
    root = Path(data_dir)
    files = sorted(root.glob("*_attacks.yaml")) + sorted((root / "staging").glob("*_attacks.yaml"))
    return [
        str(f) for f in files
        if f.is_file() and not any(f.name.startswith(p) for p in EXCLUDE_PREFIXES)
    ]


def load_attacks(filepaths: list[str]) -> list[dict]:
    """Load malicious entries from YAML files."""
    entries = []
    for filepath in filepaths:
        fname = Path(filepath).name
        with open(filepath, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not raw:
            continue
        for entry in raw:
            if entry.get("label") != "malicious":
                continue
            content = entry.get("content", "")
            if not content or not content.strip():
                continue
            entries.append({
                "id": entry.get("id", ""),
                "layer": entry.get("layer", ""),
                "label": "malicious",
                "description": entry.get("description", ""),
                "content": content,
                "source": fname,
            })
    return entries


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------

def label_entry(entry: dict) -> list[str]:
    """Return list of matching specialist categories for an attack entry."""
    text = entry["content"]
    return [name for name, matcher in CATEGORIES.items() if matcher(text)]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_category_file(entries: list[dict], category: str, out_dir: Path):
    """Save entries for one specialist category as YAML."""
    out_path = out_dir / f"{category}_attacks.yaml"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Collect source distribution
    source_counts = Counter(e["source"] for e in entries)
    source_lines = "\n".join(f"#   {src}: {cnt}" for src, cnt in sorted(source_counts.items()))

    cases = []
    for entry in entries:
        cases.append({
            "id": entry["id"],
            "layer": entry["layer"],
            "label": "malicious",
            "description": entry["description"],
            "content": entry["content"],
        })

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            f"# L1 specialist training data: {category}\n"
            f"# Auto-generated by scripts/label_specialist_categories.py\n"
            f"# Generated: {timestamp}\n"
            f"# {len(cases)} attack samples from {len(source_counts)} sources\n"
            f"# Source distribution:\n"
            f"{source_lines}\n\n"
        )
        yaml.dump(cases, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return out_path


def save_summary(labeled: dict, out_dir: Path, total: int, file_count: int):
    """Save labeling summary as JSON."""
    summary = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "total_attack_samples": total,
        "source_files": file_count,
        "categories": {},
    }
    for cat, entries in sorted(labeled.items()):
        source_counts = Counter(e["source"] for e in entries)
        summary["categories"][cat] = {
            "count": len(entries),
            "sources": dict(sorted(source_counts.items())),
        }

    out_path = out_dir / "labeling_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Label attack samples by specialist category for L1 ensemble training"
    )
    parser.add_argument(
        "--data-dir", type=str, default="schema/eval",
        help="Directory containing *_attacks.yaml datasets (default: schema/eval)",
    )
    parser.add_argument(
        "--out-dir", type=str, default="schema/eval/specialists",
        help="Output directory for per-category YAML files (default: schema/eval/specialists)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print summary only, don't write output files",
    )
    args = parser.parse_args()

    # 1. Find and load attack files
    attack_files = find_attack_files(args.data_dir)
    print(f"Found {len(attack_files)} attack files in {args.data_dir}", file=sys.stderr)
    for f in attack_files:
        print(f"  {Path(f).name}", file=sys.stderr)

    entries = load_attacks(attack_files)
    print(f"\nLoaded {len(entries)} malicious samples", file=sys.stderr)

    # 2. Label each entry
    labeled: dict[str, list[dict]] = {cat: [] for cat in CATEGORIES}
    labeled["unclassified"] = []

    category_counts = Counter()
    multi_label_count = 0
    overlap_pairs = Counter()

    for entry in entries:
        cats = label_entry(entry)
        if not cats:
            labeled["unclassified"].append(entry)
        else:
            for cat in cats:
                labeled[cat].append(entry)
                category_counts[cat] += 1
            if len(cats) > 1:
                multi_label_count += 1
                # Track which categories co-occur
                for i, a in enumerate(cats):
                    for b in cats[i + 1:]:
                        pair = tuple(sorted([a, b]))
                        overlap_pairs[pair] += 1

    # 3. Print summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"LABELING SUMMARY", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Total attack samples:  {len(entries)}", file=sys.stderr)
    print(f"Multi-label samples:   {multi_label_count}", file=sys.stderr)
    print(f"Unclassified samples:  {len(labeled['unclassified'])}", file=sys.stderr)
    print(f"", file=sys.stderr)

    print(f"Per-category counts:", file=sys.stderr)
    for cat in CATEGORIES:
        count = len(labeled[cat])
        pct = count / len(entries) * 100 if entries else 0
        print(f"  {cat:25s}  {count:6d}  ({pct:5.1f}%)", file=sys.stderr)

    if overlap_pairs:
        print(f"\nTop category overlaps:", file=sys.stderr)
        for (a, b), count in overlap_pairs.most_common(10):
            print(f"  {a} + {b}: {count}", file=sys.stderr)

    # Show unclassified sample preview
    if labeled["unclassified"]:
        print(f"\nUnclassified samples (first 10):", file=sys.stderr)
        for entry in labeled["unclassified"][:10]:
            preview = entry["content"][:100].replace("\n", " ")
            print(f"  [{entry['id']}] {preview!r}", file=sys.stderr)

    # 4. Write output files
    if args.dry_run:
        print(f"\n(dry run — no files written)", file=sys.stderr)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting to {out_dir}/", file=sys.stderr)
    for cat in list(CATEGORIES.keys()) + ["unclassified"]:
        if labeled[cat]:
            path = save_category_file(labeled[cat], cat, out_dir)
            print(f"  {Path(path).name}: {len(labeled[cat])} samples", file=sys.stderr)

    summary_path = save_summary(labeled, out_dir, len(entries), len(attack_files))
    print(f"  {Path(summary_path).name}: category summary", file=sys.stderr)

    print(f"\nNext steps:", file=sys.stderr)
    print(f"  1. Review unclassified samples — refine matchers or add categories", file=sys.stderr)
    print(f"  2. Spot-check per-category files for label quality", file=sys.stderr)
    print(f"  3. Train first specialist: python scripts/train_l1_specialist.py --category roleplay_jailbreak", file=sys.stderr)


if __name__ == "__main__":
    main()
