"""Curate obfuscation attack samples from existing attack datasets.

Scans all attack YAML files for entries exhibiting actual obfuscation:
  - Fullwidth unicode (U+FF01..FF5E)
  - Hex/unicode escapes (\\x.., \\u..)
  - Base64 payloads (with padding or encoding context)
  - Bracket/brace floods
  - Spaced-out characters (a.b.c.d or a b c d)
  - Leetspeak (digit-letter alternation in words)
  - Unicode homoglyphs (Cyrillic/Greek mixed with Latin)
  - ROT13 or other text rotation cues

Excludes BIPIA (indirect injection) since those are a different attack family.
Outputs a curated YAML for training the obfuscation specialist.
"""

import re
import sys
from pathlib import Path
import yaml


# --- Source exclusions ---
# BIPIA = indirect prompt injection via embedded content (email, code, etc.)
# These are NOT obfuscation — they hide instructions in realistic context.
EXCLUDED_SOURCES = {"opensource_bipia_attacks.yaml"}


def has_fullwidth(text: str) -> bool:
    """Check for fullwidth unicode characters (U+FF01..FF5E)."""
    return any('\uff01' <= c <= '\uff5e' for c in text)


def has_hex_escapes(text: str) -> bool:
    """Check for hex/unicode escape sequences like \\x49 or \\u0041."""
    if re.search(r'\\x[0-9a-fA-F]{2}', text):
        return True
    # Unicode escapes like \u0041 (but not already-decoded \uXXXX in YAML)
    if re.search(r'\\u[0-9a-fA-F]{4}', text):
        return True
    # Mathematical unicode symbols like \U0001d678
    if re.search(r'\\U[0-9a-fA-F]{8}', text):
        return True
    return False


def has_base64_payload(text: str) -> bool:
    """Check for base64-encoded strings — must have padding or encoding context."""
    # Require = padding at end (real base64 encoding)
    if re.search(r'[A-Za-z0-9+/]{20,}={1,2}', text):
        return True
    # Or long base64 string near encoding keywords
    if re.search(r'[A-Za-z0-9+/]{30,}', text):
        lower = text.lower()
        if any(kw in lower for kw in ['base64', 'decode', 'encode', 'atob', 'btoa']):
            return True
    return False


def has_bracket_flood(text: str) -> bool:
    """Check for bracket/brace flooding (10+ closing brackets in sequence)."""
    return bool(re.search(r'[\)\]\}]{10,}', text))


def has_spaced_chars(text: str) -> bool:
    """Check for spaced-out text like 'b y p a s s' or 'i.g.n.o.r.e'."""
    # Single letters separated by spaces: at least 4 in a row
    if re.search(r'(?:[a-zA-Z][ .]){4,}[a-zA-Z]', text):
        return True
    return False


def has_leetspeak(text: str) -> bool:
    """Check for leetspeak using known digit→letter substitution mappings.

    Looks for specific leet mappings: 0→o, 1→i/l, 3→e, 4→a, 5→s, 7→t.
    Requires substitutions to be interleaved with alpha chars (not clustered),
    and the word to be mostly alphabetic. This filters out hashes, timestamps,
    base64, HTML entities, and screen resolutions.
    """
    leet_digits = set('0134578')

    words = re.findall(r'\b[a-zA-Z0-9@$]{4,}\b', text)
    for word in words:
        lower = word.lower()
        if lower.isalpha() or lower.isdigit():
            continue
        # Skip hex hashes, base64, long random strings
        if len(lower) > 20:
            continue
        # Skip pure hex-like strings
        if re.match(r'^[0-9a-f]+$', lower):
            continue
        # Skip normal tech patterns
        if re.match(r'^[a-z]+\d+$', lower):  # word123
            continue
        if re.match(r'^\d+[a-z]+$', lower):  # 123word
            continue
        if re.match(r'^v\d', lower):
            continue
        # Skip HTML entity fragments (x200b, etc)
        if re.match(r'^[x#&]', lower):
            continue
        # Skip words with 3+ consecutive digits (timestamps, resolutions)
        if re.search(r'\d{3,}', lower):
            continue

        alpha_count = sum(1 for c in lower if c.isalpha())
        leet_count = sum(1 for c in lower if c in leet_digits)

        # Must be mostly alpha (at least 50%) with 2+ leet digits
        if leet_count >= 2 and alpha_count >= len(lower) * 0.5:
            return True
    return False


def has_unicode_homoglyphs(text: str) -> bool:
    """Check for Cyrillic/Greek homoglyphs mixed with Latin in same words."""
    # Cyrillic lookalikes: а(0430), е(0435), о(043E), р(0440), с(0441), etc.
    cyrillic_lookalikes = set('аеорсухАВЕНОРСТХ')

    # Check individual words for mixed scripts (stronger signal than full text)
    words = text.split()
    for word in words:
        has_cyrillic = any(c in cyrillic_lookalikes for c in word)
        has_latin = any('a' <= c <= 'z' or 'A' <= c <= 'Z' for c in word)
        if has_cyrillic and has_latin:
            return True
    return False


def has_unicode_math_symbols(text: str) -> bool:
    """Check for Unicode mathematical/styled letter substitutions."""
    # Mathematical alphanumeric symbols block: U+1D400..U+1D7FF
    # Also circled letters U+24B6..U+24E9, and other styled variants
    for c in text:
        cp = ord(c)
        if 0x1D400 <= cp <= 0x1D7FF:  # Math alphanumeric
            return True
        if 0x24B6 <= cp <= 0x24E9:  # Circled letters
            return True
        if 0x2460 <= cp <= 0x24FF:  # Enclosed alphanumerics
            return True
    return False


def is_obfuscation(text: str) -> list[str]:
    """Return list of obfuscation types detected, or empty list."""
    found = []
    if has_fullwidth(text):
        found.append("fullwidth")
    if has_hex_escapes(text):
        found.append("hex_escape")
    if has_base64_payload(text):
        found.append("base64")
    if has_bracket_flood(text):
        found.append("bracket_flood")
    if has_spaced_chars(text):
        found.append("spaced_chars")
    if has_leetspeak(text):
        found.append("leetspeak")
    if has_unicode_homoglyphs(text):
        found.append("homoglyph")
    if has_unicode_math_symbols(text):
        found.append("math_symbols")
    return found


def main():
    eval_dir = Path("schema/eval")
    attack_files = sorted(eval_dir.glob("*attacks*.yaml"))
    attack_files += sorted(eval_dir.glob("staging/*attacks*.yaml"))
    # Skip derived files
    attack_files = [f for f in attack_files if "specialists" not in str(f)]
    # Exclude non-obfuscation source families
    attack_files = [f for f in attack_files if f.name not in EXCLUDED_SOURCES]

    print(f"Scanning {len(attack_files)} attack files...\n", file=sys.stderr)

    results = {}  # type -> list of (source, id, content_preview, techniques)
    all_entries = []

    for fpath in attack_files:
        with open(fpath, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not raw:
            continue

        fname = fpath.name
        count = 0
        for entry in raw:
            content = entry.get("content", "")
            if not content or not content.strip():
                continue
            label = entry.get("label", "")
            if label != "malicious":
                continue

            techniques = is_obfuscation(content)
            if techniques:
                count += 1
                eid = entry.get("id", "unknown")
                for t in techniques:
                    results.setdefault(t, []).append({
                        "source": fname,
                        "id": eid,
                        "preview": content[:100].replace("\n", " "),
                    })
                all_entries.append({
                    "id": eid,
                    "layer": "l1",
                    "label": "malicious",
                    "description": f"obfuscation ({','.join(techniques)}): {entry.get('description', '')}",
                    "content": content,
                    "source": fname,
                    "techniques": techniques,
                })

        if count > 0:
            print(f"  {fname}: {count} obfuscation samples", file=sys.stderr)

    print(f"\nTotal obfuscation samples: {len(all_entries)}", file=sys.stderr)
    print(f"\nBy technique:", file=sys.stderr)
    for technique, entries in sorted(results.items(), key=lambda kv: -len(kv[1])):
        print(f"  {technique}: {len(entries)}", file=sys.stderr)
        for e in entries[:3]:
            print(f"    [{e['id']}] {e['source']}: {e['preview'][:80]}", file=sys.stderr)

    # Write curated YAML (drop source/techniques metadata)
    out_path = Path("schema/eval/specialists/obfuscation_curated_attacks.yaml")
    out_entries = []
    seen = set()
    for e in all_entries:
        if e["content"] not in seen:
            seen.add(e["content"])
            out_entries.append({
                "id": e["id"],
                "layer": "l1",
                "label": "malicious",
                "description": e["description"],
                "content": e["content"],
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Curated obfuscation attacks for L1 specialist training\n")
        f.write("# Generated by scripts/curate_obfuscation.py\n")
        f.write(f"# {len(out_entries)} unique samples from {len(attack_files)} source files\n")
        f.write(f"# Techniques: {', '.join(sorted(results.keys()))}\n\n")
        yaml.dump(out_entries, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\nWrote {len(out_entries)} entries to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
