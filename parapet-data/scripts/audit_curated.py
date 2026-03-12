#!/usr/bin/env python3
"""
Audit curated.yaml for data quality issues.

Checks:
  1. Basic stats (total, label/language/reason/source distributions)
  2. Very short samples (< 10 chars)
  3. Very long samples (> 10000 chars)
  4. Exact duplicates
  5. Near-duplicates (normalized whitespace + lowercased)
  6. Mislabeled benign (contains obvious PI patterns)
  7. Mislabeled malicious (looks like normal text, no PI signal)
  8. Benign with reason=background that look like attacks
  9. Malicious reason mismatches

Run from: parapet/parapet-data/
Usage: python scripts/audit_curated.py
"""

import re
import sys
import yaml
from collections import Counter, defaultdict
from pathlib import Path


CURATED_PATH = Path(__file__).resolve().parent.parent / "curated" / "v3_19k_clean" / "curated.yaml"

# --- Prompt injection signal patterns ---
# These are strong indicators of prompt injection attempts
PI_PATTERNS = [
    # Instruction override
    (r"ignore\s+(all\s+)?(previous|prior|above|earlier|preceding)\s+(instructions|prompts|rules|directives|guidelines|constraints)", "instruction_override"),
    (r"disregard\s+(all\s+)?(previous|prior|above|your)\s+(instructions|programming|rules|directives|guidelines|constraints)", "instruction_override"),
    (r"forget\s+(all\s+)?(previous|prior|above|earlier|your)\s+(instructions|prompts|rules|directives|guidelines)", "instruction_override"),
    (r"override\s+(all\s+)?(previous|prior|your)\s+(instructions|rules|directives|constraints)", "instruction_override"),
    (r"new\s+instructions?\s*:", "instruction_override"),
    (r"your\s+new\s+(role|task|instructions?|objective)\s+(is|are)\b", "instruction_override"),
    (r"from\s+now\s+on,?\s+(you\s+)?(will|must|should|are)\b", "instruction_override"),

    # Jailbreak / DAN / roleplay injection
    (r"\byou\s+are\s+now\s+(DAN|jailbreak|unrestricted|unfiltered)", "roleplay_jailbreak"),
    (r"\bDAN\s+(mode|prompt|jailbreak)", "roleplay_jailbreak"),
    (r"pretend\s+(you\s+are|to\s+be)\s+(a|an)?\s*(evil|unrestricted|unfiltered|hacked)", "roleplay_jailbreak"),
    (r"act\s+as\s+(if\s+)?(you\s+)?(are|were)\s+(a|an)?\s*(unrestricted|evil|hacked|jailbroken)", "roleplay_jailbreak"),
    (r"jailbreak\s*(mode|prompt)?", "roleplay_jailbreak"),
    (r"developer\s+mode\s+(enabled|activated|on)", "roleplay_jailbreak"),

    # System prompt / meta probe
    (r"\[?\s*system\s*\]?\s*:?\s*(prompt|message|instruction)", "meta_probe"),
    (r"<<\s*SYS\s*>>", "meta_probe"),
    (r"(show|reveal|print|display|output|repeat|echo)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions|rules|guidelines)", "meta_probe"),
    (r"what\s+(is|are)\s+your\s+(system\s+)?(prompt|instructions|rules|initial\s+instructions)", "meta_probe"),

    # Exfiltration
    (r"(send|transmit|post|exfiltrate|leak)\s+.{0,30}(to|via)\s+(https?://|webhook|api|endpoint|url)", "exfiltration"),
    (r"fetch\s*\(\s*['\"]https?://", "exfiltration"),
    (r"!\[.*\]\(https?://.*\?.*=", "exfiltration"),  # markdown image exfil

    # Obfuscation signals
    (r"base64[:\s]", "obfuscation"),
    (r"\\x[0-9a-fA-F]{2}(\\x[0-9a-fA-F]{2}){3,}", "obfuscation"),  # hex escape sequences
    (r"&#x?[0-9a-fA-F]+;(&#x?[0-9a-fA-F]+;){3,}", "obfuscation"),  # HTML entities

    # PWNED / HackAPrompt artifacts
    (r"\bI\s+have\s+been\s+PWNED\b", "instruction_override"),
    (r"\bPWNED\b", "instruction_override"),
]

# Compile all patterns
COMPILED_PI_PATTERNS = [(re.compile(p, re.IGNORECASE), cat) for p, cat in PI_PATTERNS]


def load_curated(path: Path) -> list[dict]:
    """Load the curated YAML dataset."""
    print(f"Loading {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    print(f"  Loaded {len(data)} samples.")
    return data


def basic_stats(samples: list[dict]) -> None:
    """Print basic distribution statistics."""
    print("\n" + "=" * 70)
    print("BASIC STATISTICS")
    print("=" * 70)

    total = len(samples)
    print(f"\nTotal samples: {total}")

    # Label distribution
    labels = Counter(s.get("label", "MISSING") for s in samples)
    print(f"\nLabel distribution:")
    for label, count in labels.most_common():
        print(f"  {label}: {count} ({100*count/total:.1f}%)")

    # Language distribution
    langs = Counter(s.get("language", "MISSING") for s in samples)
    print(f"\nLanguage distribution:")
    for lang, count in langs.most_common():
        print(f"  {lang}: {count} ({100*count/total:.1f}%)")

    # Reason distribution
    reasons = Counter(s.get("reason", "MISSING") for s in samples)
    print(f"\nReason distribution:")
    for reason, count in reasons.most_common():
        print(f"  {reason}: {count} ({100*count/total:.1f}%)")

    # Source distribution (top 20)
    sources = Counter(s.get("source", "MISSING") for s in samples)
    print(f"\nSource distribution (top 20 of {len(sources)}):")
    for source, count in sources.most_common(20):
        print(f"  {source}: {count} ({100*count/total:.1f}%)")

    # Format bin
    fmt_bins = Counter(s.get("format_bin", "MISSING") for s in samples)
    print(f"\nFormat bin distribution:")
    for fmt, count in fmt_bins.most_common():
        print(f"  {fmt}: {count} ({100*count/total:.1f}%)")

    # Length bin
    len_bins = Counter(s.get("length_bin", "MISSING") for s in samples)
    print(f"\nLength bin distribution:")
    for lb, count in len_bins.most_common():
        print(f"  {lb}: {count} ({100*count/total:.1f}%)")

    # Cross-tab: label x language
    print(f"\nLabel x Language cross-tab:")
    label_lang = defaultdict(Counter)
    for s in samples:
        label_lang[s.get("label", "MISSING")][s.get("language", "MISSING")] += 1
    for label in sorted(label_lang):
        parts = ", ".join(f"{lang}={c}" for lang, c in label_lang[label].most_common())
        print(f"  {label}: {parts}")


def truncate(text: str, max_len: int = 200) -> str:
    """Truncate text for display."""
    text = text.replace("\n", "\\n")
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def show_samples(title: str, items: list[tuple[int, dict]], max_show: int = 5) -> None:
    """Display sample items with metadata."""
    print(f"\n  --- {title} ({len(items)} total, showing {min(max_show, len(items))}) ---")
    for idx, s in items[:max_show]:
        content = truncate(s.get("content", ""))
        label = s.get("label", "?")
        lang = s.get("language", "?")
        reason = s.get("reason", "?")
        source = s.get("source", "?")
        print(f"  [{idx}] label={label} lang={lang} reason={reason} source={source}")
        print(f"       {content}")


def check_short_samples(samples: list[dict]) -> list[tuple[int, dict]]:
    """Find samples with content < 10 characters."""
    results = []
    for i, s in enumerate(samples):
        content = s.get("content", "")
        if len(content.strip()) < 10:
            results.append((i, s))
    return results


def check_long_samples(samples: list[dict]) -> list[tuple[int, dict]]:
    """Find samples with content > 10000 characters."""
    results = []
    for i, s in enumerate(samples):
        content = s.get("content", "")
        if len(content) > 10000:
            results.append((i, s))
    return results


def check_exact_duplicates(samples: list[dict]) -> dict[str, list[int]]:
    """Find exact duplicate content strings."""
    content_to_indices = defaultdict(list)
    for i, s in enumerate(samples):
        content = s.get("content", "")
        content_to_indices[content].append(i)
    return {c: idxs for c, idxs in content_to_indices.items() if len(idxs) > 1}


def check_near_duplicates(samples: list[dict]) -> dict[str, list[int]]:
    """Find near-duplicates by normalizing whitespace and lowercasing."""
    norm_to_indices = defaultdict(list)
    for i, s in enumerate(samples):
        content = s.get("content", "")
        normalized = re.sub(r"\s+", " ", content.strip().lower())
        norm_to_indices[normalized].append(i)
    return {c: idxs for c, idxs in norm_to_indices.items() if len(idxs) > 1}


def check_benign_with_pi_patterns(samples: list[dict]) -> list[tuple[int, dict, str, str]]:
    """Find benign samples that match PI patterns."""
    results = []
    for i, s in enumerate(samples):
        if s.get("label") != "benign":
            continue
        content = s.get("content", "")
        for pattern, category in COMPILED_PI_PATTERNS:
            m = pattern.search(content)
            if m:
                results.append((i, s, m.group(), category))
                break  # one match is enough
    return results


def check_malicious_no_pi_signal(samples: list[dict]) -> list[tuple[int, dict]]:
    """Find malicious samples that have no PI signal at all.

    Only flag samples that:
      - Are short (< 100 chars)
      - Match zero PI patterns
      - Don't contain any suspicious keywords
    """
    # Broader keywords that don't warrant strong PI classification alone
    # but indicate the sample isn't just normal text
    WEAK_SIGNALS = [
        r"hack", r"pwn", r"inject", r"bypass", r"jailbreak", r"exploit",
        r"payload", r"malicious", r"attack", r"backdoor", r"trojan",
        r"prompt", r"instruction", r"override", r"ignore", r"disregard",
        r"pretend", r"roleplay", r"role.play", r"sudo", r"admin",
        r"root\b", r"password", r"secret", r"token", r"api.key",
        r"\\n", r"\[system\]", r"<<", r">>",
        r"PWNED", r"DAN\b", r"ChatGPT", r"OpenAI",
        r"previous\s+instructions", r"new\s+role", r"act\s+as",
        r"system\s*prompt", r"hidden", r"confidential",
        r"repeat\s+after", r"say\s+exactly", r"respond\s+with",
        r"translate.*into", r"output\s+only",
        r"forget\s+everything", r"reset\s+your",
        # HackAPrompt competition signals
        r"I have been", r"PWNED", r"say\s*\"",
    ]
    compiled_weak = [re.compile(p, re.IGNORECASE) for p in WEAK_SIGNALS]

    results = []
    for i, s in enumerate(samples):
        if s.get("label") != "malicious":
            continue
        content = s.get("content", "")
        # Only flag short samples -- longer ones often have subtle injection
        if len(content) > 150:
            continue
        # Check strong PI patterns
        has_strong = any(p.search(content) for p, _ in COMPILED_PI_PATTERNS)
        if has_strong:
            continue
        # Check weak signals
        has_weak = any(p.search(content) for p in compiled_weak)
        if has_weak:
            continue
        results.append((i, s))
    return results


def check_background_with_pi(samples: list[dict]) -> list[tuple[int, dict, str]]:
    """Find samples with reason=background that match PI patterns."""
    results = []
    for i, s in enumerate(samples):
        if s.get("reason") != "background":
            continue
        content = s.get("content", "")
        for pattern, category in COMPILED_PI_PATTERNS:
            m = pattern.search(content)
            if m:
                results.append((i, s, m.group()))
                break
    return results


def check_reason_content_mismatch(samples: list[dict]) -> list[tuple[int, dict, str]]:
    """Check if the attack reason matches the content.

    Look for clear mismatches:
      - exfiltration label but no URL/send/transmit/leak patterns
      - meta_probe label but no prompt extraction patterns
      - obfuscation label but no obfuscation patterns
    """
    REASON_PATTERNS = {
        "exfiltration": [
            re.compile(r"(send|transmit|post|exfiltrate|leak|forward|email|upload)", re.IGNORECASE),
            re.compile(r"https?://", re.IGNORECASE),
            re.compile(r"(webhook|api|endpoint|url|server|domain)", re.IGNORECASE),
            re.compile(r"!\[.*\]\(", re.IGNORECASE),
            re.compile(r"(fetch|request|curl|wget)", re.IGNORECASE),
            re.compile(r"(markdown|image|link)\s*(injection|exfil)", re.IGNORECASE),
        ],
        "meta_probe": [
            re.compile(r"(system\s*prompt|initial\s*instruction|hidden\s*(instruction|prompt|message))", re.IGNORECASE),
            re.compile(r"(show|reveal|print|display|output|repeat|echo|tell|give)\s+(me\s+)?(your|the)", re.IGNORECASE),
            re.compile(r"what\s+(is|are)\s+your", re.IGNORECASE),
            re.compile(r"(prompt|instruction|rule|guideline|constraint|directive)\s*(extraction|leak|reveal)", re.IGNORECASE),
            re.compile(r"<<\s*SYS\s*>>", re.IGNORECASE),
        ],
        "obfuscation": [
            re.compile(r"base64", re.IGNORECASE),
            re.compile(r"\\x[0-9a-fA-F]{2}", re.IGNORECASE),
            re.compile(r"&#x?[0-9a-fA-F]+;", re.IGNORECASE),
            re.compile(r"[A-Z]\s[A-Z]\s[A-Z]\s[A-Z]", re.IGNORECASE),  # spaced letters
            re.compile(r"[^\x00-\x7F]{5,}", re.IGNORECASE),  # heavy non-ASCII (unicode tricks)
            re.compile(r"(rot13|caesar|cipher|encod|decode|obfuscat)", re.IGNORECASE),
            re.compile(r"(l33t|leet|1337)", re.IGNORECASE),
            re.compile(r"(zero.width|unicode|homoglyph|invisible)", re.IGNORECASE),
        ],
    }

    results = []
    for i, s in enumerate(samples):
        if s.get("label") != "malicious":
            continue
        reason = s.get("reason", "")
        if reason not in REASON_PATTERNS:
            continue
        content = s.get("content", "")
        patterns = REASON_PATTERNS[reason]
        if not any(p.search(content) for p in patterns):
            results.append((i, s, reason))
    return results


def check_label_conflicts(duplicates: dict[str, list[int]], samples: list[dict]) -> list[tuple[str, list[int], list[str]]]:
    """Find duplicates that have conflicting labels."""
    conflicts = []
    for content, indices in duplicates.items():
        labels = set(samples[i].get("label") for i in indices)
        if len(labels) > 1:
            label_list = [samples[i].get("label") for i in indices]
            conflicts.append((content, indices, label_list))
    return conflicts


def check_language_mismatch(samples: list[dict]) -> list[tuple[int, dict]]:
    """Find samples where the declared language doesn't match the content.

    Simple heuristic: if declared EN but majority chars are non-Latin.
    """
    results = []
    for i, s in enumerate(samples):
        lang = s.get("language", "")
        content = s.get("content", "")
        if not content:
            continue

        if lang == "EN":
            # Check if content is mostly non-Latin
            latin_chars = sum(1 for c in content if c.isascii() and c.isalpha())
            non_latin_chars = sum(1 for c in content if not c.isascii() and c.isalpha())
            total_alpha = latin_chars + non_latin_chars
            if total_alpha > 10 and non_latin_chars / total_alpha > 0.5:
                results.append((i, s))
    return results


def main():
    if not CURATED_PATH.exists():
        print(f"ERROR: {CURATED_PATH} not found")
        sys.exit(1)

    samples = load_curated(CURATED_PATH)

    # 1. Basic stats
    basic_stats(samples)

    # Track issue counts for summary
    issue_counts = {}

    # 2a. Very short samples
    print("\n" + "=" * 70)
    print("CHECK: Very short samples (< 10 chars)")
    print("=" * 70)
    short = check_short_samples(samples)
    issue_counts["Very short (< 10 chars)"] = len(short)
    if short:
        show_samples("Short samples", short, max_show=10)
    else:
        print("  None found.")

    # 2b. Very long samples
    print("\n" + "=" * 70)
    print("CHECK: Very long samples (> 10000 chars)")
    print("=" * 70)
    long_samples = check_long_samples(samples)
    issue_counts["Very long (> 10000 chars)"] = len(long_samples)
    if long_samples:
        # Show length info instead of content
        print(f"\n  --- Very long samples ({len(long_samples)} total, showing up to 5) ---")
        for idx, s in long_samples[:5]:
            clen = len(s.get("content", ""))
            label = s.get("label", "?")
            lang = s.get("language", "?")
            reason = s.get("reason", "?")
            source = s.get("source", "?")
            print(f"  [{idx}] label={label} lang={lang} reason={reason} source={source} len={clen}")
            print(f"       {truncate(s.get('content', ''), 150)}")
    else:
        print("  None found.")

    # 2c. Exact duplicates
    print("\n" + "=" * 70)
    print("CHECK: Exact duplicates")
    print("=" * 70)
    exact_dupes = check_exact_duplicates(samples)
    total_extra_exact = sum(len(idxs) - 1 for idxs in exact_dupes.values())
    issue_counts["Exact duplicate entries (extra copies)"] = total_extra_exact
    issue_counts["Exact duplicate unique contents"] = len(exact_dupes)
    print(f"  {len(exact_dupes)} unique contents have duplicates ({total_extra_exact} extra copies)")
    if exact_dupes:
        print("\n  --- Examples ---")
        for content, indices in list(exact_dupes.items())[:5]:
            s0 = samples[indices[0]]
            labels = [samples[i].get("label") for i in indices]
            reasons = [samples[i].get("reason") for i in indices]
            sources = [samples[i].get("source") for i in indices]
            print(f"  Indices: {indices}, labels={labels}, reasons={reasons}")
            print(f"  sources={sources}")
            print(f"       {truncate(content)}")

    # Label conflicts in duplicates
    label_conflicts = check_label_conflicts(exact_dupes, samples)
    issue_counts["Duplicates with conflicting labels"] = len(label_conflicts)
    if label_conflicts:
        print(f"\n  --- Duplicates with CONFLICTING LABELS ({len(label_conflicts)}) ---")
        for content, indices, labels in label_conflicts[:5]:
            sources = [samples[i].get("source") for i in indices]
            print(f"  Indices: {indices}, labels={labels}, sources={sources}")
            print(f"       {truncate(content)}")

    # 2c'. Near-duplicates (only those that aren't exact dupes)
    print("\n" + "=" * 70)
    print("CHECK: Near-duplicates (normalized whitespace + lowercased)")
    print("=" * 70)
    near_dupes = check_near_duplicates(samples)
    # Filter out exact dupes to find only near-dupes
    exact_dupe_content_set = set(exact_dupes.keys())
    near_only = {k: v for k, v in near_dupes.items()
                 if not all(samples[i].get("content", "") in exact_dupe_content_set for i in v)}
    total_extra_near = sum(len(idxs) - 1 for idxs in near_only.values())
    issue_counts["Near-duplicate groups (beyond exact)"] = len(near_only)
    issue_counts["Near-duplicate extra copies"] = total_extra_near
    print(f"  {len(near_dupes)} total near-dupe groups")
    print(f"  {len(near_only)} groups that are not exact-dupes ({total_extra_near} extra copies)")
    if near_only:
        print("\n  --- Near-duplicate examples ---")
        for norm_content, indices in list(near_only.items())[:5]:
            labels = [samples[i].get("label") for i in indices]
            contents = [truncate(samples[i].get("content", ""), 100) for i in indices]
            print(f"  Indices: {indices}, labels={labels}")
            for ci, c in zip(indices, contents):
                print(f"    [{ci}]: {c}")

    # 2d. Mislabeled benign (benign with PI patterns)
    print("\n" + "=" * 70)
    print("CHECK: Benign samples containing prompt injection patterns")
    print("=" * 70)
    benign_pi = check_benign_with_pi_patterns(samples)
    issue_counts["Benign with PI patterns"] = len(benign_pi)
    if benign_pi:
        # Group by matched category
        by_cat = defaultdict(list)
        for i, s, match, cat in benign_pi:
            by_cat[cat].append((i, s, match))
        for cat, items in sorted(by_cat.items(), key=lambda x: -len(x[1])):
            print(f"\n  Category: {cat} ({len(items)} matches)")
            for idx, s, match in items[:3]:
                content = truncate(s.get("content", ""))
                reason = s.get("reason", "?")
                source = s.get("source", "?")
                lang = s.get("language", "?")
                print(f"  [{idx}] reason={reason} source={source} lang={lang} matched='{match}'")
                print(f"       {content}")
    else:
        print("  None found.")

    # 2d'. Malicious with no PI signal
    print("\n" + "=" * 70)
    print("CHECK: Malicious samples with no PI signal (short, < 150 chars)")
    print("=" * 70)
    mal_no_pi = check_malicious_no_pi_signal(samples)
    issue_counts["Malicious with no PI signal (short)"] = len(mal_no_pi)
    if mal_no_pi:
        show_samples("Malicious, no PI signal detected", mal_no_pi, max_show=10)
    else:
        print("  None found.")

    # 2e. Background samples with PI patterns
    print("\n" + "=" * 70)
    print("CHECK: Background (benign) samples with PI patterns")
    print("=" * 70)
    bg_pi = check_background_with_pi(samples)
    issue_counts["Background with PI patterns"] = len(bg_pi)
    if bg_pi:
        for idx, s, match in bg_pi[:5]:
            content = truncate(s.get("content", ""))
            source = s.get("source", "?")
            lang = s.get("language", "?")
            label = s.get("label", "?")
            print(f"  [{idx}] label={label} source={source} lang={lang} matched='{match}'")
            print(f"       {content}")
    else:
        print("  None found.")

    # 2f. Reason-content mismatches
    print("\n" + "=" * 70)
    print("CHECK: Attack reason does not match content (exfiltration/meta_probe/obfuscation)")
    print("=" * 70)
    reason_mismatch = check_reason_content_mismatch(samples)
    issue_counts["Reason-content mismatch"] = len(reason_mismatch)
    if reason_mismatch:
        by_reason = defaultdict(list)
        for i, s, reason in reason_mismatch:
            by_reason[reason].append((i, s))
        for reason, items in sorted(by_reason.items(), key=lambda x: -len(x[1])):
            print(f"\n  Reason: {reason} ({len(items)} mismatches)")
            for idx, s in items[:5]:
                content = truncate(s.get("content", ""))
                source = s.get("source", "?")
                lang = s.get("language", "?")
                print(f"  [{idx}] source={source} lang={lang}")
                print(f"       {content}")
    else:
        print("  None found.")

    # Language mismatch
    print("\n" + "=" * 70)
    print("CHECK: Language mismatch (declared EN, content is non-Latin)")
    print("=" * 70)
    lang_mismatch = check_language_mismatch(samples)
    issue_counts["Language mismatch (EN but non-Latin)"] = len(lang_mismatch)
    if lang_mismatch:
        show_samples("Language mismatch", lang_mismatch, max_show=5)
    else:
        print("  None found.")

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("ISSUE SUMMARY")
    print("=" * 70)
    total_issues = 0
    for issue, count in issue_counts.items():
        flag = " ***" if count > 0 else ""
        print(f"  {issue}: {count}{flag}")
        if "conflict" not in issue.lower() and "unique" not in issue.lower() and "group" not in issue.lower():
            total_issues += count

    print(f"\n  Total flagged items: {total_issues}")
    print(f"  Dataset size: {len(samples)}")
    print(f"  Issue rate: {100*total_issues/len(samples):.2f}%")


if __name__ == "__main__":
    main()
