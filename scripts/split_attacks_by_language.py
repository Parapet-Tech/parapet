"""Split attacks_121624.yaml into per-language files using script-ratio detection.

Output:
  schema/eval/training/multilingual/attacks_121624_en.yaml  (EN-primary)
  schema/eval/training/multilingual/attacks_121624_zh.yaml  (ZH-primary, from merged corpus)
  schema/eval/training/multilingual/attacks_121624_ar.yaml  (AR-primary, from merged corpus)
  schema/eval/training/multilingual/attacks_121624_ru.yaml  (RU-primary, from merged corpus)

These are the non-EN samples ALREADY in the merged corpus.
Combine with zh/ar/ru_attacks.yaml (novel extractions) for full coverage.

Usage: python scripts/split_attacks_by_language.py
"""
import sys
import yaml
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

CJK_RANGES = [
    (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF),
    (0xF900, 0xFAFF), (0x3000, 0x303F), (0x3040, 0x309F), (0x30A0, 0x30FF),
]
ARABIC_RANGES = [
    (0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF), (0xFE70, 0xFEFF),
]
CYRILLIC_RANGES = [
    (0x0400, 0x04FF), (0x0500, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F),
]

# Known non-PI source prefixes to exclude from non-EN attack files
NON_PI_PREFIXES = [
    'arabic-hallucination',  # content safety, not PI
    'multijail',             # content safety, not PI
]


def _in_ranges(cp, ranges):
    for lo, hi in ranges:
        if lo <= cp <= hi:
            return True
    return False


def classify_language(text):
    """Return 'zh', 'ar', 'ru', or 'en' based on script ratios."""
    if not text or not isinstance(text, str):
        return 'en'
    total = cjk = arabic = cyrillic = 0
    for ch in text:
        cp = ord(ch)
        if ch.isspace() or ch in '.,;:!?()[]{}"\'-_/\\@#$%^&*+=<>~`|0123456789':
            continue
        total += 1
        if _in_ranges(cp, CJK_RANGES):
            cjk += 1
        elif _in_ranges(cp, ARABIC_RANGES):
            arabic += 1
        elif _in_ranges(cp, CYRILLIC_RANGES):
            cyrillic += 1
    if total == 0:
        return 'en'
    cjk_r = cjk / total
    ar_r = arabic / total
    cy_r = cyrillic / total
    # Primary language if >10% script ratio
    scores = [('zh', cjk_r), ('ar', ar_r), ('ru', cy_r)]
    scores.sort(key=lambda x: x[1], reverse=True)
    if scores[0][1] >= 0.10:
        return scores[0][0]
    return 'en'


def is_non_pi_source(item):
    """Check if sample comes from a known non-PI source."""
    item_id = item.get('id', '').lower()
    desc = item.get('description', '').lower()
    for prefix in NON_PI_PREFIXES:
        if prefix in item_id or prefix in desc:
            return True
    return False


def main():
    src = Path('schema/eval/malicious/attacks_121624.yaml')
    out_dir = Path('schema/eval/training/multilingual')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {src}...")
    try:
        loader = yaml.CSafeLoader
    except AttributeError:
        loader = yaml.SafeLoader
    with open(src, 'r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=loader)
    print(f"Loaded {len(data)} entries")

    by_lang = {'en': [], 'zh': [], 'ar': [], 'ru': []}
    non_pi_excluded = Counter()

    for item in data:
        content = item.get('content', '')
        lang = classify_language(content)

        # For non-EN, exclude known non-PI sources
        if lang != 'en' and is_non_pi_source(item):
            non_pi_excluded[lang] += 1
            continue

        by_lang[lang].append(item)

    print(f"\nLanguage distribution:")
    for lang in ['en', 'zh', 'ar', 'ru']:
        print(f"  {lang.upper()}: {len(by_lang[lang])}")
    print(f"\nNon-PI excluded: {dict(non_pi_excluded)}")

    # Write per-language files (skip EN — it's too large and not needed as a separate file)
    for lang in ['zh', 'ar', 'ru']:
        items = by_lang[lang]
        if not items:
            continue
        out_path = out_dir / f"attacks_121624_{lang}.yaml"
        with open(out_path, 'w', encoding='utf-8') as f:
            yaml.dump(items, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"  Wrote {len(items)} {lang.upper()} entries to {out_path}")


if __name__ == '__main__':
    main()
