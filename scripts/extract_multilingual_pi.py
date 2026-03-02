"""Extract multilingual PI samples from TheWall datasets into v2-ready YAML slices.

Sources:
  - hackaprompt-dataset (ZH, AR, RU)
  - wambosec/prompt-injections (ZH, AR, RU)
  - nueralchemy/Prompt-injection-dataset (ZH, AR, RU)

Output:
  parapet/schema/eval/training/multilingual/
    zh_attacks.yaml
    ar_attacks.yaml
    ru_attacks.yaml

Each output file has format matching attacks51042.yaml:
  - content: <text>
    id: <source>-<lang>-<index>
    label: attack
    layer: l1

Usage: python scripts/extract_multilingual_pi.py
  (run from parapet/ directory)
"""
import sys
import os
import hashlib
import yaml
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')

THEWALL = Path(__file__).resolve().parent.parent.parent / "TheWall"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "schema" / "eval" / "training" / "multilingual"

# Unicode script detection
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


def _in_ranges(cp, ranges):
    for lo, hi in ranges:
        if lo <= cp <= hi:
            return True
    return False


def script_ratios(text):
    """Return (cjk_ratio, arabic_ratio, cyrillic_ratio) for text."""
    if not text or not isinstance(text, str):
        return 0, 0, 0
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
        return 0, 0, 0
    return cjk / total, arabic / total, cyrillic / total


def text_hash(text):
    return hashlib.sha256(text.strip().encode('utf-8')).hexdigest()[:16]


def classify_lang(cjk_r, ar_r, cy_r, threshold=0.10):
    """Return primary non-EN language if above threshold, or None."""
    # Priority: whichever script has highest ratio
    scores = [('zh', cjk_r), ('ar', ar_r), ('ru', cy_r)]
    scores.sort(key=lambda x: x[1], reverse=True)
    if scores[0][1] >= threshold:
        return scores[0][0]
    return None


def load_existing_hashes():
    """Load hashes from attacks_121624.yaml for dedup."""
    path = Path(__file__).resolve().parent.parent / "schema" / "eval" / "malicious" / "attacks_121624.yaml"
    if not path.exists():
        print(f"  WARN: {path} not found, skipping dedup")
        return set()
    print(f"  Loading {path.name} for dedup...")
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    hashes = set()
    for item in data:
        c = item.get('content', '')
        if c and isinstance(c, str):
            hashes.add(text_hash(c))
    print(f"  Loaded {len(hashes)} existing hashes")
    return hashes


def extract_hackaprompt(existing_hashes):
    """Extract non-EN PI from hackaprompt parquet."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("  SKIP: pyarrow not installed")
        return {}

    path = THEWALL / "hackaprompt-dataset" / "hackaprompt.parquet"
    if not path.exists():
        print(f"  NOT FOUND: {path}")
        return {}

    table = pq.read_table(path)
    texts = table.column('user_input').to_pylist()
    levels = table.column('level').to_pylist() if 'level' in table.column_names else [None] * len(texts)
    correct = table.column('correct').to_pylist() if 'correct' in table.column_names else [None] * len(texts)

    by_lang = defaultdict(dict)  # lang -> {hash: sample}
    for i, text in enumerate(texts):
        if not text or not isinstance(text, str):
            continue
        if len(text.strip()) < 10:
            continue

        cjk_r, ar_r, cy_r = script_ratios(text)
        lang = classify_lang(cjk_r, ar_r, cy_r)
        if not lang:
            continue

        h = text_hash(text)
        if h in existing_hashes:
            continue

        if h not in by_lang[lang]:
            by_lang[lang][h] = {
                'content': text.strip(),
                'id': f'hackaprompt-{lang}-{i}',
                'label': 'attack',
                'layer': 'l1',
                'level': levels[i],
                'correct': correct[i],
            }

    return by_lang


def extract_parquet_dataset(name, path, text_col, label_col=None, attack_values=None, existing_hashes=set()):
    """Extract non-EN PI from a parquet dataset."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {}

    by_lang = defaultdict(dict)
    for parquet_file in sorted(Path(path).glob("*.parquet")):
        table = pq.read_table(parquet_file)
        if text_col not in table.column_names:
            continue

        texts = table.column(text_col).to_pylist()
        labels = None
        if label_col and label_col in table.column_names:
            labels = table.column(label_col).to_pylist()

        for i, text in enumerate(texts):
            if not text or not isinstance(text, str):
                continue
            if len(text.strip()) < 10:
                continue
            if labels is not None and attack_values:
                if str(labels[i]) not in attack_values:
                    continue

            cjk_r, ar_r, cy_r = script_ratios(text)
            lang = classify_lang(cjk_r, ar_r, cy_r)
            if not lang:
                continue

            h = text_hash(text)
            if h in existing_hashes:
                continue

            if h not in by_lang[lang]:
                by_lang[lang][h] = {
                    'content': text.strip(),
                    'id': f'{name}-{lang}-{i}',
                    'label': 'attack',
                    'layer': 'l1',
                }

    return by_lang


def merge_results(*dicts):
    """Merge multiple by_lang dicts, deduplicating by hash."""
    merged = defaultdict(dict)
    for d in dicts:
        for lang, samples in d.items():
            for h, sample in samples.items():
                if h not in merged[lang]:
                    merged[lang][h] = sample
    return merged


def write_yaml(lang, samples, output_dir):
    """Write samples to YAML file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{lang}_attacks.yaml"

    # Convert to list, sorted by id for stability
    items = sorted(samples.values(), key=lambda x: x['id'])

    # Strip extra metadata (level, correct) — keep only v2 fields
    clean = []
    for item in items:
        clean.append({
            'content': item['content'],
            'id': item['id'],
            'label': item['label'],
            'layer': item['layer'],
        })

    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(clean, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    return path, len(clean)


def main():
    print("=" * 70)
    print("Multilingual PI Extractor")
    print("=" * 70)

    # Load existing hashes for dedup
    existing_hashes = load_existing_hashes()

    # Track all hashes across sources to avoid cross-source duplication
    all_seen = set(existing_hashes)

    # 1. HackAPrompt (richest source)
    print("\n[1] Extracting from hackaprompt-dataset...")
    hacka = extract_hackaprompt(all_seen)
    for lang in hacka:
        print(f"  {lang.upper()}: {len(hacka[lang])} unique novel samples")
        all_seen.update(hacka[lang].keys())

    # 2. wambosec
    print("\n[2] Extracting from wambosec/prompt-injections...")
    wambo = extract_parquet_dataset(
        'wambosec',
        THEWALL / "wambosec" / "prompt-injections" / "data",
        'prompt', label_col='label', attack_values={'1'},
        existing_hashes=all_seen,
    )
    for lang in wambo:
        print(f"  {lang.upper()}: {len(wambo[lang])} unique novel samples")
        all_seen.update(wambo[lang].keys())

    # 3. nueralchemy
    print("\n[3] Extracting from nueralchemy/Prompt-injection-dataset...")
    neur = extract_parquet_dataset(
        'nueralchemy',
        THEWALL / "nueralchemy" / "Prompt-injection-dataset" / "data",
        'prompt', label_col='label', attack_values={'1'},
        existing_hashes=all_seen,
    )
    for lang in neur:
        print(f"  {lang.upper()}: {len(neur[lang])} unique novel samples")
        all_seen.update(neur[lang].keys())

    # 4. imoxto (prompt_injection_cleaned_dataset-v2) — likely heavy overlap with hackaprompt
    print("\n[4] Extracting from prompt_injection_cleaned_dataset-v2...")
    imoxto = extract_parquet_dataset(
        'imoxto-v2',
        THEWALL / "prompt_injection_cleaned_dataset-v2" / "data",
        'text', label_col='label', attack_values={'1'},
        existing_hashes=all_seen,
    )
    for lang in imoxto:
        print(f"  {lang.upper()}: {len(imoxto[lang])} unique novel samples")
        all_seen.update(imoxto[lang].keys())

    # Merge all
    print("\n" + "-" * 70)
    print("Merging and writing output...")
    merged = merge_results(hacka, wambo, neur, imoxto)

    for lang in ['zh', 'ar', 'ru']:
        samples = merged.get(lang, {})
        if not samples:
            print(f"  {lang.upper()}: no samples to write")
            continue
        path, count = write_yaml(lang, samples, OUTPUT_DIR)
        print(f"  {lang.upper()}: wrote {count} samples to {path}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    for lang in ['zh', 'ar', 'ru']:
        count = len(merged.get(lang, {}))
        print(f"  {lang.upper()}: {count} unique novel PI samples extracted")
    print("=" * 70)


if __name__ == '__main__':
    main()
