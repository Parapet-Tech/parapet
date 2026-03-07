"""Scan TheWall datasets for hidden multilingual prompt injection content.

Uses Unicode script ratios to detect ZH (CJK), AR (Arabic), RU (Cyrillic)
content that was ingested without language tagging.

Usage: python scripts/scan_multilingual_pi.py
  (run from parapet/ directory)
"""
import sys
import os
import json
import csv
import re
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')

THEWALL = Path(__file__).resolve().parent.parent.parent / "TheWall"

# Unicode script detection ranges
CJK_RANGES = [
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Ext A
    (0x20000, 0x2A6DF),  # CJK Ext B
    (0x2A700, 0x2B73F),  # CJK Ext C
    (0x2B740, 0x2B81F),  # CJK Ext D
    (0xF900, 0xFAFF),    # CJK Compat
    (0x3000, 0x303F),    # CJK Symbols
    (0x3040, 0x309F),    # Hiragana
    (0x30A0, 0x30FF),    # Katakana
]

ARABIC_RANGES = [
    (0x0600, 0x06FF),    # Arabic
    (0x0750, 0x077F),    # Arabic Supplement
    (0x08A0, 0x08FF),    # Arabic Extended-A
    (0xFB50, 0xFDFF),    # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),    # Arabic Presentation Forms-B
]

CYRILLIC_RANGES = [
    (0x0400, 0x04FF),    # Cyrillic
    (0x0500, 0x052F),    # Cyrillic Supplement
    (0x2DE0, 0x2DFF),    # Cyrillic Extended-A
    (0xA640, 0xA69F),    # Cyrillic Extended-B
]


def in_ranges(cp, ranges):
    for lo, hi in ranges:
        if lo <= cp <= hi:
            return True
    return False


def detect_scripts(text):
    """Return dict of script ratios for a text string."""
    if not text or not isinstance(text, str):
        return {}
    # Only count actual script characters (letters), skip ASCII punctuation/digits
    total = 0
    cjk = 0
    arabic = 0
    cyrillic = 0
    latin = 0
    for ch in text:
        cp = ord(ch)
        if ch.isspace() or ch in '.,;:!?()[]{}"\'-_/\\@#$%^&*+=<>~`|0123456789':
            continue
        total += 1
        if in_ranges(cp, CJK_RANGES):
            cjk += 1
        elif in_ranges(cp, ARABIC_RANGES):
            arabic += 1
        elif in_ranges(cp, CYRILLIC_RANGES):
            cyrillic += 1
        elif 0x0041 <= cp <= 0x007A or 0x00C0 <= cp <= 0x024F:
            latin += 1
    if total == 0:
        return {}
    return {
        'total_chars': total,
        'cjk': cjk / total,
        'arabic': arabic / total,
        'cyrillic': cyrillic / total,
        'latin': latin / total,
    }


def classify_language(scripts, threshold=0.10):
    """Classify primary non-EN language if above threshold."""
    langs = []
    if scripts.get('cjk', 0) >= threshold:
        langs.append(('zh', scripts['cjk']))
    if scripts.get('arabic', 0) >= threshold:
        langs.append(('ar', scripts['arabic']))
    if scripts.get('cyrillic', 0) >= threshold:
        langs.append(('ru', scripts['cyrillic']))
    return langs


def scan_parquet(path, text_col, label_col=None, attack_values=None):
    """Scan a parquet file for non-EN content."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print(f"  SKIP (pyarrow not installed): {path}")
        return []

    table = pq.read_table(path)
    df_cols = table.column_names
    if text_col not in df_cols:
        print(f"  WARN: column '{text_col}' not in {df_cols}")
        return []

    texts = table.column(text_col).to_pylist()
    labels = None
    if label_col and label_col in df_cols:
        labels = table.column(label_col).to_pylist()

    results = []
    for i, text in enumerate(texts):
        if not text or not isinstance(text, str):
            continue
        # If filtering by label (attacks only)
        if labels is not None and attack_values is not None:
            lbl = labels[i]
            if str(lbl) not in attack_values:
                continue

        scripts = detect_scripts(text)
        if not scripts:
            continue
        langs = classify_language(scripts)
        if langs:
            results.append({
                'index': i,
                'text': text[:200],
                'langs': langs,
                'scripts': scripts,
            })
    return results


def scan_csv(path, text_col, label_col=None, attack_values=None, delimiter=','):
    """Scan a CSV/TSV file for non-EN content."""
    results = []
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for i, row in enumerate(reader):
                text = row.get(text_col, '')
                if not text:
                    continue
                if label_col and attack_values:
                    lbl = row.get(label_col, '')
                    if str(lbl) not in attack_values:
                        continue
                scripts = detect_scripts(text)
                if not scripts:
                    continue
                langs = classify_language(scripts)
                if langs:
                    results.append({
                        'index': i,
                        'text': text[:200],
                        'langs': langs,
                        'scripts': scripts,
                    })
    except Exception as e:
        print(f"  ERROR reading {path}: {e}")
    return results


def scan_json(path, text_extractor):
    """Scan a JSON file for non-EN content using a custom text extractor."""
    results = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = list(data.values()) if not isinstance(list(data.values())[0], str) else [data]
        else:
            return results

        for i, item in enumerate(items):
            text = text_extractor(item)
            if not text:
                continue
            scripts = detect_scripts(text)
            if not scripts:
                continue
            langs = classify_language(scripts)
            if langs:
                results.append({
                    'index': i,
                    'text': text[:200],
                    'langs': langs,
                    'scripts': scripts,
                })
    except Exception as e:
        print(f"  ERROR reading {path}: {e}")
    return results


def summarize_results(name, results):
    """Print summary of detected multilingual content."""
    if not results:
        print(f"  {name}: 0 non-EN samples found")
        return

    by_lang = defaultdict(list)
    for r in results:
        for lang, ratio in r['langs']:
            by_lang[lang].append((r, ratio))

    print(f"  {name}: {len(results)} non-EN samples found")
    for lang in ['zh', 'ar', 'ru']:
        items = by_lang.get(lang, [])
        if not items:
            continue
        primary = [r for r, ratio in items if ratio >= 0.30]
        mixed = [r for r, ratio in items if 0.10 <= ratio < 0.30]
        print(f"    {lang.upper()}: {len(primary)} primary (>=30%), {len(mixed)} mixed (10-30%)")
        # Show a few examples
        for r, ratio in items[:3]:
            preview = r['text'][:100].replace('\n', ' ')
            print(f"      [{ratio:.0%}] {preview}...")


def main():
    print("=" * 70)
    print("TheWall Multilingual PI Scanner")
    print("=" * 70)

    # 1. HackAPrompt (100K parquet)
    print("\n[1] hackaprompt-dataset (100K)")
    path = THEWALL / "hackaprompt-dataset" / "hackaprompt.parquet"
    if path.exists():
        results = scan_parquet(path, 'user_input')
        summarize_results("hackaprompt", results)
    else:
        print(f"  NOT FOUND: {path}")

    # 2. prompt_injection_cleaned_dataset-v2 (2 shards)
    print("\n[2] prompt_injection_cleaned_dataset-v2")
    all_results = []
    for shard in sorted((THEWALL / "prompt_injection_cleaned_dataset-v2" / "data").glob("*.parquet")):
        results = scan_parquet(shard, 'text', label_col='label', attack_values={'1'})
        all_results.extend(results)
    summarize_results("imoxto-v2 (attacks only)", all_results)

    # 3. SPML_Chatbot_Prompt_Injection (86K CSV)
    print("\n[3] SPML_Chatbot_Prompt_Injection (86K)")
    path = THEWALL / "SPML_Chatbot_Prompt_Injection" / "spml_prompt_injection.csv"
    if path.exists():
        results = scan_csv(path, 'User Prompt', label_col='Prompt injection', attack_values={'1'})
        summarize_results("SPML", results)
    else:
        print(f"  NOT FOUND: {path}")

    # 4. jailbreak_llms CSVs
    print("\n[4] jailbreak_llms")
    jb_dir = THEWALL / "jailbreak_llms" / "data" / "prompts"
    for csv_file in sorted(jb_dir.glob("jailbreak_*.csv")):
        results = scan_csv(csv_file, 'prompt')
        summarize_results(csv_file.name, results)

    # 5. wildjailbreak (TSV — could be large)
    print("\n[5] wildjailbreak")
    wj_train = THEWALL / "wildjailbreak" / "train" / "train.tsv"
    wj_eval = THEWALL / "wildjailbreak" / "eval" / "eval.tsv"
    for tsv_path in [wj_train, wj_eval]:
        if tsv_path.exists():
            # Check file size first
            size_mb = tsv_path.stat().st_size / (1024 * 1024)
            if size_mb > 500:
                print(f"  {tsv_path.name}: {size_mb:.0f}MB — sampling first 50K rows")
                # Sample by reading with limit
                results = []
                try:
                    with open(tsv_path, 'r', encoding='utf-8', errors='replace') as f:
                        reader = csv.DictReader(f, delimiter='\t')
                        for i, row in enumerate(reader):
                            if i >= 50000:
                                break
                            # adversarial_harmful or vanilla_harmful
                            dtype = row.get('data_type', '')
                            if 'harmful' not in dtype:
                                continue
                            text = row.get('adversarial', '') or row.get('vanilla', '')
                            if not text:
                                continue
                            scripts = detect_scripts(text)
                            if not scripts:
                                continue
                            langs = classify_language(scripts)
                            if langs:
                                results.append({
                                    'index': i,
                                    'text': text[:200],
                                    'langs': langs,
                                    'scripts': scripts,
                                })
                except Exception as e:
                    print(f"  ERROR: {e}")
                summarize_results(tsv_path.name, results)
            else:
                results = scan_csv(tsv_path, 'adversarial', delimiter='\t')
                summarize_results(tsv_path.name, results)
        else:
            print(f"  LFS pointer? {tsv_path} size: checking...")
            if tsv_path.exists():
                with open(tsv_path, 'r') as f:
                    head = f.read(100)
                if 'oid sha256' in head:
                    print(f"  LFS POINTER — needs `git lfs pull`")

    # 6. ctf-satml24 chat data
    print("\n[6] ctf-satml24 (137K chats)")
    chat_path = THEWALL / "ctf-satml24" / "chat.json"
    if chat_path.exists():
        size_mb = chat_path.stat().st_size / (1024 * 1024)
        print(f"  chat.json: {size_mb:.0f}MB")
        # Sample the file — it's large
        results = []
        try:
            with open(chat_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"  Loaded {len(data)} entries, scanning...")
            # data is likely a list of chat objects
            count = 0
            for item in data:
                if count >= 50000:
                    break
                count += 1
                # Extract user messages from chat
                if isinstance(item, dict):
                    messages = item.get('messages', item.get('chat', []))
                    if isinstance(messages, list):
                        user_texts = [m.get('content', '') for m in messages
                                      if isinstance(m, dict) and m.get('role') == 'user']
                        text = ' '.join(user_texts)
                    else:
                        text = str(item.get('content', ''))
                elif isinstance(item, str):
                    text = item
                else:
                    continue
                scripts = detect_scripts(text)
                if not scripts:
                    continue
                langs = classify_language(scripts)
                if langs:
                    results.append({
                        'index': count,
                        'text': text[:200],
                        'langs': langs,
                        'scripts': scripts,
                    })
        except Exception as e:
            print(f"  ERROR: {e}")
        summarize_results("ctf-satml24", results)
    else:
        print(f"  NOT FOUND: {chat_path}")

    # 7. wambosec (5.7K parquet)
    print("\n[7] wambosec/prompt-injections")
    wambo_dir = THEWALL / "wambosec" / "prompt-injections" / "data"
    all_results = []
    for shard in sorted(wambo_dir.glob("*.parquet")):
        results = scan_parquet(shard, 'prompt', label_col='label', attack_values={'1'})
        all_results.extend(results)
    summarize_results("wambosec", all_results)

    # 8. nueralchemy (10.6K parquet)
    print("\n[8] nueralchemy/Prompt-injection-dataset")
    neur_path = THEWALL / "nueralchemy" / "Prompt-injection-dataset" / "data" / "train-00000-of-00001.parquet"
    if neur_path.exists():
        results = scan_parquet(neur_path, 'prompt', label_col='label', attack_values={'1'})
        summarize_results("nueralchemy", results)

    # 9. ai_safety_50k (Portuguese!)
    print("\n[9] ai_safety_50k (EN + PT)")
    pt_path = THEWALL / "ai_safety_50k" / "data" / "portuguese-00000-of-00001.parquet"
    if pt_path.exists():
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(pt_path)
            print(f"  Portuguese shard: {table.num_rows} rows, columns: {table.column_names}")
            # Check a sample
            if 'conversation' in table.column_names:
                sample = table.column('conversation')[0].as_py()
                print(f"  Sample: {str(sample)[:150]}...")
        except Exception as e:
            print(f"  ERROR: {e}")

    # 10. Check for LFS pointers we might need to pull
    print("\n[10] LFS pointer check")
    for name in ['wildjailbreak', 'SPML_Chatbot_Prompt_Injection', 'hackaprompt-dataset']:
        data_dir = THEWALL / name
        for f in data_dir.rglob('*.parquet'):
            size = f.stat().st_size
            if size < 1000:  # likely LFS pointer
                with open(f, 'r', errors='ignore') as fh:
                    head = fh.read(100)
                if 'oid sha256' in head:
                    print(f"  LFS POINTER: {f.relative_to(THEWALL)} ({size}B)")
        for f in data_dir.rglob('*.csv'):
            size = f.stat().st_size
            if size < 1000:
                with open(f, 'r', errors='ignore') as fh:
                    head = fh.read(100)
                if 'oid sha256' in head:
                    print(f"  LFS POINTER: {f.relative_to(THEWALL)} ({size}B)")
        for f in data_dir.rglob('*.tsv'):
            size = f.stat().st_size
            if size < 1000:
                with open(f, 'r', errors='ignore') as fh:
                    head = fh.read(100)
                if 'oid sha256' in head:
                    print(f"  LFS POINTER: {f.relative_to(THEWALL)} ({size}B)")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == '__main__':
    main()
