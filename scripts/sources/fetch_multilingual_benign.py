#!/usr/bin/env python3
"""
Sample multilingual benign data from Wikipedia (RU/ZH/AR) and XQuAD.

Wikipedia articles → creative writing / general knowledge (narrative text)
XQuAD → knowledge / Q&A

Usage:
    python scripts/fetch_multilingual_benign.py
"""

import random
import sys
from pathlib import Path

import pandas as pd
import yaml


WIKI_BASE = Path("C:/Users/anyth/MINE/dev/TheWall/benign/wikipedia/notlooseyoufuckingbaby")
XQUAD_BASE = Path("C:/Users/anyth/MINE/dev/TheWall/benign/xquad")
OUT_DIR = Path("schema/eval/staging")

# Wikipedia: sample narrative-heavy articles (longer text = more narrative)
WIKI_LANGS = {
    "ru": {"creative": 500, "knowledge": 500, "chat": 500, "hard_negatives": 500},
    "zh": {"creative": 300, "knowledge": 300, "chat": 400, "hard_negatives": 400},
    "ar": {"creative": 300, "knowledge": 300, "chat": 350, "hard_negatives": 350},
}

# XQuAD: Q&A pairs
XQUAD_LANGS = {"ru": 300, "zh": 240, "ar": 210}


def load_wiki_parquets(lang_dir: Path) -> pd.DataFrame:
    """Load all parquet files from a Wikipedia language dir."""
    files = sorted(lang_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def sample_wiki(lang: str, targets: dict, rng: random.Random) -> list[dict]:
    """Sample Wikipedia articles, bin by length for different categories."""
    lang_dir = WIKI_BASE / lang
    df = load_wiki_parquets(lang_dir)
    if df.empty:
        print(f"  WARNING: no data for wiki/{lang}", file=sys.stderr)
        return []

    # Filter to articles with real content
    df = df[df["text"].notna()]
    df["word_count"] = df["text"].str.split().str.len()
    df = df[df["word_count"] >= 20]

    print(f"  wiki/{lang}: {len(df)} valid articles", file=sys.stderr)

    entries = []

    # Long articles → creative (narrative-heavy, plot summaries, biographies)
    long_df = df[df["word_count"] >= 150].copy()
    n_creative = min(targets.get("creative", 0), len(long_df))
    if n_creative > 0:
        sampled = long_df.sample(n=n_creative, random_state=rng.randint(0, 2**31))
        for i, (_, row) in enumerate(sampled.iterrows()):
            # Take first ~500 words to keep reasonable size
            text = " ".join(str(row["text"]).split()[:500])
            entries.append({
                "id": f"wiki-{lang}-creative-{i:04d}",
                "content": text,
                "label": "benign",
                "layer": "l1",
                "source": f"wikipedia_{lang}",
                "description": f"Wikipedia {lang}: {row.get('title', '')}",
            })
        used_ids = set(sampled.index)
    else:
        used_ids = set()

    # Medium articles → knowledge
    remaining = df[~df.index.isin(used_ids)]
    med_df = remaining[(remaining["word_count"] >= 25) & (remaining["word_count"] < 300)]
    n_knowledge = min(targets.get("knowledge", 0), len(med_df))
    if n_knowledge > 0:
        sampled = med_df.sample(n=n_knowledge, random_state=rng.randint(0, 2**31))
        for i, (_, row) in enumerate(sampled.iterrows()):
            entries.append({
                "id": f"wiki-{lang}-knowledge-{i:04d}",
                "content": str(row["text"]).strip(),
                "label": "benign",
                "layer": "l1",
                "source": f"wikipedia_{lang}",
                "description": f"Wikipedia {lang}: {row.get('title', '')}",
            })
        used_ids.update(sampled.index)

    # Short-medium → chat (diverse short text)
    remaining = df[~df.index.isin(used_ids)]
    n_chat = min(targets.get("chat", 0), len(remaining))
    if n_chat > 0:
        sampled = remaining.sample(n=n_chat, random_state=rng.randint(0, 2**31))
        for i, (_, row) in enumerate(sampled.iterrows()):
            text = " ".join(str(row["text"]).split()[:200])
            entries.append({
                "id": f"wiki-{lang}-chat-{i:04d}",
                "content": text,
                "label": "benign",
                "layer": "l1",
                "source": f"wikipedia_{lang}",
                "description": f"Wikipedia {lang}: {row.get('title', '')}",
            })
        used_ids.update(sampled.index)

    # Any remaining → hard negatives
    remaining = df[~df.index.isin(used_ids)]
    n_hn = min(targets.get("hard_negatives", 0), len(remaining))
    if n_hn > 0:
        sampled = remaining.sample(n=n_hn, random_state=rng.randint(0, 2**31))
        for i, (_, row) in enumerate(sampled.iterrows()):
            text = " ".join(str(row["text"]).split()[:200])
            entries.append({
                "id": f"wiki-{lang}-hn-{i:04d}",
                "content": text,
                "label": "benign",
                "layer": "l1",
                "source": f"wikipedia_{lang}",
                "description": f"Wikipedia {lang}: {row.get('title', '')}",
            })

    print(f"    sampled {len(entries)} entries", file=sys.stderr)
    return entries


def sample_xquad(lang: str, n: int, rng: random.Random) -> list[dict]:
    """Sample XQuAD Q&A pairs."""
    lang_dir = XQUAD_BASE / f"xquad.{lang}"
    parquet = list(lang_dir.glob("*.parquet"))
    if not parquet:
        print(f"  WARNING: no data for xquad.{lang}", file=sys.stderr)
        return []

    df = pd.read_parquet(parquet[0])
    df = df[df["context"].notna() & df["question"].notna()]

    print(f"  xquad.{lang}: {len(df)} rows", file=sys.stderr)
    sampled = df.sample(n=min(n, len(df)), random_state=rng.randint(0, 2**31))

    entries = []
    for i, (_, row) in enumerate(sampled.iterrows()):
        # Combine question + context as natural Q&A text
        content = f"{row['question']}\n\n{row['context']}"
        entries.append({
            "id": f"xquad-{lang}-{i:04d}",
            "content": content,
            "label": "benign",
            "layer": "l1",
            "source": f"xquad_{lang}",
            "description": f"XQuAD {lang} Q&A",
        })
    return entries


def write_yaml(entries: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(entries, f, default_flow_style=False, allow_unicode=True, width=1000)
    print(f"  Wrote {len(entries)} → {path}", file=sys.stderr)


def main():
    rng = random.Random(42)

    all_wiki = []
    all_xquad = []

    # Wikipedia per language
    for lang, targets in WIKI_LANGS.items():
        print(f"\n=== Wikipedia {lang} ===", file=sys.stderr)
        entries = sample_wiki(lang, targets, rng)
        all_wiki.extend(entries)

    # XQuAD per language
    for lang, n in XQUAD_LANGS.items():
        print(f"\n=== XQuAD {lang} ===", file=sys.stderr)
        entries = sample_xquad(lang, n, rng)
        all_xquad.extend(entries)

    # Write per-language Wikipedia files
    for lang in WIKI_LANGS:
        lang_entries = [e for e in all_wiki if e["source"] == f"wikipedia_{lang}"]
        write_yaml(lang_entries, OUT_DIR / f"opensource_wikipedia_{lang}_benign.yaml")

    # Write per-language XQuAD files
    for lang in XQUAD_LANGS:
        lang_entries = [e for e in all_xquad if e["source"] == f"xquad_{lang}"]
        write_yaml(lang_entries, OUT_DIR / f"opensource_xquad_{lang}_benign.yaml")

    # Summary
    print(f"\n{'='*50}", file=sys.stderr)
    print(f"Total Wikipedia: {len(all_wiki)}", file=sys.stderr)
    print(f"Total XQuAD: {len(all_xquad)}", file=sys.stderr)
    print(f"Grand total: {len(all_wiki) + len(all_xquad)}", file=sys.stderr)


if __name__ == "__main__":
    main()
