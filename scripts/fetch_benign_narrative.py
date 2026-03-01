#!/usr/bin/env python3
"""
Sample benign narrative/noisy data from locally cloned HuggingFace datasets.

Sources:
  - Reddit WritingPrompts  → creative writing / fiction
  - Reddit gaming + Showerthoughts → noisy internet text (entropy vaccine)
  - BookSum → book/chapter plot summaries
  - Wiki Movie Plots → movie plot summaries

Usage:
    python scripts/fetch_benign_narrative.py \
        --thewall-dir C:/Users/anyth/MINE/dev/TheWall/benign \
        --out-dir schema/eval/staging
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import yaml

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas required. pip install pandas pyarrow", file=sys.stderr)
    sys.exit(1)


def sample_reddit(data_dir: Path, subreddit: str, n: int, rng: random.Random) -> list[dict]:
    """Read parquet files for one subreddit, sample n posts with selftext."""
    pattern = f"{subreddit}-*.parquet"
    files = sorted(data_dir.glob(pattern))
    if not files:
        print(f"  WARNING: no parquet files for {subreddit}", file=sys.stderr)
        return []

    print(f"  Reading {len(files)} parquet files for r/{subreddit}...", file=sys.stderr)
    dfs = [pd.read_parquet(f, columns=["title", "selftext"]) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Filter to posts with real text content
    df = df[df["selftext"].notna()]
    df = df[~df["selftext"].isin(["[removed]", "[deleted]", ""])]
    df["word_count"] = df["selftext"].str.split().str.len()
    df = df[df["word_count"] >= 5]

    print(f"    {len(df)} valid posts, sampling {min(n, len(df))}", file=sys.stderr)
    sampled = df.sample(n=min(n, len(df)), random_state=rng.randint(0, 2**31))

    entries = []
    for i, (_, row) in enumerate(sampled.iterrows()):
        title = str(row.get("title") or "").strip()
        text = str(row["selftext"]).strip()
        content = f"{title}\n\n{text}" if title else text
        entries.append({
            "id": f"reddit-{subreddit.lower()}-{i:04d}",
            "content": content,
            "label": "benign",
            "layer": "l1",
            "source": f"reddit_{subreddit.lower()}",
            "description": f"Reddit r/{subreddit} post",
        })
    return entries


def sample_booksum(thewall_dir: Path, n: int, rng: random.Random) -> list[dict]:
    """Sample from booksum train.csv — chapter/book summaries."""
    csv_path = thewall_dir / "booksum" / "train.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found", file=sys.stderr)
        return []

    print(f"  Reading booksum...", file=sys.stderr)
    # booksum CSV has very long fields
    csv.field_size_limit(sys.maxsize)
    rows = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Use summary_text (plot summary) as the content
            text = (row.get("summary_text") or "").strip()
            if not text or len(text.split()) < 20:
                continue
            rows.append({
                "text": text,
                "book": row.get("book_id", ""),
            })

    print(f"    {len(rows)} valid summaries, sampling {min(n, len(rows))}", file=sys.stderr)
    sampled = rng.sample(rows, min(n, len(rows)))

    entries = []
    for i, row in enumerate(sampled):
        entries.append({
            "id": f"booksum-{i:04d}",
            "content": row["text"],
            "label": "benign",
            "layer": "l1",
            "source": "booksum",
            "description": f"Book summary: {row['book']}",
        })
    return entries


def sample_wiki_movie_plots(thewall_dir: Path, n: int, rng: random.Random) -> list[dict]:
    """Sample from wiki movie plots CSV."""
    csv_path = thewall_dir / "wiki-movie-plots-with-summaries" / "wiki_movie_plots_deduped_with_summaries.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found", file=sys.stderr)
        return []

    print(f"  Reading wiki movie plots...", file=sys.stderr)
    csv.field_size_limit(sys.maxsize)
    df = pd.read_csv(csv_path)

    # Try common column names for plot text
    plot_col = None
    for col in ["Plot", "plot", "Plot Summary", "summary", "Wiki Plot"]:
        if col in df.columns:
            plot_col = col
            break

    if plot_col is None:
        # Just use the longest text column
        print(f"    Columns: {list(df.columns)}", file=sys.stderr)
        text_cols = [c for c in df.columns if df[c].dtype == "object"]
        if text_cols:
            plot_col = max(text_cols, key=lambda c: df[c].str.len().mean())
            print(f"    Using column: {plot_col}", file=sys.stderr)
        else:
            print(f"    ERROR: no text columns found", file=sys.stderr)
            return []

    df = df[df[plot_col].notna()]
    df = df[df[plot_col].str.split().str.len() >= 20]

    print(f"    {len(df)} valid plots, sampling {min(n, len(df))}", file=sys.stderr)
    sampled = df.sample(n=min(n, len(df)), random_state=rng.randint(0, 2**31))

    title_col = None
    for col in ["Title", "title", "Movie", "movie"]:
        if col in sampled.columns:
            title_col = col
            break

    entries = []
    for i, (_, row) in enumerate(sampled.iterrows()):
        title = str(row[title_col]).strip() if title_col else ""
        text = str(row[plot_col]).strip()
        entries.append({
            "id": f"wiki-movie-{i:04d}",
            "content": text,
            "label": "benign",
            "layer": "l1",
            "source": "wiki_movie_plots",
            "description": f"Movie plot: {title}" if title else "Movie plot summary",
        })
    return entries


def write_yaml(entries: list[dict], path: Path):
    """Write entries to YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(entries, f, default_flow_style=False, allow_unicode=True, width=1000)
    print(f"  Wrote {len(entries)} entries to {path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Fetch benign narrative/noisy data")
    parser.add_argument("--thewall-dir", type=Path,
                        default=Path("C:/Users/anyth/MINE/dev/TheWall/benign"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("schema/eval/staging"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    reddit_data = args.thewall_dir / "REDDIT_submissions" / "data"

    # 1. WritingPrompts → creative writing
    print("\n=== WritingPrompts (creative writing) ===", file=sys.stderr)
    wp = sample_reddit(reddit_data, "WritingPrompts", 3000, rng)
    write_yaml(wp, args.out_dir / "opensource_reddit_writingprompts_benign.yaml")

    # 2. gaming + Showerthoughts → noisy internet text
    print("\n=== gaming + Showerthoughts (noisy text) ===", file=sys.stderr)
    gaming = sample_reddit(reddit_data, "gaming", 750, rng)
    shower = sample_reddit(reddit_data, "Showerthoughts", 750, rng)
    noisy = gaming + shower
    rng.shuffle(noisy)
    # Re-index
    for i, e in enumerate(noisy):
        e["id"] = f"reddit-noisy-{i:04d}"
    write_yaml(noisy, args.out_dir / "opensource_reddit_noisy_benign.yaml")

    # 3. BookSum → fiction/narrative summaries
    print("\n=== BookSum (fiction summaries) ===", file=sys.stderr)
    books = sample_booksum(args.thewall_dir, 2000, rng)
    write_yaml(books, args.out_dir / "opensource_booksum_benign.yaml")

    # 4. Wiki Movie Plots → movie plot summaries
    print("\n=== Wiki Movie Plots ===", file=sys.stderr)
    movies = sample_wiki_movie_plots(args.thewall_dir, 2000, rng)
    write_yaml(movies, args.out_dir / "opensource_wiki_movie_plots_benign.yaml")

    # Summary
    total = len(wp) + len(noisy) + len(books) + len(movies)
    print(f"\n{'='*50}", file=sys.stderr)
    print(f"Total: {total} entries", file=sys.stderr)
    print(f"  WritingPrompts:   {len(wp)}", file=sys.stderr)
    print(f"  Noisy (gaming+ST): {len(noisy)}", file=sys.stderr)
    print(f"  BookSum:          {len(books)}", file=sys.stderr)
    print(f"  Wiki Movie Plots: {len(movies)}", file=sys.stderr)


if __name__ == "__main__":
    main()
