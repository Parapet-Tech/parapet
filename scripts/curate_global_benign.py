#!/usr/bin/env python3
"""
Curate global benign dataset for L1 retrain from TheWall raw data.

This script is intentionally defensive:
- Resolves TheWall root from CLI/env/common workspace locations
- Reads raw files directly (no staging dependency)
- Enforces category/language targets from strategy/l1_retrain_plan.md
- Backfills missing language buckets within each category
- Avoids crashing on empty runs and prints actionable diagnostics

Usage:
  python scripts/curate_global_benign.py
  python scripts/curate_global_benign.py --scale 8
  python scripts/curate_global_benign.py --thewall-root ../TheWall --scale 8
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

try:
    YAML_LOADER = yaml.CSafeLoader
except AttributeError:
    YAML_LOADER = yaml.SafeLoader

try:
    YAML_DUMPER = yaml.CSafeDumper
except AttributeError:
    YAML_DUMPER = yaml.SafeDumper


# ---------------------------------------------------------------------------
# Paths / defaults
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent

DEFAULT_OUT = REPO_ROOT / "schema" / "eval" / "training" / "global_benign_curated.yaml"

THEWALL_CANDIDATES = [
    Path.cwd() / "TheWall",
    REPO_ROOT / "TheWall",
    WORKSPACE_ROOT / "TheWall",
    Path(r"C:\Users\anyth\MINE\dev\DefenseSector\TheWall"),
    Path(r"C:\Users\anyth\MINE\dev\TheWall"),
]

ATTACK_SIGNATURE_PATTERNS = [
    # HackAPrompt-style canary payloads.
    re.compile(r"(?i)\bi\s+have\s+been\s+p[\W_]*w[\W_]*n[\W_]*e[\W_]*d\b"),
    re.compile(r"(?i)\bp[\W_]*w[\W_]*n[\W_]*e[\W_]*d\b"),
    # Generic instruction-override directives.
    re.compile(
        r"(?is)\b(ignore|disregard|forget)\b.{0,60}\b(previous|prior|all|above)\b.{0,60}\b"
        r"(instruction|instructions|prompt|rules?)\b"
    ),
    # System prompt exfiltration asks.
    re.compile(
        r"(?is)\b(reveal|show|print|dump|leak|expose)\b.{0,80}\b"
        r"(system\s*prompt|developer\s*message|hidden\s*prompt|instructions?)\b"
    ),
    # Secret/canary extraction asks.
    re.compile(r"(?is)\b(output|return|write)\b.{0,80}\b(canary|secret|api key|password)\b"),
]
INVALID_TEXT_CTRL_RE = re.compile(
    # YAML-safe text filter:
    # - C0 controls except TAB/LF/CR
    # - C1 controls (except NEL 0x85)
    # - UTF-16 surrogate range
    # - noncharacters U+FFFE/U+FFFF
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x84\x86-\x9F\uD800-\uDFFF\uFFFE\uFFFF]"
)


def resolve_thewall_root(cli_path: str | None) -> Path:
    """Resolve TheWall root from --thewall-root, env, or known local defaults."""
    if cli_path:
        p = Path(cli_path).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"ERROR: --thewall-root not found: {p}")
        return p

    if os.environ.get("THEWALL_ROOT"):
        p = Path(os.environ["THEWALL_ROOT"]).expanduser().resolve()
        if p.exists():
            return p

    for candidate in THEWALL_CANDIDATES:
        p = candidate.expanduser().resolve()
        if p.exists():
            return p

    tried = "\n".join(f"  - {c}" for c in THEWALL_CANDIDATES)
    raise SystemExit(
        "ERROR: Could not locate TheWall root.\n"
        "Pass --thewall-root explicitly or set THEWALL_ROOT.\n"
        f"Tried:\n{tried}"
    )


# ---------------------------------------------------------------------------
# Text extractors
# ---------------------------------------------------------------------------


def _clean_text(text: str) -> str:
    text = str(text or "")
    # Drop control bytes invalid for YAML/JSON payloads (keep \t \n \r).
    text = INVALID_TEXT_CTRL_RE.sub("", text)
    text = text.strip()
    if not text:
        return ""
    # Strip trailing spaces per line, preserve line breaks.
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    return text if len(text) >= 5 else ""


def extract_instruction_response(row):
    """Combine instruction + optional context + response/output."""
    inst = str(row.get("instruction", row.get("指令", "")))
    resp = str(row.get("output", row.get("response", row.get("回复", row.get("输出", "")))))
    ctx = str(row.get("context", row.get("input", row.get("上下文", ""))))
    parts = [p for p in [inst, ctx, resp] if str(p).strip()]
    return _clean_text("\n".join(parts))


def _parse_conv(val):
    """Normalize conversation field that may be list, JSON string, or pyarrow array."""
    if val is None:
        return []
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return [val]
    if hasattr(val, "tolist"):
        val = val.tolist()
    if isinstance(val, list):
        return [dict(m) if hasattr(m, "keys") else m for m in val]
    return []


def _first_user_turn(msgs):
    for msg in msgs:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return _clean_text(str(msg.get("content", "")))
    if msgs and isinstance(msgs[0], dict):
        return _clean_text(str(msgs[0].get("content", "")))
    return ""


def extract_wildchat(row):
    return _first_user_turn(_parse_conv(row.get("conversation")))


def extract_conversation_a(row):
    conv = row.get("conversation_a", row.get("conversation"))
    return _first_user_turn(_parse_conv(conv))


def extract_saiga(row):
    return _first_user_turn(_parse_conv(row.get("messages")))


def extract_writingprompt(row):
    story = str(row.get("story", ""))
    prompt = str(row.get("prompt", ""))
    return _clean_text(story if len(story) > len(prompt) else prompt)


def extract_plot(row):
    return _clean_text(str(row.get("Plot", row.get("plot", row.get("summary", "")))))


def extract_wildjailbreak(row):
    # train.tsv uses "vanilla"; eval.tsv uses "adversarial"
    text = row.get("adversarial", "") or row.get("vanilla", "")
    return _clean_text(str(text))


def extract_col(col: str) -> Callable:
    def _extract(row):
        return _clean_text(str(row.get(col, "")))

    return _extract


def extract_xquad_question(row):
    question = str(row.get("question", ""))
    context = str(row.get("context", ""))
    # Include context lightly so knowledge samples are not all ultra-short.
    if context and len(context) > 280:
        context = context[:280]
    parts = [p for p in [question, context] if p.strip()]
    return _clean_text("\n".join(parts))


def extract_prompt_chosen(row):
    """Combine RLHF prompt with preferred answer."""
    prompt = str(row.get("prompt", ""))
    chosen = str(row.get("chosen", ""))
    parts = [p for p in [prompt, chosen] if p.strip()]
    return _clean_text("\n".join(parts))


SYSTEM_LIKE_RE = re.compile(
    r"(?i)(^\s*system\s*:|\byou are\b|\brole\s*:\s*system\b|\bassistant\b|\binstructions?\b|\bact as\b)"
)


def is_system_like(text: str) -> bool:
    return bool(SYSTEM_LIKE_RE.search(text))


def looks_like_attack_payload(text: str) -> bool:
    """Heuristic blocklist for obvious prompt-injection payload text."""
    t = text.strip()
    if not t:
        return False
    return any(p.search(t) for p in ATTACK_SIGNATURE_PATTERNS)


# ---------------------------------------------------------------------------
# Source definitions
#
# Fields:
#   name, path, category, language, max_rows, extractor, label_filter, text_filter?
#
# label_filter:
#   None => no filtering
#   (column, [allowed_values]) => keep rows whose normalized column value
#                                 matches one of allowed_values
# ---------------------------------------------------------------------------

SOURCES = [
    # Instructions/tasks
    {
        "name": "dolly-15k",
        "path": "benign/databricks-dolly-15k",
        "category": "instructions",
        "language": "EN",
        "max_rows": 0,
        "extractor": extract_instruction_response,
        "label_filter": None,
    },
    {
        "name": "ru-alpaca",
        "path": "benign/ru_turbo_alpaca",
        "category": "instructions",
        "language": "RU",
        "max_rows": 0,
        "extractor": extract_instruction_response,
        # Keep only "ok" quality rows.
        "label_filter": ("label", ["ok"]),
    },
    {
        "name": "dolly-15k-zh",
        "path": "benign/databricksdatabricks-dolly-15k-chinese",
        "category": "instructions",
        "language": "ZH",
        "max_rows": 0,
        "extractor": extract_instruction_response,
        "label_filter": None,
    },
    {
        "name": "alpaca-zh",
        "path": "benign/alpaca-zh",
        "category": "instructions",
        "language": "ZH",
        "max_rows": 0,
        "extractor": extract_instruction_response,
        "label_filter": None,
    },
    {
        "name": "aya-ar",
        "path": "benign/CohereForAI_aya_dataset_Arabic",
        "category": "instructions",
        "language": "AR",
        "max_rows": 0,
        "extractor": extract_col("inputs"),
        "label_filter": None,
    },
    # Chat/conversation
    {
        "name": "wildchat",
        "path": "benign/WildChat-1M",
        "category": "chat",
        "language": "EN",
        "max_rows": 50000,
        "extractor": extract_wildchat,
        "label_filter": None,
    },
    {
        "name": "chatbot-arena",
        "path": "benign/chatbot_arena_conversations",
        "category": "chat",
        "language": "EN",
        "max_rows": 0,
        "extractor": extract_conversation_a,
        "label_filter": None,
    },
    {
        "name": "saiga",
        "path": "benign/ru_turbo_saiga",
        "category": "chat",
        "language": "RU",
        "max_rows": 0,
        "extractor": extract_saiga,
        "label_filter": None,
    },
    {
        "name": "arabic-chatml",
        "path": "benign/arabic-fine-tuning-chatML",
        "category": "chat",
        "language": "AR",
        "max_rows": 0,
        "extractor": extract_col("text"),
        "label_filter": None,
    },
    {
        "name": "gpt-arabic-chatml",
        "path": "benign/GPT-Arabic-ChatML-data",
        "category": "chat",
        "language": "AR",
        "max_rows": 0,
        "extractor": extract_col("text"),
        "label_filter": None,
    },
    {
        "name": "rlhf-zh-chat",
        "path": "benign/rlhf-reward-single-round-trans_chinese",
        "category": "chat",
        "language": "ZH",
        "max_rows": 0,
        "extractor": extract_prompt_chosen,
        "label_filter": None,
    },
    # Creative writing / roleplay
    {
        "name": "writingprompts",
        "path": "benign/writingprompts",
        "category": "creative",
        "language": "EN",
        "max_rows": 50000,
        "extractor": extract_writingprompt,
        "label_filter": None,
    },
    {
        "name": "movie-plots",
        "path": "benign/wiki-movie-plots-with-summaries",
        "category": "creative",
        "language": "EN",
        "max_rows": 0,
        "extractor": extract_plot,
        "label_filter": None,
    },
    {
        "name": "movie-plots-ru",
        "path": "benign/movie-plots-from-wikipedia-in-russian",
        "category": "creative",
        "language": "RU",
        "max_rows": 0,
        "extractor": extract_plot,
        "label_filter": None,
    },
    # Code
    {
        "name": "codealpaca",
        "path": "benign/evol-codealpaca-v1",
        "category": "code",
        "language": "EN",
        "max_rows": 50000,
        "extractor": extract_instruction_response,
        "label_filter": None,
    },
    # Knowledge / Q&A
    {
        "name": "trivia-qa",
        "path": "benign/trivia-qa",
        "category": "knowledge",
        "language": "EN",
        "max_rows": 50000,
        "extractor": extract_col("query"),
        "label_filter": None,
    },
    {
        "name": "xstest",
        "path": "benign/XSTest",
        "category": "knowledge",
        "language": "EN",
        "max_rows": 0,
        "extractor": extract_col("prompt"),
        # Keep only safe prompts.
        "label_filter": ("label", ["safe"]),
    },
    {
        "name": "xquad-ru",
        "path": "benign/xquad/xquad.ru",
        "category": "knowledge",
        "language": "RU",
        "max_rows": 0,
        "extractor": extract_xquad_question,
        "label_filter": None,
    },
    {
        "name": "xquad-zh",
        "path": "benign/xquad/xquad.zh",
        "category": "knowledge",
        "language": "ZH",
        "max_rows": 0,
        "extractor": extract_xquad_question,
        "label_filter": None,
    },
    {
        "name": "stem-zh-instruction",
        "path": "benign/stem_zh_instruction",
        "category": "knowledge",
        "language": "ZH",
        "max_rows": 50000,
        "extractor": extract_instruction_response,
        "label_filter": None,
    },
    {
        "name": "xquad-ar",
        "path": "benign/xquad/xquad.ar",
        "category": "knowledge",
        "language": "AR",
        "max_rows": 0,
        "extractor": extract_xquad_question,
        "label_filter": None,
    },
    # System prompts
    {
        "name": "spml",
        "path": "SPML_Chatbot_Prompt_Injection",
        "category": "system_prompts",
        "language": "EN",
        "max_rows": 0,
        "extractor": extract_col("User Prompt"),
        "label_filter": ("Prompt injection", ["0", 0, False]),
    },
    {
        "name": "promptshield-system",
        "path": "PromptShield",
        "category": "system_prompts",
        "language": "EN",
        "max_rows": 0,
        "extractor": extract_col("prompt"),
        "label_filter": ("label", ["0", 0, False]),
        "text_filter": is_system_like,
    },
    # Hard negatives (benign slices from attack-heavy datasets)
    {
        "name": "promptshield-neg",
        "path": "PromptShield",
        "category": "hard_negatives",
        "language": "EN",
        "max_rows": 0,
        "extractor": extract_col("prompt"),
        "label_filter": ("label", ["0", 0, False]),
    },
    {
        "name": "wildjailbreak-neg",
        "path": "wildjailbreak",
        "category": "hard_negatives",
        "language": "EN",
        "max_rows": 50000,
        "extractor": extract_wildjailbreak,
        "label_filter": ("data_type", ["vanilla_benign", "adversarial_benign"]),
    },
    {
        "name": "safeguard-neg",
        "path": "safe-guard-prompt-injection/data",
        "category": "hard_negatives",
        "language": "EN",
        "max_rows": 0,
        "extractor": extract_col("text"),
        "label_filter": ("label", ["0", 0, False]),
    },
    {
        "name": "jailbreak-llms-neg",
        "path": "jailbreak_llms/data/prompts",
        "category": "hard_negatives",
        "language": "EN",
        "max_rows": 50000,
        "extractor": extract_col("prompt"),
        "label_filter": ("jailbreak", [False, "false", "False", 0, "0"]),
    },
    {
        "name": "resa-neg",
        "path": "ReSA",
        "category": "hard_negatives",
        "language": "EN",
        "max_rows": 0,
        "extractor": extract_col("instruction"),
        "label_filter": ("data_type", ["vanilla_benign", "adversarial_benign"]),
    },
    {
        "name": "browsesafe-neg",
        "path": "browsesafe-bench",
        "category": "hard_negatives",
        "language": "EN",
        "max_rows": 0,
        "extractor": extract_col("content"),
        "label_filter": ("label", ["no", False, 0, "0"]),
    },
    {
        "name": "nueralchemy-neg",
        "path": "nueralchemy/Prompt-injection-dataset",
        "category": "hard_negatives",
        "language": "EN",
        "max_rows": 0,
        "extractor": extract_col("prompt"),
        "label_filter": ("label", ["0", 0, False]),
    },
    {
        "name": "wambosec-neg",
        "path": "wambosec/prompt-injections",
        "category": "hard_negatives",
        "language": "EN",
        "max_rows": 0,
        "extractor": extract_col("prompt"),
        "label_filter": ("label", ["0", 0, False]),
    },
    {
        "name": "arabguard-neg",
        "path": "ArabGuard-Egyptian-V1",
        "category": "hard_negatives",
        "language": "AR",
        "max_rows": 0,
        "extractor": extract_col("text"),
        "label_filter": ("label", ["0", 0, False]),
    },
]


# ---------------------------------------------------------------------------
# Targets (base 25k, multiply by --scale)
# ---------------------------------------------------------------------------

BASE_TARGETS = {
    ("instructions", "EN"): 3000,
    ("instructions", "RU"): 400,
    ("instructions", "ZH"): 320,
    ("instructions", "AR"): 280,
    ("chat", "EN"): 3000,
    ("chat", "RU"): 400,
    ("chat", "ZH"): 320,
    ("chat", "AR"): 280,
    ("creative", "EN"): 3600,
    ("creative", "RU"): 200,
    ("creative", "ZH"): 100,
    ("creative", "AR"): 100,
    ("code", "EN"): 3600,
    ("code", "RU"): 200,
    ("code", "ZH"): 100,
    ("code", "AR"): 100,
    ("system_prompts", "EN"): 2000,
    ("hard_negatives", "EN"): 3000,
    ("hard_negatives", "RU"): 400,
    ("hard_negatives", "ZH"): 320,
    ("hard_negatives", "AR"): 280,
    ("knowledge", "EN"): 2250,
    ("knowledge", "RU"): 300,
    ("knowledge", "ZH"): 240,
    ("knowledge", "AR"): 210,
}

LENGTH_PROPORTIONS = {
    "instructions": {"short": 0.25, "medium": 0.40, "long": 0.25, "structured": 0.10},
    "chat": {"short": 0.25, "medium": 0.40, "long": 0.25, "structured": 0.10},
    "creative": {"short": 0.10, "medium": 0.30, "long": 0.55, "structured": 0.05},
    "code": {"short": 0.10, "medium": 0.30, "long": 0.30, "structured": 0.30},
    "system_prompts": {"short": 0.30, "medium": 0.40, "long": 0.20, "structured": 0.10},
    "hard_negatives": {"short": 0.30, "medium": 0.40, "long": 0.20, "structured": 0.10},
    "knowledge": {"short": 0.30, "medium": 0.40, "long": 0.20, "structured": 0.10},
}

DEFAULT_LENGTH_PROPS = {"short": 0.25, "medium": 0.40, "long": 0.25, "structured": 0.10}


# ---------------------------------------------------------------------------
# File discovery / loading
# ---------------------------------------------------------------------------


def find_data_files(dataset_dir: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for ext, fmt in [
        ("**/*.parquet", "parquet"),
        ("**/*.jsonl", "jsonl"),
        ("**/*.csv", "csv"),
        ("**/*.tsv", "tsv"),
        ("**/*.json", "json"),
    ]:
        for p in sorted(dataset_dir.glob(ext)):
            if p.name.startswith("."):
                continue
            files.append((fmt, p))
    return files


def _read_data_file(fmt: str, path: Path) -> pd.DataFrame:
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "jsonl":
        return pd.read_json(path, lines=True, encoding="utf-8")
    if fmt in ("csv", "tsv"):
        sep = "\t" if fmt == "tsv" else ","
        return pd.read_csv(path, sep=sep, encoding="utf-8", on_bad_lines="skip")
    if fmt == "json":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            if isinstance(data, dict):
                for key in ["data", "examples", "items", "train"]:
                    if key in data and isinstance(data[key], list):
                        return pd.DataFrame(data[key])
            return pd.DataFrame()
        except json.JSONDecodeError:
            # Some datasets ship JSON Lines with a .json extension.
            try:
                return pd.read_json(path, lines=True, encoding="utf-8")
            except Exception:
                return pd.DataFrame()
    return pd.DataFrame()


def _norm_label(v) -> str:
    return str(v).strip().lower()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8", errors="replace")).hexdigest()


def load_attack_hashes(paths: list[Path]) -> set[str]:
    """Load attack corpus rows and return exact content-hash set."""
    hashes: set[str] = set()
    if not paths:
        return hashes

    print("\nLoading attack corpus hashes for benign decontamination...", file=sys.stderr)
    for p in paths:
        path = p.expanduser().resolve()
        if not path.exists():
            print(f"  WARN attack corpus missing: {path}", file=sys.stderr)
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.load(f, Loader=YAML_LOADER)
        except Exception as exc:
            print(f"  WARN failed reading attack corpus {path.name}: {exc}", file=sys.stderr)
            continue

        if not isinstance(raw, list):
            print(f"  WARN attack corpus not list: {path}", file=sys.stderr)
            continue

        added = 0
        for row in raw:
            if not isinstance(row, dict):
                continue
            content = str(row.get("content", ""))
            if not content.strip():
                continue
            hashes.add(_content_hash(content))
            added += 1
        print(f"  {path.name}: {added:,} rows", file=sys.stderr)

    print(f"  Attack hash set: {len(hashes):,} unique", file=sys.stderr)
    return hashes


def drop_exact_attack_overlaps(entries: list[dict], attack_hashes: set[str]) -> tuple[list[dict], int]:
    if not attack_hashes:
        return entries, 0
    kept = [e for e in entries if _content_hash(e["content"]) not in attack_hashes]
    return kept, len(entries) - len(kept)


def load_source(
    source: dict,
    thewall_root: Path,
    seed: int,
    drop_attack_signatures: bool = False,
) -> list[str]:
    """Load one source and return extracted benign text rows."""
    name = source["name"]
    dataset_dir = thewall_root / source["path"]
    max_rows = int(source["max_rows"])
    extractor = source["extractor"]
    label_filter = source["label_filter"]
    text_filter = source.get("text_filter")

    if not dataset_dir.exists():
        print(f"  SKIP {name}: {dataset_dir} not found", file=sys.stderr)
        return []

    files = find_data_files(dataset_dir)
    if not files:
        print(f"  SKIP {name}: no data files in {dataset_dir}", file=sys.stderr)
        return []

    rng = random.Random(seed)
    files = files.copy()
    rng.shuffle(files)

    dfs = []
    total_rows = 0
    row_budget = max_rows * 3 if max_rows > 0 else 0

    for fmt, path in files:
        try:
            df = _read_data_file(fmt, path)
        except Exception as exc:
            print(f"    WARN {name}: failed to read {path.name}: {exc}", file=sys.stderr)
            continue
        if df.empty:
            continue
        dfs.append(df)
        total_rows += len(df)
        if row_budget and total_rows >= row_budget:
            break

    if not dfs:
        print(f"  SKIP {name}: all files empty/unreadable", file=sys.stderr)
        return []

    df = pd.concat(dfs, ignore_index=True)

    if label_filter:
        col, allowed_values = label_filter
        if col not in df.columns:
            print(
                f"    WARN {name}: filter column '{col}' missing, cols={list(df.columns)[:10]}",
                file=sys.stderr,
            )
            return []
        allowed = {_norm_label(v) for v in allowed_values}
        before = len(df)
        df = df[df[col].apply(lambda x: _norm_label(x) in allowed)]
        if df.empty:
            sample_vals = [str(v) for v in pd.Series(dfs[0][col]).dropna().unique()[:10]]
            print(
                f"    WARN {name}: filter '{col}' in {allowed_values} matched 0/{before}; "
                f"sample={sample_vals}",
                file=sys.stderr,
            )
            return []

    if max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)

    out: list[str] = []
    dropped_attack_like = 0
    for row in df.itertuples(index=False):
        text = extractor(row._asdict())
        if not text:
            continue
        if text_filter and not text_filter(text):
            continue
        if drop_attack_signatures and looks_like_attack_payload(text):
            dropped_attack_like += 1
            continue
        out.append(text)

    if not out:
        print(f"  {name:<25}       0 rows (from {len(df)} loaded)", file=sys.stderr)
        print(f"    columns: {list(df.columns)[:10]}", file=sys.stderr)
    else:
        dropped_msg = f", dropped {dropped_attack_like} attack-like" if dropped_attack_like else ""
        print(f"  {name:<25} {len(out):>7} rows{dropped_msg}", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# Length / sampling
# ---------------------------------------------------------------------------


def bin_length(text: str) -> str:
    """Bucket text format/size for balancing."""
    words = len(text.split())
    has_structure = any(
        marker in text for marker in ["{", "```", "<html", "<div", "<?xml", "def ", "class "]
    )
    if has_structure and words > 10:
        return "structured"
    if words < 25:
        return "short"
    if words < 150:
        return "medium"
    return "long"


def sample_with_length_balance(entries: list[dict], target_n: int, length_props: dict, rng: random.Random) -> list[dict]:
    by_length = defaultdict(list)
    for entry in entries:
        by_length[entry["length_bin"]].append(entry)

    sampled: list[dict] = []
    for bucket, prop in length_props.items():
        bucket_target = int(target_n * prop)
        available = by_length.get(bucket, [])
        take = min(bucket_target, len(available))
        if take > 0:
            sampled.extend(rng.sample(available, take))

    remaining = target_n - len(sampled)
    if remaining > 0:
        sampled_ids = {e["_hid"] for e in sampled}
        leftovers = [e for e in entries if e["_hid"] not in sampled_ids]
        if leftovers:
            sampled.extend(rng.sample(leftovers, min(remaining, len(leftovers))))
    return sampled


# ---------------------------------------------------------------------------
# Curation
# ---------------------------------------------------------------------------


def curate(
    thewall_root: Path,
    scale: int,
    seed: int,
    drop_attack_signatures: bool = False,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    targets = {k: v * scale for k, v in BASE_TARGETS.items()}
    total_target = sum(targets.values())
    print(f"Target: {total_target:,} (base 25K x {scale})\n", file=sys.stderr)

    print(f"TheWall root: {thewall_root}", file=sys.stderr)
    print("Loading from TheWall...", file=sys.stderr)

    pool: dict[tuple[str, str], list[dict]] = defaultdict(list)
    seen_hashes: set[str] = set()

    for source in SOURCES:
        texts = load_source(
            source,
            thewall_root=thewall_root,
            seed=seed,
            drop_attack_signatures=drop_attack_signatures,
        )
        for text in texts:
            h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            pool[(source["category"], source["language"])].append(
                {
                    "_hid": h,
                    "id": f"{source['name']}-{h[:10]}",
                    "content": text,
                    "label": "benign",
                    "layer": "l1",
                    "source": source["name"],
                    "category": source["category"],
                    "language": source["language"],
                    "length_bin": bin_length(text),
                    "description": "",
                }
            )

    print("\nPool sizes:", file=sys.stderr)
    for key in sorted(pool.keys()):
        print(f"  {str(key):<35} {len(pool[key]):>7}", file=sys.stderr)
    print(f"  {'TOTAL':<35} {sum(len(v) for v in pool.values()):>7}", file=sys.stderr)

    selected: list[dict] = []
    selected_ids: set[str] = set()
    category_deficits: Counter = Counter()
    per_bucket_stats: list[tuple[str, str, int, int]] = []

    print("\nSampling by category/language target...", file=sys.stderr)
    for (category, language), target_n in sorted(targets.items()):
        available = [e for e in pool.get((category, language), []) if e["_hid"] not in selected_ids]
        length_props = LENGTH_PROPORTIONS.get(category, DEFAULT_LENGTH_PROPS)
        sampled = sample_with_length_balance(available, target_n, length_props, rng)
        for entry in sampled:
            selected_ids.add(entry["_hid"])
        selected.extend(sampled)

        taken = len(sampled)
        per_bucket_stats.append((category, language, target_n, taken))
        gap = target_n - taken
        if gap > 0:
            category_deficits[category] += gap
            print(f"  ({category}, {language}): {taken:,}/{target_n:,} - short {gap:,}", file=sys.stderr)
        else:
            print(f"  ({category}, {language}): {taken:,}/{target_n:,}", file=sys.stderr)

    print("\nCategory backfill (same category, any language)...", file=sys.stderr)
    category_backfill_stats: list[tuple[str, int, int]] = []
    for category, deficit in sorted(category_deficits.items()):
        if deficit <= 0:
            continue
        leftovers = [
            e
            for (cat, _lang), entries in pool.items()
            if cat == category
            for e in entries
            if e["_hid"] not in selected_ids
        ]
        if not leftovers:
            category_backfill_stats.append((category, deficit, 0))
            print(f"  {category:<16} 0/{deficit:,} (no leftovers)", file=sys.stderr)
            continue
        sampled = sample_with_length_balance(
            leftovers, deficit, LENGTH_PROPORTIONS.get(category, DEFAULT_LENGTH_PROPS), rng
        )
        for entry in sampled:
            selected_ids.add(entry["_hid"])
        selected.extend(sampled)
        category_backfill_stats.append((category, deficit, len(sampled)))
        print(f"  {category:<16} {len(sampled):,}/{deficit:,}", file=sys.stderr)
        category_deficits[category] -= len(sampled)

    unresolved_category_deficit = int(sum(max(0, d) for d in category_deficits.values()))

    global_gap = total_target - len(selected)
    global_backfill = 0
    if global_gap > 0:
        print(f"\nGlobal backfill: need {global_gap:,} more rows...", file=sys.stderr)
        leftovers = [e for entries in pool.values() for e in entries if e["_hid"] not in selected_ids]
        if leftovers:
            sampled = sample_with_length_balance(leftovers, global_gap, DEFAULT_LENGTH_PROPS, rng)
            for entry in sampled:
                selected_ids.add(entry["_hid"])
            selected.extend(sampled)
            global_backfill = len(sampled)
            global_gap -= global_backfill
        print(f"  added {global_backfill:,}, remaining gap {max(0, global_gap):,}", file=sys.stderr)

    summary = {
        "target_total": total_target,
        "actual_total": len(selected),
        "remaining_gap": max(0, global_gap),
        "unresolved_category_deficit": unresolved_category_deficit,
        "bucket_stats": per_bucket_stats,
        "category_backfill_stats": category_backfill_stats,
    }
    return selected, summary


def report(entries: list[dict], summary: dict) -> None:
    n = len(entries)
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Final dataset: {n:,} entries (target {summary['target_total']:,})", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    if n == 0:
        print("ERROR: No entries curated. Check TheWall root and source mappings.", file=sys.stderr)
        return

    cat_counts = Counter(e["category"] for e in entries)
    print("\nBy category:", file=sys.stderr)
    for k in sorted(cat_counts):
        print(f"  {k:<20} {cat_counts[k]:>7} ({cat_counts[k] / n * 100:.1f}%)", file=sys.stderr)

    lang_counts = Counter(e["language"] for e in entries)
    print("\nBy language:", file=sys.stderr)
    for k in sorted(lang_counts):
        print(f"  {k:<5} {lang_counts[k]:>7} ({lang_counts[k] / n * 100:.1f}%)", file=sys.stderr)

    len_counts = Counter(e["length_bin"] for e in entries)
    print("\nBy length:", file=sys.stderr)
    for k in ["short", "medium", "long", "structured"]:
        c = len_counts.get(k, 0)
        print(f"  {k:<12} {c:>7} ({c / n * 100:.1f}%)", file=sys.stderr)

    if summary["remaining_gap"] > 0:
        print(
            f"\nWARN: dataset is short by {summary['remaining_gap']:,} rows vs target.",
            file=sys.stderr,
        )
    if summary["unresolved_category_deficit"] > 0:
        print(
            f"WARN: unresolved category deficit = {summary['unresolved_category_deficit']:,}.",
            file=sys.stderr,
        )


def build_composition(entries: list[dict], summary: dict) -> dict:
    """Build composition stats for reproducibility and training audits."""
    n = len(entries)
    if n == 0:
        return {
            "total": 0,
            "by_category": [],
            "by_language": [],
            "by_length_bin": [],
            "by_source": [],
            "category_language_matrix": {},
            "summary": summary,
        }

    def _rows(counter: Counter) -> list[dict]:
        return [
            {"name": k, "count": int(v), "pct": round((v / n) * 100.0, 4)}
            for k, v in counter.most_common()
        ]

    by_category = Counter(e["category"] for e in entries)
    by_language = Counter(e["language"] for e in entries)
    by_length_bin = Counter(e["length_bin"] for e in entries)
    by_source = Counter(e["source"] for e in entries)

    matrix: dict[str, dict[str, int]] = defaultdict(dict)
    languages = sorted(set(e["language"] for e in entries))
    for category in sorted(set(e["category"] for e in entries)):
        for language in languages:
            matrix[category][language] = 0
    for e in entries:
        matrix[e["category"]][e["language"]] += 1

    matrix_clean = {category: dict(counts) for category, counts in matrix.items()}

    return {
        "total": n,
        "by_category": _rows(by_category),
        "by_language": _rows(by_language),
        "by_length_bin": _rows(by_length_bin),
        "by_source": _rows(by_source),
        "category_language_matrix": matrix_clean,
        "summary": summary,
    }


def write_yaml(path: Path, data, progress_label: str | None = None) -> None:
    """Write YAML with C dumper when available, with progress for large lists."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(data, list) and len(data) > 20_000:
            # Stream as JSON array (valid YAML 1.2) for speed and reliability on huge corpora.
            total = len(data)
            f.write("[\n")
            for idx, row in enumerate(data, start=1):
                json.dump(row, f, ensure_ascii=False, separators=(",", ":"))
                if idx < total:
                    f.write(",\n")
                else:
                    f.write("\n")
                if progress_label and (idx % 5000 == 0 or idx == total):
                    print(f"Writing {progress_label}: {idx:,}/{total:,}", file=sys.stderr, flush=True)
            f.write("]\n")
        else:
            yaml.dump(
                data,
                f,
                Dumper=YAML_DUMPER,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=1000,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate global benign dataset from TheWall")
    parser.add_argument(
        "--scale",
        type=int,
        default=8,
        help="Multiplier for base 25K targets (default: 8 => ~200K)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--thewall-root",
        type=str,
        default=None,
        help="Path to TheWall root (optional; auto-resolved if omitted)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output YAML path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--composition-out",
        type=Path,
        default=None,
        help="Optional composition report path (YAML). Default: <out>_composition.yaml",
    )
    parser.add_argument(
        "--drop-attack-signatures",
        action="store_true",
        help="Drop rows matching attack-signature heuristics (default: keep labeled benign rows)",
    )
    parser.add_argument(
        "--attack-corpus-files",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Optional attack YAML files for exact hash decontamination. "
            "Default: none (disabled unless provided)."
        ),
    )
    parser.add_argument(
        "--skip-attack-corpus-dedup",
        action="store_true",
        help="Skip exact overlap removal against attack corpus hashes",
    )
    args = parser.parse_args()

    if args.scale <= 0:
        raise SystemExit("ERROR: --scale must be > 0")

    thewall_root = resolve_thewall_root(args.thewall_root)
    dataset, summary = curate(
        thewall_root=thewall_root,
        scale=args.scale,
        seed=args.seed,
        drop_attack_signatures=args.drop_attack_signatures,
    )

    attack_hashes: set[str] = set()
    if not args.skip_attack_corpus_dedup:
        attack_files = args.attack_corpus_files or []
        if attack_files:
            attack_hashes = load_attack_hashes(attack_files)
            dataset, dropped = drop_exact_attack_overlaps(dataset, attack_hashes)
            summary["attack_exact_overlap_dropped"] = dropped
            if dropped > 0:
                print(
                    f"Dropped {dropped:,} rows due to exact overlap with attack corpus",
                    file=sys.stderr,
                )
        else:
            print("No attack corpus files provided; skipping exact attack-hash decontamination.", file=sys.stderr)

    summary["actual_total"] = len(dataset)
    summary["remaining_gap"] = max(0, summary["target_total"] - len(dataset))
    report(dataset, summary)

    if not dataset:
        raise SystemExit("ERROR: No benign data curated; refusing to write empty output.")

    clean = [
        {
            "id": e["id"],
            "layer": e["layer"],
            "label": e["label"],
            "description": e["description"],
            "content": e["content"],
            "source": e["source"],
            "category": e["category"],
            "language": e["language"],
            "length_bin": e["length_bin"],
        }
        for e in dataset
    ]

    out_path = args.out.expanduser().resolve()
    write_yaml(out_path, clean, progress_label="dataset")

    print(f"\nWrote {len(clean):,} entries to {out_path}", file=sys.stderr)

    comp_path = (
        args.composition_out.expanduser().resolve()
        if args.composition_out
        else out_path.with_name(f"{out_path.stem}_composition.yaml")
    )
    composition = build_composition(clean, summary)
    write_yaml(comp_path, composition, progress_label=None)

    print(f"Wrote composition to {comp_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
