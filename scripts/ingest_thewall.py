#!/usr/bin/env python3
"""Ingest TheWall datasets into staging YAML files."""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
import warnings
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
import yaml

SUPPORTED_EXTENSIONS = {".csv", ".jsonl", ".parquet", ".json", ".tsv"}
MULTIJAIL_LANGS = ("en", "zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv")
TEXT_FALLBACK_COLUMNS = (
    "prompt",
    "text",
    "content",
    "body",
    "instruction",
    "question",
    "user_input",
    "user",
    "goal",
    "behavior",
    "behavior_text",
    "poison_content",
    "contestant_message",
    "submission_message",
    "conversation",
)

DATASET_TEXT_OVERRIDES: Dict[str, Sequence[str]] = {
    "llmail-inject-challenge": ("content", "body"),
    "PromptShield": ("text", "prompt"),
    "mosscap_prompt_injection": ("text", "prompt"),
    "tensor-trust-data": ("prompt", "attack"),
    "browsesafe-bench": ("html_content", "content"),
    "prompt-injection-dataset": ("text", "prompt"),
    "prompt_injection_cleaned_dataset-v2": ("text", "prompt"),
    "AmpleGCG": ("goal", "prompt", "target"),
    "generative-ai-red-teaming": (
        "prompt",
        "contestant_message",
        "submission_message",
        "user",
        "conversation",
    ),
}

DATASET_LABEL_COLUMN_OVERRIDES = {
    "ReSA": "data_type",
    "prompt_injection_cleaned_dataset-v2": "labels",
}

FORCE_ALL_POSITIVE_DATASETS = {
    "mosscap_prompt_injection",
    "prompt-injections",
    "russian_prompt_injections",
    "galtea-red-teaming-clustered-data",
}

DEFAULT_GCG_SUFFIX = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
HTML_BLOCK_TAG_RE = re.compile(
    r"(?is)</?(?:p|div|li|ul|ol|h[1-6]|tr|td|th|section|article|header|footer|main|aside|pre|code|br)[^>]*>"
)
HTML_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style)[^>]*>.*?</\1>")
HTML_TAG_RE = re.compile(r"(?is)<[^>]+>")
INJECTION_HINTS = (
    "ignore",
    "instruction",
    "system prompt",
    "developer mode",
    "jailbreak",
    "reveal",
    "bypass",
    "secret",
    "api key",
    "password",
    "tool call",
    "send",
    "exfil",
    "curl",
    "wget",
)


@dataclass
class PendingEntry:
    content: str
    description: str
    custom_id: Optional[str] = None


def eprint(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "dataset"


def parse_format_extensions(format_value: Any) -> set[str]:
    if format_value is None:
        return set()
    fmt = str(format_value).lower()
    extensions: set[str] = set()
    if "csv" in fmt:
        extensions.update({".csv", ".tsv"})
    if "jsonl" in fmt:
        extensions.add(".jsonl")
    if "parquet" in fmt:
        extensions.add(".parquet")
    if "json" in fmt and "jsonl" not in fmt:
        extensions.add(".json")
    return extensions


def load_index(index_path: Path) -> List[Dict[str, Any]]:
    with index_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid index structure in {index_path}")

    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError(f"Invalid index structure in {index_path}")

    merged: List[Dict[str, Any]] = []
    for item in datasets:
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        entry.setdefault("__default_label", "positive")
        merged.append(entry)

    benign = payload.get("benign")
    if isinstance(benign, list):
        for item in benign:
            if not isinstance(item, dict):
                continue
            entry = dict(item)
            # Benign corpus is unlabeled but semantically negative for attack classifiers.
            entry.setdefault("__default_label", "negative")
            merged.append(entry)

    return merged


def resolve_dataset_dir(thewall_root: Path, dataset: Dict[str, Any]) -> Path:
    path_override = dataset.get("path")
    if isinstance(path_override, str) and path_override.strip():
        override_path = Path(path_override.strip())
        if override_path.is_absolute():
            return override_path
        return thewall_root / override_path
    return thewall_root / str(dataset.get("name", ""))


def discover_candidate_files(dataset_dir: Path, preferred_extensions: set[str]) -> List[Path]:
    all_candidates = [
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    all_candidates.sort()
    if preferred_extensions:
        preferred = [path for path in all_candidates if path.suffix.lower() in preferred_extensions]
        if preferred:
            return preferred
    return all_candidates


def iter_records_from_file(path: Path) -> Iterator[Any]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            for chunk in pd.read_csv(path, chunksize=2000, low_memory=False):
                for record in chunk.to_dict(orient="records"):
                    yield record
            return
        if suffix == ".tsv":
            for chunk in pd.read_csv(path, sep="\t", chunksize=2000, low_memory=False):
                for record in chunk.to_dict(orient="records"):
                    yield record
            return
        if suffix == ".parquet":
            frame = pd.read_parquet(path)
            for record in frame.to_dict(orient="records"):
                yield record
            return
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
            return
        if suffix == ".json":
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except json.JSONDecodeError:
                # Some repos use newline-delimited JSON despite `.json` extension.
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
                return
            if isinstance(payload, list):
                for item in payload:
                    yield item
                return
            if isinstance(payload, dict):
                for key in ("data", "records", "items", "examples"):
                    value = payload.get(key)
                    if isinstance(value, list):
                        for item in value:
                            yield item
                        return
                if payload and all(isinstance(v, list) for v in payload.values()):
                    for value in payload.values():
                        for item in value:
                            yield item
                    return
                if len(payload) <= 100:
                    yield payload
            return
    except Exception as exc:  # pragma: no cover - best effort for heterogeneous datasets
        eprint(f"[warn] failed reading {path}: {exc}")


def get_field(record: Any, field_path: str) -> Any:
    if not isinstance(record, dict):
        return None
    if field_path in record:
        return record.get(field_path)
    current: Any = record
    for part in field_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def parse_string_container(text: str) -> Any:
    trimmed = text.strip()
    if not trimmed or trimmed[0] not in "[{":
        return None
    try:
        return json.loads(trimmed)
    except Exception:
        pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.literal_eval(trimmed)
    except Exception:
        return None


def normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        parsed = parse_string_container(text)
        if parsed is not None:
            parsed_text = normalize_text(parsed)
            if parsed_text:
                return parsed_text
        return text
    if isinstance(value, bytes):
        return normalize_text(value.decode("utf-8", errors="ignore"))
    if isinstance(value, (bool, int, float)):
        text = str(value).strip()
        return text or None
    if isinstance(value, dict):
        for key in (
            "content",
            "text",
            "body",
            "prompt",
            "question",
            "instruction",
            "goal",
            "attack",
            "user",
            "message",
            "value",
        ):
            if key in value:
                candidate = normalize_text(value.get(key))
                if candidate:
                    return candidate
        return None
    if hasattr(value, "tolist"):
        try:
            return normalize_text(value.tolist())
        except Exception:
            pass
    if isinstance(value, Iterable):
        parts: List[str] = []
        for item in value:
            candidate = normalize_text(item)
            if candidate:
                parts.append(candidate)
        if parts:
            return "\n---\n".join(parts)
        return None
    text = str(value).strip()
    return text or None


def extract_conversation_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        parsed = parse_string_container(value)
        if parsed is not None:
            return extract_conversation_text(parsed)
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, dict):
        return normalize_text(value.get("content")) or normalize_text(value.get("text"))
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        parts: List[str] = []
        for item in value:
            if isinstance(item, dict):
                candidate = (
                    normalize_text(item.get("content"))
                    or normalize_text(item.get("text"))
                    or normalize_text(item.get("body"))
                )
            else:
                candidate = normalize_text(item)
            if candidate:
                parts.append(candidate)
        if parts:
            return "\n---\n".join(parts)
    return normalize_text(value)


def extract_turns_by_role(value: Any, allowed_roles: set[str]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        parsed = parse_string_container(value)
        if parsed is not None:
            return extract_turns_by_role(parsed, allowed_roles)
        return None
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if not (isinstance(value, Iterable) and not isinstance(value, (str, bytes))):
        return None

    parts: List[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = normalize_text(item.get("role"))
        role_token = role.lower() if role else ""
        if role_token not in allowed_roles:
            continue
        candidate = (
            normalize_text(item.get("content"))
            or normalize_text(item.get("text"))
            or normalize_text(item.get("body"))
            or normalize_text(item.get("message"))
        )
        if candidate:
            parts.append(candidate)

    if parts:
        return "\n---\n".join(parts)
    return None


def extract_browsesafe_text(value: Any) -> Optional[str]:
    raw = normalize_text(value)
    if not raw:
        return None
    text = unescape(raw)
    text = HTML_SCRIPT_STYLE_RE.sub(" ", text)
    text = HTML_BLOCK_TAG_RE.sub("\n", text)
    text = HTML_TAG_RE.sub(" ", text)

    lines: List[str] = []
    for line in text.splitlines():
        collapsed = " ".join(line.split()).strip()
        if collapsed:
            lines.append(collapsed)
    if not lines:
        return None

    hint_lines = []
    for line in lines:
        lowered = line.lower()
        if any(hint in lowered for hint in INJECTION_HINTS):
            hint_lines.append(line)

    selected = hint_lines if hint_lines else lines
    selected = ordered_unique(selected)
    compact = "\n".join(selected)
    if len(compact) > 4000:
        compact = compact[:4000].rstrip()
    return compact or None


def extract_amplegcg_text(record: Any) -> Optional[str]:
    if isinstance(record, dict):
        suffix = (
            normalize_text(record.get("prompt"))
            or normalize_text(record.get("suffix"))
            or normalize_text(record.get("adv_string"))
            or normalize_text(record.get("control"))
            or normalize_text(record.get("control_str"))
            or normalize_text(record.get("p"))
        )
        goal = (
            normalize_text(record.get("goal"))
            or normalize_text(record.get("instruction"))
            or normalize_text(record.get("text"))
            or normalize_text(record.get("target"))
        )
    else:
        suffix = None
        goal = normalize_text(record)

    if suffix and goal:
        return f"{goal}\n---\n{suffix}"
    if suffix:
        return suffix
    if goal:
        # Fall back to a canonical GCG control-seed pattern so this corpus
        # carries suffix-shape signal instead of pure harmful goals.
        return f"{goal}\n---\n{DEFAULT_GCG_SUFFIX}"
    return None


def extract_nemotron_text(record: Any) -> Optional[str]:
    attack_text = (
        get_field(record, "attack_snapshot.attack.injection_string")
        or get_field(record, "attack_snapshot.attack.topic")
        or get_field(record, "attack_snapshot.attack.prompt")
    )
    normalized = normalize_text(attack_text)
    if normalized:
        return normalized
    trace_value = get_field(record, "trace")
    if hasattr(trace_value, "tolist"):
        trace_value = trace_value.tolist()
    if isinstance(trace_value, list):
        for item in trace_value:
            if not isinstance(item, dict):
                continue
            for key in ("attributes.input.value", "attributes.output.value"):
                candidate = item.get(key)
                candidate_text = normalize_text(candidate)
                if not candidate_text:
                    continue
                parsed = parse_string_container(candidate_text)
                if isinstance(parsed, dict):
                    topic = parsed.get("topic") or parsed.get("injection_string")
                    topic_text = normalize_text(topic)
                    if topic_text:
                        return topic_text
                return candidate_text
    return None


def is_placeholder_text_column(text_column: Any) -> bool:
    if not isinstance(text_column, str):
        return True
    lowered = text_column.strip().lower()
    if not lowered:
        return True
    markers = ("varies", "(list items)", "(and")
    return any(marker in lowered for marker in markers)


def ordered_unique(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    output: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def extract_text_for_record(dataset_name: str, text_column: Any, record: Any) -> Optional[str]:
    if dataset_name == "ai_safety_50k":
        return extract_conversation_text(get_field(record, "conversation"))
    if dataset_name == "atlas":
        return extract_turns_by_role(get_field(record, "conversation"), {"user", "attacker", "human"}) or extract_conversation_text(
            get_field(record, "conversation")
        )
    if dataset_name == "ctf-satml24":
        return extract_turns_by_role(get_field(record, "history"), {"user", "attacker", "human"}) or extract_conversation_text(
            get_field(record, "history")
        )
    if dataset_name == "browsesafe-bench":
        return extract_browsesafe_text(get_field(record, "html_content") or get_field(record, "content"))
    if dataset_name == "AmpleGCG":
        return extract_amplegcg_text(record)
    if dataset_name == "Nemotron-AIQ-Agentic-Safety-Dataset-1.0":
        return extract_nemotron_text(record)

    candidate_columns: List[str] = []
    candidate_columns.extend(DATASET_TEXT_OVERRIDES.get(dataset_name, ()))
    if isinstance(text_column, str) and not is_placeholder_text_column(text_column):
        candidate_columns.append(text_column)
    candidate_columns.extend(TEXT_FALLBACK_COLUMNS)

    for column in ordered_unique(candidate_columns):
        value = get_field(record, column)
        candidate = normalize_text(value)
        if candidate:
            return candidate

    return normalize_text(record)


def normalize_label_token(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return str(value).strip().lower()
    return str(value).strip().lower()


def classify_token(token: str) -> Optional[str]:
    if not token:
        return None

    positive_exact = {"1", "1.0", "true", "yes", "y"}
    negative_exact = {"0", "0.0", "false", "no", "n"}
    if token in positive_exact:
        return "positive"
    if token in negative_exact:
        return "negative"

    positive_keywords = ("harmful", "attack", "inject", "jailbreak", "malicious", "unsafe")
    negative_keywords = ("benign", "legitimate", "safe")
    if any(keyword in token for keyword in positive_keywords):
        return "positive"
    if any(keyword in token for keyword in negative_keywords):
        return "negative"
    return None


def build_label_map(label_values: Any) -> Dict[str, str]:
    if not isinstance(label_values, list):
        return {}
    mapping: Dict[str, str] = {}
    for value in label_values:
        token = normalize_label_token(value)
        classification = classify_token(token)
        if classification:
            mapping[token] = classification
    if not mapping and len(label_values) == 2:
        first = normalize_label_token(label_values[0])
        second = normalize_label_token(label_values[1])
        mapping[first] = "positive"
        mapping[second] = "negative"
    return mapping


def classify_record_label(dataset_name: str, dataset: Dict[str, Any], record: Any) -> Optional[str]:
    default_label = str(dataset.get("__default_label", "positive")).strip().lower()
    if default_label not in {"positive", "negative"}:
        default_label = "positive"

    if dataset_name == "ReSA":
        token = normalize_label_token(get_field(record, "data_type") or get_field(record, "type"))
        if "harmful" in token:
            return "positive"
        if "benign" in token:
            return "negative"
        return None

    if dataset_name == "jailbreak_llms":
        token = normalize_label_token(get_field(record, "jailbreak"))
        return classify_token(token)

    label_column = DATASET_LABEL_COLUMN_OVERRIDES.get(dataset_name, dataset.get("label_column"))
    if label_column is None:
        return default_label

    raw_label = get_field(record, label_column)
    if raw_label is None and dataset.get("label_column") is not None:
        raw_label = get_field(record, dataset["label_column"])
    if raw_label is None:
        for alias in ("label", "labels", "data_type", "type", "class", "cluster", "jailbreak"):
            raw_label = get_field(record, alias)
            if raw_label is not None:
                break

    if raw_label is None:
        if dataset_name in FORCE_ALL_POSITIVE_DATASETS:
            return "positive"
        return None

    token = normalize_label_token(raw_label)
    direct = classify_token(token)
    if direct:
        return direct

    mapping = build_label_map(dataset.get("label_values"))
    if token in mapping:
        return mapping[token]

    if dataset_name in FORCE_ALL_POSITIVE_DATASETS:
        return "positive"
    return None


def collect_entries_from_record(
    dataset_name: str,
    dataset: Dict[str, Any],
    record: Any,
    row_number: int,
) -> List[Tuple[str, PendingEntry]]:
    if dataset_name == "MultiJail":
        collected: List[Tuple[str, PendingEntry]] = []
        for lang in MULTIJAIL_LANGS:
            text = normalize_text(get_field(record, lang))
            if not text:
                continue
            entry = PendingEntry(
                content=text,
                description=f"multijail ({lang})",
                custom_id=f"multijail-{lang}-{row_number:04d}",
            )
            collected.append(("positive", entry))
        return collected

    text = extract_text_for_record(dataset_name, dataset.get("text_column"), record)
    if not text:
        return []

    label = classify_record_label(dataset_name, dataset, record)
    if label not in {"positive", "negative"}:
        return []

    return [(label, PendingEntry(content=text, description=dataset_name))]


def dedupe_entries(entries: List[PendingEntry]) -> List[PendingEntry]:
    seen: set[str] = set()
    deduped: List[PendingEntry] = []
    for entry in entries:
        if entry.content in seen:
            continue
        seen.add(entry.content)
        deduped.append(entry)
    return deduped


def cap_entries(entries: List[PendingEntry], cap: int, rng: random.Random) -> List[PendingEntry]:
    if len(entries) <= cap:
        return entries
    sampled = rng.sample(entries, cap)
    return sampled


def build_yaml_records(dataset_name: str, label: str, entries: List[PendingEntry]) -> List[Dict[str, str]]:
    if not entries:
        return []
    width = max(4, len(str(len(entries))))
    slug = slugify(dataset_name)
    records: List[Dict[str, str]] = []
    for index, entry in enumerate(entries, start=1):
        record_id = entry.custom_id or f"{slug}-{index:0{width}d}"
        records.append(
            {
                "id": record_id,
                "layer": "l1",
                "label": label,
                "content": entry.content,
                "description": entry.description,
            }
        )
    return records


def write_yaml(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(rows, handle, sort_keys=False, allow_unicode=True)


def write_or_remove_yaml(path: Path, rows: List[Dict[str, str]], dataset_name: str) -> None:
    if rows:
        write_yaml(path, rows)
        eprint(f"[write] {dataset_name}: {path} ({len(rows)})")
        return
    if path.exists():
        path.unlink()
        eprint(f"[remove] {dataset_name}: removed empty output {path}")


def process_dataset(
    dataset: Dict[str, Any],
    thewall_root: Path,
    output_dir: Path,
    cap: int,
    rng: random.Random,
) -> None:
    name = dataset.get("name")
    if not isinstance(name, str) or not name:
        return

    routing = dataset.get("specialist_routing")
    if isinstance(routing, list) and len(routing) == 0:
        eprint(f"[skip] {name}: specialist_routing is empty")
        return

    dataset_dir = resolve_dataset_dir(thewall_root, dataset)
    if not dataset_dir.exists():
        eprint(f"[skip] {name}: directory not found at {dataset_dir}")
        return

    preferred_extensions = parse_format_extensions(dataset.get("format"))
    candidates = discover_candidate_files(dataset_dir, preferred_extensions)
    if not candidates:
        eprint(f"[skip] {name}: no supported files found")
        return

    eprint(f"[start] {name}: {len(candidates)} candidate files")
    positive_entries: List[PendingEntry] = []
    negative_entries: List[PendingEntry] = []
    row_counter = 0

    for candidate in candidates:
        eprint(f"[file] {name}: {candidate.relative_to(dataset_dir)}")
        for record in iter_records_from_file(candidate):
            row_counter += 1
            for label, entry in collect_entries_from_record(name, dataset, record, row_counter):
                if label == "positive":
                    positive_entries.append(entry)
                else:
                    negative_entries.append(entry)

    if not positive_entries and not negative_entries:
        eprint(f"[skip] {name}: no usable text rows")
        return

    positive_entries = dedupe_entries(positive_entries)
    negative_entries = dedupe_entries(negative_entries)

    if len(positive_entries) > cap:
        eprint(f"[cap] {name}: positive {len(positive_entries)} -> {cap}")
        positive_entries = cap_entries(positive_entries, cap, rng)
    if len(negative_entries) > cap:
        eprint(f"[cap] {name}: negative {len(negative_entries)} -> {cap}")
        negative_entries = cap_entries(negative_entries, cap, rng)

    slug = slugify(name)
    has_label_column = dataset.get("label_column") is not None
    default_label = str(dataset.get("__default_label", "positive")).strip().lower()

    positive_path = output_dir / f"thewall_{slug}_pos.yaml"
    negative_path = output_dir / f"thewall_{slug}_neg.yaml"

    positive_rows = build_yaml_records(name, "positive", positive_entries)
    negative_rows = build_yaml_records(name, "negative", negative_entries)

    write_positive = has_label_column or default_label == "positive" or bool(positive_rows)
    write_negative = has_label_column or default_label == "negative" or bool(negative_rows)

    if write_positive:
        write_or_remove_yaml(positive_path, positive_rows, name)
    elif positive_path.exists():
        positive_path.unlink()
        eprint(f"[remove] {name}: removed stale output {positive_path}")

    if write_negative:
        write_or_remove_yaml(negative_path, negative_rows, name)
    elif negative_path.exists():
        negative_path.unlink()
        eprint(f"[remove] {name}: removed stale output {negative_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest TheWall datasets into staging YAML files.")
    parser.add_argument(
        "--thewall-root",
        type=Path,
        default=Path("..") / "TheWall",
        help="Root path for TheWall datasets.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Path to INDEX.yaml (defaults to <thewall-root>/INDEX.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("schema") / "eval" / "staging",
        help="Output directory for generated YAML files.",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=5000,
        help="Maximum rows per output file after dedupe.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling when rows exceed --cap.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        help="Only process these dataset names (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thewall_root = args.thewall_root.resolve()
    index_path = args.index.resolve() if args.index else (thewall_root / "INDEX.yaml").resolve()
    output_dir = args.output_dir.resolve()
    rng = random.Random(args.seed)

    eprint(f"[init] root={thewall_root}")
    eprint(f"[init] index={index_path}")
    eprint(f"[init] output={output_dir}")

    datasets = load_index(index_path)
    if args.dataset:
        allowed = set(args.dataset)
        datasets = [d for d in datasets if d.get("name") in allowed]
        eprint(f"[filter] --dataset {args.dataset} -> {len(datasets)} matched")
    for dataset in datasets:
        process_dataset(dataset, thewall_root, output_dir, args.cap, rng)

    eprint("[done] ingestion complete")


if __name__ == "__main__":
    main()
