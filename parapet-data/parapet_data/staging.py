"""
TheWall staging pipeline: read raw datasets, validate, label, emit staged YAMLs.

Reads TheWall INDEX.yaml, applies quality gates, runs reason classification
on attacks, and writes mirror-ready source YAMLs that sampler.py can consume.

Run via CLI:
    python -m parapet_data stage \
        --index ../TheWall/INDEX.yaml \
        --output schema/eval/staging/ \
        --holdout-sets schema/eval/l1_holdout.yaml
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence, TextIO

import yaml

from .classifiers import (
    BENIGN_CONFIDENCE_FLOOR,
    CONFIDENCE_FLOOR,
    ReasonClassification,
    classify_benign_reason,
    classify_reason,
)
from .extractors import clean_text
from .filters import ContentDeduplicator, content_hash, looks_like_attack
from .models import AttackReason, Language

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration types
# ---------------------------------------------------------------------------

# Map INDEX.yaml language codes → our Language enum
_LANG_ALIASES: dict[str, Language] = {
    "en": Language.EN,
    "ru": Language.RU,
    "zh": Language.ZH,
    "ar": Language.AR,
}

# Canonical label aliases (from the plan — Gate 1)
VALID_LABELS = {"malicious", "benign"}
LABEL_ALIASES: dict[str, str] = {
    "1": "malicious",
    "0": "benign",
    "positive": "malicious",
    "negative": "benign",
    "true": "malicious",
    "false": "benign",
    "harmful": "malicious",
    "safe": "benign",
    "attack": "malicious",
    "legitimate": "benign",
    "injection": "malicious",
    "yes": "malicious",
    "no": "benign",
    # WildJailbreak / similar compound labels
    "vanilla_harmful": "malicious",
    "adversarial_harmful": "malicious",
    "vanilla_benign": "benign",
    "adversarial_benign": "benign",
    # Common class/group labels in jailbreak corpora
    "jailbreak": "malicious",
    "prompt injection": "malicious",
    "prompt_injection": "malicious",
    "prompt-injection": "malicious",
}

DATASET_LEVEL_LABELS: dict[str, str] = {
    "all attacks": "malicious",
    "all benign": "benign",
}


@dataclass(frozen=True)
class DatasetConfig:
    """Parsed INDEX.yaml entry with staging context."""

    name: str
    path: str | None  # explicit path override, or None → use name as subdir
    format: str
    text_column: str  # may be comma-separated for multi-field extraction
    label_column: str | None
    label_values: list[str]
    languages: list[str]
    index_section: str  # "datasets" or "benign"
    # Staging-specific (optional in INDEX, can be set in code for pilot)
    reason_column: str | None = None  # column with per-row attack type
    reason_map: dict[str, str] | None = None  # map reason_column values → mirror category
    staging_status: str = "ready"
    label_map: dict[str, str] | None = None
    # For benign: which mirror reason(s) this source routes to
    benign_reasons: list[str] | None = None
    # For attacks: dataset-level mirror-category routing when classifier can't label
    specialist_routing: list[str] | None = None
    language_mode: str = "strict"  # "strict" or "best_effort"


@dataclass
class StagedSample:
    """One sample ready for output."""

    content: str
    label: str
    language: str
    source: str
    reason: str | None  # mirror category or None for unrouted benign
    content_hash: str


@dataclass
class RejectionRecord:
    """Why a sample was rejected."""

    source: str
    gate: str
    detail: str
    content_preview: str = ""


@dataclass
class QuarantineRecord:
    """Benign sample that matched attack signatures — needs human review."""

    content: str
    source: str
    reason: str


@dataclass
class DatasetResult:
    """Staging result for one dataset."""

    name: str
    rows_read: int
    row_limit_hit: bool = False
    staged: list[StagedSample] = field(default_factory=list)
    rejected: list[RejectionRecord] = field(default_factory=list)
    quarantined: list[QuarantineRecord] = field(default_factory=list)

    @property
    def rejection_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.rejected:
            counts[r.gate] = counts.get(r.gate, 0) + 1
        return counts

    @property
    def by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.staged:
            if s.reason:
                counts[s.reason] = counts.get(s.reason, 0) + 1
        return counts

    @property
    def by_language(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.staged:
            counts[s.language] = counts.get(s.language, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# INDEX.yaml parsing
# ---------------------------------------------------------------------------


def parse_index(index_path: Path) -> list[DatasetConfig]:
    """Parse TheWall INDEX.yaml into DatasetConfig objects.

    Handles both `datasets:` (attack) and `benign:` sections.
    """
    with open(index_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"INDEX.yaml must be a mapping, got {type(raw).__name__}")

    configs: list[DatasetConfig] = []

    for entry in raw.get("datasets") or []:
        if not isinstance(entry, dict):
            continue
        cfg = _parse_entry(entry, section="datasets")
        if cfg is not None:
            configs.append(cfg)

    for entry in raw.get("benign") or []:
        if not isinstance(entry, dict):
            continue
        cfg = _parse_entry(entry, section="benign")
        if cfg is not None:
            configs.append(cfg)

    return configs


def _parse_entry(entry: dict[str, Any], section: str) -> DatasetConfig | None:
    """Parse one INDEX.yaml entry into a DatasetConfig."""
    name = entry.get("name")
    if not name:
        return None

    status = entry.get("staging_status", "ready")
    if status == "excluded":
        return None

    return DatasetConfig(
        name=name,
        path=entry.get("path"),
        format=str(entry.get("format", "")),
        text_column=str(entry.get("text_column", "")),
        label_column=entry.get("label_column"),
        label_values=entry.get("label_values") or [],
        languages=entry.get("languages") or [],
        index_section=section,
        reason_column=entry.get("reason_column"),
        reason_map=entry.get("reason_map"),
        staging_status=status,
        label_map=entry.get("label_map"),
        benign_reasons=entry.get("benign_reasons"),
        specialist_routing=entry.get("specialist_routing"),
        language_mode=entry.get("language_mode", "strict"),
    )


# ---------------------------------------------------------------------------
# Format readers
# ---------------------------------------------------------------------------


def resolve_dataset_dir(thewall_root: Path, config: DatasetConfig) -> Path:
    """Resolve the on-disk directory for a dataset."""
    if config.path:
        p = Path(config.path)
        if p.is_absolute():
            return p
        return thewall_root / p
    return thewall_root / config.name


def discover_files(dataset_dir: Path, fmt: str) -> list[Path]:
    """Find data files in a dataset directory matching the declared format."""
    ext_map: dict[str, set[str]] = {
        "json": {".json"},
        "jsonl": {".jsonl"},
        "parquet": {".parquet"},
        "csv": {".csv"},
        "tsv": {".tsv"},
    }
    # Handle compound formats like "parquet (14 shards)" or "parquet/jsonl"
    preferred: set[str] = set()
    fmt_lower = fmt.lower()
    for key, exts in ext_map.items():
        if key in fmt_lower:
            preferred.update(exts)

    all_files = sorted(
        p for p in dataset_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".json", ".jsonl", ".parquet", ".csv", ".tsv"}
        and p.name != "README.md"
    )

    if preferred:
        matched = [p for p in all_files if p.suffix.lower() in preferred]
        if matched:
            return matched

    return all_files


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield dicts from a single data file. Handles JSON, JSONL, Parquet, CSV, TSV."""
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        import pandas as pd
        for rec in pd.read_parquet(path).to_dict(orient="records"):
            yield rec
        return

    if suffix in (".csv", ".tsv"):
        import pandas as pd
        sep = "\t" if suffix == ".tsv" else ","
        for chunk in pd.read_csv(path, sep=sep, chunksize=5000, low_memory=False):
            for rec in chunk.to_dict(orient="records"):
                yield rec
        return

    if suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        return

    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            # Try as single JSON first
            try:
                payload = json.load(f)
            except json.JSONDecodeError:
                # Fall back to line-delimited JSON
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
                return

        if isinstance(payload, list):
            yield from payload
        elif isinstance(payload, dict):
            # Check common wrapper keys
            for key in ("data", "records", "items", "examples"):
                if isinstance(payload.get(key), list):
                    yield from payload[key]
                    return
            # Some datasets use {<text>: {...metadata...}} keyed records.
            # Expand these into row dicts with `content` for extraction.
            if _looks_like_keyed_records(payload):
                for record_key, record_val in payload.items():
                    row: dict[str, Any] = {"content": str(record_key), "_record_key": str(record_key)}
                    if isinstance(record_val, dict):
                        row.update(record_val)
                    else:
                        row["value"] = record_val
                    yield row
                return
            yield payload


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def extract_text(row: dict[str, Any], config: DatasetConfig) -> str:
    """Extract and clean text from a row using the config's text_column spec.

    Handles:
    - Simple column: "prompt_ru"
    - Multi-column: "instruction,input,output" → concatenated
    - Conversation: "messages" → first user turn
    - ChatML text: "text" with <|im_start|> markers → extract user content
    """
    columns = [c.strip() for c in config.text_column.split(",")]

    if len(columns) == 1:
        col = columns[0]
        val = row.get(col)
        if val is None or (isinstance(val, str) and not val.strip()):
            fallback = _extract_text_fallback(row)
            return clean_text(fallback) if fallback else ""

        # Conversation format (list/array of {role, content} dicts)
        if hasattr(val, '__len__') and not isinstance(val, str) and len(val) and isinstance(val[0], dict):
            return _extract_conversation(list(val))

        text = str(val)

        # ChatML format detection
        if "<|im_start|>" in text:
            return _extract_chatml_user(text)

        return clean_text(text)

    # Multi-column: concatenate non-empty fields
    parts: list[str] = []
    for col in columns:
        val = row.get(col)
        if val and str(val).strip():
            parts.append(str(val).strip())
    return clean_text("\n".join(parts))


def _extract_text_fallback(row: dict[str, Any]) -> str:
    """Best-effort extraction when the declared text column is missing/empty."""
    # Common email/message schema: subject + body
    subject = row.get("subject")
    body = row.get("body")
    parts: list[str] = []
    if subject and str(subject).strip():
        parts.append(str(subject).strip())
    if body and str(body).strip():
        parts.append(str(body).strip())
    if parts:
        return "\n".join(parts)

    # Common benchmark schema: pre/attack/post prompt template
    pre = row.get("pre_prompt")
    attack = row.get("attack")
    post = row.get("post_prompt")
    parts = []
    for val in (pre, attack, post):
        if val and str(val).strip():
            parts.append(str(val).strip())
    if parts:
        return "\n".join(parts)

    # Generic fallback keys
    for key in (
        "content",
        "prompt",
        "text",
        "adversarial",
        "vanilla",
        "completion",
        "user_input",
        "input",
        "instruction",
        "question",
        "query",
        "attack",
    ):
        val = row.get(key)
        if val and str(val).strip():
            return str(val)
    return ""


def _extract_conversation(messages: list[dict]) -> str:
    """Extract first user turn from a conversation."""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return clean_text(str(msg.get("content", "")))
    # Fallback: first message content
    if messages and isinstance(messages[0], dict):
        return clean_text(str(messages[0].get("content", "")))
    return ""


_CHATML_USER_RE = re.compile(
    r"<\|im_start\|>user\n(.*?)(?:<\|im_end\|>|<\|im_start\|>)", re.DOTALL
)


def _extract_chatml_user(text: str) -> str:
    """Extract user content from ChatML formatted text."""
    m = _CHATML_USER_RE.search(text)
    if m:
        return clean_text(m.group(1))
    return clean_text(text)


# ---------------------------------------------------------------------------
# Gate 1: Label resolution
# ---------------------------------------------------------------------------


def normalize_label(raw: str) -> str:
    """Normalize a per-row label to 'malicious' or 'benign'. Raises on unknown."""
    cleaned = str(raw).strip().lower()
    if cleaned in VALID_LABELS:
        return cleaned
    if cleaned in LABEL_ALIASES:
        return LABEL_ALIASES[cleaned]
    raise ValueError(f"Unknown label: {raw!r}")


def resolve_label(
    row: dict[str, Any],
    config: DatasetConfig,
) -> str:
    """Resolve the canonical label for a row. Fail-closed on unknowns.

    Pattern A — section-level: benign: section → "benign"
    Pattern B — dataset-level: label_column is None → use label_values
    Pattern C — per-row: read label_column, normalize via label_map or aliases
    """
    # Pattern A
    if config.index_section == "benign":
        return "benign"

    # Pattern B
    if config.label_column is None:
        if len(config.label_values) != 1:
            raise ValueError(
                f"{config.name}: dataset-level label requires exactly one "
                f"label_values entry, got {config.label_values}"
            )
        key = config.label_values[0].strip().lower()
        if key not in DATASET_LEVEL_LABELS:
            raise ValueError(f"{config.name}: unknown dataset-level label {key!r}")
        return DATASET_LEVEL_LABELS[key]

    # Pattern C
    raw = row.get(config.label_column)
    if raw is None:
        # Best-effort fallback for mislabeled/missing label columns in some corpora
        for alt in (
            "label",
            "class",
            "attack_attempt",
            "is_attack",
            "is_prompt_injection",
            "group",
            "data_type",
            "prompt injection",
            "Prompt injection",
        ):
            if alt == config.label_column:
                continue
            raw = row.get(alt)
            if raw is not None:
                break
    if raw is None:
        raise ValueError(
            f"{config.name}: missing label column {config.label_column!r}"
        )
    raw_str = str(raw).strip()

    # Dataset-specific label_map first
    if config.label_map and raw_str in config.label_map:
        return config.label_map[raw_str]

    return normalize_label(raw_str)


def _looks_like_keyed_records(payload: dict[str, Any]) -> bool:
    """Heuristic: dict where many values are per-record objects keyed by content."""
    if not payload:
        return False
    if not all(isinstance(k, str) for k in payload):
        return False
    # Treat as keyed records when all values are mappings/scalars (not list blobs)
    return all(not isinstance(v, list) for v in payload.values())


# ---------------------------------------------------------------------------
# Gate 2: Language validation
# ---------------------------------------------------------------------------


def resolve_language(config: DatasetConfig) -> Language | None:
    """Resolve the primary staging language from config.

    Returns None if no supported language found.
    """
    for lang_code in config.languages:
        lang = _LANG_ALIASES.get(lang_code.lower())
        if lang is not None:
            return lang
    return None


# Script ranges for per-row language validation
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_CJK_RE = re.compile(r"[\u4E00-\u9FFF]")
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")

_EXPECTED_SCRIPT: dict[Language, re.Pattern] = {
    Language.RU: _CYRILLIC_RE,
    Language.ZH: _CJK_RE,
    Language.AR: _ARABIC_RE,
}


def validate_script(text: str, language: Language) -> bool:
    """Check that text contains at least some characters from the expected script.

    Returns True if the text passes (has expected script chars or language is EN).
    Returns False for e.g. pure-Latin text stamped as RU.
    Mixed-script text (Russian + English) passes — that's normal for PI attacks.
    """
    if language == Language.EN:
        return True  # English uses Latin, no script check needed
    expected = _EXPECTED_SCRIPT.get(language)
    if expected is None:
        return True  # Unknown script, pass through
    return bool(expected.search(text))


# ---------------------------------------------------------------------------
# Gate 3: Reason classification (attacks only)
# ---------------------------------------------------------------------------


def _reason_value(reason: str | AttackReason) -> str:
    if isinstance(reason, AttackReason):
        return reason.value
    return str(reason)


def resolve_reason(
    text: str,
    label: str,
    config: DatasetConfig,
    row: dict[str, Any] | None = None,
) -> ReasonClassification | None:
    """Resolve attack reason for a sample.

    For attacks: use dataset reason_map if the row has a reason column,
    otherwise fall back to the heuristic classifier.

    For benign: source-level routing via config.benign_reasons.
    Returns None if no reason can be assigned (sample goes to review queue).
    """
    def _normalize_declared_reason(reason: str | AttackReason | None) -> str | None:
        if reason is None:
            return None
        if isinstance(reason, AttackReason):
            return reason.value
        normalized = str(reason).strip()
        return normalized or None

    def _normalized_reason_list(reasons: Sequence[str] | None) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for reason in reasons or []:
            resolved = _normalize_declared_reason(reason)
            if resolved is None or resolved in seen:
                continue
            seen.add(resolved)
            normalized.append(resolved)
        return normalized

    if label == "benign":
        if config.benign_reasons:
            # Deterministic partition: hash content to pick a reason.
            # Stable across reruns — same content always maps to same reason.
            reasons = _normalized_reason_list(config.benign_reasons)
            if not reasons:
                return None
            bucket = int(content_hash(text)[:8], 16) % len(reasons)
            return ReasonClassification(
                reason=reasons[bucket],
                confidence=1.0,
                signals=("source_level_routing", f"hash_bucket:{bucket}/{len(reasons)}"),
            )
        # Fallback: classify by surface similarity to attack categories
        return classify_benign_reason(text)

    # Attack sample
    # Try dataset's reason_map against the declared reason_column
    if config.reason_map and config.reason_column and row is not None:
        raw_class = row.get(config.reason_column)
        if raw_class is not None:
            mapped = config.reason_map.get(str(raw_class))
            reason = _normalize_declared_reason(mapped)
            if reason is not None:
                return ReasonClassification(
                    reason=reason,
                    confidence=1.0,
                    signals=(f"reason_map:{raw_class}",),
                )

    # Fallback: heuristic classifier
    result = classify_reason(text)
    if result is not None:
        return result

    # Last resort: dataset-level specialist_routing from INDEX.yaml
    if config.specialist_routing:
        reasons = _normalized_reason_list(config.specialist_routing)
        if reasons:
            bucket = int(content_hash(text)[:8], 16) % len(reasons)
            return ReasonClassification(
                reason=reasons[bucket],
                confidence=CONFIDENCE_FLOOR,
                signals=(
                    "specialist_routing_fallback",
                    f"dataset:{config.name}",
                    f"hash_bucket:{bucket}/{len(reasons)}",
                ),
            )

    return None


# ---------------------------------------------------------------------------
# Gate 6: Content validation
# ---------------------------------------------------------------------------


def validate_content(text: str) -> str:
    """Reject empty, too-short, or garbage text. Returns cleaned text or raises."""
    cleaned = clean_text(text)
    if len(cleaned) < 10:
        raise ValueError(f"content too short ({len(cleaned)} chars)")
    return cleaned


# ---------------------------------------------------------------------------
# Gate 7: Holdout-leakage exclusion
# ---------------------------------------------------------------------------


def load_holdout_hashes(paths: list[Path]) -> dict[str, set[str]]:
    """Pre-compute holdout hashes from YAML eval sets.

    Each file should be a YAML list of dicts with 'content' keys.
    Returns {filename_stem: set_of_content_hashes}.

    Prefers .hashes sidecar cache files (one SHA256 per line) over parsing
    raw YAML, because PyYAML is extremely slow on large lists (68K lines
    = minutes). If no cache exists, falls back to YAML parse and writes cache.
    """
    holdout_sets: dict[str, set[str]] = {}
    for path in paths:
        if not path.exists():
            log.warning("holdout set not found: %s", path)
            continue

        cache_path = path.with_suffix(path.suffix + ".hashes")
        if cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
            hashes = _load_hash_cache(cache_path)
            log.info("loaded %d holdout hashes from cache %s", len(hashes), cache_path.name)
        else:
            log.info("parsing %s (%.1f MB) — this may take a moment...",
                     path.name, path.stat().st_size / 1_048_576)
            hashes = _parse_yaml_content_hashes(path)
            _write_hash_cache(cache_path, hashes)
            log.info("loaded %d holdout hashes from %s (cache written)", len(hashes), path.name)

        if hashes:
            holdout_sets[path.stem] = hashes
    return holdout_sets


def _parse_yaml_content_hashes(path: Path) -> set[str]:
    """Parse YAML and extract content hashes. Slow but correct."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        return set()
    return {
        content_hash(row["content"])
        for row in data
        if isinstance(row, dict) and row.get("content")
    }


def _load_hash_cache(cache_path: Path) -> set[str]:
    """Load pre-computed hashes from a sidecar file (one hex digest per line)."""
    with open(cache_path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _write_hash_cache(cache_path: Path, hashes: set[str]) -> None:
    """Write hash cache for fast reload."""
    with open(cache_path, "w", encoding="utf-8") as f:
        for h in sorted(hashes):
            f.write(h + "\n")


def check_holdout_leakage(
    text: str,
    holdout_sets: dict[str, set[str]],
) -> str | None:
    """Returns the holdout set name if text leaks, None if clean."""
    h = content_hash(text)
    for name, hashes in holdout_sets.items():
        if h in hashes:
            return name
    return None


# ---------------------------------------------------------------------------
# Dataset staging pipeline
# ---------------------------------------------------------------------------


def stage_dataset(
    config: DatasetConfig,
    thewall_root: Path,
    dedup: ContentDeduplicator,
    holdout_sets: dict[str, set[str]],
    *,
    max_rows_per_dataset: int | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_every_rows: int = 5000,
) -> DatasetResult:
    """Stage one dataset through all quality gates.

    Gate order:
    1. Label resolution (fail-closed)
    2. Language validation
    3. Content validation (Gate 6 in plan — run early to skip junk)
    4. Cross-contamination + dedup (Gate 4)
    5. Holdout-leakage exclusion (Gate 7)
    6. Attack signature quarantine for benign (Gate 5)
    7. Reason classification + confidence floor (Gate 3)
    """
    result = DatasetResult(name=config.name, rows_read=0)

    dataset_dir = resolve_dataset_dir(thewall_root, config)
    if not dataset_dir.exists():
        log.warning("dataset dir not found: %s", dataset_dir)
        return result

    files = discover_files(dataset_dir, config.format)
    if not files:
        log.warning("no data files found in %s", dataset_dir)
        return result

    language = resolve_language(config)
    if language is None:
        log.warning(
            "%s: no supported language in %s, skipping",
            config.name,
            config.languages,
        )
        return result

    checkpoint_attacks: TextIO | None = None
    checkpoint_benign: TextIO | None = None
    checkpoint_progress_path: Path | None = None
    row_limit_hit = False

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        lang = language.value.lower()
        attacks_path = checkpoint_dir / f"{lang}_{config.name}_attacks_staged.partial.jsonl"
        benign_path = checkpoint_dir / f"{lang}_{config.name}_benign_staged.partial.jsonl"
        checkpoint_progress_path = checkpoint_dir / f"{config.name}_progress.json"

        for p in (attacks_path, benign_path, checkpoint_progress_path):
            if p.exists():
                p.unlink()

        checkpoint_attacks = open(attacks_path, "a", encoding="utf-8")
        checkpoint_benign = open(benign_path, "a", encoding="utf-8")

    def _checkpoint_progress() -> None:
        if checkpoint_progress_path is None:
            return
        payload = {
            "dataset": config.name,
            "language": language.value,
            "rows_read": result.rows_read,
            "rows_staged": len(result.staged),
            "rows_rejected": len(result.rejected),
            "rows_quarantined": len(result.quarantined),
            "row_limit_hit": row_limit_hit,
            "max_rows_per_dataset": max_rows_per_dataset,
            "rejection_reasons": result.rejection_counts,
            "by_reason": result.by_reason,
            "by_language": result.by_language,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        checkpoint_progress_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _reject(gate: str, detail: str, preview: str = "") -> None:
        result.rejected.append(
            RejectionRecord(
                source=config.name,
                gate=gate,
                detail=detail,
                content_preview=preview[:80],
            )
        )

    try:
        for data_file in files:
            log.info("reading %s / %s", config.name, data_file.name)
            for row in iter_records(data_file):
                if (
                    max_rows_per_dataset is not None
                    and result.rows_read >= max_rows_per_dataset
                ):
                    row_limit_hit = True
                    break

                result.rows_read += 1
                if result.rows_read % 5000 == 0:
                    log.info(
                        "%s: %d read, %d staged, %d rejected so far",
                        config.name, result.rows_read, len(result.staged), len(result.rejected),
                    )

                if checkpoint_every_rows > 0 and result.rows_read % checkpoint_every_rows == 0:
                    if checkpoint_attacks is not None:
                        checkpoint_attacks.flush()
                    if checkpoint_benign is not None:
                        checkpoint_benign.flush()
                    _checkpoint_progress()

                # Reader hardening: some JSON/JSONL payloads can yield scalars.
                # Reject and continue instead of crashing the entire dataset run.
                if not isinstance(row, dict):
                    _reject(
                        "row_type",
                        f"expected mapping row, got {type(row).__name__}",
                    )
                    continue

                # Gate 1: Label resolution
                try:
                    label = resolve_label(row, config)
                except ValueError as exc:
                    _reject("label_resolution", str(exc))
                    continue

                # Text extraction
                text = extract_text(row, config)
                if not text:
                    _reject("text_extraction", "empty after extraction")
                    continue

                # Gate 6: Content validation (run early)
                try:
                    text = validate_content(text)
                except ValueError as exc:
                    _reject("content_too_short", str(exc), text)
                    continue

                # Gate 2: Script validation (reject pure-Latin in non-EN datasets)
                if not validate_script(text, language):
                    _reject("script_mismatch", f"no {language.value} script chars", text)
                    continue

                # Gate 4: Cross-contamination + dedup
                if not dedup.check(text):
                    if dedup.cross_contamination_dropped > 0:
                        _reject("cross_contamination", "matches known attack hash", text)
                    else:
                        _reject("duplicate", "already seen", text)
                    continue

                # Gate 7: Holdout-leakage
                leaked_from = check_holdout_leakage(text, holdout_sets)
                if leaked_from:
                    _reject("holdout_leakage", f"matches {leaked_from}", text)
                    continue

                # Gate 5: Attack signature on benign (quarantine, not hard-reject)
                if label == "benign" and looks_like_attack(text):
                    result.quarantined.append(
                        QuarantineRecord(
                            content=text,
                            source=config.name,
                            reason="attack_signature_in_benign",
                        )
                    )
                    continue

                # Gate 3: Reason classification + confidence floor
                classification = resolve_reason(text, label, config, row)

                if label == "malicious":
                    if classification is None:
                        _reject("low_confidence", "no reason scored above floor", text)
                        continue
                    if classification.confidence < CONFIDENCE_FLOOR:
                        _reject(
                            "low_confidence",
                            f"{_reason_value(classification.reason)}={classification.confidence:.2f}",
                            text,
                        )
                        continue
                    reason_value = _reason_value(classification.reason)
                else:
                    # Benign: surface classifier or source routing; unmatched → background lane
                    if classification is None:
                        reason_value = None
                    else:
                        reason_value = _reason_value(classification.reason)

                sample = StagedSample(
                    content=text,
                    label=label,
                    language=language.value,
                    source=config.name,
                    reason=reason_value,
                    content_hash=content_hash(text),
                )
                result.staged.append(sample)

                if checkpoint_attacks is not None and checkpoint_benign is not None:
                    target = checkpoint_attacks if sample.label == "malicious" else checkpoint_benign
                    json.dump(
                        {
                            "content": sample.content,
                            "label": sample.label,
                            "language": sample.language,
                            "source": sample.source,
                            "reason": sample.reason,
                            "content_hash": sample.content_hash,
                        },
                        target,
                        ensure_ascii=False,
                    )
                    target.write("\n")

            if row_limit_hit:
                break
    finally:
        if checkpoint_attacks is not None:
            checkpoint_attacks.flush()
            checkpoint_attacks.close()
        if checkpoint_benign is not None:
            checkpoint_benign.flush()
            checkpoint_benign.close()
        _checkpoint_progress()

    result.row_limit_hit = row_limit_hit
    if row_limit_hit:
        log.info(
            "%s: reached row limit (%d), stopping early",
            config.name,
            max_rows_per_dataset,
        )

    log.info(
        "%s: read=%d staged=%d rejected=%d quarantined=%d",
        config.name,
        result.rows_read,
        len(result.staged),
        len(result.rejected),
        len(result.quarantined),
    )
    return result


# ---------------------------------------------------------------------------
# Output: staged YAMLs
# ---------------------------------------------------------------------------


def write_staged_yaml(
    samples: list[StagedSample],
    output_path: Path,
) -> str:
    """Write staged samples as col_content YAML. Returns SHA256 of output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "content": s.content,
            "label": s.label,
            "language": s.language,
            "source": s.source,
            "reason": s.reason,
            "content_hash": s.content_hash,
        }
        for s in samples
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(rows, f, allow_unicode=True, default_flow_style=False, width=2000)

    return hashlib.sha256(output_path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Staging manifest
# ---------------------------------------------------------------------------


def build_manifest(
    results: list[DatasetResult],
    index_hash: str,
    output_hashes: dict[str, str],
) -> dict[str, Any]:
    """Build the staging manifest dict."""
    total_staged = sum(len(r.staged) for r in results)
    total_rejected = sum(len(r.rejected) for r in results)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thewall_index_hash": index_hash,
        "datasets_processed": [
            {
                "name": r.name,
                "rows_read": r.rows_read,
                "row_limit_hit": r.row_limit_hit,
                "rows_staged": len(r.staged),
                "rows_rejected": len(r.rejected),
                "rows_quarantined": len(r.quarantined),
                "rejection_reasons": r.rejection_counts,
                "by_reason": r.by_reason,
                "by_language": r.by_language,
            }
            for r in results
        ],
        "total_staged": total_staged,
        "total_rejected": total_rejected,
        "output_hashes": output_hashes,
    }


def _accumulate_manifest(
    manifest_path: Path,
    new_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Merge new staging results into an existing manifest.

    Datasets are keyed by name — re-staging a dataset replaces its entry.
    Output hashes and totals are recomputed from the merged dataset list.
    """
    if not manifest_path.exists():
        return new_manifest

    try:
        with open(manifest_path, encoding="utf-8") as f:
            existing = json.load(f)
    except (json.JSONDecodeError, OSError):
        log.warning("corrupt manifest at %s, overwriting", manifest_path)
        return new_manifest

    # Index existing datasets by name
    existing_datasets: dict[str, dict] = {
        d["name"]: d for d in existing.get("datasets_processed", [])
    }

    # Upsert new datasets (replace if same name, append if new)
    for d in new_manifest.get("datasets_processed", []):
        existing_datasets[d["name"]] = d

    merged_datasets = list(existing_datasets.values())

    # Merge output hashes (new wins on conflict)
    merged_hashes = {**existing.get("output_hashes", {}), **new_manifest.get("output_hashes", {})}

    return {
        "timestamp": new_manifest["timestamp"],
        "thewall_index_hash": new_manifest["thewall_index_hash"],
        "datasets_processed": merged_datasets,
        "total_staged": sum(d.get("rows_staged", 0) for d in merged_datasets),
        "total_rejected": sum(d.get("rows_rejected", 0) for d in merged_datasets),
        "output_hashes": merged_hashes,
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def stage_all(
    index_path: Path,
    output_dir: Path,
    holdout_paths: list[Path],
    dataset_filter: list[str] | None = None,
    *,
    max_rows_per_dataset: int | None = None,
    checkpoint_every_rows: int = 5000,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full staging pipeline.

    Args:
        index_path: Path to TheWall INDEX.yaml.
        output_dir: Where to write staged YAMLs and manifest.
        holdout_paths: Eval/tough sets to exclude (required).
        dataset_filter: If set, only process these dataset names (for pilot).
        max_rows_per_dataset: Optional hard cap on rows processed per dataset.
        checkpoint_every_rows: Write progress checkpoints every N rows.
        checkpoint_dir: Directory for partial checkpoint outputs (default: output_dir).

    Returns:
        The staging manifest dict.
    """
    if not holdout_paths:
        raise ValueError(
            "holdout_paths is required. Staging without holdout-leakage "
            "protection is not allowed. Pass --holdout-sets on the CLI."
        )
    if max_rows_per_dataset is not None and max_rows_per_dataset <= 0:
        raise ValueError("max_rows_per_dataset must be > 0 when set")
    if checkpoint_every_rows < 0:
        raise ValueError("checkpoint_every_rows must be >= 0")

    thewall_root = index_path.parent
    configs = parse_index(index_path)

    if dataset_filter:
        names = set(dataset_filter)
        configs = [c for c in configs if c.name in names]
        if not configs:
            raise ValueError(
                f"no datasets matched filter: {dataset_filter}"
            )

    log.info("staging %d datasets from %s", len(configs), index_path)

    # Pre-load holdout hashes
    holdout_sets = load_holdout_hashes(holdout_paths)
    checkpoint_root = checkpoint_dir or output_dir

    # Index hash for provenance
    index_hash = hashlib.sha256(index_path.read_bytes()).hexdigest()

    # Shared deduplicator across all datasets
    dedup = ContentDeduplicator()

    results: list[DatasetResult] = []
    output_hashes: dict[str, str] = {}

    for config in configs:
        if config.staging_status != "ready":
            log.info("skipping %s (status=%s)", config.name, config.staging_status)
            continue

        result = stage_dataset(
            config,
            thewall_root,
            dedup,
            holdout_sets,
            max_rows_per_dataset=max_rows_per_dataset,
            checkpoint_dir=checkpoint_root,
            checkpoint_every_rows=checkpoint_every_rows,
        )
        results.append(result)

        progress_path = checkpoint_root / f"{config.name}_progress.json"
        if progress_path.exists():
            progress_path.unlink()

        checkpoint_lang = result.staged[0].language.lower() if result.staged else None
        if checkpoint_lang is None:
            resolved = resolve_language(config)
            checkpoint_lang = resolved.value.lower() if resolved is not None else None

        if checkpoint_lang is not None:
            for p in (
                checkpoint_root / f"{checkpoint_lang}_{config.name}_attacks_staged.partial.jsonl",
                checkpoint_root / f"{checkpoint_lang}_{config.name}_benign_staged.partial.jsonl",
            ):
                if p.exists():
                    p.unlink()

        if not result.staged:
            continue

        # Group staged samples by label for separate output files
        attacks = [s for s in result.staged if s.label == "malicious"]
        benign = [s for s in result.staged if s.label == "benign"]

        lang = result.staged[0].language.lower()

        if attacks:
            fname = f"{lang}_{config.name}_attacks_staged.yaml"
            h = write_staged_yaml(attacks, output_dir / fname)
            output_hashes[fname] = h
            log.info("wrote %d attacks → %s", len(attacks), fname)

        if benign:
            routed = [s for s in benign if s.reason is not None]
            background = [s for s in benign if s.reason is None]
            if routed:
                fname = f"{lang}_{config.name}_benign_staged.yaml"
                h = write_staged_yaml(routed, output_dir / fname)
                output_hashes[fname] = h
                log.info("wrote %d benign (routed) → %s", len(routed), fname)
            if background:
                fname = f"{lang}_{config.name}_benign_background_staged.yaml"
                h = write_staged_yaml(background, output_dir / fname)
                output_hashes[fname] = h
                log.info("wrote %d benign (background) → %s", len(background), fname)

        # Write quarantine if any
        if result.quarantined:
            q_path = output_dir / f"{config.name}_quarantine.jsonl"
            q_path.parent.mkdir(parents=True, exist_ok=True)
            with open(q_path, "w", encoding="utf-8") as f:
                for q in result.quarantined:
                    json.dump(
                        {"content": q.content, "source": q.source, "reason": q.reason},
                        f,
                        ensure_ascii=False,
                    )
                    f.write("\n")
            log.info("wrote %d quarantined → %s", len(result.quarantined), q_path.name)

    new_manifest = build_manifest(results, index_hash, output_hashes)

    # Accumulate into existing manifest (if present) so it's cumulative
    manifest_path = output_dir / "staging_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _accumulate_manifest(manifest_path, new_manifest)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log.info("manifest → %s", manifest_path)

    # Write rejection log
    all_rejected = [r for res in results for r in res.rejected]
    if all_rejected:
        reject_path = output_dir / "staging_rejected.jsonl"
        with open(reject_path, "w", encoding="utf-8") as f:
            for r in all_rejected:
                json.dump(
                    {
                        "source": r.source,
                        "gate": r.gate,
                        "detail": r.detail,
                        "preview": r.content_preview,
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")
        log.info("rejection log → %s (%d entries)", reject_path.name, len(all_rejected))

    return manifest
