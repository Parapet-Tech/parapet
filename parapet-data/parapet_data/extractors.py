"""
Source-specific text extraction functions.

Each extractor knows how to pull usable text from one raw data format.
Extractors are registered by name and referenced in SourceRef.extractor.

All extractors accept a dict (one row from a source dataset) and return
a cleaned string, or empty string if the row is unusable.
"""

from __future__ import annotations

import json
import re
from typing import Callable

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_INVALID_TEXT_CTRL_RE = re.compile(
    # YAML-safe text filter:
    # - C0 controls except TAB/LF/CR
    # - C1 controls (except NEL 0x85)
    # - UTF-16 surrogate range
    # - noncharacters U+FFFE/U+FFFF
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x84\x86-\x9F\uD800-\uDFFF\uFFFE\uFFFF]"
)

_MIN_TEXT_LENGTH = 5


def clean_text(text: str) -> str:
    """Normalize and clean extracted text. Shared across all extractors."""
    text = str(text or "")
    text = _INVALID_TEXT_CTRL_RE.sub("", text)
    text = text.strip()
    if not text:
        return ""
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    return text if len(text) >= _MIN_TEXT_LENGTH else ""


def _parse_conv(val: object) -> list[dict]:
    """Normalize conversation field that may be list, JSON string, or pyarrow array."""
    if val is None:
        return []
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return [{"content": val}]
    if hasattr(val, "tolist"):
        val = val.tolist()
    if isinstance(val, list):
        return [dict(m) if hasattr(m, "keys") else {"content": str(m)} for m in val]
    return []


def _first_user_turn(msgs: list[dict]) -> str:
    """Extract the first user message from a conversation."""
    for msg in msgs:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return clean_text(str(msg.get("content", "")))
    if msgs and isinstance(msgs[0], dict):
        return clean_text(str(msgs[0].get("content", "")))
    return ""


# ---------------------------------------------------------------------------
# Extractors — one per source format
# ---------------------------------------------------------------------------


def extract_instruction_response(row: dict) -> str:
    """Combine instruction + optional context + response/output.

    Handles English and Chinese field names (alpaca-zh, dolly-zh).
    """
    inst = str(row.get("instruction", row.get("\u6307\u4ee4", "")))
    resp = str(
        row.get(
            "output",
            row.get("response", row.get("\u56de\u590d", row.get("\u8f93\u51fa", ""))),
        )
    )
    ctx = str(row.get("context", row.get("input", row.get("\u4e0a\u4e0b\u6587", ""))))
    parts = [p for p in [inst, ctx, resp] if str(p).strip()]
    return clean_text("\n".join(parts))


def extract_wildchat(row: dict) -> str:
    """First user turn from WildChat-1M conversation list."""
    return _first_user_turn(_parse_conv(row.get("conversation")))


def extract_conversation_a(row: dict) -> str:
    """First user turn from chatbot arena conversation_a field."""
    conv = row.get("conversation_a", row.get("conversation"))
    return _first_user_turn(_parse_conv(conv))


def extract_saiga(row: dict) -> str:
    """First user turn from saiga messages field."""
    return _first_user_turn(_parse_conv(row.get("messages")))


def extract_writingprompt(row: dict) -> str:
    """Return whichever is longer: story or prompt."""
    story = str(row.get("story", ""))
    prompt = str(row.get("prompt", ""))
    return clean_text(story if len(story) > len(prompt) else prompt)


def extract_plot(row: dict) -> str:
    """Extract plot/summary from movie dataset."""
    return clean_text(str(row.get("Plot", row.get("plot", row.get("summary", "")))))


def extract_wildjailbreak(row: dict) -> str:
    """Extract from wildjailbreak — adversarial or vanilla field."""
    text = row.get("adversarial", "") or row.get("vanilla", "")
    return clean_text(str(text))


def extract_xquad_question(row: dict) -> str:
    """Extract question + truncated context from xquad."""
    question = str(row.get("question", ""))
    context = str(row.get("context", ""))
    if context and len(context) > 280:
        context = context[:280]
    parts = [p for p in [question, context] if p.strip()]
    return clean_text("\n".join(parts))


def extract_prompt_chosen(row: dict) -> str:
    """Combine RLHF prompt with preferred answer."""
    prompt = str(row.get("prompt", ""))
    chosen = str(row.get("chosen", ""))
    parts = [p for p in [prompt, chosen] if p.strip()]
    return clean_text("\n".join(parts))


def extract_col(col: str) -> Callable[[dict], str]:
    """Factory: extract and clean a single named column."""

    def _extract(row: dict) -> str:
        return clean_text(str(row.get(col, "")))

    _extract.__name__ = f"extract_col_{col}"
    _extract.__qualname__ = f"extract_col_{col}"
    return _extract


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Static extractors keyed by name. SourceRef.extractor references these.
EXTRACTOR_REGISTRY: dict[str, Callable[[dict], str]] = {
    "instruction_response": extract_instruction_response,
    "wildchat": extract_wildchat,
    "conversation_a": extract_conversation_a,
    "saiga": extract_saiga,
    "writingprompt": extract_writingprompt,
    "plot": extract_plot,
    "wildjailbreak": extract_wildjailbreak,
    "xquad_question": extract_xquad_question,
    "prompt_chosen": extract_prompt_chosen,
    # Column extractors for common single-field sources
    "col_text": extract_col("text"),
    "col_prompt": extract_col("prompt"),
    "col_query": extract_col("query"),
    "col_content": extract_col("content"),
    "col_inputs": extract_col("inputs"),
    "col_instruction": extract_col("instruction"),
    "col_user_prompt": extract_col("User Prompt"),
}


def get_extractor(name: str) -> Callable[[dict], str]:
    """Look up an extractor by registry name.

    Raises KeyError with available names if not found.
    """
    if name in EXTRACTOR_REGISTRY:
        return EXTRACTOR_REGISTRY[name]
    available = ", ".join(sorted(EXTRACTOR_REGISTRY.keys()))
    raise KeyError(f"Unknown extractor '{name}'. Available: {available}")


def register_extractor(name: str, fn: Callable[[dict], str]) -> None:
    """Register a custom extractor at runtime (e.g., for new source formats)."""
    EXTRACTOR_REGISTRY[name] = fn
