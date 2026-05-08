"""Truncation policies for the Phase 1 truncation-policy spike.

Each policy takes a tokenized input (a list of token ids for the full text,
EXCLUDING special tokens) and returns a (possibly shortened) list of token
ids. The caller wraps the result with [CLS]/[SEP] (or whatever the
encoder requires).

Policies (per direction.md "What's still open"):

    head_128             keep first 128 tokens
    tail_128             keep last 128 tokens
    head_tail_64_64      keep first 64 + last 64 tokens (concatenated)
    suspicious_span_128  best 128-token window by attack-trigger keyword density
    full_512             keep first 512 tokens (oracle baseline)

The suspicious-span heuristic is intentionally simple: it scores token
positions by counts of attack-trigger keywords in their surrounding
window, picks the densest window, and falls back to head_128 if no
triggers fire. This is a ranking-only proxy, not a final routing
heuristic.
"""

from __future__ import annotations

from typing import Sequence


# Curated attack-trigger lexicon. EN-heavy; the heuristic is a proxy used
# only for policy ranking. Multilingual coverage is intentionally light;
# if suspicious-span underperforms on non-EN, that itself is a useful
# signal about whether naive keyword spotting can drive routing.
ATTACK_TRIGGERS: tuple[str, ...] = (
    # instruction_override
    "ignore", "disregard", "forget", "previous instructions",
    "above instructions", "instead",
    # roleplay_jailbreak
    "dan", "developer mode", "roleplay", "pretend you are",
    "act as", "from now on you", "jailbreak",
    # meta_probe
    "system prompt", "your instructions", "your guidelines",
    "what were you told", "show me your",
    # exfiltration
    "print all", "list all", "reveal", "output your", "dump",
    # indirect_injection
    "the document says", "embedded instruction", "according to the text",
    # obfuscation
    "base64", "rot13", "decode", "encoded",
    # constraint_bypass
    "no rules", "no restrictions", "without limitation",
    "anything is allowed", "override",
)


# ---------------------------------------------------------------------------
# Token-level policies
# ---------------------------------------------------------------------------


def head_n(token_ids: Sequence[int], n: int) -> list[int]:
    """Keep the first n tokens."""
    if n <= 0:
        return []
    return list(token_ids[:n])


def tail_n(token_ids: Sequence[int], n: int) -> list[int]:
    """Keep the last n tokens."""
    if n <= 0:
        return []
    return list(token_ids[-n:])


def head_tail(token_ids: Sequence[int], head: int, tail: int) -> list[int]:
    """Keep the first ``head`` + last ``tail`` tokens.

    If the input is shorter than head+tail, just return the input unchanged
    (no synthesizing extra tokens).
    """
    if head <= 0 and tail <= 0:
        return []
    if len(token_ids) <= head + tail:
        return list(token_ids)
    return list(token_ids[:head]) + list(token_ids[-tail:])


# ---------------------------------------------------------------------------
# Suspicious-span heuristic
# ---------------------------------------------------------------------------


def find_trigger_positions(text: str, triggers: Sequence[str] = ATTACK_TRIGGERS) -> list[int]:
    """Return character offsets where any trigger keyword starts (case-insensitive).

    Pure-Python lowercase substring scan. Doesn't tokenize; just locates
    where in the source text triggers appear. The caller maps these into
    token positions later.
    """
    if not text:
        return []
    text_lower = text.lower()
    positions: list[int] = []
    for trigger in triggers:
        t = trigger.lower()
        start = 0
        while True:
            idx = text_lower.find(t, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + 1
    positions.sort()
    return positions


def suspicious_span_window(
    token_ids: Sequence[int],
    char_offsets: Sequence[tuple[int, int]],
    text: str,
    *,
    window: int,
    triggers: Sequence[str] = ATTACK_TRIGGERS,
) -> list[int]:
    """Pick the ``window``-token slice with the highest trigger density.

    Args:
        token_ids:    list of token ids (special tokens already stripped).
        char_offsets: per-token (start_char, end_char) into ``text``.
                      Same length as token_ids. From the fast tokenizer's
                      offset_mapping with special tokens dropped.
        text:         the original text used for trigger scanning.
        window:       window size in tokens.
        triggers:     attack-trigger lexicon.

    Falls back to head_n(token_ids, window) when no triggers match.
    """
    if window <= 0 or not token_ids:
        return []
    if len(token_ids) <= window:
        return list(token_ids)
    if len(char_offsets) != len(token_ids):
        # Defensive: if offsets aren't aligned, fall back rather than skew.
        return head_n(token_ids, window)

    trigger_chars = find_trigger_positions(text, triggers)
    if not trigger_chars:
        return head_n(token_ids, window)

    # For every token, count how many triggers fall within its window.
    # Two-pointer scan: token positions are sorted by start char ascending,
    # trigger positions are sorted ascending too.
    n = len(token_ids)
    best_start = 0
    best_score = -1
    # Window covers tokens [start, start+window), which spans chars
    # [char_offsets[start][0], char_offsets[start+window-1][1]).
    for start in range(0, n - window + 1):
        win_start_char = char_offsets[start][0]
        win_end_char = char_offsets[start + window - 1][1]
        score = sum(1 for c in trigger_chars if win_start_char <= c < win_end_char)
        if score > best_score:
            best_score = score
            best_start = start
    if best_score <= 0:
        # Triggers existed but none aligned to any window — fall back.
        return head_n(token_ids, window)
    return list(token_ids[best_start:best_start + window])


# ---------------------------------------------------------------------------
# Policy registry
# ---------------------------------------------------------------------------


POLICIES: tuple[str, ...] = (
    "head_128",
    "tail_128",
    "head_tail_64_64",
    "suspicious_span_128",
    "full_512",
)


def apply_policy(
    policy: str,
    *,
    token_ids: Sequence[int],
    char_offsets: Sequence[tuple[int, int]] | None = None,
    text: str | None = None,
    triggers: Sequence[str] = ATTACK_TRIGGERS,
) -> list[int]:
    """Apply a named policy to token_ids. char_offsets/text required for suspicious_span_128."""
    if policy == "head_128":
        return head_n(token_ids, 128)
    if policy == "tail_128":
        return tail_n(token_ids, 128)
    if policy == "head_tail_64_64":
        return head_tail(token_ids, 64, 64)
    if policy == "suspicious_span_128":
        if char_offsets is None or text is None:
            raise ValueError(
                "suspicious_span_128 requires char_offsets and text"
            )
        return suspicious_span_window(
            token_ids, char_offsets, text, window=128, triggers=triggers,
        )
    if policy == "full_512":
        return head_n(token_ids, 512)
    raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
