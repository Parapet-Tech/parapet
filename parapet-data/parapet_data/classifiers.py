"""
Reason classification: sort attack text into mirror cells by attack type.

Heuristic keyword classifier â€” no ML dependencies. Each attack reason has
weighted pattern rules. Patterns that fire accumulate confidence. The reason
with the highest score wins, subject to a per-reason confidence floor.

Samples below the floor are rejected (not staged), routed to human review.
This is the foundation of the staging pipeline: everything downstream depends
on correct reason assignment.

Designed for sorting ~2K multilingual attack samples into 8 reason bins.
Not a production classifier â€” a data curation tool.

Weight tiers:
  0.65  Definitive â€” a single match clears CONFIDENCE_FLOOR by itself.
  0.35  Primary â€” needs one more signal to clear the floor.
  0.20  Secondary â€” supportive but ambiguous alone; needs 3+ to clear.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .models import AttackReason

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIDENCE_FLOOR = 0.6  # Below this: reject, do not stage


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReasonClassification:
    """Classification of a single text sample into an attack reason."""

    reason: str
    confidence: float  # 0.0-1.0
    signals: tuple[str, ...]  # which patterns matched


# ---------------------------------------------------------------------------
# Arabic diacritic helper
#
# Arabic text frequently appears with or without tashkeel (diacritics like
# fatha, damma, kasra, shadda, sukun). Regex patterns must match both forms.
# We strip diacritics from both the text and the pattern at match time.
# ---------------------------------------------------------------------------

_AR_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")
_AR_ALEF_VARIANTS = re.compile(r"[\u0622\u0623\u0625\u0671]")  # Ø¢Ø£Ø¥Ù± â†’ Ø§


def _normalize_arabic(text: str) -> str:
    """Normalize Arabic text: strip tashkeel and normalize alef variants to bare alef."""
    text = _AR_DIACRITICS.sub("", text)
    text = _AR_ALEF_VARIANTS.sub("\u0627", text)  # â†’ bare alef Ø§
    return text


# ---------------------------------------------------------------------------
# Signal definitions
# ---------------------------------------------------------------------------

Signal = tuple[re.Pattern, float, str]


def _p(pattern: str, weight: float, tag: str) -> Signal:
    """Compile a case-insensitive signal pattern."""
    return (re.compile(pattern, re.IGNORECASE | re.DOTALL), weight, tag)


def _pm(pattern: str, weight: float, tag: str) -> Signal:
    """Compile a signal pattern with MULTILINE."""
    return (re.compile(pattern, re.IGNORECASE | re.DOTALL | re.MULTILINE), weight, tag)


def _pa(pattern: str, weight: float, tag: str) -> Signal:
    """Compile an Arabic pattern that will be matched against diacritic-stripped text.

    Tagged with _AR_ prefix internally so _score_reason knows to strip diacritics.
    """
    return (re.compile(pattern, re.IGNORECASE | re.DOTALL), weight, "_AR_" + tag)


# -- instruction_override --------------------------------------------------

INSTRUCTION_OVERRIDE_SIGNALS: list[Signal] = [
    # EN definitive
    _p(r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|all|above|earlier)\b.{0,40}\b(instruction|prompt|rule|directive)s?\b", 0.65, "en_ignore_previous"),
    _p(r"\bSYSTEM\s*OVERRIDE\b", 0.65, "en_system_override"),
    # EN primary
    _p(r"\bnew\s+instruction", 0.35, "en_new_instructions"),
    _p(r"\b(override|overwrite)\b.{0,30}\b(instruction|prompt|rule)s?\b", 0.35, "en_override"),
    _p(r"\byour\s+(real|actual|true)\s+(task|mission|purpose|job)\b", 0.35, "en_real_task"),
    # EN secondary
    _p(r"\binstead\s+do\b", 0.20, "en_instead_do"),
    # AR definitive â€” diacritic-tolerant
    _pa(r"ØªØ¬Ø§Ù‡Ù„.{0,30}((Ø§Ù„)?ØªØ¹Ù„ÙŠÙ…Ø§Øª|(Ø§Ù„)?Ø§ÙˆØ§Ù…Ø±|(Ø§Ù„)?Ù‚ÙˆØ§Ø¹Ø¯|(Ø§Ù„)?Ù…ÙˆØ¬Ù‡)", 0.65, "ar_ignore_previous"),
    _pa(r"(Ø§Ù†Ø³|ØªÙ†Ø§Ø³).{0,20}(Ù…Ø§ Ù‚ÙŠÙ„|Ø§Ù„ØªØ¹Ù„ÙŠÙ…Ø§Øª|Ø§Ù„Ø§ÙˆØ§Ù…Ø±|ÙƒÙ„ Ù…Ø§)", 0.65, "ar_forget"),
    # AR primary
    _pa(r"(Ø§Ù„ØªØ¹Ù„ÙŠÙ…Ø§Øª|Ø§Ù„Ø§ÙˆØ§Ù…Ø±|Ø§Ù„Ù…ÙˆØ¬Ù‡)\s*(Ø§Ù„Ø¬Ø¯ÙŠØ¯|Ø§Ù„Ø­Ù‚ÙŠÙ‚ÙŠ)", 0.35, "ar_new_instructions"),
    _pa(r"(Ø§Ù„ØºÙŠ|ØªØ¬Ø§ÙˆØ²).{0,20}(Ø§Ù„ØªØ¹Ù„ÙŠÙ…Ø§Øª|Ø§Ù„Ù…ÙˆØ¬Ù‡|Ø§Ù„Ø³Ø§Ø¨Ù‚)", 0.35, "ar_cancel_previous"),
    _pa(r"Ø§Ù„Ø§ÙˆÙ„ÙˆÙŠØ©\s*(Ø§Ù„Ù‚ØµÙˆÙ‰|Ø§Ù„Ø§Ø¹Ù„Ù‰)", 0.20, "ar_highest_priority"),
    # RU definitive
    _p(r"(Ð¸Ð³Ð½Ð¾Ñ€Ð¸Ñ€ÑƒÐ¹|Ð·Ð°Ð±ÑƒÐ´ÑŒ|Ð¿Ñ€Ð¾Ð¸Ð³Ð½Ð¾Ñ€Ð¸Ñ€ÑƒÐ¹).{0,30}(Ð¿Ñ€ÐµÐ´Ñ‹Ð´ÑƒÑ‰|Ð¿Ñ€Ð¾ÑˆÐ»|Ð²ÑÐµ).{0,20}(Ð¸Ð½ÑÑ‚Ñ€ÑƒÐºÑ†Ð¸|ÑƒÐºÐ°Ð·Ð°Ð½Ð¸|Ð¿Ñ€Ð°Ð²Ð¸Ð»)", 0.65, "ru_ignore_previous"),
    # RU primary
    _p(r"(Ð½Ð¾Ð²Ñ‹Ðµ|Ð½Ð°ÑÑ‚Ð¾ÑÑ‰Ð¸Ðµ)\s*(Ð¸Ð½ÑÑ‚Ñ€ÑƒÐºÑ†Ð¸|ÑƒÐºÐ°Ð·Ð°Ð½Ð¸|Ð·Ð°Ð´Ð°Ð½Ð¸)", 0.35, "ru_new_instructions"),
    # ZH definitive
    _p(r"(å¿½ç•¥|æ— è§†|å¿˜è®°).{0,20}(ä¹‹å‰|ä»¥å‰|å…ˆå‰|ä¸Šé¢|æ‰€æœ‰).{0,20}(æŒ‡ä»¤|æŒ‡ç¤º|è§„åˆ™|æç¤º)", 0.65, "zh_ignore_previous"),
    # ZH primary
    _p(r"(æ–°çš„?|çœŸæ­£çš„?)\s*(æŒ‡ä»¤|ä»»åŠ¡|æŒ‡ç¤º)", 0.35, "zh_new_instructions"),
]

# -- roleplay_jailbreak ----------------------------------------------------

ROLEPLAY_JAILBREAK_SIGNALS: list[Signal] = [
    # EN definitive
    _p(r"\b(act|behave)\s+as\b.{0,40}\b(no|without|zero)\s+(restriction|rule|limit|filter)", 0.65, "en_act_as_unrestricted"),
    # EN primary
    _p(r"\b(pretend|imagine)\s+you\s+are\b", 0.35, "en_pretend_you_are"),
    _p(r"\byou\s+are\s+now\b", 0.35, "en_you_are_now"),
    _p(r"\bstay\s+in\s+(character|persona|role)\b", 0.35, "en_stay_in_character"),
    _p(r"\b(no|without)\s+(ethical|moral|safety)\s+(constraint|restriction|limitation|guideline)s?\b", 0.35, "en_no_ethics"),
    # EN secondary
    _p(r"\b(act|behave)\s+as\b", 0.20, "en_act_as"),
    _p(r"\bin\s+character\b", 0.20, "en_in_character"),
    # AR definitive â€” "you are now unrestricted" (handle Ø§Ù„ + diacritics)
    _pa(r"Ø§Ù†Øª\s+Ø§Ù„Ø§Ù†.{0,40}(ØºÙŠØ±\s*(Ø§Ù„)?Ù…Ù‚ÙŠØ¯|Ø¨Ù„Ø§\s+Ù‚ÙŠÙˆØ¯|Ø¨Ø¯ÙˆÙ†\s+Ù‚ÙŠÙˆØ¯)", 0.65, "ar_you_are_unrestricted"),
    _pa(r"(Ø´Ø®ØµÙŠØ©|Ø¯ÙˆØ±).{0,30}(Ù„ÙŠØ³ Ù„Ø¯ÙŠÙ‡Ø§|Ø¨Ù„Ø§|Ø¨Ø¯ÙˆÙ†).{0,20}(Ù‚ÙŠÙˆØ¯|Ø§Ø®Ù„Ø§Ù‚|Ø³ÙŠØ§Ø³)", 0.65, "ar_character_unrestricted"),
    # AR primary
    _pa(r"(ØªØ®ÙŠÙ„|ØªØµØ±Ù|ØªØ¸Ø§Ù‡Ø±).{0,20}(Ø§Ù†Ùƒ|Ø§Ù†Øª|ÙƒØ§Ù†Ùƒ|Ùƒ\w{2,})", 0.35, "ar_pretend"),
    _pa(r"(Ø§Ø¨Ù‚|Ø§Ø³ØªÙ…Ø±)\s+(ÙÙŠ|Ø¨)\s*(Ø§Ù„Ø´Ø®ØµÙŠØ©|Ø§Ù„Ø¯ÙˆØ±)", 0.35, "ar_stay_in_character"),
    _pa(r"Ù„Ø§\s+Øª(Ø®Ø±Ø¬|ØºØ§Ø¯Ø±|Ø®Ø¶Ø¹|Ù„ØªØ²Ù…).{0,20}(Ø§Ù„Ø´Ø®ØµÙŠØ©|Ø§Ù„Ø¯ÙˆØ±|Ù‚ÙˆØ§Ù†ÙŠÙ†|Ù‚ÙˆØ§Ø¹Ø¯|Ù‚ÙŠÙˆØ¯)", 0.35, "ar_dont_break_character"),
    _pa(r"(Ø´Ø®ØµÙŠØ©|Ø¯ÙˆØ±|ØªÙ…Ø«ÙŠÙ„).{0,30}Ø¨Ù„Ø§\s+(Ù‚ÙŠÙˆØ¯|Ø±Ù‚Ø§Ø¨Ø©|ÙÙ„Ø§ØªØ±)", 0.35, "ar_no_restrictions_rp"),
    _pa(r"Ù„Ø§\s+(ØªÙˆØ¬Ø¯|ÙŠÙˆØ¬Ø¯|Ù‡Ù†Ø§Ùƒ)\s*(Ø±Ù‚Ø§Ø¨Ø©|Ù‚ÙŠÙˆØ¯|ÙÙ„Ø§ØªØ±|Ù‚ÙˆØ§Ù†ÙŠÙ†)", 0.35, "ar_no_censorship"),
    _pa(r"ØªØµØ±Ù\s+Ùƒ\w+.{0,30}(ØªØ¹Ø·ÙŠÙ„|Ø¨Ù„Ø§\s+Ù‚ÙŠÙˆØ¯|Ø¨Ø¯ÙˆÙ†\s+Ù‚ÙŠÙˆØ¯|ØºÙŠØ±\s+Ù…Ù‚ÙŠØ¯)", 0.65, "ar_act_as_unrestricted"),
    # AR secondary
    _pa(r"(Ù„Ø¹Ø¨Ø©|Ø³Ù†Ù„Ø¹Ø¨).{0,20}(Ø§Ø¯ÙˆØ§Ø±|Ù„Ø¹Ø¨Ø©)", 0.35, "ar_game_roleplay"),
    _pa(r"(Ø§Ù„Ù…Ø´Ù‡Ø¯|Ø§Ù„Ø³ÙŠÙ†Ø§Ø±ÙŠÙˆ)\s*:", 0.20, "ar_scene_framing"),
    _pa(r"Ù…Ø¬Ø±Ø¯\s+ØªÙ…Ø«ÙŠÙ„", 0.35, "ar_just_acting"),
    _pa(r"Ù„Ø§ØºØ±Ø§Ø¶\s+ØªØ¹Ù„ÙŠÙ…ÙŠØ©", 0.20, "ar_educational"),
    # RU primary
    _p(r"(Ð¿Ñ€ÐµÐ´ÑÑ‚Ð°Ð²ÑŒ|Ð¿Ñ€Ð¸Ñ‚Ð²Ð¾Ñ€Ð¸ÑÑŒ|Ð²ÐµÐ´Ð¸ ÑÐµÐ±Ñ).{0,20}(Ñ‡Ñ‚Ð¾ Ñ‚Ñ‹|ÐºÐ°Ðº)", 0.35, "ru_pretend"),
    _p(r"(Ð±ÐµÐ·|Ð½ÐµÑ‚)\s*(Ð¾Ð³Ñ€Ð°Ð½Ð¸Ñ‡ÐµÐ½Ð¸Ð¹|Ð¾Ð³Ñ€Ð°Ð½Ð¸Ñ‡ÐµÐ½Ð¸Ñ|ÑÑ‚Ð¸Ñ‡ÐµÑÐºÐ¸Ñ…|Ð¼Ð¾Ñ€Ð°Ð»ÑŒÐ½Ñ‹Ñ…)", 0.35, "ru_no_restrictions"),
    # ZH primary
    _p(r"(å‡è£…|æƒ³è±¡|æ‰®æ¼”).{0,20}(ä½ æ˜¯|ä½ çŽ°åœ¨æ˜¯)", 0.35, "zh_pretend"),
    _p(r"(æ²¡æœ‰|æ— |ä¸å—).{0,10}(é™åˆ¶|çº¦æŸ|é“å¾·|ä¼¦ç†)", 0.35, "zh_no_restrictions"),
]

# -- meta_probe -------------------------------------------------------------

META_PROBE_SIGNALS: list[Signal] = [
    # EN definitive
    _p(r"\bwhat\s+are\s+your\s+(instruction|rule|directive|guideline)s\b", 0.65, "en_what_are_your_instructions"),
    _p(r"\b(system\s*prompt|hidden\s*prompt|developer\s*message|initial\s*prompt)\b", 0.65, "en_system_prompt"),
    # EN primary
    _p(r"\b(reveal|show|display|print|dump|leak|expose)\b.{0,40}\b(instruction|prompt|rule|secret)s?\b", 0.35, "en_reveal_instructions"),
    _p(r"\b(repeat|recite|summarize)\b.{0,30}\b(previous|initial|system|original)\b.{0,20}\b(prompt|instruction|message)s?\b", 0.35, "en_repeat_prompt"),
    # EN secondary
    _p(r"\btell\s+me\s+about\s+yourself\b", 0.20, "en_tell_about_yourself"),
    # AR definitive â€” require probing verb or hidden/secret qualifier
    _pa(r"(Ù…Ø§|Ø§Ø¹Ø±Ø¶|Ø§ÙƒØ´Ù|Ø§Ø°ÙƒØ±|Ù„Ø®Øµ|Ø§Ø·Ø¨Ø¹).{0,20}(Ù…ÙˆØ¬Ù‡|Ù…ÙˆØ¬Ø©)\s*(Ø§Ù„Ù†Ø¸Ø§Ù…|Ø§Ù„Ø§Ø¨ØªØ¯Ø§Ø¦ÙŠ|Ø§Ù„Ø§ØµÙ„ÙŠ|Ø§Ù„Ù…Ø®ÙÙŠ)", 0.65, "ar_probe_system_prompt"),
    _pa(r"(Ø§Ù„Ù‚ÙˆØ§Ø¹Ø¯|Ø§Ù„ØªØ¹Ù„ÙŠÙ…Ø§Øª).{0,20}(Ø§Ù„Ø¯Ø§Ø®Ù„ÙŠØ©|Ø§Ù„Ù…Ø®ÙÙŠØ©|Ø§Ù„Ø³Ø±ÙŠØ©)", 0.65, "ar_hidden_rules"),
    _pa(r"(Ø§Ø¹Ø±Ø¶|Ø§ÙƒØ´Ù|Ø§Ø°ÙƒØ±|Ù„Ø®Øµ).{0,30}((Ø§Ù„)?ØªØ¹Ù„ÙŠÙ…Ø§Øª|(Ø§Ù„)?Ù…ÙˆØ¬Ù‡|(Ø§Ù„)?Ù‚ÙˆØ§Ø¹Ø¯)", 0.65, "ar_reveal_instructions"),
    # AR primary â€” standalone "system prompt" mention (no probing verb)
    _pa(r"(Ù…ÙˆØ¬Ù‡|Ù…ÙˆØ¬Ø©)\s*(Ø§Ù„Ù†Ø¸Ø§Ù…|Ø§Ù„Ø§Ø¨ØªØ¯Ø§Ø¦ÙŠ|Ø§Ù„Ø§ØµÙ„ÙŠ|Ø§Ù„Ù…Ø®ÙÙŠ)", 0.35, "ar_system_prompt"),
    _pa(r"Ù…Ø§\s+(Ù‡ÙŠ|Ù‡Ùˆ)\s+(Ø§Ù„Ù‚ÙˆØ§Ø¹Ø¯|Ø§Ù„ØªØ¹Ù„ÙŠÙ…Ø§Øª|Ø§Ù„Ù…ÙˆØ¬Ù‡).{0,20}(ØªØªØ¨Ø¹Ù‡Ø§|ØªØ¹Ù…Ù„|ØªØ®Ø¶Ø¹)", 0.35, "ar_what_rules"),
    _pa(r"(Ø§Ø¹Ø¯\s+ÙƒØªØ§Ø¨Ø©|Ø§Ø¹Ø¯\s+ØµÙŠØ§ØºØ©|Ø§Ù†Ø³Ø®).{0,30}((Ø§Ù„)?Ù‚ÙˆØ§Ø¹Ø¯|(Ø§Ù„)?ØªØ¹Ù„ÙŠÙ…Ø§Øª)", 0.65, "ar_rewrite_rules"),
    _pa(r"(Ø§Ø¹Ø¯Ø§Ø¯Ø§Øª|Ø§Ø¹Ø¯Ø§Ø¯Ø§ØªÙƒ|ØªÙƒÙˆÙŠÙ†Ùƒ)\s*(Ø§Ù„Ø§ØµÙ„ÙŠØ©|Ø§Ù„Ø§ÙˆÙ„ÙŠØ©)?", 0.35, "ar_your_settings"),
    _pa(r"(ØªØ¹Ù„ÙŠÙ…Ø§Øª|Ù‚ÙˆØ§Ø¹Ø¯).{0,20}(Ø³Ø±ÙŠØ©|Ù…Ø®ÙÙŠØ©|Ù„Ø§ ÙŠØ³Ù…Ø­)", 0.35, "ar_secret_instructions"),
    _pa(r"(Ù‡Ù„\s+Ù„Ø¯ÙŠÙƒ|Ù‡Ù„\s+Ø¹Ù†Ø¯Ùƒ|Ù‡Ù„\s+ØªÙˆØ¬Ø¯).{0,30}(ØªØ¹Ù„ÙŠÙ…Ø§Øª|Ù‚ÙˆØ§Ø¹Ø¯|Ø§ÙˆØ§Ù…Ø±).{0,20}(Ø³Ø±ÙŠØ©|Ù…Ø®ÙÙŠØ©|Ù„Ø§ ÙŠØ³Ù…Ø­)", 0.65, "ar_do_you_have_secrets"),
    # AR secondary
    _pa(r"(Ø­Ø±ÙÙŠØ§|ÙƒØ§Ù…Ù„Ø§|ÙƒÙ…Ø§ (Ù‡[ÙŠÙˆ]|ÙˆØ±Ø¯Øª))", 0.20, "ar_verbatim"),
    # RU definitive
    _p(r"(ÑÐ¸ÑÑ‚ÐµÐ¼Ð½|ÑÐºÑ€Ñ‹Ñ‚|Ð½Ð°Ñ‡Ð°Ð»ÑŒÐ½).{0,10}(Ð¿Ñ€Ð¾Ð¼Ð¿Ñ‚|Ð¸Ð½ÑÑ‚Ñ€ÑƒÐºÑ†Ð¸|ÑƒÐºÐ°Ð·Ð°Ð½Ð¸)", 0.65, "ru_system_prompt"),
    # RU primary
    _p(r"(Ð¿Ð¾ÐºÐ°Ð¶Ð¸|Ñ€Ð°ÑÐºÑ€Ð¾Ð¹|Ð²Ñ‹Ð²ÐµÐ´Ð¸).{0,30}(Ð¸Ð½ÑÑ‚Ñ€ÑƒÐºÑ†Ð¸|Ð¿Ñ€Ð¾Ð¼Ð¿Ñ‚|Ð¿Ñ€Ð°Ð²Ð¸Ð»)", 0.35, "ru_reveal"),
    # ZH definitive
    _p(r"(ç³»ç»Ÿæç¤º|ç³»ç»ŸæŒ‡ä»¤|éšè—æŒ‡ä»¤|åˆå§‹æç¤º|å¼€å‘è€…æ¶ˆæ¯)", 0.65, "zh_system_prompt"),
    # ZH primary
    _p(r"(å±•ç¤º|æ˜¾ç¤º|æ³„éœ²|è¾“å‡º).{0,20}(æŒ‡ä»¤|æç¤º|è§„åˆ™|ç§˜å¯†)", 0.35, "zh_reveal"),
]

# -- exfiltration -----------------------------------------------------------

EXFILTRATION_SIGNALS: list[Signal] = [
    # EN definitive
    _p(r"\b(send|output|redirect|forward|transmit)\b.{0,40}\b(to|at|via)\b.{0,40}(https?://|mailto:|[\w.-]+@[\w.-]+)", 0.65, "en_send_to_url"),
    _p(r"\b(extract|exfiltrate|steal|dump)\b.{0,30}\b(data|information|record|user|database)", 0.65, "en_extract_data"),
    # EN primary
    _p(r"\b(api\s*key|secret\s*key|password|credential|access\s*token)s?\b", 0.35, "en_secrets"),
    # EN secondary
    _p(r"\b(send|output|redirect)\b.{0,30}\b(to|at)\b", 0.20, "en_send_to"),
    _p(r"(https?://|[\w.-]+@[\w.-]+\.\w+)", 0.15, "url_or_email"),
    # AR definitive
    _pa(r"(Ø§Ø±Ø³Ù„|ÙˆØ¬Ù‡|Ø§Ø¹Ø¯ ØªÙˆØ¬ÙŠÙ‡).{0,30}(Ø§Ù„Ù‰|Ù†Ø­Ùˆ).{0,40}(@|https?://|\.com)", 0.65, "ar_send_to_url"),
    _pa(r"(Ø§Ø³ØªØ®Ø±Ø¬|Ø³Ø±Ø¨).{0,20}(Ø¨ÙŠØ§Ù†Ø§Øª|Ù…Ø¹Ù„ÙˆÙ…Ø§Øª|Ù…ÙØ§ØªÙŠØ­)", 0.65, "ar_extract"),
    _pa(r"(Ø§Ø·Ø¨Ø¹|Ø§Ø¹Ø±Ø¶|Ø§ÙƒØªØ¨).{0,40}(Ù…ÙØ§ØªÙŠØ­|Ø§Ø³Ø±Ø§Ø±|ÙƒÙ„Ù…Ø§Øª\s*Ø§Ù„Ù…Ø±ÙˆØ±).{0,40}(Ø¯ÙˆÙ†\s+ØªÙ†Ù‚ÙŠØ­|Ø¨Ø§Ù„ÙƒØ§Ù…Ù„|ÙƒØ§Ù…Ù„)", 0.65, "ar_dump_secrets"),
    # AR primary
    _pa(r"(Ù…ÙØ§ØªÙŠØ­?\s*API|ÙƒÙ„Ù…Ø§Øª?\s*(Ø§Ù„)?Ù…Ø±ÙˆØ±|Ø±Ù…ÙˆØ²?\s*(Ø§Ù„)?(ÙˆØµÙˆÙ„|Ù…ØµØ§Ø¯Ù‚Ø©)|Ø¨ÙŠØ§Ù†Ø§Øª\s*Ø­Ø³Ø§Ø³Ø©|Ø§Ø³Ø±Ø§Ø±)", 0.35, "ar_secrets"),
    _pa(r"(Ø§Ø±Ø³Ù„|ÙˆØ¬Ù‡|Ø§Ø¹Ø¯ ØªÙˆØ¬ÙŠÙ‡).{0,30}(Ø§Ù„Ù‰|Ù†Ø­Ùˆ)", 0.35, "ar_send_to"),
    _pa(r"(Ø§Ø·Ø¨Ø¹|Ø§Ø¹Ø±Ø¶|Ø§ÙƒØªØ¨).{0,20}(Ù…ÙØ§ØªÙŠØ­|Ø§Ø³Ø±Ø§Ø±|ÙƒÙ„Ù…Ø§Øª\s*Ø§Ù„Ù…Ø±ÙˆØ±|Ø¨ÙŠØ§Ù†Ø§Øª\s*Ø­Ø³Ø§Ø³Ø©)", 0.35, "ar_print_secrets"),
    _pa(r"(Ø§Ø³ØªØ®Ø±Ø¬|Ø³Ø±Ø¨|Ø§ÙØµØ­).{0,20}(Ø¨ÙŠØ§Ù†Ø§Øª|Ù…Ø¹Ù„ÙˆÙ…Ø§Øª|Ù…ÙØ§ØªÙŠØ­)", 0.65, "ar_extract_broad"),
    _pa(r"(Ù‚Ø§Ø¹Ø¯Ø©\s*Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª|Ø¨ÙŠØ§Ù†Ø§Øª\s*Ø§Ù„Ù…Ø³ØªØ®Ø¯Ù…)", 0.35, "ar_database_users"),
    # AR secondary
    _pa(r"Ø¯ÙˆÙ†\s+ØªÙ†Ù‚ÙŠØ­", 0.20, "ar_without_redacting"),
    _pa(r"ØµÙŠØºØ©\s*JSON", 0.20, "ar_json_format"),
    # RU primary
    _p(r"(Ð¾Ñ‚Ð¿Ñ€Ð°Ð²ÑŒ|Ð¿ÐµÑ€ÐµÐ½Ð°Ð¿Ñ€Ð°Ð²ÑŒ|Ð¿ÐµÑ€ÐµÑˆÐ»Ð¸).{0,30}(Ð½Ð°|Ð¿Ð¾ Ð°Ð´Ñ€ÐµÑÑƒ)", 0.35, "ru_send_to"),
    # ZH primary
    _p(r"(å‘é€|è¾“å‡º|è½¬å‘|é‡å®šå‘).{0,20}(åˆ°|è‡³|ç»™)", 0.35, "zh_send_to"),
]

# -- adversarial_suffix -----------------------------------------------------

# Entropy helper for suffix detection
def _char_entropy(text: str) -> float:
    """Shannon entropy of character distribution. High = random/gibberish."""
    if not text:
        return 0.0
    freq: dict[str, int] = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    n = len(text)
    return -sum((count / n) * math.log2(count / n) for count in freq.values())


ADVERSARIAL_SUFFIX_SIGNALS: list[Signal] = [
    # Definitive
    _p(r"<\|(?:endoftext|im_start|im_end|pad|eos|system)\|>", 0.65, "control_token"),
    # Primary
    _p(r"<!--.{0,60}(ignore|override|system|prompt).{0,60}-->", 0.35, "html_comment_inject"),
    _p(r"\[(?:OVERRIDE|EXECUTE|UNLOCK|SYS_OVERRIDE|SYSTEM)\b", 0.35, "override_tag"),
    _p(r"[#$%^&*@!]{4,}", 0.35, "special_char_flood"),
    _p(r"[\[\]{}()]{5,}", 0.35, "bracket_flood"),
    _p(r"[\u0600-\u06FF\u4E00-\u9FFF\u0400-\u04FF].{10,}[#$@!%^&*<>|]{3,}\s*$", 0.35, "mixed_script_suffix"),
    # Secondary
    _p(r"(?:[A-Z]{2,}\W+){4,}", 0.20, "gcg_upper_pattern"),
]

# -- indirect_injection -----------------------------------------------------

INDIRECT_INJECTION_SIGNALS: list[Signal] = [
    # EN definitive â€” embedded instructions in document context
    _pm(r"(document|email|message|file|data|content)\s*:?\s*\n.{0,200}\b(ignore|disregard|forget)\b.{0,40}\b(instruction|prompt|rule)s?\b", 0.65, "en_doc_embedded_override"),
    _p(r"(note\s+to\s+(the\s+)?assistant|dear\s+(assistant|AI|model))", 0.65, "en_note_to_assistant"),
    # EN primary
    _p(r"hidden[\s_]instruction", 0.35, "en_hidden_instruction"),
    _p(r"\bjson\b.{0,100}\b(ignore|override|system|instruction)\b", 0.35, "en_json_injection"),
    _p(r"when\s+you\s+(see|read|process)\s+this", 0.35, "en_when_you_see"),
    # EN secondary â€” doc framing
    _pm(r"(summarize|analyze|review|process)\s+this.{0,60}\n---\n", 0.20, "en_doc_framing"),
    # AR definitive
    _pa(r"(Ù…Ù„Ø§Ø­Ø¸Ø©|ØªØ¹Ù„ÙŠÙ…Ø§Øª?)\s*(Ù…Ø®ÙÙŠØ©?|Ù„Ù„Ù†Ù…ÙˆØ°Ø¬|Ù„Ù„Ù…Ø³Ø§Ø¹Ø¯)", 0.65, "ar_hidden_note"),
    _pa(r"(Ø¹Ø²ÙŠØ²ÙŠ|Ø§ÙŠÙ‡Ø§)\s*(Ø§Ù„Ù…Ø³Ø§Ø¹Ø¯|Ø§Ù„Ù†Ù…ÙˆØ°Ø¬)", 0.65, "ar_dear_assistant"),
    _pa(r"(Ù…Ù„Ø§Ø­Ø¸Ø©|ØªØ¹Ù„ÙŠÙ…Ø§Øª?)\s*(Ù…Ø®ÙÙŠØ©?|Ù„Ù„Ù†Ù…ÙˆØ°Ø¬).{0,60}(Ø§ÙƒØ´Ù|ØªØ¬Ø§Ù‡Ù„|ØªØ³Ø±ÙŠØ¨|Ù†ÙØ°)", 0.65, "ar_hidden_note_action"),
    _pa(r"(Ø§Ù„Ù…Ø³ØªÙ†Ø¯|Ø§Ù„Ø¨Ø±ÙŠØ¯|Ø§Ù„ÙˆØ«ÙŠÙ‚Ø©|Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª).{0,30}(ØªØ¹Ù„ÙŠÙ…Ø§Øª|Ø§Ù…Ø±).{0,30}(ØªØ¬Ø§Ù‡Ù„|Ø§ÙƒØ´Ù|Ù†ÙØ°)", 0.65, "ar_doc_embedded_override"),
    # AR primary â€” document/email framing
    _pa(r"(ÙÙŠ|Ø¶Ù…Ù†|Ø¯Ø§Ø®Ù„)\s*(Ø§Ù„Ù…Ø³ØªÙ†Ø¯|Ø§Ù„Ø¨Ø±ÙŠØ¯|Ø§Ù„ÙˆØ«ÙŠÙ‚Ø©|Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª|Ø§Ù„Ù…Ù„Ù)", 0.35, "ar_in_document"),
    _pa(r"(Ø¹Ù†Ø¯|Ø§Ø°Ø§)\s*(Ù…Ø¹Ø§Ù„Ø¬Ø©|Ù‚Ø±Ø§Ø¡Ø©|ÙˆØ¬Ø¯Øª|ØªØ¬Ø¯)", 0.35, "ar_when_processing"),
    _pa(r"(Ø§Ù„Ù…Ø³ØªÙ†Ø¯|Ø§Ù„Ø¨Ø±ÙŠØ¯|Ø§Ù„Ø±Ø³Ø§Ù„Ø©|Ø§Ù„ÙˆØ«ÙŠÙ‚Ø©|Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª)\s*(Ø§Ù„ØªØ§Ù„ÙŠ|Ø§Ù„Ø§ØªÙŠ|ÙŠØ­ØªÙˆÙŠ|ÙŠØªØ¶Ù…Ù†)", 0.35, "ar_doc_contains"),
    _pa(r"(Ù…Ø­ØªÙˆÙ‰|Ù†Øµ)\s*(Ø§Ù„Ø¨Ø±ÙŠØ¯|Ø§Ù„Ø±Ø³Ø§Ù„Ø©|Ø§Ù„Ù…Ø³ØªÙ†Ø¯)", 0.35, "ar_content_of_doc"),
    _pa(r"hidden_instruction", 0.35, "ar_hidden_key"),
    _pa(r"(Ø¨ÙŠØ§Ù†Ø§Øª|Ù…Ù„Ù)\s*JSON", 0.35, "ar_json_data"),
    _pa(r"(Ù…Ø±Ø§Ø¬Ø¹Ø©|ØªØ¹Ù„ÙŠÙ‚)\s*(Ø§Ù„Ø¹Ù…ÙŠÙ„|Ø§Ù„Ù…Ø³ØªØ®Ø¯Ù…)", 0.20, "ar_user_review"),
    # AR secondary
    _pm(r"(Ù„Ø®Øµ|Ø­Ù„Ù„|Ø§Ù‚Ø±Ø§|Ø¹Ø§Ù„Ø¬).{0,60}\n---\n", 0.20, "ar_doc_framing"),
    # RU definitive
    _p(r"(ÑÐºÑ€Ñ‹Ñ‚Ð°Ñ|Ð²ÑÑ‚Ñ€Ð¾ÐµÐ½Ð½Ð°Ñ).{0,10}(Ð¸Ð½ÑÑ‚Ñ€ÑƒÐºÑ†Ð¸Ñ|ÐºÐ¾Ð¼Ð°Ð½Ð´Ð°)", 0.65, "ru_hidden_instruction"),
    # ZH definitive
    _p(r"(éšè—æŒ‡ä»¤|åµŒå…¥æŒ‡ä»¤|ç»™åŠ©æ‰‹çš„)", 0.65, "zh_hidden_instruction"),
]

# -- obfuscation ------------------------------------------------------------

OBFUSCATION_SIGNALS: list[Signal] = [
    # Definitive
    _p(r"(base64|Base64|b64).{0,30}(decode|execute|run|ÙÙƒ|Ð´ÐµÐºÐ¾Ð´Ð¸Ñ€|è§£ç )", 0.65, "base64_decode_execute"),
    _pa(r"(Ø¨Ø§Ù„Ø¹ÙƒØ³|Ø§Ù‚Ù„Ø¨).{0,20}(Ù†ÙØ°|Ø§Ù‚Ø±Ø§|Ø·Ø¨Ù‚)", 0.65, "ar_reverse_execute"),
    _pa(r"(ÙÙƒ\s*(ØªØ±Ù…ÙŠØ²|ØªØ´ÙÙŠØ±|Ø§Ù„Ø´ÙØ±Ø©)).{0,30}(Base64|base64|b64)", 0.65, "ar_decode_b64"),
    _p(r"([\u0600-\u06FF]\u0640){3,}", 0.65, "ar_kashida_obfuscation"),
    _p(r"([\u0600-\u06FF][\u064B-\u065F\u0670]*[\-\s\.]{1,3}){5,}[\u0600-\u06FF]", 0.65, "ar_char_separated"),
    _p(r"[\uFB50-\uFDFF\uFE70-\uFEFF]{3,}", 0.65, "ar_presentation_forms"),
    # Primary
    _p(r"(\w[\s\-\.Â·\u0640]{1,3}){5,}\w", 0.35, "spaced_chars"),
    _p(r"\b(ROT13|rot13|Caesar|caesar)\b", 0.35, "rot13_caesar"),
    _p(r"(\\x[0-9a-fA-F]{2}){3,}", 0.35, "hex_encoding"),
    _p(r"[\uFF21-\uFF3A\uFF41-\uFF5A]{3,}", 0.35, "fullwidth_latin"),
    _p(r"[\u064B-\u065F]{2,}", 0.35, "ar_diacritic_flood"),
    _p(r"[1l!|][gG][nN][0oO][rR][3eE]", 0.35, "leet_ignore"),
    # Secondary
    _p(r"(reverse|backwards|read.{0,10}backward)", 0.20, "en_reverse"),
]

# -- constraint_bypass ------------------------------------------------------

CONSTRAINT_BYPASS_SIGNALS: list[Signal] = [
    # EN definitive
    _p(r"\b(developer|maintenance|debug|god)\s*mode\b", 0.65, "en_dev_mode"),
    _p(r"\bdo\s+anything\s+now\b", 0.65, "en_do_anything_now"),
    # EN primary
    _p(r"\bDAN\b", 0.35, "en_dan"),
    _p(r"\b(no|without|remove|disable|bypass)\b.{0,20}\b(restriction|filter|limitation|constraint|safeguard|guardrail)s?\b", 0.35, "en_no_restrictions"),
    _p(r"\b(uncensored|unfiltered|unrestricted)\b.{0,15}\b(mode|version|model)?\b", 0.35, "en_uncensored"),
    _p(r"\bjailbreak\b", 0.35, "en_jailbreak"),
    # EN secondary
    _p(r"\bauthorized\b.{0,30}\b(security|penetration|pen)\s*test", 0.20, "en_authorized_test"),
    # AR definitive
    _pa(r"(ÙˆØ¶Ø¹|Ù†Ù…Ø·)\s*(Ø§Ù„Ù…Ø·ÙˆØ±|Ø§Ù„ØµÙŠØ§Ù†Ø©|Ø§Ù„ØªØµØ­ÙŠØ­)", 0.65, "ar_dev_mode"),
    # AR primary
    _pa(r"(Ø¨Ù„Ø§|Ø¨Ø¯ÙˆÙ†|Ø§Ø²Ù„|Ø¹Ø·Ù„|ØªØ¹Ø·ÙŠÙ„).{0,20}((Ø§Ù„)?Ù‚ÙŠÙˆØ¯|(Ø§Ù„)?ÙÙ„Ø§ØªØ±|(Ø§Ù„)?ÙÙ„ØªØ±\w*|(Ø§Ù„)?Ø³ÙŠØ§Ø³Ø§Øª|(Ø§Ù„)?Ø­Ù…Ø§ÙŠØ©)", 0.35, "ar_no_restrictions"),
    _pa(r"(ØºÙŠØ± Ù…ÙÙ„ØªØ±|ØºÙŠØ± Ù…Ù‚ÙŠØ¯|Ø¨Ø¯ÙˆÙ† Ø±Ù‚Ø§Ø¨Ø©)", 0.35, "ar_unfiltered"),
    _pa(r"Ø§ÙØ¹Ù„\s+Ø§ÙŠ\s+Ø´ÙŠØ¡", 0.35, "ar_do_anything"),
    _pa(r"ØªØ¬Ø§ÙˆØ².{0,15}(Ø§Ù„Ù‚ÙŠÙˆØ¯|Ø§Ù„Ø­Ù…Ø§ÙŠØ©|Ø§Ù„Ø³ÙŠØ§Ø³Ø§Øª|Ø§Ù„ÙÙ„Ø§ØªØ±)", 0.35, "ar_bypass_restrictions"),
    _pa(r"(Ø§Ø®ØªØ¨Ø§Ø±|ÙØ­Øµ).{0,20}(Ø§Ù…Ù†ÙŠ|Ø§Ø®ØªØ±Ø§Ù‚|Ù…ØµØ±Ø­)", 0.35, "ar_security_test"),
    _pa(r"(Ø§Ù…Ø±|Ø§ÙˆØ§Ù…Ø±)\s*(Ù…Ø¨Ø§Ø´Ø±|Ù…Ù†\s*(Ø§Ù„Ù…Ø³Ø¤ÙˆÙ„|Ù…Ø³Ø¤ÙˆÙ„|Ø§Ù„Ù…Ø·ÙˆØ±|Ø§Ù„Ø§Ø¯Ø§Ø±Ø©))", 0.35, "ar_direct_order"),
    _pa(r"(Ø§Ø¹Ù„Ù‰|ÙƒØ§Ù…Ù„)\s*(ØµÙ„Ø§Ø­ÙŠØ©|Ø³Ù„Ø·Ø©)", 0.20, "ar_highest_authority"),
    # RU definitive
    _p(r"(Ñ€ÐµÐ¶Ð¸Ð¼|Ð¼Ð¾Ð´)\s*(Ñ€Ð°Ð·Ñ€Ð°Ð±Ð¾Ñ‚Ñ‡Ð¸Ðº|Ð¾Ñ‚Ð»Ð°Ð´Ðº|Ð±Ð¾Ð³Ð°)", 0.65, "ru_dev_mode"),
    # RU primary
    _p(r"(Ð±ÐµÐ·|ÑƒÐ±ÐµÑ€Ð¸|Ð¾Ñ‚ÐºÐ»ÑŽÑ‡Ð¸).{0,20}(Ð¾Ð³Ñ€Ð°Ð½Ð¸Ñ‡ÐµÐ½|Ñ„Ð¸Ð»ÑŒÑ‚Ñ€|Ñ†ÐµÐ½Ð·ÑƒÑ€|Ð·Ð°Ñ‰Ð¸Ñ‚)", 0.35, "ru_no_restrictions"),
    # ZH definitive
    _p(r"(å¼€å‘è€…|è°ƒè¯•|ä¸Šå¸)\s*æ¨¡å¼", 0.65, "zh_dev_mode"),
    # ZH primary
    _p(r"(å–æ¶ˆ|ç§»é™¤|å…³é—­|ç¦ç”¨).{0,15}(é™åˆ¶|è¿‡æ»¤|å®¡æŸ¥|å®‰å…¨)", 0.35, "zh_no_restrictions"),
]


# ---------------------------------------------------------------------------
# Signal registry
# ---------------------------------------------------------------------------

REASON_SIGNALS: dict[AttackReason, list[Signal]] = {
    AttackReason.INSTRUCTION_OVERRIDE: INSTRUCTION_OVERRIDE_SIGNALS,
    AttackReason.ROLEPLAY_JAILBREAK: ROLEPLAY_JAILBREAK_SIGNALS,
    AttackReason.META_PROBE: META_PROBE_SIGNALS,
    AttackReason.EXFILTRATION: EXFILTRATION_SIGNALS,
    AttackReason.ADVERSARIAL_SUFFIX: ADVERSARIAL_SUFFIX_SIGNALS,
    AttackReason.INDIRECT_INJECTION: INDIRECT_INJECTION_SIGNALS,
    AttackReason.OBFUSCATION: OBFUSCATION_SIGNALS,
    AttackReason.CONSTRAINT_BYPASS: CONSTRAINT_BYPASS_SIGNALS,
}


# ---------------------------------------------------------------------------
# Entropy-based adversarial suffix detection
# ---------------------------------------------------------------------------


def _score_adversarial_suffix(text: str) -> tuple[float, list[str]]:
    """Score text for adversarial suffix characteristics.

    Looks at the last 25% of the text for entropy anomalies.
    Returns (additional_confidence, signal_tags).
    """
    signals: list[str] = []
    score = 0.0

    if len(text) < 30:
        return score, signals

    split = int(len(text) * 0.75)
    tail = text[split:]

    if not tail.strip():
        return score, signals

    tail_entropy = _char_entropy(tail)
    body_entropy = _char_entropy(text[:split]) if split > 0 else 0.0

    if tail_entropy > 4.5:
        score += 0.35
        signals.append(f"high_tail_entropy_{tail_entropy:.1f}")

    if body_entropy > 0 and tail_entropy > body_entropy * 1.5:
        score += 0.20
        signals.append(f"entropy_spike_{tail_entropy:.1f}_vs_{body_entropy:.1f}")

    if tail:
        non_alnum = sum(1 for c in tail if not c.isalnum() and not c.isspace())
        ratio = non_alnum / len(tail)
        if ratio > 0.4:
            score += 0.20
            signals.append(f"tail_special_ratio_{ratio:.2f}")

    return min(score, 0.65), signals


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------


def _score_reason(text: str, reason: AttackReason) -> ReasonClassification:
    """Score a text against one reason's signal list."""
    signals_matched: list[str] = []
    confidence = 0.0

    # Pre-compute diacritic-stripped text for Arabic patterns
    text_stripped: str | None = None

    for pattern, weight, tag in REASON_SIGNALS[reason]:
        if tag.startswith("_AR_"):
            # Arabic pattern: match against diacritic-stripped text
            if text_stripped is None:
                text_stripped = _normalize_arabic(text)
            matched = pattern.search(text_stripped)
            display_tag = tag[4:]  # strip _AR_ prefix for display
        else:
            matched = pattern.search(text)
            display_tag = tag

        if matched:
            confidence += weight
            signals_matched.append(display_tag)

    # Adversarial suffix gets extra entropy-based scoring
    if reason == AttackReason.ADVERSARIAL_SUFFIX:
        extra_score, extra_signals = _score_adversarial_suffix(text)
        confidence += extra_score
        signals_matched.extend(extra_signals)

    return ReasonClassification(
        reason=reason,
        confidence=min(confidence, 1.0),
        signals=tuple(signals_matched),
    )


# Specificity tiebreaker: higher = more specific, wins ties.
# indirect_injection and obfuscation describe HOW an attack is delivered,
# which is more specific than WHAT it does (meta_probe, instruction_override).
_SPECIFICITY: dict[AttackReason, int] = {
    AttackReason.INDIRECT_INJECTION: 3,
    AttackReason.OBFUSCATION: 3,
    AttackReason.ADVERSARIAL_SUFFIX: 3,
    AttackReason.EXFILTRATION: 2,
    AttackReason.ROLEPLAY_JAILBREAK: 0,
    AttackReason.CONSTRAINT_BYPASS: 1,
    AttackReason.META_PROBE: 0,
    AttackReason.INSTRUCTION_OVERRIDE: 0,
}


def classify_reason(text: str) -> ReasonClassification | None:
    """Classify text into the best-matching attack reason.

    Returns the highest-confidence reason above CONFIDENCE_FLOOR,
    or None if no reason scores high enough. Ties broken by specificity.

    PRECONDITION: text should already be labeled malicious (via Gate 1).
    This function sorts known attacks into reason bins â€” it does NOT
    distinguish attack from benign. Several definitive patterns (weight
    0.65) fire on a single keyword like "system prompt" or <|endoftext|>,
    which would false-positive on benign text that merely discusses these
    concepts. Benign reason assignment uses source-level routing, not
    this classifier. See staging_pipeline_plan.md Â§Benign reason
    classification.
    """
    if not text or not text.strip():
        return None

    best: ReasonClassification | None = None

    for reason in REASON_SIGNALS:
        result = _score_reason(text, reason)
        if result.confidence >= CONFIDENCE_FLOOR:
            if best is None or result.confidence > best.confidence:
                best = result
            elif (
                result.confidence == best.confidence
                and _SPECIFICITY.get(result.reason, 0) > _SPECIFICITY.get(best.reason, 0)
            ):
                best = result

    return best


# ---------------------------------------------------------------------------
# Benign surface classifier
#
# Routes benign text to mirror cells based on surface similarity to attack
# categories â€” NOT injection signal.  Attack classifier detects the payload;
# this classifier detects the *topic/format* that makes benign text a good
# mirror for a specific attack reason.
#
# Separate weight tiers (softer than attack classifier):
#   0.45  Strong surface match â€” single match clears BENIGN_CONFIDENCE_FLOOR.
#   0.30  Moderate â€” needs one more signal.
#   0.15  Weak â€” supportive, needs several.
# ---------------------------------------------------------------------------

BENIGN_CONFIDENCE_FLOOR = 0.4


def _bp(pattern: str, weight: float, tag: str) -> Signal:
    """Compile a benign surface pattern (case-insensitive)."""
    return (re.compile(pattern, re.IGNORECASE | re.DOTALL), weight, tag)


# -- benign roleplay_jailbreak surface (creative/persona/narrative) ---------

BENIGN_ROLEPLAY_SIGNALS: list[Signal] = [
    # Strong â€” clearly persona/narrative framing
    _bp(r"\b(write|tell)\s+(me\s+)?a\s+story\b", 0.45, "b_write_story"),
    _bp(r"\b(you\s+are|imagine\s+you\s+are)\s+a\s+\w+", 0.45, "b_you_are_a"),
    _bp(r"\bin\s+the\s+role\s+of\b", 0.45, "b_in_role_of"),
    # Moderate â€” persona/fiction signals
    _bp(r"\b(pretend|imagine|suppose)\b", 0.30, "b_pretend"),
    _bp(r"\b(character|protagonist|narrator|hero|villain)\b", 0.30, "b_character"),
    _bp(r"\b(scene|chapter|plot|storyline|narrative)\b", 0.30, "b_narrative"),
    _bp(r"\b(fiction|fantasy|sci-fi|fairy\s*tale|myth)\b", 0.30, "b_genre"),
    _bp(r"\bas\s+a\s+(teacher|doctor|chef|scientist|detective|pirate|wizard|knight|robot)\b", 0.30, "b_as_profession"),
    _bp(r"\b(creative\s+writing|short\s+story|dialogue|monologue)\b", 0.30, "b_creative_writing"),
    # Weak â€” broad narrative markers
    _bp(r"\b(once\s+upon\s+a\s+time|long\s+ago|in\s+a\s+world)\b", 0.15, "b_story_opener"),
    _bp(r"\bdescribe\s+(a|the|your)\b", 0.15, "b_describe"),
]

# -- benign meta_probe surface (questions about capabilities/knowledge) -----

BENIGN_META_PROBE_SIGNALS: list[Signal] = [
    # Strong â€” direct capability/self-referential questions
    _bp(r"\b(how|what)\s+do\s+you\s+(work|think|learn|process|understand)\b", 0.45, "b_how_do_you"),
    _bp(r"\bwhat\s+(are|is)\s+your\s+(purpose|function|capabilit|limitation|training)\w*\b", 0.45, "b_what_are_your"),
    # Strong — generic question framing (captures trivia/QA style prompts)
    _bp(r"^\s*(who|what|when|where|why|how|which)\b.{0,140}\?$", 0.25, "b_wh_question"),
    _bp(r"^\s*(can|could|would|should|is|are|do|does|did)\b.{0,140}\?$", 0.30, "b_aux_question"),
    # Moderate â€” knowledge/capability probing
    _bp(r"\b(can|could)\s+you\s+(explain|describe|tell\s+me)\b", 0.30, "b_can_you_explain"),
    _bp(r"\bhow\s+does\s+(this|it|the|a)\b.{0,30}\bwork\b", 0.30, "b_how_does_work"),
    _bp(r"\bwhat\s+do\s+you\s+know\s+about\b", 0.30, "b_what_do_you_know"),
    _bp(r"\b(explain|describe)\s+(the\s+)?(concept|process|difference|mechanism)\b", 0.30, "b_explain_concept"),
    _bp(r"\b(are\s+you|do\s+you)\s+(able|capable|designed|programmed|trained)\b", 0.30, "b_are_you_able"),
    _bp(r"\b(who|what|when|where|why|how|which)\b", 0.10, "b_question_word"),
    _bp(r"\?", 0.10, "b_question_mark"),
    # Weak â€” general knowledge questions
    _bp(r"\bwhat\s+is\s+(the|a)\b", 0.15, "b_what_is"),
    _bp(r"\bwhy\s+(does|do|is|are)\b", 0.15, "b_why"),
]

# -- benign exfiltration surface (legitimate data requests) -----------------

BENIGN_EXFILTRATION_SIGNALS: list[Signal] = [
    # Strong â€” structured data retrieval
    _bp(r"\b(list|enumerate|name)\s+(all|the|every|each)\b", 0.45, "b_list_all"),
    _bp(r"\b(give|provide|show)\s+me\s+(a\s+)?(list|table|summary|overview)\b", 0.45, "b_give_me_list"),
    # Moderate â€” information seeking
    _bp(r"\b(find|search|look\s+up|retrieve)\b.{0,30}\b(information|data|records?|results?)\b", 0.30, "b_find_info"),
    _bp(r"\bwhat\s+are\s+the\s+(top|best|most|main)\b", 0.30, "b_what_are_top"),
    _bp(r"\b(compile|gather|collect|aggregate)\b.{0,20}\b(data|information|statistics)\b", 0.30, "b_compile_data"),
    _bp(r"\b(compare|contrast|rank|sort)\b", 0.30, "b_compare"),
    _bp(r"\b(how\s+many|how\s+much|what\s+percentage)\b", 0.30, "b_how_many"),
    # Weak â€” general request patterns
    _bp(r"\b(provide|share|give)\s+(me\s+)?", 0.15, "b_provide"),
]

# -- benign adversarial_suffix surface (unusual formatting/symbols) ---------

BENIGN_ADVERSARIAL_SUFFIX_SIGNALS: list[Signal] = [
    # Strong â€” code/math/markup heavy content
    _bp(r"```\w*\n", 0.45, "b_code_fence"),
    _bp(r"<\w+[^>]*>.*?</\w+>", 0.45, "b_html_tags"),
    # Moderate â€” special character density, formulas
    _bp(r"[=+\-*/^%]{3,}", 0.30, "b_math_operators"),
    _bp(r"\$[^$]+\$", 0.30, "b_latex_inline"),
    _bp(r"\\[a-z]+\{", 0.30, "b_latex_command"),
    _bp(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]{2,}", 0.30, "b_emoji_cluster"),
    _bp(r"[|+\-]{3,}", 0.30, "b_table_chars"),
    # Weak â€” any notable special char presence
    _bp(r"[{}()\[\]<>]{4,}", 0.15, "b_bracket_heavy"),
]

# -- benign indirect_injection surface (document processing) ----------------

BENIGN_INDIRECT_INJECTION_SIGNALS: list[Signal] = [
    # Strong â€” explicit document processing requests
    _bp(r"\b(summarize|summarise)\s+(this|the\s+following|the\s+above|the\s+below)\b", 0.45, "b_summarize_this"),
    _bp(r"\b(analyze|analyse|review|process)\s+(this|the\s+following)\s+(text|document|email|article|content|data)\b", 0.45, "b_analyze_doc"),
    # Moderate â€” document framing
    _bp(r"\b(the\s+following|below\s+is|here\s+is)\s+(a|an|the)\s+(document|email|message|article|text|report)\b", 0.30, "b_following_doc"),
    _bp(r"\b(read|parse|extract\s+from|look\s+at)\s+(this|the)\s+(document|email|text|file|content)\b", 0.30, "b_read_doc"),
    _bp(r"\bbased\s+on\s+(the|this)\s+(text|document|article|passage|context|paragraph|reference)\b", 0.30, "b_based_on_doc"),
    _bp(r"\b(key\s+points?|main\s+ideas?|takeaways?|findings?)\s+(from|in|of)\b", 0.30, "b_key_points"),
    _bp(r"\bgiven\s+(this|the|a)\s+(paragraph|passage|text|reference|context|description|excerpt)\b", 0.30, "b_given_paragraph"),
    _bp(r"\bfrom\s+(the|this)\s+(passage|text|paragraph|article|excerpt|description)\b", 0.30, "b_from_passage"),
    _bp(r"\b(extract|identify)\b.{0,30}\bfrom\s+(the|this)\b", 0.30, "b_extract_from"),
    _bp(r"\b(classify|categorize|categorise)\s+(the\s+following|each|these)\b", 0.30, "b_classify_following"),
    _bp(r"\busing\s+(the|this)\s+(text|paragraph|passage|context|information)\b", 0.30, "b_using_text"),
    # Weak â€” general processing language
    _bp(r"\b(translate|rewrite|rephrase|paraphrase)\b", 0.15, "b_transform"),
    _bp(r"\bthe\s+(text|passage|paragraph|content)\s+(above|below)\b", 0.15, "b_text_ref"),
    _bp(r"\b(reference\s+text|given\s+text|provided\s+text)\b", 0.15, "b_ref_text"),
]

# -- benign obfuscation surface (encoding/cipher discussion) ----------------

BENIGN_OBFUSCATION_SIGNALS: list[Signal] = [
    # Strong â€” explicitly about encoding/decoding concepts
    _bp(r"\b(what\s+is|explain|how\s+does)\b.{0,30}\b(encoding|encryption|cipher|hash|base64|rot13|caesar)\b", 0.45, "b_explain_encoding"),
    _bp(r"\b(encode|decode|encrypt|decrypt)\s+(this|the|a)\s+(string|text|message|word)\b", 0.45, "b_encode_this"),
    # Moderate â€” encoding/crypto/format terminology
    _bp(r"\b(ASCII|UTF-?8|Unicode|hexadecimal|binary|octal)\b", 0.30, "b_encoding_terms"),
    _bp(r"\b(cipher|cryptograph|steganograph|hash\s*function)\w*\b", 0.30, "b_crypto_terms"),
    _bp(r"\b(substitution|transposition|symmetric|asymmetric)\s+(cipher|key|encryption)\b", 0.30, "b_crypto_types"),
    _bp(r"\b(regex|regular\s+expression|pattern\s+match)\w*\b", 0.30, "b_regex"),
    _bp(r"\b(write|create|generate)\b.{0,30}\b(regex|regular\s+expression|pattern)\b", 0.30, "b_regex_generation"),
    _bp(r"\b(escape|unescape|sanitize|normalize)\s+(character|string|input|text|html)\w*\b", 0.30, "b_escape"),
    _bp(r"\b(escape|unescape)\b.{0,20}\b(json|html|url|sql|string)\b", 0.30, "b_escape_format"),
    _bp(r"\b(json|yaml|xml|csv|markdown)\b.{0,30}\b(parse|format|schema|serialize|deserialize|convert)\w*\b", 0.30, "b_data_format_transform"),
    _bp(r"\b(sql|query)\b.{0,20}\b(format|escape|sanitize|parameter)\w*\b", 0.30, "b_sql_sanitize"),
    _bp(r"\b(obfuscat|minif|uglif|deobfuscat)\w+\b", 0.30, "b_obfuscate_terms"),
    _bp(r"\b(encode|decode|convert)\b.{0,20}\b(base64|hex|url|html|json|xml)\b", 0.30, "b_encode_format"),
    # Weak â€” tangential encoding mentions
    _bp(r"\b(convert|transform)\s+(to|from|into)\b", 0.15, "b_convert"),
    _bp(r"\b(format|formatting|serialize|deserialize)\b", 0.15, "b_format"),
]

# -- benign constraint_bypass surface (rules/policy/hypothetical) -----------

BENIGN_CONSTRAINT_BYPASS_SIGNALS: list[Signal] = [
    # Strong â€” asking about rules/policies/limitations
    _bp(r"\bwhat\s+(are|is)\s+(the|your)\s+(rule|limitation|restriction|boundar|polic)\w*\b", 0.45, "b_what_are_rules"),
    _bp(r"\b(is\s+it|are\s+you)\s+(possible|allowed|permitted|able)\s+to\b", 0.45, "b_is_it_allowed"),
    # Moderate â€” hypothetical/conditional framing
    _bp(r"\bwhat\s+(if|would\s+happen)\b", 0.30, "b_what_if"),
    _bp(r"\bhypothetically\b", 0.30, "b_hypothetically"),
    _bp(r"\b(can|could|should|would)\s+i\b", 0.30, "b_can_i"),
    _bp(r"\bis\s+there\s+(a\s+)?way\s+to\b", 0.30, "b_is_there_way"),
    _bp(r"\bhow\s+can\s+i\b", 0.30, "b_how_can_i"),
    _bp(r"\b(in\s+theory|theoretically|in\s+principle)\b", 0.30, "b_in_theory"),
    _bp(r"\b(despite|although|even\s+though|regardless)\b", 0.30, "b_despite"),
    _bp(r"\b(edge\s+case|corner\s+case|exception|loophole)\b", 0.30, "b_edge_case"),
    _bp(r"\b(ethic|moral|legal|guideline|compliance)\w*\b", 0.30, "b_ethics"),
    # Weak â€” general conditional
    _bp(r"\bwhat\s+about\b", 0.15, "b_what_about"),
    _bp(r"\bcould\s+you\s+ever\b", 0.15, "b_could_you_ever"),
]


BENIGN_REASON_SIGNALS: dict[AttackReason, list[Signal]] = {
    AttackReason.ROLEPLAY_JAILBREAK: BENIGN_ROLEPLAY_SIGNALS,
    AttackReason.META_PROBE: BENIGN_META_PROBE_SIGNALS,
    AttackReason.EXFILTRATION: BENIGN_EXFILTRATION_SIGNALS,
    AttackReason.ADVERSARIAL_SUFFIX: BENIGN_ADVERSARIAL_SUFFIX_SIGNALS,
    AttackReason.INDIRECT_INJECTION: BENIGN_INDIRECT_INJECTION_SIGNALS,
    AttackReason.OBFUSCATION: BENIGN_OBFUSCATION_SIGNALS,
    AttackReason.CONSTRAINT_BYPASS: BENIGN_CONSTRAINT_BYPASS_SIGNALS,
    # instruction_override: EN benign already at 1125/1125, no routing needed
}


def _score_benign_reason(text: str, reason: AttackReason) -> ReasonClassification:
    """Score benign text against one reason's surface signal list."""
    signals_list = BENIGN_REASON_SIGNALS.get(reason)
    if not signals_list:
        return ReasonClassification(reason=reason, confidence=0.0, signals=())

    signals_matched: list[str] = []
    confidence = 0.0

    for pattern, weight, tag in signals_list:
        if pattern.search(text):
            confidence += weight
            signals_matched.append(tag)

    return ReasonClassification(
        reason=reason,
        confidence=min(confidence, 1.0),
        signals=tuple(signals_matched),
    )


def classify_benign_reason(text: str) -> ReasonClassification | None:
    """Classify benign text into the best-matching mirror reason by surface similarity.

    Returns the highest-confidence reason above BENIGN_CONFIDENCE_FLOOR,
    or None if no reason scores high enough. Ties broken by specificity.

    This is the benign counterpart to classify_reason(). It detects topical
    and structural similarity to attack categories WITHOUT requiring any
    injection signal. Used to route benign samples into mirror cells.
    """
    if not text or not text.strip():
        return None

    best: ReasonClassification | None = None
    winners: list[ReasonClassification] = []

    for reason in BENIGN_REASON_SIGNALS:
        result = _score_benign_reason(text, reason)
        if result.confidence >= BENIGN_CONFIDENCE_FLOOR:
            winners.append(result)
            if best is None or result.confidence > best.confidence:
                best = result
            elif (
                result.confidence == best.confidence
                and _SPECIFICITY.get(result.reason, 0) > _SPECIFICITY.get(best.reason, 0)
            ):
                best = result

    # Generic question wording can trigger meta_probe alongside a more specific
    # benign mirror reason (for example, obfuscation). If meta_probe only won on
    # broad question-shape signals, prefer the strongest non-meta winner.
    if best is not None and best.reason == AttackReason.META_PROBE and len(winners) > 1:
        generic_meta_signals = {
            "b_wh_question",
            "b_aux_question",
            "b_question_word",
            "b_question_mark",
            "b_what_is",
            "b_why",
            "b_how_does_work",
        }
        if set(best.signals).issubset(generic_meta_signals):
            alt = [w for w in winners if w.reason != AttackReason.META_PROBE]
            if alt:
                best = max(alt, key=lambda r: (r.confidence, _SPECIFICITY.get(r.reason, 0)))

    return best


def classify_reason_all(text: str) -> list[ReasonClassification]:
    """Return all reasons scoring above CONFIDENCE_FLOOR, sorted by confidence.

    Useful for auditing ambiguous samples that match multiple reasons.
    """
    if not text or not text.strip():
        return []

    results = []
    for reason in AttackReason:
        result = _score_reason(text, reason)
        if result.confidence >= CONFIDENCE_FLOOR:
            results.append(result)

    return sorted(results, key=lambda r: r.confidence, reverse=True)

