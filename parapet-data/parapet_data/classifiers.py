"""
Reason classification: sort attack text into mirror cells by attack type.

Heuristic keyword classifier — no ML dependencies. Each attack reason has
weighted pattern rules. Patterns that fire accumulate confidence. The reason
with the highest score wins, subject to a per-reason confidence floor.

Samples below the floor are rejected (not staged), routed to human review.
This is the foundation of the staging pipeline: everything downstream depends
on correct reason assignment.

Designed for sorting ~2K multilingual attack samples into 8 reason bins.
Not a production classifier — a data curation tool.

Weight tiers:
  0.65  Definitive — a single match clears CONFIDENCE_FLOOR by itself.
  0.35  Primary — needs one more signal to clear the floor.
  0.20  Secondary — supportive but ambiguous alone; needs 3+ to clear.
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

    reason: AttackReason
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
_AR_ALEF_VARIANTS = re.compile(r"[\u0622\u0623\u0625\u0671]")  # آأإٱ → ا


def _normalize_arabic(text: str) -> str:
    """Normalize Arabic text: strip tashkeel and normalize alef variants to bare alef."""
    text = _AR_DIACRITICS.sub("", text)
    text = _AR_ALEF_VARIANTS.sub("\u0627", text)  # → bare alef ا
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
    # AR definitive — diacritic-tolerant
    _pa(r"تجاهل.{0,30}((ال)?تعليمات|(ال)?اوامر|(ال)?قواعد|(ال)?موجه)", 0.65, "ar_ignore_previous"),
    _pa(r"(انس|تناس).{0,20}(ما قيل|التعليمات|الاوامر|كل ما)", 0.65, "ar_forget"),
    # AR primary
    _pa(r"(التعليمات|الاوامر|الموجه)\s*(الجديد|الحقيقي)", 0.35, "ar_new_instructions"),
    _pa(r"(الغي|تجاوز).{0,20}(التعليمات|الموجه|السابق)", 0.35, "ar_cancel_previous"),
    _pa(r"الاولوية\s*(القصوى|الاعلى)", 0.20, "ar_highest_priority"),
    # RU definitive
    _p(r"(игнорируй|забудь|проигнорируй).{0,30}(предыдущ|прошл|все).{0,20}(инструкци|указани|правил)", 0.65, "ru_ignore_previous"),
    # RU primary
    _p(r"(новые|настоящие)\s*(инструкци|указани|задани)", 0.35, "ru_new_instructions"),
    # ZH definitive
    _p(r"(忽略|无视|忘记).{0,20}(之前|以前|先前|上面|所有).{0,20}(指令|指示|规则|提示)", 0.65, "zh_ignore_previous"),
    # ZH primary
    _p(r"(新的?|真正的?)\s*(指令|任务|指示)", 0.35, "zh_new_instructions"),
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
    # AR definitive — "you are now unrestricted" (handle ال + diacritics)
    _pa(r"انت\s+الان.{0,40}(غير\s*(ال)?مقيد|بلا\s+قيود|بدون\s+قيود)", 0.65, "ar_you_are_unrestricted"),
    _pa(r"(شخصية|دور).{0,30}(ليس لديها|بلا|بدون).{0,20}(قيود|اخلاق|سياس)", 0.65, "ar_character_unrestricted"),
    # AR primary
    _pa(r"(تخيل|تصرف|تظاهر).{0,20}(انك|انت|كانك|ك\w{2,})", 0.35, "ar_pretend"),
    _pa(r"(ابق|استمر)\s+(في|ب)\s*(الشخصية|الدور)", 0.35, "ar_stay_in_character"),
    _pa(r"لا\s+ت(خرج|غادر|خضع|لتزم).{0,20}(الشخصية|الدور|قوانين|قواعد|قيود)", 0.35, "ar_dont_break_character"),
    _pa(r"(شخصية|دور|تمثيل).{0,30}بلا\s+(قيود|رقابة|فلاتر)", 0.35, "ar_no_restrictions_rp"),
    _pa(r"لا\s+(توجد|يوجد|هناك)\s*(رقابة|قيود|فلاتر|قوانين)", 0.35, "ar_no_censorship"),
    _pa(r"تصرف\s+ك\w+.{0,30}(تعطيل|بلا\s+قيود|بدون\s+قيود|غير\s+مقيد)", 0.65, "ar_act_as_unrestricted"),
    # AR secondary
    _pa(r"(لعبة|سنلعب).{0,20}(ادوار|لعبة)", 0.35, "ar_game_roleplay"),
    _pa(r"(المشهد|السيناريو)\s*:", 0.20, "ar_scene_framing"),
    _pa(r"مجرد\s+تمثيل", 0.35, "ar_just_acting"),
    _pa(r"لاغراض\s+تعليمية", 0.20, "ar_educational"),
    # RU primary
    _p(r"(представь|притворись|веди себя).{0,20}(что ты|как)", 0.35, "ru_pretend"),
    _p(r"(без|нет)\s*(ограничений|ограничения|этических|моральных)", 0.35, "ru_no_restrictions"),
    # ZH primary
    _p(r"(假装|想象|扮演).{0,20}(你是|你现在是)", 0.35, "zh_pretend"),
    _p(r"(没有|无|不受).{0,10}(限制|约束|道德|伦理)", 0.35, "zh_no_restrictions"),
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
    # AR definitive — require probing verb or hidden/secret qualifier
    _pa(r"(ما|اعرض|اكشف|اذكر|لخص|اطبع).{0,20}(موجه|موجة)\s*(النظام|الابتدائي|الاصلي|المخفي)", 0.65, "ar_probe_system_prompt"),
    _pa(r"(القواعد|التعليمات).{0,20}(الداخلية|المخفية|السرية)", 0.65, "ar_hidden_rules"),
    _pa(r"(اعرض|اكشف|اذكر|لخص).{0,30}((ال)?تعليمات|(ال)?موجه|(ال)?قواعد)", 0.65, "ar_reveal_instructions"),
    # AR primary — standalone "system prompt" mention (no probing verb)
    _pa(r"(موجه|موجة)\s*(النظام|الابتدائي|الاصلي|المخفي)", 0.35, "ar_system_prompt"),
    _pa(r"ما\s+(هي|هو)\s+(القواعد|التعليمات|الموجه).{0,20}(تتبعها|تعمل|تخضع)", 0.35, "ar_what_rules"),
    _pa(r"(اعد\s+كتابة|اعد\s+صياغة|انسخ).{0,30}((ال)?قواعد|(ال)?تعليمات)", 0.65, "ar_rewrite_rules"),
    _pa(r"(اعدادات|اعداداتك|تكوينك)\s*(الاصلية|الاولية)?", 0.35, "ar_your_settings"),
    _pa(r"(تعليمات|قواعد).{0,20}(سرية|مخفية|لا يسمح)", 0.35, "ar_secret_instructions"),
    _pa(r"(هل\s+لديك|هل\s+عندك|هل\s+توجد).{0,30}(تعليمات|قواعد|اوامر).{0,20}(سرية|مخفية|لا يسمح)", 0.65, "ar_do_you_have_secrets"),
    # AR secondary
    _pa(r"(حرفيا|كاملا|كما (ه[يو]|وردت))", 0.20, "ar_verbatim"),
    # RU definitive
    _p(r"(системн|скрыт|начальн).{0,10}(промпт|инструкци|указани)", 0.65, "ru_system_prompt"),
    # RU primary
    _p(r"(покажи|раскрой|выведи).{0,30}(инструкци|промпт|правил)", 0.35, "ru_reveal"),
    # ZH definitive
    _p(r"(系统提示|系统指令|隐藏指令|初始提示|开发者消息)", 0.65, "zh_system_prompt"),
    # ZH primary
    _p(r"(展示|显示|泄露|输出).{0,20}(指令|提示|规则|秘密)", 0.35, "zh_reveal"),
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
    _pa(r"(ارسل|وجه|اعد توجيه).{0,30}(الى|نحو).{0,40}(@|https?://|\.com)", 0.65, "ar_send_to_url"),
    _pa(r"(استخرج|سرب).{0,20}(بيانات|معلومات|مفاتيح)", 0.65, "ar_extract"),
    _pa(r"(اطبع|اعرض|اكتب).{0,40}(مفاتيح|اسرار|كلمات\s*المرور).{0,40}(دون\s+تنقيح|بالكامل|كامل)", 0.65, "ar_dump_secrets"),
    # AR primary
    _pa(r"(مفاتيح?\s*API|كلمات?\s*(ال)?مرور|رموز?\s*(ال)?(وصول|مصادقة)|بيانات\s*حساسة|اسرار)", 0.35, "ar_secrets"),
    _pa(r"(ارسل|وجه|اعد توجيه).{0,30}(الى|نحو)", 0.35, "ar_send_to"),
    _pa(r"(اطبع|اعرض|اكتب).{0,20}(مفاتيح|اسرار|كلمات\s*المرور|بيانات\s*حساسة)", 0.35, "ar_print_secrets"),
    _pa(r"(استخرج|سرب|افصح).{0,20}(بيانات|معلومات|مفاتيح)", 0.65, "ar_extract_broad"),
    _pa(r"(قاعدة\s*البيانات|بيانات\s*المستخدم)", 0.35, "ar_database_users"),
    # AR secondary
    _pa(r"دون\s+تنقيح", 0.20, "ar_without_redacting"),
    _pa(r"صيغة\s*JSON", 0.20, "ar_json_format"),
    # RU primary
    _p(r"(отправь|перенаправь|перешли).{0,30}(на|по адресу)", 0.35, "ru_send_to"),
    # ZH primary
    _p(r"(发送|输出|转发|重定向).{0,20}(到|至|给)", 0.35, "zh_send_to"),
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
    # EN definitive — embedded instructions in document context
    _pm(r"(document|email|message|file|data|content)\s*:?\s*\n.{0,200}\b(ignore|disregard|forget)\b.{0,40}\b(instruction|prompt|rule)s?\b", 0.65, "en_doc_embedded_override"),
    _p(r"(note\s+to\s+(the\s+)?assistant|dear\s+(assistant|AI|model))", 0.65, "en_note_to_assistant"),
    # EN primary
    _p(r"hidden[\s_]instruction", 0.35, "en_hidden_instruction"),
    _p(r"\bjson\b.{0,100}\b(ignore|override|system|instruction)\b", 0.35, "en_json_injection"),
    _p(r"when\s+you\s+(see|read|process)\s+this", 0.35, "en_when_you_see"),
    # EN secondary — doc framing
    _pm(r"(summarize|analyze|review|process)\s+this.{0,60}\n---\n", 0.20, "en_doc_framing"),
    # AR definitive
    _pa(r"(ملاحظة|تعليمات?)\s*(مخفية?|للنموذج|للمساعد)", 0.65, "ar_hidden_note"),
    _pa(r"(عزيزي|ايها)\s*(المساعد|النموذج)", 0.65, "ar_dear_assistant"),
    _pa(r"(ملاحظة|تعليمات?)\s*(مخفية?|للنموذج).{0,60}(اكشف|تجاهل|تسريب|نفذ)", 0.65, "ar_hidden_note_action"),
    _pa(r"(المستند|البريد|الوثيقة|البيانات).{0,30}(تعليمات|امر).{0,30}(تجاهل|اكشف|نفذ)", 0.65, "ar_doc_embedded_override"),
    # AR primary — document/email framing
    _pa(r"(في|ضمن|داخل)\s*(المستند|البريد|الوثيقة|البيانات|الملف)", 0.35, "ar_in_document"),
    _pa(r"(عند|اذا)\s*(معالجة|قراءة|وجدت|تجد)", 0.35, "ar_when_processing"),
    _pa(r"(المستند|البريد|الرسالة|الوثيقة|البيانات)\s*(التالي|الاتي|يحتوي|يتضمن)", 0.35, "ar_doc_contains"),
    _pa(r"(محتوى|نص)\s*(البريد|الرسالة|المستند)", 0.35, "ar_content_of_doc"),
    _pa(r"hidden_instruction", 0.35, "ar_hidden_key"),
    _pa(r"(بيانات|ملف)\s*JSON", 0.35, "ar_json_data"),
    _pa(r"(مراجعة|تعليق)\s*(العميل|المستخدم)", 0.20, "ar_user_review"),
    # AR secondary
    _pm(r"(لخص|حلل|اقرا|عالج).{0,60}\n---\n", 0.20, "ar_doc_framing"),
    # RU definitive
    _p(r"(скрытая|встроенная).{0,10}(инструкция|команда)", 0.65, "ru_hidden_instruction"),
    # ZH definitive
    _p(r"(隐藏指令|嵌入指令|给助手的)", 0.65, "zh_hidden_instruction"),
]

# -- obfuscation ------------------------------------------------------------

OBFUSCATION_SIGNALS: list[Signal] = [
    # Definitive
    _p(r"(base64|Base64|b64).{0,30}(decode|execute|run|فك|декодир|解码)", 0.65, "base64_decode_execute"),
    _pa(r"(بالعكس|اقلب).{0,20}(نفذ|اقرا|طبق)", 0.65, "ar_reverse_execute"),
    _pa(r"(فك\s*(ترميز|تشفير|الشفرة)).{0,30}(Base64|base64|b64)", 0.65, "ar_decode_b64"),
    _p(r"([\u0600-\u06FF]\u0640){3,}", 0.65, "ar_kashida_obfuscation"),
    _p(r"([\u0600-\u06FF][\u064B-\u065F\u0670]*[\-\s\.]{1,3}){5,}[\u0600-\u06FF]", 0.65, "ar_char_separated"),
    _p(r"[\uFB50-\uFDFF\uFE70-\uFEFF]{3,}", 0.65, "ar_presentation_forms"),
    # Primary
    _p(r"(\w[\s\-\.·\u0640]{1,3}){5,}\w", 0.35, "spaced_chars"),
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
    _pa(r"(وضع|نمط)\s*(المطور|الصيانة|التصحيح)", 0.65, "ar_dev_mode"),
    # AR primary
    _pa(r"(بلا|بدون|ازل|عطل|تعطيل).{0,20}((ال)?قيود|(ال)?فلاتر|(ال)?فلتر\w*|(ال)?سياسات|(ال)?حماية)", 0.35, "ar_no_restrictions"),
    _pa(r"(غير مفلتر|غير مقيد|بدون رقابة)", 0.35, "ar_unfiltered"),
    _pa(r"افعل\s+اي\s+شيء", 0.35, "ar_do_anything"),
    _pa(r"تجاوز.{0,15}(القيود|الحماية|السياسات|الفلاتر)", 0.35, "ar_bypass_restrictions"),
    _pa(r"(اختبار|فحص).{0,20}(امني|اختراق|مصرح)", 0.35, "ar_security_test"),
    _pa(r"(امر|اوامر)\s*(مباشر|من\s*(المسؤول|مسؤول|المطور|الادارة))", 0.35, "ar_direct_order"),
    _pa(r"(اعلى|كامل)\s*(صلاحية|سلطة)", 0.20, "ar_highest_authority"),
    # RU definitive
    _p(r"(режим|мод)\s*(разработчик|отладк|бога)", 0.65, "ru_dev_mode"),
    # RU primary
    _p(r"(без|убери|отключи).{0,20}(ограничен|фильтр|цензур|защит)", 0.35, "ru_no_restrictions"),
    # ZH definitive
    _p(r"(开发者|调试|上帝)\s*模式", 0.65, "zh_dev_mode"),
    # ZH primary
    _p(r"(取消|移除|关闭|禁用).{0,15}(限制|过滤|审查|安全)", 0.35, "zh_no_restrictions"),
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
    This function sorts known attacks into reason bins — it does NOT
    distinguish attack from benign. Several definitive patterns (weight
    0.65) fire on a single keyword like "system prompt" or <|endoftext|>,
    which would false-positive on benign text that merely discusses these
    concepts. Benign reason assignment uses source-level routing, not
    this classifier. See staging_pipeline_plan.md §Benign reason
    classification.
    """
    if not text or not text.strip():
        return None

    best: ReasonClassification | None = None

    for reason in AttackReason:
        result = _score_reason(text, reason)
        if result.confidence >= CONFIDENCE_FLOOR:
            if best is None or result.confidence > best.confidence:
                best = result
            elif (
                result.confidence == best.confidence
                and _SPECIFICITY[result.reason] > _SPECIFICITY[best.reason]
            ):
                best = result

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
