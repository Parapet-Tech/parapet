// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! L1 Harness — separates model scoring from infrastructure.
//!
//! The harness owns transforms (squash, quote detection, quote stripping),
//! multi-pass scoring orchestration, calibration, and signal assembly.
//! The model owns one thing: `fn score(&self, text: &str) -> f64`.

use crate::message::{Message, TrustLevel, Role};

// ---------------------------------------------------------------------------
// Thresholds shared between the engine verdict path and the signal extractor
// ---------------------------------------------------------------------------

/// Minimum raw margin delta to trigger mention dampening.
/// In raw SVM margin space: a delta of 1.0 means the quoted content
/// contributed ~1.0 margin units to the full-text score. Given benign
/// median ~= -4.4 and malicious median ~= 1.8, this is a meaningful shift.
pub const MENTION_RAW_DELTA_THRESHOLD: f64 = 1.0;

// ---------------------------------------------------------------------------
// Model trait
// ---------------------------------------------------------------------------

/// Calibration parameters for sigmoid mapping: `P = 1 / (1 + exp(-a * (score + b)))`.
#[derive(Debug, Clone, Copy)]
pub struct CalibrationParams {
    /// Sigmoid steepness. Controls confidence spread.
    pub a: f64,
    /// Sigmoid offset. Centers the decision boundary.
    pub b: f64,
}

/// A pluggable L1 scoring model.
///
/// The model's only job: given text, return a number.
/// Higher = more attack-like surface form.
/// The harness handles everything else.
pub trait L1Model: Send + Sync {
    /// Score a text fragment. Returns raw, unbounded margin.
    fn score(&self, text: &str) -> f64;

    /// Calibration parameters for this model's score distribution.
    fn calibration(&self) -> CalibrationParams;
}

// ---------------------------------------------------------------------------
// L1Signal — harness output contract
// ---------------------------------------------------------------------------

/// Structured evidence emitted by the L1 harness.
///
/// This is the stable contract between L1 and the signal bus.
/// Downstream consumers depend on this shape, not on the model.
///
/// No field in L1Signal is a verdict. Every field is evidence.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct L1Signal {
    /// Message index in the input slice.
    pub message_index: usize,

    /// Message role.
    pub role: Role,

    // -- Raw margins (canonical geometric values) --

    /// Full-text raw margin from the model.
    pub raw_score: f64,

    /// Raw margin after stripping quoted regions.
    pub raw_unquoted_score: f64,

    /// Raw margin after squash / deobfuscation.
    pub raw_squash_score: f64,

    // -- Scores (calibrated [0.0, 1.0]) --

    /// Full text score. "Does this text contain attack-like language?"
    pub score: f32,

    /// Score of text with quoted regions stripped.
    /// Low unquoted + high full = signal is coming from quoted regions.
    pub unquoted_score: f32,

    /// Score of squashed (deobfuscated) text.
    /// High squash + low raw = obfuscation was masking attack patterns.
    pub squash_score: f32,

    // -- Context bits (deterministic, model-independent) --

    /// Structural quoting markers detected.
    pub quote_detected: bool,

    // -- Derived --

    /// `raw_score - raw_unquoted_score`. Canonical quote-concentration signal.
    /// Routing thresholds operate on this value in raw margin space.
    pub raw_score_delta: f64,
}

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------

/// Convert a raw model margin to a calibrated probability [0.0, 1.0].
fn calibrate(raw_score: f64, params: &CalibrationParams) -> f32 {
    let p = 1.0 / (1.0 + (-params.a * (raw_score + params.b)).exp());
    p as f32
}

// ---------------------------------------------------------------------------
// Transforms — pure functions on text, model-independent
// ---------------------------------------------------------------------------

/// Strip all non-alphanumeric characters and lowercase. O(N) deobfuscation.
///
/// `"i. g-n o r e, P.R.E.V.I.O.U.S!"` → `"ignoreprevious"`
///
/// Moved from l1.rs — same logic, now harness-owned.
pub fn squash(text: &str) -> String {
    text.chars()
        .flat_map(|c| c.to_lowercase())
        .filter(|c| c.is_alphanumeric())
        .collect()
}

/// Check whether `text` contains at least one word-boundary-delimited `'...'` pair.
///
/// A `'` is a quote delimiter if NOT preceded by an alphanumeric char (or at SOL)
/// and the matching close `'` is NOT followed by an alphanumeric char (or at EOL).
/// This distinguishes `'attack'` (quoting) from `don't` (apostrophe).
fn has_word_boundary_single_quotes(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;
    while i < len {
        if chars[i] == '\'' {
            // Opening boundary: preceded by non-alphanumeric or SOL.
            let open_ok = i == 0 || !chars[i - 1].is_alphanumeric();
            if open_ok {
                // Scan for a closing boundary quote.
                for j in (i + 1)..len {
                    if chars[j] == '\'' {
                        let close_ok = j + 1 >= len || !chars[j + 1].is_alphanumeric();
                        if close_ok && j > i + 1 {
                            return true;
                        }
                    }
                }
            }
        }
        i += 1;
    }
    false
}

/// Find all word-boundary-delimited `'...'` ranges in `text`.
/// Returns byte-offset pairs `(start, end)` where `start` is the position of `'`
/// and `end` is one past the closing `'`. Pairs are greedy-first, non-overlapping.
///
/// Returns `None` if any word-boundary opening `'` has no matching close
/// (imbalance → caller must return original text unchanged).
fn find_single_quote_ranges(text: &str) -> Option<Vec<(usize, usize)>> {
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    // Build byte-offset map.
    let mut byte_offsets: Vec<usize> = Vec::with_capacity(len + 1);
    let mut pos = 0;
    for &ch in &chars {
        byte_offsets.push(pos);
        pos += ch.len_utf8();
    }
    byte_offsets.push(pos); // sentinel for end

    let mut ranges = Vec::new();
    let mut i = 0;
    while i < len {
        if chars[i] == '\'' {
            let open_ok = i == 0 || !chars[i - 1].is_alphanumeric();
            if open_ok {
                let mut found_close = false;
                for j in (i + 1)..len {
                    if chars[j] == '\'' {
                        let close_ok = j + 1 >= len || !chars[j + 1].is_alphanumeric();
                        if close_ok && j > i + 1 {
                            ranges.push((byte_offsets[i], byte_offsets[j + 1]));
                            i = j + 1;
                            found_close = true;
                            break;
                        }
                    }
                }
                if !found_close {
                    // Unmatched word-boundary opening quote → imbalance.
                    return None;
                }
                continue; // i already advanced past the close
            }
        }
        i += 1;
    }
    Some(ranges)
}

/// Coarse binary: does this text have structural quoting markers?
///
/// Detects paired delimiters and blockquote prefixes. Does NOT attempt
/// to identify what is quoted or parse nested structures.
///
/// Fail-safe: false negatives are harmless (raw_score_delta will be zero,
/// routing proceeds normally).
pub fn quote_detect(text: &str) -> bool {
    // Paired delimiter check — any matching open/close pair.
    const PAIRS: &[(char, char)] = &[
        ('"', '"'),               // ASCII double quote
        ('\u{201C}', '\u{201D}'), // " "
        ('\u{2018}', '\u{2019}'), // ' '
        ('\u{300C}', '\u{300D}'), // 「 」
        ('\u{300E}', '\u{300F}'), // 『 』
        ('\u{00AB}', '\u{00BB}'), // « »
    ];

    for &(open, close) in PAIRS {
        if let Some(start) = text.find(open) {
            // For same-char delimiters, look past the opening character.
            let after = start + open.len_utf8();
            if after < text.len() && text[after..].contains(close) {
                return true;
            }
        }
    }

    // ASCII single quotes — word-boundary heuristic to avoid apostrophes.
    // A ' is a quote delimiter if preceded by non-alphanumeric (or SOL)
    // and the matching close ' is followed by non-alphanumeric (or EOL).
    // "don't" → apostrophe (preceded by 'n'). 'attack' → delimiter.
    if has_word_boundary_single_quotes(text) {
        return true;
    }

    // Backtick fences: ``` or single ` pairs
    if text.contains("```") {
        let count = text.matches("```").count();
        if count >= 2 {
            return true;
        }
    }
    if let Some(first) = text.find('`') {
        if first + 1 < text.len() && text[first + 1..].contains('`') {
            return true;
        }
    }

    // Markdown blockquote: line starting with "> "
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("> ") || trimmed == ">" {
            return true;
        }
    }

    false
}

/// Remove content between detected quote boundaries. Returns the remaining text.
///
/// Stripping rules:
/// - Remove content between paired delimiters (greedy outermost)
/// - Remove fenced code blocks (``` delimited)
/// - Remove inline backtick content
/// - Remove content after `> ` line prefix (per line)
/// - Preserve surrounding text and whitespace
///
/// Fail-safe: if stripping produces empty text (entire message was quoted),
/// returns None. Caller should treat this as "no unquoted content" and
/// set raw_unquoted_score = raw_score (a quote with no surrounding context
/// is a command, not a mention).
///
/// If any delimiter is unbalanced, the entire function returns the original
/// text unchanged (fail-closed on parser confusion). Partial stripping never
/// leaks through.
pub fn strip_quotes(text: &str) -> Option<String> {
    // Work on a candidate. If any step encounters an imbalance, bail and
    // return the original text.
    let mut result = text.to_string();
    let mut any_stripped = false;

    // 1. Remove fenced code blocks (``` ... ```)
    loop {
        if let Some(start) = result.find("```") {
            let after_open = start + 3;
            if let Some(end_offset) = result[after_open..].find("```") {
                let end = after_open + end_offset + 3;
                result = format!("{}{}", &result[..start], &result[end..]);
                any_stripped = true;
                continue;
            } else {
                // Unbalanced fence — return original unchanged.
                return Some(text.to_string());
            }
        }
        break;
    }

    // 2. Remove paired delimiters (ASCII + Unicode)
    const PAIRS: &[(char, char)] = &[
        ('\u{201C}', '\u{201D}'), // " "
        ('\u{2018}', '\u{2019}'), // ' '
        ('\u{300C}', '\u{300D}'), // 「 」
        ('\u{300E}', '\u{300F}'), // 『 』
        ('\u{00AB}', '\u{00BB}'), // « »
    ];

    for &(open, close) in PAIRS {
        loop {
            if let Some(start) = result.find(open) {
                let after_open = start + open.len_utf8();
                if let Some(end_offset) = result[after_open..].find(close) {
                    let end = after_open + end_offset + close.len_utf8();
                    result = format!("{}{}", &result[..start], &result[end..]);
                    any_stripped = true;
                    continue;
                } else {
                    // Unbalanced — return original.
                    return Some(text.to_string());
                }
            }
            break;
        }
    }

    // 3. ASCII double quotes ("...") — same open/close character.
    loop {
        if let Some(start) = result.find('"') {
            let after_open = start + 1;
            if after_open < result.len() {
                if let Some(end_offset) = result[after_open..].find('"') {
                    let end = after_open + end_offset + 1;
                    result = format!("{}{}", &result[..start], &result[end..]);
                    any_stripped = true;
                    continue;
                } else {
                    // Unbalanced — return original.
                    return Some(text.to_string());
                }
            }
        }
        break;
    }

    // 4. ASCII single quotes ('...') — word-boundary heuristic.
    // Only strip pairs where the opening ' is NOT preceded by alphanumeric
    // and the closing ' is NOT followed by alphanumeric.
    // "don't do what's right" → no stripping (apostrophes between letters).
    // "He said 'attack' loudly" → strip 'attack'.
    match find_single_quote_ranges(&result) {
        None => {
            // Unbalanced word-boundary single quote — return original.
            return Some(text.to_string());
        }
        Some(ranges) if !ranges.is_empty() => {
            // Remove ranges in reverse order to preserve byte offsets.
            let mut r = result.clone();
            for &(start, end) in ranges.iter().rev() {
                r = format!("{}{}", &r[..start], &r[end..]);
            }
            result = r;
            any_stripped = true;
        }
        _ => {}
    }

    // 5. Remove inline backtick content (single `)
    loop {
        if let Some(start) = result.find('`') {
            let after_open = start + 1;
            if after_open < result.len() {
                if let Some(end_offset) = result[after_open..].find('`') {
                    let end = after_open + end_offset + 1;
                    result = format!("{}{}", &result[..start], &result[end..]);
                    any_stripped = true;
                    continue;
                } else {
                    // Unbalanced — return original.
                    return Some(text.to_string());
                }
            }
        }
        break;
    }

    // 6. Remove blockquote lines (> prefix)
    let lines: Vec<&str> = result.lines().collect();
    let filtered: Vec<&str> = lines
        .into_iter()
        .filter(|line| {
            let trimmed = line.trim_start();
            if trimmed.starts_with("> ") || trimmed == ">" {
                any_stripped = true;
                false
            } else {
                true
            }
        })
        .collect();
    result = filtered.join("\n");

    // Check if anything meaningful remains
    let trimmed = result.trim();
    if trimmed.is_empty() {
        if any_stripped {
            None // Fully quoted
        } else {
            Some(text.to_string()) // Nothing was stripped, nothing remains — degenerate input
        }
    } else {
        Some(result)
    }
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// L1 Harness — wraps any L1Model with deterministic transforms and
/// emits structured L1Signal evidence.
pub struct L1Harness;

impl L1Harness {
    /// Scan messages and emit per-message L1Signals.
    ///
    /// Only untrusted, non-empty messages are scored. For each:
    /// 1. Run transforms (squash, quote_detect, strip_quotes)
    /// 2. Score raw, squashed, and unquoted text variants
    /// 3. Calibrate all scores
    /// 4. Assemble L1Signal
    pub fn scan(messages: &[Message], model: &dyn L1Model) -> Vec<L1Signal> {
        let cal = model.calibration();
        let mut signals = Vec::new();

        for (i, msg) in messages.iter().enumerate() {
            if msg.trust == TrustLevel::Trusted {
                continue;
            }
            if msg.content.is_empty() {
                continue;
            }

            let text = &msg.content;

            // Transforms
            let squashed = squash(text);
            let has_quotes = quote_detect(text);
            let unquoted = strip_quotes(text);

            // Score raw text
            let raw_score = model.score(text);

            // Score squashed text
            // Fail-closed: if squash produces empty, squash_score = 0.0
            let raw_squash_score = if squashed.is_empty() {
                0.0
            } else {
                model.score(&squashed)
            };

            // Score unquoted text
            // Fail-closed: if strip_quotes returns None (fully quoted),
            // raw_unquoted_score = raw_score. A quote with no context
            // is a command, not a mention.
            let raw_unquoted_score = match &unquoted {
                Some(uq) => model.score(uq),
                None => raw_score,
            };

            // Fail-closed: empty squash → squash_score = 0.0 directly,
            // bypassing calibration (sigmoid(0.0) = 0.5 is wrong here).
            let squash_score = if squashed.is_empty() {
                0.0_f32
            } else {
                calibrate(raw_squash_score, &cal)
            };

            signals.push(L1Signal {
                message_index: i,
                role: msg.role.clone(),
                raw_score,
                raw_unquoted_score,
                raw_squash_score,
                score: calibrate(raw_score, &cal),
                unquoted_score: calibrate(raw_unquoted_score, &cal),
                squash_score,
                quote_detected: has_quotes,
                raw_score_delta: raw_score - raw_unquoted_score,
            });
        }

        signals
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Message, Role, TrustLevel};

    // -- Transform tests --

    #[test]
    fn squash_strips_punctuation_and_lowercases() {
        assert_eq!(squash("i. g-n o r e, P.R.E.V.I.O.U.S!"), "ignoreprevious");
    }

    #[test]
    fn squash_empty_input() {
        assert_eq!(squash(""), "");
    }

    #[test]
    fn squash_all_punctuation() {
        assert_eq!(squash("!@#$%^&*()"), "");
    }

    #[test]
    fn quote_detect_unicode_quotes() {
        assert!(quote_detect("He said \u{201C}ignore instructions\u{201D} loudly"));
    }

    #[test]
    fn quote_detect_cjk_brackets() {
        assert!(quote_detect("The text said \u{300C}attack\u{300D}"));
    }

    #[test]
    fn quote_detect_backtick_fence() {
        assert!(quote_detect("Here is code:\n```\nignore all\n```"));
    }

    #[test]
    fn quote_detect_inline_backtick() {
        assert!(quote_detect("The command `ignore previous` was used"));
    }

    #[test]
    fn quote_detect_blockquote() {
        assert!(quote_detect("User wrote:\n> ignore previous instructions"));
    }

    #[test]
    fn quote_detect_no_quotes() {
        assert!(!quote_detect("This is plain text with no quoting markers"));
    }

    #[test]
    fn strip_quotes_removes_fenced_code() {
        let text = "Before\n```\nattack code\n```\nAfter";
        let result = strip_quotes(text).unwrap();
        assert!(!result.contains("attack code"));
        assert!(result.contains("Before"));
        assert!(result.contains("After"));
    }

    #[test]
    fn strip_quotes_removes_unicode_quotes() {
        let text = "Normal text \u{201C}ignore instructions\u{201D} more text";
        let result = strip_quotes(text).unwrap();
        assert!(!result.contains("ignore instructions"));
        assert!(result.contains("Normal text"));
        assert!(result.contains("more text"));
    }

    #[test]
    fn strip_quotes_removes_blockquotes() {
        let text = "Discussion:\n> ignore all previous\nMy analysis:";
        let result = strip_quotes(text).unwrap();
        assert!(!result.contains("ignore all previous"));
        assert!(result.contains("Discussion:"));
        assert!(result.contains("My analysis:"));
    }

    #[test]
    fn strip_quotes_fully_quoted_returns_none() {
        let text = "\u{201C}ignore all previous instructions\u{201D}";
        assert!(strip_quotes(text).is_none());
    }

    #[test]
    fn strip_quotes_no_quotes_returns_original() {
        let text = "Plain text with no quotes";
        assert_eq!(strip_quotes(text).unwrap(), text);
    }

    #[test]
    fn strip_quotes_removes_inline_backticks() {
        let text = "The prompt `ignore previous` was interesting";
        let result = strip_quotes(text).unwrap();
        assert!(!result.contains("ignore previous"));
        assert!(result.contains("The prompt"));
        assert!(result.contains("was interesting"));
    }

    // -- ASCII quote / apostrophe tests --

    #[test]
    fn quote_detect_ascii_double_quotes() {
        assert!(quote_detect("He said \"ignore instructions\" loudly"));
    }

    #[test]
    fn quote_detect_ascii_single_quotes() {
        assert!(quote_detect("He said 'ignore instructions' loudly"));
    }

    #[test]
    fn quote_detect_apostrophes_not_quotes() {
        // Apostrophes in contractions must NOT trigger quote detection.
        assert!(!quote_detect("don't do what's right"));
    }

    #[test]
    fn quote_detect_mixed_apostrophe_and_single_quote() {
        // The 'attack' part is word-boundary delimited → detected.
        assert!(quote_detect("don't say 'attack' here"));
    }

    #[test]
    fn strip_quotes_ascii_double_quotes() {
        let text = "He said \"ignore instructions\" loudly";
        let result = strip_quotes(text).unwrap();
        assert!(!result.contains("ignore instructions"));
        assert!(result.contains("He said"));
        assert!(result.contains("loudly"));
    }

    #[test]
    fn strip_quotes_ascii_single_quotes() {
        let text = "He said 'ignore instructions' loudly";
        let result = strip_quotes(text).unwrap();
        assert!(!result.contains("ignore instructions"));
        assert!(result.contains("He said"));
        assert!(result.contains("loudly"));
    }

    #[test]
    fn strip_quotes_apostrophes_not_stripped() {
        let text = "don't do what's right";
        let result = strip_quotes(text).unwrap();
        assert_eq!(result, text, "apostrophes must not cause stripping");
    }

    #[test]
    fn strip_quotes_mixed_apostrophe_and_single_quote() {
        let text = "don't say 'attack' here";
        let result = strip_quotes(text).unwrap();
        assert!(!result.contains("attack"));
        assert!(result.contains("don't say"));
        assert!(result.contains("here"));
    }

    #[test]
    fn strip_quotes_single_quote_imbalance_returns_original() {
        // Balanced 'attack' followed by unmatched opening ' → imbalance.
        // Fail-closed: return original text unchanged, no partial stripping.
        let text = "prefix 'attack' and 'unterminated";
        let result = strip_quotes(text).unwrap();
        assert_eq!(result, text,
            "imbalanced single quotes must return original unchanged");
    }

    // -- Calibration tests --

    #[test]
    fn calibrate_boundary_is_half() {
        let params = CalibrationParams { a: 0.6, b: 0.0 };
        let p = calibrate(0.0, &params);
        assert!((p - 0.5).abs() < 1e-6);
    }

    #[test]
    fn calibrate_monotonic() {
        let params = CalibrationParams { a: 0.6, b: 0.0 };
        let scores = [-5.0, -2.0, 0.0, 2.0, 5.0];
        let calibrated: Vec<f32> = scores.iter().map(|&s| calibrate(s, &params)).collect();
        for w in calibrated.windows(2) {
            assert!(w[1] > w[0], "calibrated scores must be monotonically increasing");
        }
    }

    #[test]
    fn calibrate_strong_values() {
        let params = CalibrationParams { a: 0.6, b: 0.0 };
        assert!(calibrate(5.0, &params) > 0.9);
        assert!(calibrate(-5.0, &params) < 0.1);
    }

    // -- Harness integration tests --

    /// Trivial test model: returns a fixed score for any text containing "attack",
    /// negative otherwise.
    struct FixedModel;

    impl L1Model for FixedModel {
        fn score(&self, text: &str) -> f64 {
            if text.contains("attack") {
                3.0
            } else {
                -3.0
            }
        }

        fn calibration(&self) -> CalibrationParams {
            CalibrationParams { a: 0.6, b: 0.0 }
        }
    }

    fn make_msg(content: &str, trust: TrustLevel) -> Message {
        Message {
            role: Role::User,
            content: content.to_string(),
            tool_calls: vec![],
            tool_call_id: None,
            tool_name: None,
            trust,
            trust_spans: vec![],
        }
    }

    #[test]
    fn harness_skips_trusted_messages() {
        let messages = vec![
            make_msg("attack payload", TrustLevel::Trusted),
            make_msg("benign text", TrustLevel::Untrusted),
        ];
        let signals = L1Harness::scan(&messages, &FixedModel);
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].message_index, 1);
    }

    #[test]
    fn harness_skips_empty_messages() {
        let messages = vec![make_msg("", TrustLevel::Untrusted)];
        let signals = L1Harness::scan(&messages, &FixedModel);
        assert!(signals.is_empty());
    }

    #[test]
    fn harness_scores_attack_text() {
        let messages = vec![make_msg("this is an attack", TrustLevel::Untrusted)];
        let signals = L1Harness::scan(&messages, &FixedModel);
        assert_eq!(signals.len(), 1);
        assert!(signals[0].raw_score > 0.0);
        assert!(signals[0].score > 0.5);
    }

    #[test]
    fn harness_scores_benign_text() {
        let messages = vec![make_msg("hello world", TrustLevel::Untrusted)];
        let signals = L1Harness::scan(&messages, &FixedModel);
        assert_eq!(signals.len(), 1);
        assert!(signals[0].raw_score < 0.0);
        assert!(signals[0].score < 0.5);
    }

    #[test]
    fn harness_quoted_attack_produces_delta() {
        // Attack text is inside quotes; surrounding text is benign.
        let text = "Here is what the user wrote: \u{201C}attack payload\u{201D} — seems fine";
        let messages = vec![make_msg(text, TrustLevel::Untrusted)];
        let signals = L1Harness::scan(&messages, &FixedModel);
        let s = &signals[0];

        // Full text contains "attack" → high raw_score
        assert!(s.raw_score > 0.0);
        // Unquoted text has "attack" stripped → low raw_unquoted_score
        assert!(s.raw_unquoted_score < 0.0);
        // Delta should be positive (attack signal came from quoted region)
        assert!(s.raw_score_delta > 0.0);
        assert!(s.quote_detected);
    }

    #[test]
    fn harness_fully_quoted_failsafe() {
        // Entire message is quoted — strip_quotes returns None.
        // Fail-closed: raw_unquoted_score = raw_score, delta = 0.
        let text = "\u{201C}attack payload\u{201D}";
        let messages = vec![make_msg(text, TrustLevel::Untrusted)];
        let signals = L1Harness::scan(&messages, &FixedModel);
        let s = &signals[0];

        assert!(s.quote_detected);
        assert!((s.raw_score_delta).abs() < f64::EPSILON,
            "fully-quoted message must have zero delta (fail-closed)");
        assert_eq!(s.raw_unquoted_score, s.raw_score);
    }

    #[test]
    fn harness_squash_empty_failsafe() {
        // All-punctuation message: squash produces empty string.
        // Fail-closed: squash_score = 0.0.
        let text = "!@#$%^&*()";
        let messages = vec![make_msg(text, TrustLevel::Untrusted)];
        let signals = L1Harness::scan(&messages, &FixedModel);
        let s = &signals[0];

        assert_eq!(s.raw_squash_score, 0.0);
        assert_eq!(s.squash_score, 0.0,
            "empty squash must emit squash_score = 0.0 (no signal), not calibrate(0.0)");
    }
}
