// NFKC normalization, HTML strip, encoding hygiene — defined in M1.6
//
// L0 normalization is a pure string transformation applied to every message
// before constraint evaluation. It is idempotent: normalizing already-normalized
// content returns the same result.

use crate::message::Message;
use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;

// ---------------------------------------------------------------------------
// Trait: Normalizer
// ---------------------------------------------------------------------------

/// Pure string normalizer. Implementations must be Send + Sync so they can be
/// shared across async tasks.
pub trait Normalizer: Send + Sync {
    /// Normalize a single string. The result must be idempotent:
    /// `normalize(normalize(x)) == normalize(x)` for all `x`.
    fn normalize(&self, input: &str) -> String;
}

// ---------------------------------------------------------------------------
// Implementation: L0Normalizer
// ---------------------------------------------------------------------------

/// L0 normalizer that applies, in order:
///
/// 1. NFKC unicode normalization (fullwidth -> ASCII, combining chars, etc.)
/// 2. HTML tag stripping (including script/style content removal)
/// 3. Zero-width / invisible character removal
pub struct L0Normalizer;

impl L0Normalizer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for L0Normalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Normalizer for L0Normalizer {
    fn normalize(&self, input: &str) -> String {
        // Step 1: NFKC normalization (handles fullwidth chars + combining chars)
        let nfkc: String = input.nfkc().collect();

        // Step 2: Strip HTML tags (including script/style content)
        let stripped = strip_html(&nfkc);

        // Step 3: Remove zero-width / invisible characters
        remove_invisible_chars(&stripped)
    }
}

// ---------------------------------------------------------------------------
// Convenience: normalize all messages in a slice
// ---------------------------------------------------------------------------

/// Normalize the `content` field of every message in the slice.
pub fn normalize_messages(normalizer: &dyn Normalizer, messages: &mut [Message]) {
    for msg in messages {
        msg.content = normalizer.normalize(&msg.content);
    }
}

// ---------------------------------------------------------------------------
// Internal: HTML stripping state machine
// ---------------------------------------------------------------------------

/// Strip HTML tags from a string using a simple state machine.
///
/// - Removes everything between `<` and `>` (including the delimiters).
/// - For `<script>` and `<style>` tags, also removes the content between
///   the opening and closing tags.
fn strip_html(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        if chars[i] == '<' {
            // Check if this is a <script or <style tag
            if let Some(tag_name) = match_dangerous_open_tag(&chars, i, len) {
                // Skip past the closing tag (e.g., </script> or </style>)
                i = skip_to_closing_tag(&chars, i, len, &tag_name);
            } else {
                // Regular tag: skip to closing >
                i = skip_tag(&chars, i, len);
            }
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

/// Check if the tag at position `start` is a <script or <style opening tag.
/// Returns the lowercase tag name if it matches, otherwise None.
fn match_dangerous_open_tag(chars: &[char], start: usize, len: usize) -> Option<String> {
    // Must start with '<' and not be a closing tag '</'
    if start + 1 >= len || chars[start] != '<' || chars[start + 1] == '/' {
        return None;
    }

    // Extract tag name (letters only, up to space or >)
    let mut name = String::new();
    let mut j = start + 1;
    while j < len && chars[j].is_ascii_alphabetic() {
        name.push(chars[j].to_ascii_lowercase());
        j += 1;
    }

    if name == "script" || name == "style" {
        // Verify the tag name is followed by whitespace, '>', or '/' (not just a prefix)
        if j < len && (chars[j] == '>' || chars[j] == ' ' || chars[j] == '/' || chars[j] == '\t' || chars[j] == '\n') {
            Some(name)
        } else if j >= len {
            // Tag name extends to end of input (malformed) -- treat as dangerous
            Some(name)
        } else {
            None
        }
    } else {
        None
    }
}

/// Skip from the opening tag at `start` past the matching closing tag.
/// For example, `<script>alert(1)</script>` -- returns the index after `>` of `</script>`.
fn skip_to_closing_tag(chars: &[char], start: usize, len: usize, tag_name: &str) -> usize {
    // First, skip past the opening tag's >
    let mut i = start;
    while i < len && chars[i] != '>' {
        i += 1;
    }
    if i < len {
        i += 1; // skip the '>'
    }

    // Now find the closing tag </tag_name>
    let closing = format!("</{}>", tag_name);
    let closing_chars: Vec<char> = closing.chars().collect();

    while i < len {
        if i + closing_chars.len() <= len {
            let mut matched = true;
            for (k, cc) in closing_chars.iter().enumerate() {
                if chars[i + k].to_ascii_lowercase() != *cc {
                    matched = false;
                    break;
                }
            }
            if matched {
                return i + closing_chars.len();
            }
        }
        i += 1;
    }

    // No closing tag found: skip to end (malformed HTML)
    len
}

/// Skip a regular HTML tag, returning the index after the closing '>'.
fn skip_tag(chars: &[char], start: usize, len: usize) -> usize {
    let mut i = start + 1;
    while i < len && chars[i] != '>' {
        i += 1;
    }
    if i < len {
        i + 1 // skip the '>'
    } else {
        len
    }
}

// ---------------------------------------------------------------------------
// Internal: invisible character removal
// ---------------------------------------------------------------------------

/// Returns true if the character is a zero-width or invisible character
/// that should be stripped during normalization.
fn is_invisible(c: char) -> bool {
    matches!(
        c,
        '\u{200B}' // Zero-width space
        | '\u{200C}' // Zero-width non-joiner
        | '\u{200D}' // Zero-width joiner
        | '\u{FEFF}' // BOM / zero-width no-break space
        | '\u{00AD}' // Soft hyphen
        | '\u{200E}' // Left-to-right mark
        | '\u{200F}' // Right-to-left mark
        | '\u{202A}' // Left-to-right embedding
        | '\u{202B}' // Right-to-left embedding
        | '\u{202C}' // Pop directional formatting
        | '\u{202D}' // Left-to-right override
        | '\u{202E}' // Right-to-left override
        | '\u{2060}' // Word joiner
        | '\u{2061}' // Function application
        | '\u{2062}' // Invisible times
        | '\u{2063}' // Invisible separator
        | '\u{2064}' // Invisible plus
        | '\u{FE00}'..='\u{FE0F}' // Variation selectors 1-16
        | '\u{180E}' // Mongolian vowel separator
    )
}

/// Remove all invisible / zero-width characters from a string.
fn remove_invisible_chars(input: &str) -> String {
    input.chars().filter(|c| !is_invisible(*c)).collect()
}

/// Apply NFKC normalization and strip invisible characters (no HTML stripping).
/// Used for security-critical string comparisons (e.g., constraint predicates)
/// where Unicode tricks could bypass checks.
pub fn normalize_for_comparison(input: &str) -> String {
    let nfkc: String = input.nfkc().collect();
    remove_invisible_chars(&nfkc)
}

// ---------------------------------------------------------------------------
// Trust span remapping (M4)
// ---------------------------------------------------------------------------

/// Maps a byte range in the original string to a byte range in the normalized string.
#[derive(Debug, Clone, PartialEq)]
pub struct OffsetMapping {
    /// Start byte in the original string.
    pub old_start: usize,
    /// End byte in the original string (exclusive).
    pub old_end: usize,
    /// Start byte in the normalized string.
    pub new_start: usize,
    /// End byte in the normalized string (exclusive).
    pub new_end: usize,
}

/// Remap trust spans from pre-normalization to post-normalization byte offsets.
///
/// Strategy: for each unique span boundary, normalize the prefix of the original
/// string up to that boundary and record the resulting length. This tells us
/// exactly where that byte position maps to in the normalized output.
///
/// Complexity: O(k * n) where k = number of unique span boundaries (typically
/// 2-40) and n = input length. Acceptable for typical message sizes.
pub fn remap_trust_spans(
    normalizer: &dyn Normalizer,
    original: &str,
    spans: &mut [crate::trust::TrustSpan],
) {
    if spans.is_empty() || original.is_empty() {
        return;
    }

    // Collect all unique boundary positions we need to map.
    let mut boundaries: Vec<usize> = Vec::new();
    for span in spans.iter() {
        boundaries.push(span.start);
        boundaries.push(span.end);
    }
    boundaries.sort_unstable();
    boundaries.dedup();

    // For each boundary, normalize the prefix up to that point and record the output length.
    let mut boundary_map: HashMap<usize, usize> = HashMap::new();

    for &boundary in &boundaries {
        // Clamp to a valid UTF-8 character boundary.
        let clamped = clamp_to_char_boundary(original, boundary);
        let prefix = &original[..clamped];
        let normalized_prefix = normalizer.normalize(prefix);
        boundary_map.insert(boundary, normalized_prefix.len());
    }

    // Remap each span.
    for span in spans.iter_mut() {
        let new_start = boundary_map.get(&span.start).copied().unwrap_or(span.start);
        let new_end = boundary_map.get(&span.end).copied().unwrap_or(span.end);

        // Conservative: if remapping produces an inverted range, swap to preserve coverage.
        if new_start <= new_end {
            span.start = new_start;
            span.end = new_end;
        } else {
            span.start = new_end;
            span.end = new_start;
        }
    }
}

/// Clamp a byte offset to the nearest valid UTF-8 character boundary.
/// If the offset falls in the middle of a multi-byte character, round down
/// to the start of that character.
fn clamp_to_char_boundary(s: &str, offset: usize) -> usize {
    if offset >= s.len() {
        return s.len();
    }
    let mut pos = offset;
    while pos > 0 && !s.is_char_boundary(pos) {
        pos -= 1;
    }
    pos
}

/// Normalize the `content` field of every message and remap trust spans.
///
/// For messages with trust spans, the original content is preserved long enough
/// to remap span offsets before being replaced with the normalized content.
/// Messages without trust spans are normalized directly (no overhead).
pub fn normalize_messages_with_spans(normalizer: &dyn Normalizer, messages: &mut [Message]) {
    for msg in messages {
        if !msg.trust_spans.is_empty() {
            let original = msg.content.clone();
            msg.content = normalizer.normalize(&original);
            remap_trust_spans(normalizer, &original, &mut msg.trust_spans);
        } else {
            msg.content = normalizer.normalize(&msg.content);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Message, Role};

    fn normalizer() -> L0Normalizer {
        L0Normalizer::new()
    }

    // -------------------------------------------------------------------
    // 1. Fullwidth chars -> ASCII
    // -------------------------------------------------------------------

    #[test]
    fn fullwidth_chars_normalized_to_ascii() {
        let n = normalizer();
        // Fullwidth 'ignore': U+FF49 U+FF47 U+FF4E U+FF4F U+FF52 U+FF45
        let input = "\u{FF49}\u{FF47}\u{FF4E}\u{FF4F}\u{FF52}\u{FF45}";
        assert_eq!(n.normalize(input), "ignore");
    }

    // -------------------------------------------------------------------
    // 2. HTML tags stripped
    // -------------------------------------------------------------------

    #[test]
    fn html_tags_stripped() {
        let n = normalizer();
        assert_eq!(n.normalize("<b>ignore</b>"), "ignore");
    }

    #[test]
    fn self_closing_html_tags_stripped() {
        let n = normalizer();
        assert_eq!(n.normalize("hello<br/>world"), "helloworld");
    }

    #[test]
    fn html_attributes_stripped() {
        let n = normalizer();
        assert_eq!(
            n.normalize("<a href=\"http://evil.com\">ignore</a>"),
            "ignore"
        );
    }

    // -------------------------------------------------------------------
    // 3. Nested HTML stripped
    // -------------------------------------------------------------------

    #[test]
    fn nested_html_with_script_stripped() {
        let n = normalizer();
        assert_eq!(
            n.normalize("<div><script>alert(1)</script>ignore</div>"),
            "ignore"
        );
    }

    #[test]
    fn deeply_nested_html_stripped() {
        let n = normalizer();
        assert_eq!(
            n.normalize("<div><p><span>ignore</span></p></div>"),
            "ignore"
        );
    }

    // -------------------------------------------------------------------
    // 4. Zero-width characters removed
    // -------------------------------------------------------------------

    #[test]
    fn zero_width_space_removed() {
        let n = normalizer();
        assert_eq!(n.normalize("ig\u{200B}nore"), "ignore");
    }

    #[test]
    fn zero_width_non_joiner_removed() {
        let n = normalizer();
        assert_eq!(n.normalize("ig\u{200C}nore"), "ignore");
    }

    #[test]
    fn zero_width_joiner_removed() {
        let n = normalizer();
        assert_eq!(n.normalize("ig\u{200D}nore"), "ignore");
    }

    #[test]
    fn bom_removed() {
        let n = normalizer();
        assert_eq!(n.normalize("\u{FEFF}ignore"), "ignore");
    }

    #[test]
    fn soft_hyphen_removed() {
        let n = normalizer();
        assert_eq!(n.normalize("ig\u{00AD}nore"), "ignore");
    }

    // -------------------------------------------------------------------
    // 5. Combining characters handled correctly (NFKC)
    // -------------------------------------------------------------------

    #[test]
    fn combining_accent_normalized_to_nfc() {
        let n = normalizer();
        // e + combining acute accent -> e-acute (U+00E9)
        let decomposed = "e\u{0301}";
        let result = n.normalize(decomposed);
        assert_eq!(result, "\u{00E9}"); // precomposed e-acute
    }

    #[test]
    fn already_composed_char_unchanged() {
        let n = normalizer();
        let result = n.normalize("\u{00E9}"); // precomposed e-acute
        assert_eq!(result, "\u{00E9}");
    }

    // -------------------------------------------------------------------
    // 6. Idempotent
    // -------------------------------------------------------------------

    #[test]
    fn idempotent_already_normalized() {
        let n = normalizer();
        let input = "already normalized content";
        let once = n.normalize(input);
        let twice = n.normalize(&once);
        assert_eq!(once, twice);
    }

    #[test]
    fn idempotent_after_complex_normalization() {
        let n = normalizer();
        let input = "<b>\u{FF48}\u{200B}ello</b>";
        let once = n.normalize(input);
        let twice = n.normalize(&once);
        assert_eq!(once, twice);
        assert_eq!(once, "hello");
    }

    // -------------------------------------------------------------------
    // 7. Empty string -> empty string
    // -------------------------------------------------------------------

    #[test]
    fn empty_string_returns_empty() {
        let n = normalizer();
        assert_eq!(n.normalize(""), "");
    }

    // -------------------------------------------------------------------
    // 8. Mixed: fullwidth + HTML + zero-width in one string
    // -------------------------------------------------------------------

    #[test]
    fn mixed_fullwidth_html_zero_width() {
        let n = normalizer();
        // Fullwidth 'h' + zero-width space + <b> tag + fullwidth 'i' + </b>
        let input = "\u{FF48}\u{200B}<b>\u{FF49}</b>";
        let result = n.normalize(input);
        assert_eq!(result, "hi");
    }

    // -------------------------------------------------------------------
    // 9. Script tag content removed (not just the tags)
    // -------------------------------------------------------------------

    #[test]
    fn script_tag_content_removed() {
        let n = normalizer();
        assert_eq!(
            n.normalize("before<script>var x = 1; alert('xss');</script>after"),
            "beforeafter"
        );
    }

    #[test]
    fn style_tag_content_removed() {
        let n = normalizer();
        assert_eq!(
            n.normalize("before<style>body { display: none; }</style>after"),
            "beforeafter"
        );
    }

    #[test]
    fn script_tag_case_insensitive() {
        let n = normalizer();
        assert_eq!(
            n.normalize("before<SCRIPT>evil()</SCRIPT>after"),
            "beforeafter"
        );
    }

    // -------------------------------------------------------------------
    // Additional edge cases
    // -------------------------------------------------------------------

    #[test]
    fn unclosed_tag_stripped() {
        let n = normalizer();
        // Unclosed tag: everything from '<' to end is stripped
        assert_eq!(n.normalize("hello<div"), "hello");
    }

    #[test]
    fn angle_brackets_in_plain_text_stripped() {
        // By design, stray '<...>' sequences are stripped.
        // This is intentional -- we prioritize safety over preserving
        // ambiguous content that looks like HTML.
        let n = normalizer();
        assert_eq!(n.normalize("3 < 5 > 2"), "3  2");
    }

    #[test]
    fn multiple_invisible_chars_all_removed() {
        let n = normalizer();
        let input = "\u{200B}\u{200C}\u{200D}\u{FEFF}\u{00AD}hello\u{200E}\u{200F}";
        assert_eq!(n.normalize(input), "hello");
    }

    #[test]
    fn normalize_messages_updates_all_content() {
        let n = normalizer();
        let mut messages = vec![
            Message::new(Role::User, "<b>hello</b>"),
            Message::new(Role::Assistant, "\u{FF48}\u{FF49}"),
            Message::new(Role::System, "clean content"),
        ];
        normalize_messages(&n, &mut messages);
        assert_eq!(messages[0].content, "hello");
        assert_eq!(messages[1].content, "hi");
        assert_eq!(messages[2].content, "clean content");
    }

    #[test]
    fn normalize_messages_empty_slice() {
        let n = normalizer();
        let mut messages: Vec<Message> = vec![];
        normalize_messages(&n, &mut messages);
        assert!(messages.is_empty());
    }

    #[test]
    fn whitespace_preserved() {
        let n = normalizer();
        assert_eq!(n.normalize("hello  world\n\ttab"), "hello  world\n\ttab");
    }

    #[test]
    fn only_html_returns_empty() {
        let n = normalizer();
        assert_eq!(n.normalize("<div><p></p></div>"), "");
    }

    #[test]
    fn script_without_closing_tag_strips_to_end() {
        let n = normalizer();
        // Malformed: no closing </script> -- everything after <script> is removed
        assert_eq!(n.normalize("before<script>evil code"), "before");
    }

    // -------------------------------------------------------------------
    // Trust span remapping tests (M4)
    // -------------------------------------------------------------------

    #[test]
    fn remap_spans_identity_ascii() {
        // ASCII-only text: no normalization changes, offsets unchanged.
        let n = normalizer();
        let original = "hello world";
        let mut spans = vec![crate::trust::TrustSpan::untrusted(6, 11, "test")];
        remap_trust_spans(&n, original, &mut spans);
        assert_eq!(spans[0].start, 6);
        assert_eq!(spans[0].end, 11);
    }

    #[test]
    fn remap_spans_invisible_char_removed() {
        // "he\u{200B}llo" -> "hello" (zero-width space at byte 2 removed)
        // ZWS is 3 bytes (E2 80 8B), so "he" = 2 bytes, ZWS = 3 bytes,
        // "llo" starts at byte 5 in the original.
        let n = normalizer();
        let original = "he\u{200B}llo";
        assert_eq!(original.len(), 8); // h=1, e=1, ZWS=3, l=1, l=1, o=1
        let mut spans = vec![crate::trust::TrustSpan::untrusted(5, 8, "test")]; // "llo" in original
        remap_trust_spans(&n, original, &mut spans);
        let normalized = n.normalize(original);
        assert_eq!(normalized, "hello");
        // "llo" in "hello" is at bytes 2..5
        assert_eq!(spans[0].start, 2);
        assert_eq!(spans[0].end, 5);
    }

    #[test]
    fn remap_spans_html_stripped() {
        // "<b>hello</b>" -> "hello"
        // "hello" in original is at bytes 3..8, in normalized at 0..5
        let n = normalizer();
        let original = "<b>hello</b>";
        let mut spans = vec![crate::trust::TrustSpan::untrusted(3, 8, "test")];
        remap_trust_spans(&n, original, &mut spans);
        assert_eq!(n.normalize(original), "hello");
        assert_eq!(spans[0].start, 0);
        assert_eq!(spans[0].end, 5);
    }

    #[test]
    fn remap_spans_nfkc_fullwidth() {
        // Fullwidth "AB" -> "AB" (each fullwidth char is 3 bytes, ASCII is 1 byte)
        let n = normalizer();
        let original = "\u{FF21}\u{FF22}"; // Fullwidth A, B (3 bytes each)
        assert_eq!(original.len(), 6);
        let mut spans = vec![crate::trust::TrustSpan::untrusted(0, 6, "test")]; // entire string
        remap_trust_spans(&n, original, &mut spans);
        assert_eq!(n.normalize(original), "AB");
        assert_eq!(spans[0].start, 0);
        assert_eq!(spans[0].end, 2);
    }

    #[test]
    fn remap_spans_empty_input() {
        let n = normalizer();
        let mut spans = vec![crate::trust::TrustSpan::untrusted(0, 0, "test")];
        remap_trust_spans(&n, "", &mut spans);
        assert_eq!(spans[0].start, 0);
        assert_eq!(spans[0].end, 0);
    }

    #[test]
    fn remap_spans_no_spans_is_noop() {
        let n = normalizer();
        let mut spans: Vec<crate::trust::TrustSpan> = Vec::new();
        remap_trust_spans(&n, "hello", &mut spans);
        assert!(spans.is_empty());
    }

    #[test]
    fn remap_spans_multiple_spans() {
        // "he<b>llo</b> wo<i>rld</i>" -> "hello world"
        let n = normalizer();
        let original = "he<b>llo</b> wo<i>rld</i>";
        let normalized = n.normalize(original);
        assert_eq!(normalized, "hello world");

        // Span 1: "llo" inside <b> tags, bytes 5..8 in original
        // Span 2: "rld" inside <i> tags, bytes 18..21 in original
        let mut spans = vec![
            crate::trust::TrustSpan::untrusted(5, 8, "test1"),
            crate::trust::TrustSpan::untrusted(18, 21, "test2"),
        ];
        remap_trust_spans(&n, original, &mut spans);

        // "llo" in "hello world" -> bytes 2..5
        assert_eq!(spans[0].start, 2);
        assert_eq!(spans[0].end, 5);
        // "rld" in "hello world" -> bytes 8..11
        assert_eq!(spans[1].start, 8);
        assert_eq!(spans[1].end, 11);
    }

    // -------------------------------------------------------------------
    // clamp_to_char_boundary tests
    // -------------------------------------------------------------------

    #[test]
    fn clamp_to_char_boundary_on_ascii() {
        let s = "hello";
        assert_eq!(clamp_to_char_boundary(s, 3), 3);
        assert_eq!(clamp_to_char_boundary(s, 0), 0);
        assert_eq!(clamp_to_char_boundary(s, 5), 5);
        assert_eq!(clamp_to_char_boundary(s, 100), 5);
    }

    #[test]
    fn clamp_to_char_boundary_on_multibyte() {
        // "h" + e-acute (U+00E9, 2 bytes: C3 A9) + "llo"
        // byte 0: 'h', byte 1: start of e-acute, byte 2: continuation, byte 3: 'l'
        let s = "h\u{00E9}llo";
        assert_eq!(clamp_to_char_boundary(s, 0), 0);
        assert_eq!(clamp_to_char_boundary(s, 1), 1); // start of e-acute (valid boundary)
        assert_eq!(clamp_to_char_boundary(s, 2), 1); // continuation byte of e-acute -> round down to 1
        assert_eq!(clamp_to_char_boundary(s, 3), 3); // 'l' (valid boundary after e-acute)
    }

    // -------------------------------------------------------------------
    // normalize_messages_with_spans tests
    // -------------------------------------------------------------------

    #[test]
    fn normalize_messages_with_spans_remaps() {
        let n = normalizer();
        let mut msg = Message::new(Role::User, "<b>untrusted</b> data");
        msg.trust_spans
            .push(crate::trust::TrustSpan::untrusted(3, 12, "rag")); // "untrusted" inside <b>

        let mut messages = vec![msg];
        normalize_messages_with_spans(&n, &mut messages);

        assert_eq!(messages[0].content, "untrusted data");
        // Span should now point to "untrusted" in the normalized string (bytes 0..9)
        assert_eq!(messages[0].trust_spans[0].start, 0);
        assert_eq!(messages[0].trust_spans[0].end, 9);
    }

    #[test]
    fn normalize_messages_with_spans_no_spans_still_normalizes() {
        let n = normalizer();
        let mut messages = vec![Message::new(Role::User, "<b>hello</b>")];
        normalize_messages_with_spans(&n, &mut messages);
        assert_eq!(messages[0].content, "hello");
        assert!(messages[0].trust_spans.is_empty());
    }

    #[test]
    fn normalize_messages_with_spans_preserves_span_metadata() {
        // Ensure source label and trust level survive remapping.
        let n = normalizer();
        let mut msg = Message::new(Role::User, "\u{200B}data");
        msg.trust_spans
            .push(crate::trust::TrustSpan::untrusted(3, 7, "web_search"));

        let mut messages = vec![msg];
        normalize_messages_with_spans(&n, &mut messages);

        assert_eq!(messages[0].content, "data");
        assert_eq!(
            messages[0].trust_spans[0].level,
            crate::message::TrustLevel::Untrusted
        );
        assert_eq!(
            messages[0].trust_spans[0].source.as_deref(),
            Some("web_search")
        );
    }

    #[test]
    fn normalize_messages_with_spans_mixed_messages() {
        // Mix of messages with and without spans -- both should be normalized.
        let n = normalizer();
        let mut msg_with_span = Message::new(Role::User, "<b>hello</b>");
        msg_with_span
            .trust_spans
            .push(crate::trust::TrustSpan::untrusted(3, 8, "rag"));
        let msg_without_span = Message::new(Role::Assistant, "\u{FF48}i");

        let mut messages = vec![msg_with_span, msg_without_span];
        normalize_messages_with_spans(&n, &mut messages);

        assert_eq!(messages[0].content, "hello");
        assert_eq!(messages[0].trust_spans[0].start, 0);
        assert_eq!(messages[0].trust_spans[0].end, 5);

        assert_eq!(messages[1].content, "hi");
        assert!(messages[1].trust_spans.is_empty());
    }
}
