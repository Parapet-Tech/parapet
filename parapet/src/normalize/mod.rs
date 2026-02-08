// NFKC normalization, HTML strip, encoding hygiene — defined in M1.6
//
// L0 normalization is a pure string transformation applied to every message
// before constraint evaluation. It is idempotent: normalizing already-normalized
// content returns the same result.

use crate::message::Message;
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
}
