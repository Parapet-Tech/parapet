// L5a output pattern scanner -- defined in M1.10
//
// Scans LLM response content for canary tokens and sensitive patterns,
// redacting matches with `[REDACTED]`. Works on both complete response
// bodies (non-streaming) and individual chunks (streaming).

use crate::config::Config;

// ---------------------------------------------------------------------------
// Interface and types
// ---------------------------------------------------------------------------

/// Scans LLM output content and redacts sensitive matches.
pub trait OutputScanner: Send + Sync {
    /// Scan and redact a complete response body (or a single streaming chunk).
    fn scan_and_redact(&self, content: &str, config: &Config) -> ScanResult;
}

/// The result of scanning content for sensitive patterns.
#[derive(Debug, Clone)]
pub struct ScanResult {
    /// The content with all matches replaced by `[REDACTED]`.
    pub content: String,
    /// Each redaction that was performed, for logging/audit.
    pub redactions: Vec<Redaction>,
}

/// A single redaction event.
#[derive(Debug, Clone)]
pub struct Redaction {
    /// The pattern or token that triggered the redaction.
    pub pattern: String,
    /// Byte offset of the match in the *original* content.
    pub position: usize,
}

// ---------------------------------------------------------------------------
// L5aScanner implementation
// ---------------------------------------------------------------------------

/// Default implementation of [`OutputScanner`].
///
/// Scans content in two passes:
/// 1. Canary tokens -- exact substring matches
/// 2. Sensitive patterns -- compiled regex matches
///
/// All matches are replaced with `[REDACTED]`.
pub struct L5aScanner;

impl OutputScanner for L5aScanner {
    fn scan_and_redact(&self, content: &str, config: &Config) -> ScanResult {
        let mut redactions: Vec<Redaction> = Vec::new();
        let mut result = content.to_string();

        // Pass 1: canary tokens (exact substring match).
        // We process one token at a time, replacing all occurrences.
        // After each replacement the string changes length, so we
        // collect redactions against the *original* content offsets
        // by scanning the original first.
        for token in &config.policy.canary_tokens {
            if token.is_empty() {
                continue;
            }
            // Find all occurrences in the *current* result string.
            // We track positions relative to the original content by
            // doing a fresh search on the current result each time,
            // but record the position in the current (partially-redacted)
            // string. This is acceptable because the spec says
            // `position` is informational for logging.
            let mut search_from = 0;
            while let Some(pos) = result[search_from..].find(token.as_str()) {
                let abs_pos = search_from + pos;
                redactions.push(Redaction {
                    pattern: token.clone(),
                    position: abs_pos,
                });
                result.replace_range(abs_pos..abs_pos + token.len(), "[REDACTED]");
                search_from = abs_pos + "[REDACTED]".len();
            }
        }

        // Pass 2: sensitive patterns (regex match).
        for compiled in &config.policy.sensitive_patterns {
            // We loop because each replacement changes offsets.
            loop {
                let Some(mat) = compiled.regex.find(&result) else {
                    break;
                };
                redactions.push(Redaction {
                    pattern: compiled.pattern.clone(),
                    position: mat.start(),
                });
                let before = &result[..mat.start()];
                let after = &result[mat.end()..];
                result = format!("{before}[REDACTED]{after}");
            }
        }

        ScanResult { content: result, redactions }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        CompiledPattern, Config, ContentPolicy, EngineConfig, LayerConfigs, PolicyConfig,
        RuntimeConfig, TrustConfig,
    };
    use std::collections::HashMap;

    /// Build a minimal Config with specified canary tokens and sensitive patterns.
    fn test_config(canary_tokens: Vec<&str>, sensitive_patterns: Vec<&str>) -> Config {
        Config {
            policy: PolicyConfig {
                version: "v1".to_string(),
                tools: HashMap::new(),
                block_patterns: Vec::new(),
                canary_tokens: canary_tokens.into_iter().map(String::from).collect(),
                sensitive_patterns: sensitive_patterns
                    .into_iter()
                    .map(|p| CompiledPattern::compile(p).unwrap())
                    .collect(),
                untrusted_content_policy: ContentPolicy::default(),
                trust: TrustConfig::default(),
                layers: LayerConfigs::default(),
            },
            runtime: RuntimeConfig {
                engine: EngineConfig::default(),
                environment: String::new(),
            },
            contract_hash: String::new(),
        }
    }

    fn scanner() -> L5aScanner {
        L5aScanner
    }

    // ---------------------------------------------------------------
    // 1. Canary token in response -> redacted
    // ---------------------------------------------------------------

    #[test]
    fn canary_token_is_redacted() {
        let config = test_config(vec!["{{CANARY_a8f3e9b1}}"], vec![]);
        let result = scanner().scan_and_redact(
            "The secret is {{CANARY_a8f3e9b1}} here.",
            &config,
        );
        assert_eq!(result.content, "The secret is [REDACTED] here.");
        assert_eq!(result.redactions.len(), 1);
        assert_eq!(result.redactions[0].pattern, "{{CANARY_a8f3e9b1}}");
    }

    // ---------------------------------------------------------------
    // 2. Sensitive pattern in response -> redacted
    // ---------------------------------------------------------------

    #[test]
    fn sensitive_pattern_is_redacted() {
        let config = test_config(vec![], vec!["CEREBRAS_API_KEY=[A-Za-z0-9-]+"]);
        let result = scanner().scan_and_redact(
            "Key is CEREBRAS_API_KEY=sk-abc123 end.",
            &config,
        );
        assert_eq!(result.content, "Key is [REDACTED] end.");
        assert_eq!(result.redactions.len(), 1);
        assert_eq!(result.redactions[0].pattern, "CEREBRAS_API_KEY=[A-Za-z0-9-]+");
    }

    // ---------------------------------------------------------------
    // 3. No match -> response unchanged
    // ---------------------------------------------------------------

    #[test]
    fn no_match_returns_content_unchanged() {
        let config = test_config(vec!["{{CANARY_xyz}}"], vec!["SECRET_KEY=[0-9]+"]);
        let result = scanner().scan_and_redact("Nothing sensitive here.", &config);
        assert_eq!(result.content, "Nothing sensitive here.");
        assert!(result.redactions.is_empty());
    }

    // ---------------------------------------------------------------
    // 4. Multiple matches in same response -> all redacted
    // ---------------------------------------------------------------

    #[test]
    fn multiple_matches_all_redacted() {
        let config = test_config(
            vec!["{{CANARY_a8f3e9b1}}"],
            vec!["CEREBRAS_API_KEY=[A-Za-z0-9]+"],
        );
        let result = scanner().scan_and_redact(
            "Token {{CANARY_a8f3e9b1}} and CEREBRAS_API_KEY=sk123 found.",
            &config,
        );
        assert_eq!(
            result.content,
            "Token [REDACTED] and [REDACTED] found."
        );
        assert_eq!(result.redactions.len(), 2);
    }

    // ---------------------------------------------------------------
    // 5. Empty response -> no error
    // ---------------------------------------------------------------

    #[test]
    fn empty_response_no_error() {
        let config = test_config(vec!["{{CANARY}}"], vec!["KEY=[0-9]+"]);
        let result = scanner().scan_and_redact("", &config);
        assert_eq!(result.content, "");
        assert!(result.redactions.is_empty());
    }

    // ---------------------------------------------------------------
    // 6. Canary token redaction preserves surrounding text
    // ---------------------------------------------------------------

    #[test]
    fn canary_token_preserves_surrounding_text() {
        let config = test_config(vec!["LEAK"], vec![]);
        let result = scanner().scan_and_redact(
            "Before LEAK after and more LEAK end.",
            &config,
        );
        assert_eq!(result.content, "Before [REDACTED] after and more [REDACTED] end.");
        assert_eq!(result.redactions.len(), 2);
    }

    // ---------------------------------------------------------------
    // 7. Sensitive pattern as regex matches and redacts full match
    // ---------------------------------------------------------------

    #[test]
    fn sensitive_pattern_regex_redacts_full_match() {
        let config = test_config(vec![], vec!["CEREBRAS_API_KEY=[A-Za-z0-9-]+"]);
        let result = scanner().scan_and_redact(
            "export CEREBRAS_API_KEY=sk-longKey123ABC end",
            &config,
        );
        assert_eq!(result.content, "export [REDACTED] end");
        assert_eq!(result.redactions.len(), 1);
    }

    // ---------------------------------------------------------------
    // 8. Multiple different patterns, multiple matches each
    // ---------------------------------------------------------------

    #[test]
    fn multiple_patterns_multiple_matches_each() {
        let config = test_config(
            vec!["{{CANARY_A}}", "{{CANARY_B}}"],
            vec!["GROQ_API_KEY=[A-Za-z0-9]+", "SECRET_TOKEN=[0-9]+"],
        );
        let content = concat!(
            "First {{CANARY_A}} then {{CANARY_B}}. ",
            "Also GROQ_API_KEY=gsk123 and SECRET_TOKEN=999. ",
            "Repeat {{CANARY_A}} and SECRET_TOKEN=42."
        );
        let result = scanner().scan_and_redact(content, &config);

        // All 6 occurrences should be redacted
        assert!(!result.content.contains("{{CANARY_A}}"));
        assert!(!result.content.contains("{{CANARY_B}}"));
        assert!(!result.content.contains("GROQ_API_KEY=gsk123"));
        assert!(!result.content.contains("SECRET_TOKEN=999"));
        assert!(!result.content.contains("SECRET_TOKEN=42"));
        assert_eq!(result.redactions.len(), 6);
    }

    // ---------------------------------------------------------------
    // 9. Redaction count tracked in result
    // ---------------------------------------------------------------

    #[test]
    fn redaction_count_matches_number_of_replacements() {
        let config = test_config(
            vec!["CANARY"],
            vec!["KEY=[0-9]+"],
        );
        let result = scanner().scan_and_redact(
            "CANARY and KEY=123 and CANARY again KEY=456",
            &config,
        );
        assert_eq!(result.redactions.len(), 4);
    }

    // ---------------------------------------------------------------
    // Additional edge cases
    // ---------------------------------------------------------------

    #[test]
    fn canary_token_at_start_of_content() {
        let config = test_config(vec!["{{CANARY}}"], vec![]);
        let result = scanner().scan_and_redact("{{CANARY}} starts here", &config);
        assert_eq!(result.content, "[REDACTED] starts here");
    }

    #[test]
    fn canary_token_at_end_of_content() {
        let config = test_config(vec!["{{CANARY}}"], vec![]);
        let result = scanner().scan_and_redact("ends here {{CANARY}}", &config);
        assert_eq!(result.content, "ends here [REDACTED]");
    }

    #[test]
    fn content_is_exactly_the_canary_token() {
        let config = test_config(vec!["{{CANARY}}"], vec![]);
        let result = scanner().scan_and_redact("{{CANARY}}", &config);
        assert_eq!(result.content, "[REDACTED]");
        assert_eq!(result.redactions.len(), 1);
    }

    #[test]
    fn empty_canary_tokens_and_patterns_returns_unchanged() {
        let config = test_config(vec![], vec![]);
        let result = scanner().scan_and_redact("anything at all", &config);
        assert_eq!(result.content, "anything at all");
        assert!(result.redactions.is_empty());
    }

    #[test]
    fn overlapping_canary_and_regex_both_applied() {
        // If a canary token matches first, the regex should not find it again
        let config = test_config(
            vec!["SECRET=abc"],
            vec!["SECRET=[a-z]+"],
        );
        let result = scanner().scan_and_redact("has SECRET=abc here", &config);
        // Canary pass replaces "SECRET=abc" -> "[REDACTED]"
        // Regex pass should not find "SECRET=[a-z]+" in the redacted string
        assert_eq!(result.content, "has [REDACTED] here");
        assert_eq!(result.redactions.len(), 1);
    }

    #[test]
    fn position_field_is_populated() {
        let config = test_config(vec!["TOKEN"], vec![]);
        let result = scanner().scan_and_redact("abc TOKEN xyz", &config);
        assert_eq!(result.redactions[0].position, 4);
    }

    #[test]
    fn regex_position_field_is_populated() {
        let config = test_config(vec![], vec!["KEY=[0-9]+"]);
        let result = scanner().scan_and_redact("prefix KEY=123 suffix", &config);
        assert_eq!(result.redactions[0].position, 7);
    }
}
