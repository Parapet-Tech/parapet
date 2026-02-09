// L4 multi-turn scanning -- defined in M9
//
// Analyzes conversation state across turns within a session.
// Detects cross-turn patterns: escalation, accumulated untrusted content,
// excessive tool usage. Topic drift is stubbed for v2 (requires embeddings).

use crate::config::{Config, DetectorConfig};
use crate::message::{Message, TrustLevel};
use crate::session::SessionState;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Action from L4 scanning.
#[derive(Debug, Clone, PartialEq)]
pub enum L4Action {
    /// Allow the request to proceed.
    Allow,
    /// Warn but allow (adds X-Guard-Warning header).
    Warn(String),
    /// Block the request.
    Block(String),
}

/// A single detection from an L4 detector.
#[derive(Debug, Clone)]
pub struct Detection {
    /// Which detector produced this.
    pub detector: String,
    /// Human-readable description.
    pub reason: String,
    /// Severity: "info", "warn", "block".
    pub severity: String,
}

/// Result from L4 multi-turn scanning.
#[derive(Debug, Clone)]
pub struct L4Result {
    /// The final action (most severe wins).
    pub action: L4Action,
    /// All detections across all detectors.
    pub detections: Vec<Detection>,
    /// Flags to update on the session state.
    pub updated_flags: HashSet<String>,
}

// ---------------------------------------------------------------------------
// MultiTurnScanner trait
// ---------------------------------------------------------------------------

/// Trait for L4 multi-turn scanning.
///
/// Implementations analyze the current request messages in the context of
/// the session's history to detect cross-turn patterns.
pub trait MultiTurnScanner: Send + Sync {
    fn scan(
        &self,
        messages: &[Message],
        session: &SessionState,
        config: &Config,
    ) -> L4Result;
}

// ---------------------------------------------------------------------------
// Default scanner with built-in detectors
// ---------------------------------------------------------------------------

/// Default L4 scanner with configurable built-in detectors.
pub struct DefaultMultiTurnScanner;

impl DefaultMultiTurnScanner {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DefaultMultiTurnScanner {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiTurnScanner for DefaultMultiTurnScanner {
    fn scan(
        &self,
        messages: &[Message],
        session: &SessionState,
        config: &Config,
    ) -> L4Result {
        let l4_config = match &config.policy.layers.l4 {
            Some(c) if c.enabled => c,
            _ => {
                return L4Result {
                    action: L4Action::Allow,
                    detections: Vec::new(),
                    updated_flags: HashSet::new(),
                };
            }
        };

        let mut detections = Vec::new();
        let mut flags = HashSet::new();
        let mut worst_action = L4Action::Allow;

        for detector_config in &l4_config.detectors {
            if !detector_config.enabled {
                continue;
            }

            let result = match detector_config.name.as_str() {
                "escalation" => detect_escalation(messages, session, detector_config),
                "accumulation" => detect_accumulation(messages, session, detector_config),
                "tool_frequency" => detect_tool_frequency(messages, session, detector_config),
                "topic_drift" => {
                    // Stub for v2 -- requires embeddings
                    None
                }
                unknown => {
                    tracing::warn!(detector = unknown, "unknown L4 detector, skipping");
                    None
                }
            };

            if let Some((detection, flag)) = result {
                worst_action = merge_action(worst_action, &detection.severity, &detection.reason);
                if let Some(f) = flag {
                    flags.insert(f);
                }
                detections.push(detection);
            }
        }

        L4Result {
            action: worst_action,
            detections,
            updated_flags: flags,
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in detectors
// ---------------------------------------------------------------------------

/// Escalation: trust level decreased across turns.
///
/// Detects when a session that previously had only trusted content now
/// includes untrusted content. This could indicate a multi-turn attack
/// where the attacker first builds trust, then injects.
fn detect_escalation(
    messages: &[Message],
    session: &SessionState,
    _config: &DetectorConfig,
) -> Option<(Detection, Option<String>)> {
    // Check if current request has untrusted content
    let has_untrusted = messages.iter().any(|m| {
        m.trust == TrustLevel::Untrusted || !m.trust_spans.is_empty()
    });

    if !has_untrusted {
        return None;
    }

    // Check if all previous turns were trusted
    let all_previous_trusted = !session.history.is_empty()
        && session
            .history
            .iter()
            .all(|h| h.trust_level == TrustLevel::Trusted);

    if !all_previous_trusted {
        return None;
    }

    // Already warned about this?
    if session.flags.contains("escalation_warned") {
        return None;
    }

    Some((
        Detection {
            detector: "escalation".to_string(),
            reason: format!(
                "trust escalation: {} previous trusted turns, now untrusted content detected",
                session.history.len()
            ),
            severity: "warn".to_string(),
        },
        Some("escalation_warned".to_string()),
    ))
}

/// Accumulation: total untrusted content volume across session exceeds threshold.
///
/// Tracks the cumulative amount of untrusted content across all turns.
/// Default threshold: 10000 characters.
fn detect_accumulation(
    messages: &[Message],
    session: &SessionState,
    config: &DetectorConfig,
) -> Option<(Detection, Option<String>)> {
    let threshold = config.threshold.unwrap_or(10000.0) as usize;

    // Count untrusted chars in current request
    let current_untrusted: usize = messages
        .iter()
        .filter(|m| m.trust == TrustLevel::Untrusted)
        .map(|m| m.content.chars().count())
        .sum();

    // Count untrusted chars from session history
    let historical_untrusted: usize = session
        .history
        .iter()
        .filter(|h| h.trust_level == TrustLevel::Untrusted)
        .map(|h| h.content_summary.chars().count())
        .sum();

    let total = current_untrusted + historical_untrusted;

    if total <= threshold {
        return None;
    }

    Some((
        Detection {
            detector: "accumulation".to_string(),
            reason: format!(
                "accumulated untrusted content ({} chars) exceeds threshold ({} chars)",
                total, threshold
            ),
            severity: "block".to_string(),
        },
        Some("accumulation_exceeded".to_string()),
    ))
}

/// Tool frequency: same tool called too many times across turns.
///
/// Default threshold: 20 calls per session.
fn detect_tool_frequency(
    messages: &[Message],
    session: &SessionState,
    config: &DetectorConfig,
) -> Option<(Detection, Option<String>)> {
    let threshold = config.threshold.unwrap_or(20.0) as usize;

    // Count tool calls in current request
    let mut tool_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    for msg in messages {
        for tc in &msg.tool_calls {
            *tool_counts.entry(tc.name.clone()).or_insert(0) += 1;
        }
    }

    // Add historical tool call counts
    for entry in &session.history {
        for tool_name in &entry.tool_calls_summary {
            *tool_counts.entry(tool_name.clone()).or_insert(0) += 1;
        }
    }

    // Find any tool that exceeds the threshold
    for (tool_name, count) in &tool_counts {
        if *count > threshold {
            return Some((
                Detection {
                    detector: "tool_frequency".to_string(),
                    reason: format!(
                        "tool '{}' called {} times (threshold: {})",
                        tool_name, count, threshold
                    ),
                    severity: "warn".to_string(),
                },
                Some(format!("tool_freq_{}_{}", tool_name, count)),
            ));
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Merge a new detection's severity into the current worst action.
/// Precedence: Block > Warn > Allow.
fn merge_action(current: L4Action, severity: &str, reason: &str) -> L4Action {
    match (severity, &current) {
        ("block", _) => L4Action::Block(reason.to_string()),
        ("warn", L4Action::Allow) => L4Action::Warn(reason.to_string()),
        ("warn", L4Action::Warn(existing)) => {
            L4Action::Warn(format!("{}; {}", existing, reason))
        }
        _ => current,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;
    use crate::message::Role;
    use crate::session::{HistoryEntry, SessionState};
    use chrono::Utc;
    use std::collections::HashMap;

    fn l4_config(detectors: Vec<DetectorConfig>) -> Config {
        Config {
            policy: PolicyConfig {
                version: "v1".to_string(),
                tools: HashMap::new(),
                block_patterns: Vec::new(),
                canary_tokens: Vec::new(),
                sensitive_patterns: Vec::new(),
                untrusted_content_policy: ContentPolicy::default(),
                trust: TrustConfig::default(),
                layers: LayerConfigs {
                    l0: None,
                    l3_inbound: None,
                    l3_outbound: None,
                    l5a: None,
                    l4: Some(L4Config {
                        enabled: true,
                        max_history: 50,
                        session_ttl_secs: 3600,
                        detectors,
                    }),
                },
            },
            runtime: RuntimeConfig {
                engine: EngineConfig::default(),
                environment: String::new(),
            },
            contract_hash: "sha256:test".to_string(),
        }
    }

    fn empty_session() -> SessionState {
        SessionState::new("test_session")
    }

    fn trusted_history_entry() -> HistoryEntry {
        HistoryEntry {
            role: "user".to_string(),
            content_summary: "trusted content".to_string(),
            timestamp: Utc::now(),
            trust_level: TrustLevel::Trusted,
            tool_calls_summary: Vec::new(),
        }
    }

    // ---------------------------------------------------------------
    // L4 disabled
    // ---------------------------------------------------------------

    #[test]
    fn l4_disabled_returns_allow() {
        let mut config = l4_config(Vec::new());
        config.policy.layers.l4.as_mut().unwrap().enabled = false;

        let scanner = DefaultMultiTurnScanner::new();
        let result = scanner.scan(&[], &empty_session(), &config);
        assert_eq!(result.action, L4Action::Allow);
    }

    #[test]
    fn l4_no_detectors_returns_allow() {
        let config = l4_config(Vec::new());
        let scanner = DefaultMultiTurnScanner::new();
        let result = scanner.scan(
            &[Message::new(Role::User, "hello")],
            &empty_session(),
            &config,
        );
        assert_eq!(result.action, L4Action::Allow);
    }

    // ---------------------------------------------------------------
    // Escalation detector
    // ---------------------------------------------------------------

    #[test]
    fn escalation_detects_trust_change() {
        let config = l4_config(vec![DetectorConfig {
            name: "escalation".to_string(),
            enabled: true,
            threshold: None,
        }]);

        let mut session = empty_session();
        // Add 3 trusted turns
        for _ in 0..3 {
            session.history.push(trusted_history_entry());
        }

        // Current request has untrusted content
        let mut msg = Message::new(Role::User, "inject here");
        msg.trust = TrustLevel::Untrusted;

        let scanner = DefaultMultiTurnScanner::new();
        let result = scanner.scan(&[msg], &session, &config);

        assert!(matches!(result.action, L4Action::Warn(_)));
        assert_eq!(result.detections.len(), 1);
        assert_eq!(result.detections[0].detector, "escalation");
        assert!(result.updated_flags.contains("escalation_warned"));
    }

    #[test]
    fn escalation_no_history_no_detection() {
        let config = l4_config(vec![DetectorConfig {
            name: "escalation".to_string(),
            enabled: true,
            threshold: None,
        }]);

        let mut msg = Message::new(Role::User, "first message");
        msg.trust = TrustLevel::Untrusted;

        let scanner = DefaultMultiTurnScanner::new();
        let result = scanner.scan(&[msg], &empty_session(), &config);
        assert_eq!(result.action, L4Action::Allow);
    }

    #[test]
    fn escalation_already_warned_skips() {
        let config = l4_config(vec![DetectorConfig {
            name: "escalation".to_string(),
            enabled: true,
            threshold: None,
        }]);

        let mut session = empty_session();
        session.history.push(trusted_history_entry());
        session.flags.insert("escalation_warned".to_string());

        let mut msg = Message::new(Role::User, "inject");
        msg.trust = TrustLevel::Untrusted;

        let scanner = DefaultMultiTurnScanner::new();
        let result = scanner.scan(&[msg], &session, &config);
        assert_eq!(result.action, L4Action::Allow);
    }

    // ---------------------------------------------------------------
    // Accumulation detector
    // ---------------------------------------------------------------

    #[test]
    fn accumulation_blocks_when_exceeded() {
        let config = l4_config(vec![DetectorConfig {
            name: "accumulation".to_string(),
            enabled: true,
            threshold: Some(20.0), // low threshold for testing
        }]);

        let session = empty_session();
        let mut msg = Message::new(Role::User, "a".repeat(30)); // 30 chars
        msg.trust = TrustLevel::Untrusted;

        let scanner = DefaultMultiTurnScanner::new();
        let result = scanner.scan(&[msg], &session, &config);
        assert!(matches!(result.action, L4Action::Block(_)));
    }

    #[test]
    fn accumulation_allows_under_threshold() {
        let config = l4_config(vec![DetectorConfig {
            name: "accumulation".to_string(),
            enabled: true,
            threshold: Some(100.0),
        }]);

        let session = empty_session();
        let mut msg = Message::new(Role::User, "short");
        msg.trust = TrustLevel::Untrusted;

        let scanner = DefaultMultiTurnScanner::new();
        let result = scanner.scan(&[msg], &session, &config);
        assert_eq!(result.action, L4Action::Allow);
    }

    // ---------------------------------------------------------------
    // Tool frequency detector
    // ---------------------------------------------------------------

    #[test]
    fn tool_frequency_warns_when_exceeded() {
        let config = l4_config(vec![DetectorConfig {
            name: "tool_frequency".to_string(),
            enabled: true,
            threshold: Some(3.0),
        }]);

        let mut session = empty_session();
        // Add history with tool calls
        for _ in 0..4 {
            session.history.push(HistoryEntry {
                role: "assistant".to_string(),
                content_summary: "response".to_string(),
                timestamp: Utc::now(),
                trust_level: TrustLevel::Trusted,
                tool_calls_summary: vec!["read_file".to_string()],
            });
        }

        let scanner = DefaultMultiTurnScanner::new();
        let result = scanner.scan(
            &[Message::new(Role::User, "next")],
            &session,
            &config,
        );
        assert!(matches!(result.action, L4Action::Warn(_)));
        assert_eq!(result.detections[0].detector, "tool_frequency");
    }

    #[test]
    fn tool_frequency_allows_under_threshold() {
        let config = l4_config(vec![DetectorConfig {
            name: "tool_frequency".to_string(),
            enabled: true,
            threshold: Some(10.0),
        }]);

        let session = empty_session();
        let scanner = DefaultMultiTurnScanner::new();
        let result = scanner.scan(
            &[Message::new(Role::User, "hello")],
            &session,
            &config,
        );
        assert_eq!(result.action, L4Action::Allow);
    }

    // ---------------------------------------------------------------
    // Multiple detectors
    // ---------------------------------------------------------------

    #[test]
    fn multiple_detectors_worst_severity_wins() {
        let config = l4_config(vec![
            DetectorConfig {
                name: "escalation".to_string(),
                enabled: true,
                threshold: None,
            },
            DetectorConfig {
                name: "accumulation".to_string(),
                enabled: true,
                threshold: Some(5.0), // Very low threshold
            },
        ]);

        let mut session = empty_session();
        session.history.push(trusted_history_entry());

        let mut msg = Message::new(Role::User, "untrusted long content");
        msg.trust = TrustLevel::Untrusted;

        let scanner = DefaultMultiTurnScanner::new();
        let result = scanner.scan(&[msg], &session, &config);

        // Accumulation blocks (severity: block), escalation warns
        // Block > Warn, so final action is Block
        assert!(matches!(result.action, L4Action::Block(_)));
        assert_eq!(result.detections.len(), 2);
    }

    // ---------------------------------------------------------------
    // merge_action
    // ---------------------------------------------------------------

    #[test]
    fn merge_action_block_wins() {
        let result = merge_action(L4Action::Warn("x".into()), "block", "blocked");
        assert!(matches!(result, L4Action::Block(_)));
    }

    #[test]
    fn merge_action_warn_upgrades_allow() {
        let result = merge_action(L4Action::Allow, "warn", "warning");
        assert!(matches!(result, L4Action::Warn(_)));
    }

    #[test]
    fn merge_action_info_preserves_current() {
        let result = merge_action(L4Action::Allow, "info", "note");
        assert_eq!(result, L4Action::Allow);
    }

    // ---------------------------------------------------------------
    // Disabled detector
    // ---------------------------------------------------------------

    #[test]
    fn disabled_detector_skipped() {
        let config = l4_config(vec![DetectorConfig {
            name: "accumulation".to_string(),
            enabled: false,
            threshold: Some(1.0), // Would definitely trigger if enabled
        }]);

        let session = empty_session();
        let mut msg = Message::new(Role::User, "lots of untrusted content");
        msg.trust = TrustLevel::Untrusted;

        let scanner = DefaultMultiTurnScanner::new();
        let result = scanner.scan(&[msg], &session, &config);
        assert_eq!(result.action, L4Action::Allow);
    }
}
