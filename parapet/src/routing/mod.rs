// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// Target L3 routing evidence surface.
//
// This module is intentionally a no-op scaffold. It gives future orthogonal
// sensor/router work a typed evidence surface without adding sensors, changing
// enforcement, or copying raw message content.

use crate::config::{Config, PatternAction};
use crate::layers::l1::L1Result;
use crate::layers::l1_harness::L1Signal;
use crate::layers::l3_inbound::{InboundResult, MatchSource};
use crate::message::{Message, Role};
use crate::signal::Signal;

/// Router interface for future target-L3 deterministic signals.
pub trait OrthogonalSensorRouter: Send + Sync {
    fn route(&self, context: &RoutingEvidenceContext<'_>) -> Vec<Signal>;
}

/// Default router used by the runtime until a measured signal is promoted.
pub struct NoopOrthogonalSensorRouter;

impl NoopOrthogonalSensorRouter {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NoopOrthogonalSensorRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl OrthogonalSensorRouter for NoopOrthogonalSensorRouter {
    fn route(&self, _context: &RoutingEvidenceContext<'_>) -> Vec<Signal> {
        Vec::new()
    }
}

/// Borrowed context available to future target-L3 routing logic.
pub struct RoutingEvidenceContext<'a> {
    pub messages: &'a [Message],
    pub l0: Option<&'a [L0Evidence]>,
    pub lexical: Vec<LexicalEvidence>,
    pub payload_scan: &'a [Signal],
    pub pattern_gate: Vec<PatternGateEvidence>,
}

impl<'a> RoutingEvidenceContext<'a> {
    pub fn new(
        messages: &'a [Message],
        l0: Option<&'a [L0Evidence]>,
        l1_signals: Option<&'a [L1Signal]>,
        l1_result: Option<&'a L1Result>,
        payload_scan: &'a [Signal],
        inbound_result: Option<&'a InboundResult>,
        config: &Config,
    ) -> Self {
        let lexical = if let Some(signals) = l1_signals {
            signals.iter().map(LexicalEvidence::from_harness).collect()
        } else if let Some(result) = l1_result {
            result
                .per_message_scores
                .iter()
                .map(LexicalEvidence::from_legacy)
                .collect()
        } else {
            Vec::new()
        };

        let pattern_gate = inbound_result
            .map(|result| {
                result
                    .matched_patterns
                    .iter()
                    .map(|m| {
                        let pattern = config.policy.block_patterns.get(m.pattern_index);
                        PatternGateEvidence {
                            message_index: m.message_index,
                            role: m.role.clone(),
                            source: m.source.clone(),
                            action: m.action.clone(),
                            pattern_index: m.pattern_index,
                            category: pattern.and_then(|p| p.category.clone()),
                            weight: pattern.map(|p| p.weight),
                            atomic: pattern.map(|p| p.atomic),
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        Self {
            messages,
            l0,
            lexical,
            payload_scan,
            pattern_gate,
        }
    }
}

/// Placeholder for future first-class L0 normalization evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct L0Evidence {
    pub message_index: usize,
    pub pre_char_len: usize,
    pub post_char_len: usize,
    pub pre_byte_len: usize,
    pub post_byte_len: usize,
    pub removed_control_count: usize,
    pub removed_invisible_count: usize,
    pub html_stripped: bool,
    pub role_marker_neutralized_count: usize,
}

/// Lexical-classifier evidence. Rich fields are optional on the legacy path.
#[derive(Debug, Clone, PartialEq)]
pub struct LexicalEvidence {
    pub message_index: usize,
    pub role: Role,
    pub raw_score: f64,
    pub calibrated_score: f32,
    pub raw_unquoted_score: Option<f64>,
    pub raw_squash_score: Option<f64>,
    pub unquoted_score: Option<f32>,
    pub squash_score: Option<f32>,
    pub quote_detected: Option<bool>,
    pub raw_score_delta: Option<f64>,
}

impl LexicalEvidence {
    fn from_harness(signal: &L1Signal) -> Self {
        Self {
            message_index: signal.message_index,
            role: signal.role.clone(),
            raw_score: signal.raw_score,
            calibrated_score: signal.score,
            raw_unquoted_score: Some(signal.raw_unquoted_score),
            raw_squash_score: Some(signal.raw_squash_score),
            unquoted_score: Some(signal.unquoted_score),
            squash_score: Some(signal.squash_score),
            quote_detected: Some(signal.quote_detected),
            raw_score_delta: Some(signal.raw_score_delta),
        }
    }

    fn from_legacy(score: &crate::layers::l1::L1MessageScore) -> Self {
        Self {
            message_index: score.message_index,
            role: score.role.clone(),
            raw_score: score.score,
            calibrated_score: score.calibrated,
            raw_unquoted_score: None,
            raw_squash_score: None,
            unquoted_score: None,
            squash_score: None,
            quote_detected: None,
            raw_score_delta: None,
        }
    }
}

/// Pattern-gate evidence with config metadata resolved at construction time.
#[derive(Debug, Clone, PartialEq)]
pub struct PatternGateEvidence {
    pub message_index: usize,
    pub role: Role,
    pub source: MatchSource,
    pub action: PatternAction,
    pub pattern_index: usize,
    pub category: Option<String>,
    pub weight: Option<f64>,
    pub atomic: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        CompiledPattern, Config, ContentPolicy, EngineConfig, LayerConfigs, PatternAction,
        PolicyConfig, RuntimeConfig, TrustConfig,
    };
    use crate::layers::l1::{L1MessageScore, L1Result, L1Verdict};
    use crate::layers::l3_inbound::{InboundResult, InboundVerdict, PatternMatch};
    use crate::message::Role;
    use std::collections::HashMap;

    fn config_with_patterns(patterns: Vec<CompiledPattern>) -> Config {
        Config {
            policy: PolicyConfig {
                version: "v1".to_string(),
                tools: HashMap::new(),
                block_patterns: patterns,
                canary_tokens: vec![],
                sensitive_patterns: vec![],
                untrusted_content_policy: ContentPolicy::default(),
                trust: TrustConfig::default(),
                layers: LayerConfigs::default(),
            },
            runtime: RuntimeConfig {
                engine: EngineConfig::default(),
                environment: "test".to_string(),
            },
            contract_hash: "sha256:test".to_string(),
        }
    }

    #[test]
    fn noop_router_returns_no_signals() {
        let config = config_with_patterns(vec![]);
        let messages = vec![Message::new(Role::User, "hello")];
        let context =
            RoutingEvidenceContext::new(&messages, None, None, None, &[], None, &config);
        let router = NoopOrthogonalSensorRouter::new();
        assert!(router.route(&context).is_empty());
    }

    #[test]
    fn harness_lexical_evidence_preserves_rich_fields() {
        let config = config_with_patterns(vec![]);
        let messages = vec![Message::new(Role::User, "hello")];
        let signals = vec![L1Signal {
            message_index: 0,
            role: Role::User,
            raw_score: 1.2,
            raw_unquoted_score: 0.4,
            raw_squash_score: 1.8,
            score: 0.7,
            unquoted_score: 0.2,
            squash_score: 0.9,
            quote_detected: true,
            raw_score_delta: 0.8,
        }];

        let context =
            RoutingEvidenceContext::new(&messages, None, Some(&signals), None, &[], None, &config);

        assert_eq!(context.lexical.len(), 1);
        let evidence = &context.lexical[0];
        assert_eq!(evidence.message_index, 0);
        assert_eq!(evidence.raw_unquoted_score, Some(0.4));
        assert_eq!(evidence.raw_squash_score, Some(1.8));
        assert_eq!(evidence.unquoted_score, Some(0.2));
        assert_eq!(evidence.squash_score, Some(0.9));
        assert_eq!(evidence.quote_detected, Some(true));
        assert_eq!(evidence.raw_score_delta, Some(0.8));
    }

    #[test]
    fn legacy_lexical_evidence_uses_none_for_missing_fields() {
        let config = config_with_patterns(vec![]);
        let messages = vec![Message::new(Role::User, "hello")];
        let result = L1Result {
            verdict: L1Verdict::Allow,
            per_message_scores: vec![L1MessageScore {
                message_index: 0,
                role: Role::User,
                score: 1.2,
                calibrated: 0.7,
                specialist_name: None,
            }],
        };

        let context =
            RoutingEvidenceContext::new(&messages, None, None, Some(&result), &[], None, &config);

        assert_eq!(context.lexical.len(), 1);
        let evidence = &context.lexical[0];
        assert_eq!(evidence.raw_score, 1.2);
        assert_eq!(evidence.calibrated_score, 0.7);
        assert_eq!(evidence.raw_unquoted_score, None);
        assert_eq!(evidence.raw_squash_score, None);
        assert_eq!(evidence.unquoted_score, None);
        assert_eq!(evidence.squash_score, None);
        assert_eq!(evidence.quote_detected, None);
        assert_eq!(evidence.raw_score_delta, None);
    }

    #[test]
    fn pattern_gate_metadata_is_resolved_into_context() {
        let pattern = CompiledPattern::compile_full(
            "ignore",
            PatternAction::Evidence,
            None,
            Some("instruction_override".to_string()),
            0.4,
            false,
        )
        .unwrap();
        let config = config_with_patterns(vec![pattern]);
        let messages = vec![Message::new(Role::User, "ignore this")];
        let inbound = InboundResult {
            verdict: InboundVerdict::Allow,
            matched_patterns: vec![PatternMatch {
                pattern_index: 0,
                message_index: 0,
                role: Role::User,
                source: MatchSource::Content,
                action: PatternAction::Evidence,
            }],
        };

        let context =
            RoutingEvidenceContext::new(&messages, None, None, None, &[], Some(&inbound), &config);

        assert_eq!(context.pattern_gate.len(), 1);
        let evidence = &context.pattern_gate[0];
        assert_eq!(evidence.category.as_deref(), Some("instruction_override"));
        assert_eq!(evidence.weight, Some(0.4));
        assert_eq!(evidence.atomic, Some(false));
    }
}
