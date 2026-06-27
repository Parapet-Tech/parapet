// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// Target L3 routing evidence surface.
//
// The runtime default remains no-op. Opt-in routers may emit attribution-only
// heuristic signals from this typed evidence surface without changing
// enforcement or copying raw message content.

use crate::config::{Config, PatternAction};
use crate::layers::l1::L1Result;
use crate::layers::l1_harness::L1Signal;
use crate::layers::l3_inbound::{InboundResult, MatchSource};
use crate::message::{Message, Role};
use crate::signal::{LayerId, Signal, SignalKind};

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

/// Opt-in shadow router for fixed-vocabulary routing attribution.
///
/// Emitted signals are `LayerId::HeuristicSignal`; the verdict combiner treats
/// that layer as attribution-only. Categories are fixed by this router and do
/// not include raw message content or policy-authored pattern categories.
/// Scores use the standard `Signal::new` clamp to keep attribution strength in
/// the normalized `[0.0, 1.0]` signal range.
pub struct ShadowHeuristicRouter;

impl ShadowHeuristicRouter {
    pub fn new() -> Self {
        Self
    }

    fn pattern_gate_signal(evidence: &PatternGateEvidence) -> Option<Signal> {
        if !matches!(evidence.action, PatternAction::Evidence) || evidence.atomic.unwrap_or(false) {
            return None;
        }

        let score = evidence.weight.unwrap_or(1.0) as f32;
        let mut signal = Signal::new(
            LayerId::HeuristicSignal,
            SignalKind::Evidence,
            Some("routing_pattern_gate_evidence".to_string()),
            score,
            1.0,
        );
        signal.message_index = Some(evidence.message_index);
        Some(signal)
    }
}

impl Default for ShadowHeuristicRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl OrthogonalSensorRouter for ShadowHeuristicRouter {
    fn route(&self, context: &RoutingEvidenceContext<'_>) -> Vec<Signal> {
        context
            .pattern_gate
            .iter()
            .filter_map(Self::pattern_gate_signal)
            .collect()
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

/// First-class L0 normalization evidence.
///
/// Fixed numeric and boolean telemetry only: no message content, no byte
/// ranges, no character offsets, no matched tokens, no policy vocabulary.
/// Counts are bounded per message and derived deterministically from the L0
/// sanitize pass. Engine population is wired: when L0 runs in `sanitize` mode,
/// `engine::process_request` collects one `L0Evidence` per message and passes
/// `Some(&[L0Evidence])` into `RoutingEvidenceContext`. This evidence is for
/// routing consumers only and must not gate enforcement decisions.
#[derive(Debug, Clone, PartialEq)]
pub struct L0Evidence {
    pub message_index: usize,
    pub pre_char_len: usize,
    pub post_char_len: usize,
    pub pre_byte_len: usize,
    pub post_byte_len: usize,
    pub removed_invisible_count: usize,
    pub confusable_replacement_count: usize,
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

    #[test]
    fn shadow_router_returns_no_signals_without_selected_context_evidence() {
        let config = config_with_patterns(vec![]);
        let messages = vec![Message::new(Role::User, "hello")];
        let context = RoutingEvidenceContext::new(&messages, None, None, None, &[], None, &config);
        let router = ShadowHeuristicRouter::new();

        assert!(router.route(&context).is_empty());
    }

    #[test]
    fn shadow_router_emits_fixed_vocabulary_evidence_attribution() {
        let pattern = CompiledPattern::compile_full(
            "ignore",
            PatternAction::Evidence,
            None,
            Some("policy_authored_category".to_string()),
            0.4,
            false,
        )
        .unwrap();
        let config = config_with_patterns(vec![pattern]);
        let messages = vec![Message::new(Role::User, "ignore this raw content")];
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
        let router = ShadowHeuristicRouter::new();

        let signals = router.route(&context);

        assert_eq!(signals.len(), 1);
        let signal = &signals[0];
        assert_eq!(signal.layer, LayerId::HeuristicSignal);
        assert_eq!(signal.kind, SignalKind::Evidence);
        assert_eq!(
            signal.category.as_deref(),
            Some("routing_pattern_gate_evidence")
        );
        assert!((signal.score - 0.4).abs() < f32::EPSILON);
        assert!((signal.confidence - 1.0).abs() < f32::EPSILON);
        assert_eq!(signal.message_index, Some(0));
        assert!(signal.segment_id.is_none());
        assert_eq!(signal.raw_model_score, None);
        assert_ne!(signal.category.as_deref(), Some("policy_authored_category"));
        assert!(!format!("{:?}", signal).contains("ignore this raw content"));
    }

    #[test]
    fn shadow_router_omits_block_action_pattern_gate_context() {
        let pattern = CompiledPattern::compile_full(
            "ignore",
            PatternAction::Block,
            None,
            Some("instruction_override".to_string()),
            1.0,
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
                action: PatternAction::Block,
            }],
        };
        let context =
            RoutingEvidenceContext::new(&messages, None, None, None, &[], Some(&inbound), &config);
        let router = ShadowHeuristicRouter::new();

        let signals = router.route(&context);

        assert!(signals.is_empty());
    }

    #[test]
    fn shadow_router_omits_atomic_evidence_pattern_gate_context() {
        let pattern = CompiledPattern::compile_full(
            "ignore",
            PatternAction::Evidence,
            None,
            Some("instruction_override".to_string()),
            0.8,
            true,
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
        let router = ShadowHeuristicRouter::new();

        let signals = router.route(&context);

        assert!(signals.is_empty());
    }

    #[test]
    fn shadow_router_preserves_order_and_message_index_for_multiple_evidence() {
        let patterns = vec![
            CompiledPattern::compile_full(
                "ignore",
                PatternAction::Evidence,
                None,
                Some("instruction_override".to_string()),
                0.3,
                false,
            )
            .unwrap(),
            CompiledPattern::compile_full(
                "reveal",
                PatternAction::Evidence,
                None,
                Some("exfil".to_string()),
                0.7,
                false,
            )
            .unwrap(),
        ];
        let config = config_with_patterns(patterns);
        let messages = vec![
            Message::new(Role::User, "ignore this"),
            Message::new(Role::User, "reveal that"),
        ];
        let inbound = InboundResult {
            verdict: InboundVerdict::Allow,
            matched_patterns: vec![
                PatternMatch {
                    pattern_index: 0,
                    message_index: 0,
                    role: Role::User,
                    source: MatchSource::Content,
                    action: PatternAction::Evidence,
                },
                PatternMatch {
                    pattern_index: 1,
                    message_index: 1,
                    role: Role::User,
                    source: MatchSource::Content,
                    action: PatternAction::Evidence,
                },
            ],
        };
        let context =
            RoutingEvidenceContext::new(&messages, None, None, None, &[], Some(&inbound), &config);
        let router = ShadowHeuristicRouter::new();

        let signals = router.route(&context);

        assert_eq!(signals.len(), 2);
        assert_eq!(signals[0].message_index, Some(0));
        assert_eq!(signals[1].message_index, Some(1));
        assert!((signals[0].score - 0.3).abs() < f32::EPSILON);
        assert!((signals[1].score - 0.7).abs() < f32::EPSILON);
        assert_eq!(
            signals
                .iter()
                .map(|s| s.category.as_deref())
                .collect::<Vec<_>>(),
            vec![
                Some("routing_pattern_gate_evidence"),
                Some("routing_pattern_gate_evidence"),
            ]
        );
    }

    #[test]
    fn shadow_router_output_is_deterministic_for_same_context() {
        let pattern = CompiledPattern::compile_full(
            "ignore",
            PatternAction::Evidence,
            None,
            Some("instruction_override".to_string()),
            0.7,
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
        let router = ShadowHeuristicRouter::new();

        let first = router.route(&context);
        let second = router.route(&context);

        assert_eq!(first.len(), second.len());
        assert_eq!(first[0].layer, second[0].layer);
        assert_eq!(first[0].kind, second[0].kind);
        assert_eq!(first[0].category, second[0].category);
        assert!((first[0].score - second[0].score).abs() < f32::EPSILON);
        assert_eq!(first[0].message_index, second[0].message_index);
    }

    #[test]
    fn l0_evidence_preserves_fixed_numeric_and_bool_fields() {
        let evidence = L0Evidence {
            message_index: 2,
            pre_char_len: 32,
            post_char_len: 28,
            pre_byte_len: 40,
            post_byte_len: 30,
            removed_invisible_count: 3,
            confusable_replacement_count: 1,
            html_stripped: true,
            role_marker_neutralized_count: 2,
        };

        assert_eq!(evidence.message_index, 2);
        assert_eq!(evidence.pre_char_len, 32);
        assert_eq!(evidence.post_char_len, 28);
        assert_eq!(evidence.pre_byte_len, 40);
        assert_eq!(evidence.post_byte_len, 30);
        assert_eq!(evidence.removed_invisible_count, 3);
        assert_eq!(evidence.confusable_replacement_count, 1);
        assert!(evidence.html_stripped);
        assert_eq!(evidence.role_marker_neutralized_count, 2);
    }

    #[test]
    fn l0_evidence_debug_format_is_content_free_by_construction() {
        // Every field is usize or bool, so Debug output cannot carry raw
        // message content, byte ranges, or matched tokens. This test pins
        // that shape and confirms the contract field names are correct.
        let evidence = L0Evidence {
            message_index: 0,
            pre_char_len: 7,
            post_char_len: 5,
            pre_byte_len: 9,
            post_byte_len: 5,
            removed_invisible_count: 2,
            confusable_replacement_count: 0,
            html_stripped: false,
            role_marker_neutralized_count: 0,
        };
        let dbg = format!("{:?}", evidence);
        let allowed = |c: char| {
            c.is_ascii_alphanumeric() || matches!(c, '_' | ' ' | ':' | ',' | '{' | '}')
        };
        assert!(
            dbg.chars().all(allowed),
            "L0Evidence Debug output contained unexpected characters: {dbg}"
        );
        assert!(dbg.contains("removed_invisible_count"));
        assert!(dbg.contains("confusable_replacement_count"));
        assert!(!dbg.contains("removed_control_count"));
    }
}
