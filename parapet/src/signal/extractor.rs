// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// Signal extraction — converts layer results into unified Signal types.

use crate::config::{Config, PatternAction};
use crate::layers::l1::L1Result;
use crate::layers::l1_harness::L1Signal;
use crate::layers::l3_inbound::InboundResult;
use crate::layers::l4::L4Result;

use super::{LayerId, Signal, SignalKind};

// ---------------------------------------------------------------------------
// Constants for L1Signal extraction
// ---------------------------------------------------------------------------

use crate::layers::l1_harness::MENTION_RAW_DELTA_THRESHOLD;

/// Minimum calibrated score gap (squash - full-text) to emit an
/// `obfuscation` signal. A gap of 0.1 in probability space means
/// squash-deobfuscation uncovered material attack signal.
const OBFUSCATION_SCORE_GAP: f32 = 0.1;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Converts layer results into signals for the verdict processor.
pub trait SignalExtractor: Send + Sync {
    fn extract_l1(&self, result: &L1Result) -> Vec<Signal>;
    fn extract_l1_signals(&self, signals: &[L1Signal]) -> Vec<Signal>;
    fn extract_l3(&self, result: &InboundResult, config: &Config) -> Vec<Signal>;
    fn extract_l4(&self, result: &L4Result) -> Vec<Signal>;
}

// ---------------------------------------------------------------------------
// Default implementation
// ---------------------------------------------------------------------------

pub struct DefaultSignalExtractor;

impl DefaultSignalExtractor {
    pub fn new() -> Self {
        Self
    }
}

impl SignalExtractor for DefaultSignalExtractor {
    fn extract_l1(&self, result: &L1Result) -> Vec<Signal> {
        result
            .per_message_scores
            .iter()
            .filter(|m| m.calibrated > 0.0)
            .map(|m| {
                Signal::new(
                    LayerId::L1,
                    SignalKind::Evidence,
                    m.specialist_name.clone(),
                    m.calibrated,
                    1.0, // L1 is calibrated; fixed constant
                )
            })
            .collect()
    }

    fn extract_l1_signals(&self, signals: &[L1Signal]) -> Vec<Signal> {
        let mut out = Vec::new();

        for s in signals {
            // Use-vs-mention dampening: if attack signal is concentrated
            // in quoted regions, emit the UNQUOTED score instead of the
            // full-text score. This dampens the primary signal — the combiner
            // naturally sees a lower score without needing a new signal kind.
            let mention = s.quote_detected
                && s.raw_score_delta > MENTION_RAW_DELTA_THRESHOLD;

            let primary_score = if mention { s.unquoted_score } else { s.score };
            let category = if mention {
                Some("mention_dampened".to_string())
            } else {
                None
            };

            let mut primary = Signal::new(
                LayerId::L1,
                SignalKind::Evidence,
                category,
                primary_score,
                1.0, // L1 is calibrated
            );
            primary.message_index = Some(s.message_index);
            out.push(primary);

            // Obfuscation signal: squash score meaningfully exceeds
            // full-text score → attack patterns were hidden by formatting.
            // Compared against original score, not dampened score.
            if s.squash_score > s.score + OBFUSCATION_SCORE_GAP {
                let mut obfusc = Signal::new(
                    LayerId::L1,
                    SignalKind::Evidence,
                    Some("obfuscation".to_string()),
                    s.squash_score,
                    0.9, // squash is deterministic and well-tested
                );
                obfusc.message_index = Some(s.message_index);
                out.push(obfusc);
            }
        }

        out
    }

    fn extract_l3(&self, result: &InboundResult, config: &Config) -> Vec<Signal> {
        result
            .matched_patterns
            .iter()
            .filter_map(|m| {
                let pattern = config.policy.block_patterns.get(m.pattern_index)?;
                let kind = if pattern.atomic {
                    SignalKind::AtomicBlock
                } else {
                    SignalKind::Evidence
                };
                let confidence = if pattern.action == PatternAction::Block {
                    1.0
                } else {
                    0.6
                };
                Some(Signal::new(
                    LayerId::L3,
                    kind,
                    pattern.category.clone(),
                    pattern.weight as f32, // clamped by Signal::new
                    confidence,
                ))
            })
            .collect()
    }

    fn extract_l4(&self, result: &L4Result) -> Vec<Signal> {
        let mut signals = Vec::new();

        // Aggregate risk score
        if result.risk_score > 0.0 {
            signals.push(Signal::new(
                LayerId::L4,
                SignalKind::Evidence,
                None,
                result.risk_score as f32,
                1.0,
            ));
        }

        // Per-category signals
        for cat in &result.matched_categories {
            if cat.weight > 0.0 {
                signals.push(Signal::new(
                    LayerId::L4,
                    SignalKind::Evidence,
                    Some(cat.category.clone()),
                    cat.weight as f32, // clamped by Signal::new
                    0.8,               // L4 categories are heuristic
                ));
            }
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
    use crate::config::{CompiledPattern, PatternAction};
    use crate::layers::l1::{L1MessageScore, L1Result, L1Verdict};
    use crate::layers::l3_inbound::{InboundResult, InboundVerdict, MatchSource, PatternMatch};
    use crate::layers::l4::{L4CategoryMatch, L4Result, L4Verdict};
    use crate::message::Role;

    fn make_config_with_patterns(patterns: Vec<CompiledPattern>) -> Config {
        use crate::config::*;
        use std::collections::HashMap;
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
    fn l1_extraction_skips_zero_scores() {
        let extractor = DefaultSignalExtractor::new();
        let result = L1Result {
            verdict: L1Verdict::Allow,
            per_message_scores: vec![
                L1MessageScore { message_index: 0, role: Role::User, score: -1.0, calibrated: 0.0, specialist_name: None },
                L1MessageScore { message_index: 1, role: Role::User, score: 0.5, calibrated: 0.4, specialist_name: None },
                L1MessageScore { message_index: 2, role: Role::User, score: 2.0, calibrated: 0.8, specialist_name: None },
            ],
        };
        let signals = extractor.extract_l1(&result);
        assert_eq!(signals.len(), 2);
        assert!((signals[0].score - 0.4).abs() < f32::EPSILON);
        assert!((signals[1].score - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn l3_extraction_atomic_pattern() {
        let extractor = DefaultSignalExtractor::new();
        let pattern = CompiledPattern::compile_full(
            "exploit",
            PatternAction::Block,
            None,
            Some("template_abuse".to_string()),
            1.0,
            true, // atomic
        )
        .unwrap();
        let config = make_config_with_patterns(vec![pattern]);

        let result = InboundResult {
            verdict: InboundVerdict::Allow,
            matched_patterns: vec![PatternMatch {
                pattern_index: 0,
                message_index: 0,
                role: Role::User,
                source: MatchSource::Content,
                action: PatternAction::Block,
            }],
        };
        let signals = extractor.extract_l3(&result, &config);
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].kind, SignalKind::AtomicBlock);
        assert_eq!(signals[0].category.as_deref(), Some("template_abuse"));
    }

    #[test]
    fn l3_extraction_evidence_pattern() {
        let extractor = DefaultSignalExtractor::new();
        let pattern = CompiledPattern::compile_full(
            "suspicious",
            PatternAction::Evidence,
            None,
            Some("roleplay".to_string()),
            0.7,
            false,
        )
        .unwrap();
        let config = make_config_with_patterns(vec![pattern]);

        let result = InboundResult {
            verdict: InboundVerdict::Allow,
            matched_patterns: vec![PatternMatch {
                pattern_index: 0,
                message_index: 0,
                role: Role::User,
                source: MatchSource::Content,
                action: PatternAction::Evidence,
            }],
        };
        let signals = extractor.extract_l3(&result, &config);
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].kind, SignalKind::Evidence);
        assert!((signals[0].score - 0.7).abs() < f32::EPSILON);
        assert!((signals[0].confidence - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn l3_extraction_resolves_metadata_from_config() {
        let extractor = DefaultSignalExtractor::new();
        let pattern = CompiledPattern::compile_full(
            "test",
            PatternAction::Block,
            None,
            Some("injection".to_string()),
            0.9,
            false,
        )
        .unwrap();
        let config = make_config_with_patterns(vec![pattern]);

        let result = InboundResult {
            verdict: InboundVerdict::Allow,
            matched_patterns: vec![PatternMatch {
                pattern_index: 0,
                message_index: 0,
                role: Role::User,
                source: MatchSource::Content,
                action: PatternAction::Block,
            }],
        };
        let signals = extractor.extract_l3(&result, &config);
        assert_eq!(signals[0].category.as_deref(), Some("injection"));
        assert!((signals[0].score - 0.9).abs() < f32::EPSILON);
        assert!((signals[0].confidence - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn l4_extraction_aggregate_and_categories() {
        let extractor = DefaultSignalExtractor::new();
        let result = L4Result {
            verdict: L4Verdict::Allow,
            risk_score: 0.6,
            matched_categories: vec![
                L4CategoryMatch {
                    category: "instruction_seeding".to_string(),
                    weight: 0.4,
                    turn_indices: vec![0, 2],
                },
                L4CategoryMatch {
                    category: "role_confusion".to_string(),
                    weight: 0.3,
                    turn_indices: vec![1],
                },
            ],
            flagged_turns: vec![0, 1, 2],
        };
        let signals = extractor.extract_l4(&result);
        assert_eq!(signals.len(), 3); // 1 aggregate + 2 categories
        // Aggregate
        assert_eq!(signals[0].layer, LayerId::L4);
        assert!(signals[0].category.is_none());
        assert!((signals[0].score - 0.6).abs() < f32::EPSILON);
        assert!((signals[0].confidence - 1.0).abs() < f32::EPSILON);
        // Categories
        assert_eq!(signals[1].category.as_deref(), Some("instruction_seeding"));
        assert!((signals[1].confidence - 0.8).abs() < f32::EPSILON);
        assert_eq!(signals[2].category.as_deref(), Some("role_confusion"));
    }

    // -- L1Signal-based extraction --

    #[test]
    fn l1_signals_emits_all_messages() {
        let extractor = DefaultSignalExtractor::new();
        let signals = vec![
            L1Signal {
                message_index: 0,
                role: Role::User,
                raw_score: -3.0,
                raw_unquoted_score: -3.0,
                raw_squash_score: -3.0,
                score: 0.14,
                unquoted_score: 0.14,
                squash_score: 0.14,
                quote_detected: false,
                raw_score_delta: 0.0,
            },
            L1Signal {
                message_index: 1,
                role: Role::User,
                raw_score: 3.0,
                raw_unquoted_score: 3.0,
                raw_squash_score: 3.0,
                score: 0.86,
                unquoted_score: 0.86,
                squash_score: 0.86,
                quote_detected: false,
                raw_score_delta: 0.0,
            },
        ];
        let extracted = extractor.extract_l1_signals(&signals);
        // No filtering — all messages emit signals.
        assert_eq!(extracted.len(), 2);
        assert!((extracted[0].score - 0.14).abs() < f32::EPSILON);
        assert!((extracted[1].score - 0.86).abs() < f32::EPSILON);
    }

    #[test]
    fn l1_signals_sets_message_index() {
        let extractor = DefaultSignalExtractor::new();
        let signals = vec![L1Signal {
            message_index: 7,
            role: Role::Assistant,
            raw_score: 1.0,
            raw_unquoted_score: 1.0,
            raw_squash_score: 1.0,
            score: 0.65,
            unquoted_score: 0.65,
            squash_score: 0.65,
            quote_detected: false,
            raw_score_delta: 0.0,
        }];
        let extracted = extractor.extract_l1_signals(&signals);
        assert_eq!(extracted.len(), 1);
        assert_eq!(extracted[0].message_index, Some(7));
        assert_eq!(extracted[0].layer, LayerId::L1);
        assert_eq!(extracted[0].kind, SignalKind::Evidence);
    }

    #[test]
    fn l1_signals_empty_input() {
        let extractor = DefaultSignalExtractor::new();
        let extracted = extractor.extract_l1_signals(&[]);
        assert!(extracted.is_empty());
    }

    #[test]
    fn l1_signals_mention_dampens_primary() {
        // quote_detected + raw_score_delta > threshold → primary dampened
        // to unquoted_score with category "mention_dampened".
        let extractor = DefaultSignalExtractor::new();
        let signals = vec![L1Signal {
            message_index: 0,
            role: Role::User,
            raw_score: 3.0,
            raw_unquoted_score: -1.0,
            raw_squash_score: 3.0,
            score: 0.86,
            unquoted_score: 0.14,
            squash_score: 0.86,
            quote_detected: true,
            raw_score_delta: 4.0, // well above threshold of 1.0
        }];
        let extracted = extractor.extract_l1_signals(&signals);
        // Only 1 signal — primary dampened to unquoted_score.
        assert_eq!(extracted.len(), 1);
        assert_eq!(extracted[0].category.as_deref(), Some("mention_dampened"));
        assert!((extracted[0].score - 0.14).abs() < f32::EPSILON,
            "mention should dampen primary to unquoted_score");
        assert_eq!(extracted[0].message_index, Some(0));
    }

    #[test]
    fn l1_signals_no_dampen_without_quotes() {
        // High delta but no quote_detected → primary at full score.
        let extractor = DefaultSignalExtractor::new();
        let signals = vec![L1Signal {
            message_index: 0,
            role: Role::User,
            raw_score: 3.0,
            raw_unquoted_score: -1.0,
            raw_squash_score: 3.0,
            score: 0.86,
            unquoted_score: 0.14,
            squash_score: 0.86,
            quote_detected: false,
            raw_score_delta: 4.0,
        }];
        let extracted = extractor.extract_l1_signals(&signals);
        assert_eq!(extracted.len(), 1);
        assert!(extracted[0].category.is_none());
        assert!((extracted[0].score - 0.86).abs() < f32::EPSILON,
            "no dampen without quote_detected");
    }

    #[test]
    fn l1_signals_no_dampen_below_threshold() {
        // Quotes detected but delta below threshold → primary at full score.
        let extractor = DefaultSignalExtractor::new();
        let signals = vec![L1Signal {
            message_index: 0,
            role: Role::User,
            raw_score: 0.5,
            raw_unquoted_score: 0.0,
            raw_squash_score: 0.5,
            score: 0.57,
            unquoted_score: 0.5,
            squash_score: 0.57,
            quote_detected: true,
            raw_score_delta: 0.5, // below threshold of 1.0
        }];
        let extracted = extractor.extract_l1_signals(&signals);
        assert_eq!(extracted.len(), 1);
        assert!(extracted[0].category.is_none());
        assert!((extracted[0].score - 0.57).abs() < f32::EPSILON,
            "small delta should not trigger dampening");
    }

    #[test]
    fn l1_signals_obfuscation_emitted() {
        // squash_score > score + gap → obfuscation signal.
        let extractor = DefaultSignalExtractor::new();
        let signals = vec![L1Signal {
            message_index: 0,
            role: Role::User,
            raw_score: 0.5,
            raw_unquoted_score: 0.5,
            raw_squash_score: 4.0,
            score: 0.57,
            unquoted_score: 0.57,
            squash_score: 0.92, // 0.35 gap, well above 0.1
            quote_detected: false,
            raw_score_delta: 0.0,
        }];
        let extracted = extractor.extract_l1_signals(&signals);
        assert_eq!(extracted.len(), 2);
        assert_eq!(extracted[1].category.as_deref(), Some("obfuscation"));
        assert!((extracted[1].score - 0.92).abs() < f32::EPSILON);
        assert!((extracted[1].confidence - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn l1_signals_obfuscation_not_emitted_small_gap() {
        // squash_score barely above score → no obfuscation signal.
        let extractor = DefaultSignalExtractor::new();
        let signals = vec![L1Signal {
            message_index: 0,
            role: Role::User,
            raw_score: 3.0,
            raw_unquoted_score: 3.0,
            raw_squash_score: 3.2,
            score: 0.86,
            unquoted_score: 0.86,
            squash_score: 0.87, // 0.01 gap, below 0.1
            quote_detected: false,
            raw_score_delta: 0.0,
        }];
        let extracted = extractor.extract_l1_signals(&signals);
        assert_eq!(extracted.len(), 1, "tiny squash gap should not trigger obfuscation");
    }

    #[test]
    fn l1_signals_mention_dampen_plus_obfuscation() {
        // Both mention dampening and obfuscation can fire on the same message.
        let extractor = DefaultSignalExtractor::new();
        let signals = vec![L1Signal {
            message_index: 0,
            role: Role::User,
            raw_score: 2.0,
            raw_unquoted_score: -1.0,
            raw_squash_score: 5.0,
            score: 0.77,
            unquoted_score: 0.14,
            squash_score: 0.95, // gap vs score = 0.18 > 0.1
            quote_detected: true,
            raw_score_delta: 3.0, // > 1.0
        }];
        let extracted = extractor.extract_l1_signals(&signals);
        // dampened primary + obfuscation = 2 signals.
        assert_eq!(extracted.len(), 2);
        assert_eq!(extracted[0].category.as_deref(), Some("mention_dampened"));
        assert!((extracted[0].score - 0.14).abs() < f32::EPSILON,
            "primary dampened to unquoted_score");
        assert_eq!(extracted[1].category.as_deref(), Some("obfuscation"));
        assert!((extracted[1].score - 0.95).abs() < f32::EPSILON);
    }
}
