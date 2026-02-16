// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// Verdict combiner — combines signals into a composite verdict.

use std::collections::HashSet;

use super::{LayerId, Signal, SignalContribution, SignalKind, Verdict, VerdictAction};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Combines signals into a composite verdict.
///
/// Returns the **recommended** action — the engine decides whether to enforce
/// (shadow logs it, hybrid enforces it).
pub trait VerdictCombiner: Send + Sync {
    fn combine(&self, signals: &[Signal]) -> Verdict;
}

// ---------------------------------------------------------------------------
// Default implementation
// ---------------------------------------------------------------------------

/// SIEM-style combiner: max baseline + cross-layer boost + multiplicative dampener.
pub struct DefaultVerdictCombiner;

impl DefaultVerdictCombiner {
    pub fn new() -> Self {
        Self
    }
}

/// Threshold above which the combiner recommends Block.
const BLOCK_THRESHOLD: f32 = 0.5;

/// Cross-layer boost factor when 2+ layers agree.
const BOOST_FACTOR: f32 = 0.2;

/// Minimum signal score to count as "agreeing" for cross-layer boost.
const AGREEMENT_THRESHOLD: f32 = 0.3;

/// L1 score below which the dampener activates.
const DAMPENER_L1_THRESHOLD: f32 = 0.2;

/// Multiplicative dampener factor applied to low-confidence L3 signals.
const DAMPENER_FACTOR: f32 = 0.4;

impl VerdictCombiner for DefaultVerdictCombiner {
    fn combine(&self, signals: &[Signal]) -> Verdict {
        // Step 1: Atomic fast-path
        for s in signals {
            if s.kind == SignalKind::AtomicBlock {
                return Verdict {
                    action: VerdictAction::Block,
                    composite_score: 1.0,
                    contributing: vec![SignalContribution {
                        layer: s.layer,
                        category: s.category.clone(),
                        raw_score: s.score,
                        weighted_score: s.score,
                    }],
                };
            }
        }

        // Collect only Evidence signals (AtomicBlock already handled).
        let evidence: Vec<&Signal> = signals
            .iter()
            .filter(|s| s.kind == SignalKind::Evidence && s.score > 0.0)
            .collect();

        if evidence.is_empty() {
            return Verdict {
                action: VerdictAction::Allow,
                composite_score: 0.0,
                contributing: vec![],
            };
        }

        // Step 2: Dampener — compute L1 aggregate, then dampen eligible L3 signals.
        let l1_max = evidence
            .iter()
            .filter(|s| s.layer == LayerId::L1)
            .map(|s| s.score)
            .fold(f32::NEG_INFINITY, f32::max);
        let l1_max = if l1_max == f32::NEG_INFINITY { 0.5 } else { l1_max };

        let dampen_l3 = l1_max < DAMPENER_L1_THRESHOLD;

        // Build contributions with effective (post-dampener) scores.
        let mut contributions: Vec<SignalContribution> = Vec::with_capacity(evidence.len());
        let mut effective_scores: Vec<(LayerId, f32, f32)> = Vec::with_capacity(evidence.len());

        for s in &evidence {
            let weighted = if dampen_l3 && s.layer == LayerId::L3 && s.confidence < 1.0 {
                s.score * DAMPENER_FACTOR
            } else {
                s.score
            };
            contributions.push(SignalContribution {
                layer: s.layer,
                category: s.category.clone(),
                raw_score: s.score,
                weighted_score: weighted,
            });
            effective_scores.push((s.layer, weighted, s.confidence));
        }

        // Step 3: Baseline = max effective score.
        let baseline = effective_scores
            .iter()
            .map(|(_, score, _)| *score)
            .fold(0.0_f32, f32::max);

        // Step 4: Cross-layer boost.
        // Collect unique layers with effective_score >= AGREEMENT_THRESHOLD.
        let agreeing_layers: HashSet<LayerId> = effective_scores
            .iter()
            .filter(|(_, score, _)| *score >= AGREEMENT_THRESHOLD)
            .map(|(layer, _, _)| *layer)
            .collect();

        let boost = if agreeing_layers.len() >= 2 {
            // min confidence among all agreeing signals
            let min_conf = effective_scores
                .iter()
                .filter(|(layer, score, _)| {
                    agreeing_layers.contains(layer) && *score >= AGREEMENT_THRESHOLD
                })
                .map(|(_, _, conf)| *conf)
                .fold(f32::INFINITY, f32::min);
            let min_conf = if min_conf == f32::INFINITY { 0.0 } else { min_conf };
            BOOST_FACTOR * min_conf
        } else {
            0.0
        };

        // Step 5: Composite + action.
        let composite = (baseline + boost).clamp(0.0, 1.0);
        let action = if composite >= BLOCK_THRESHOLD {
            VerdictAction::Block
        } else {
            VerdictAction::Allow
        };

        Verdict {
            action,
            composite_score: composite,
            contributing: contributions,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(layer: LayerId, score: f32, confidence: f32) -> Signal {
        Signal::new(layer, SignalKind::Evidence, None, score, confidence)
    }

    fn evidence_cat(
        layer: LayerId,
        score: f32,
        confidence: f32,
        category: &str,
    ) -> Signal {
        Signal::new(
            layer,
            SignalKind::Evidence,
            Some(category.to_string()),
            score,
            confidence,
        )
    }

    fn atomic(layer: LayerId) -> Signal {
        Signal::new(layer, SignalKind::AtomicBlock, None, 1.0, 1.0)
    }

    #[test]
    fn atomic_fast_path_returns_block() {
        let combiner = DefaultVerdictCombiner::new();
        let verdict = combiner.combine(&[atomic(LayerId::L3)]);
        assert_eq!(verdict.action, VerdictAction::Block);
        assert!((verdict.composite_score - 1.0).abs() < f32::EPSILON);
        assert_eq!(verdict.contributing.len(), 1);
    }

    #[test]
    fn single_evidence_baseline_only() {
        let combiner = DefaultVerdictCombiner::new();
        let verdict = combiner.combine(&[evidence(LayerId::L3, 0.5, 1.0)]);
        // No L1 signals → l1_max = 0.5, no dampening. Single layer → no boost.
        assert!((verdict.composite_score - 0.5).abs() < f32::EPSILON);
        assert_eq!(verdict.action, VerdictAction::Block); // 0.5 >= threshold
    }

    #[test]
    fn below_threshold_allows() {
        let combiner = DefaultVerdictCombiner::new();
        let verdict = combiner.combine(&[evidence(LayerId::L3, 0.3, 1.0)]);
        assert_eq!(verdict.action, VerdictAction::Allow);
        assert!((verdict.composite_score - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn cross_layer_boost_applied() {
        let combiner = DefaultVerdictCombiner::new();
        let signals = vec![
            evidence(LayerId::L1, 0.6, 1.0),
            evidence(LayerId::L3, 0.4, 1.0),
        ];
        let verdict = combiner.combine(&signals);
        // baseline = max(0.6, 0.4) = 0.6
        // Both >= 0.3, different layers → boost = 0.2 * min(1.0, 1.0) = 0.2
        // composite = 0.6 + 0.2 = 0.8
        assert!((verdict.composite_score - 0.8).abs() < f32::EPSILON);
        assert_eq!(verdict.action, VerdictAction::Block);
    }

    #[test]
    fn no_self_boost_same_layer() {
        let combiner = DefaultVerdictCombiner::new();
        let signals = vec![
            evidence_cat(LayerId::L3, 0.5, 1.0, "cat_a"),
            evidence_cat(LayerId::L3, 0.4, 1.0, "cat_b"),
        ];
        let verdict = combiner.combine(&signals);
        // Both L3 → only 1 unique layer → no boost
        // baseline = max(0.5, 0.4) = 0.5
        assert!((verdict.composite_score - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn dampener_targets_low_confidence_l3_only() {
        let combiner = DefaultVerdictCombiner::new();
        let signals = vec![
            evidence(LayerId::L1, 0.1, 1.0),          // L1 < 0.2 → dampener active
            evidence(LayerId::L3, 0.8, 1.0),           // block-action (confidence 1.0) → NOT dampened
            evidence(LayerId::L3, 0.8, 0.6),           // evidence-action (confidence 0.6) → dampened
        ];
        let verdict = combiner.combine(&signals);
        // L3 block-action: effective = 0.8 (undampened)
        // L3 evidence-action: effective = 0.8 * 0.4 = 0.32
        // baseline = max(0.1, 0.8, 0.32) = 0.8
        // L1 effective 0.1 < 0.3 → doesn't count for agreement
        // L3 effective 0.8 >= 0.3 → counts, but only one unique layer (L3) → no boost
        // composite = 0.8
        assert!((verdict.composite_score - 0.8).abs() < f32::EPSILON);

        // Verify contributions reflect dampening
        let l3_evidence = verdict
            .contributing
            .iter()
            .find(|c| c.layer == LayerId::L3 && (c.weighted_score - 0.32).abs() < f32::EPSILON);
        assert!(l3_evidence.is_some(), "dampened L3 evidence signal should have weighted_score 0.32");
    }

    #[test]
    fn no_dampener_when_l1_absent() {
        let combiner = DefaultVerdictCombiner::new();
        let signals = vec![evidence(LayerId::L3, 0.8, 0.6)];
        let verdict = combiner.combine(&signals);
        // No L1 → l1_max = 0.5, no dampening
        // effective = 0.8 (undampened)
        assert!((verdict.composite_score - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn high_composite_recommends_block() {
        let combiner = DefaultVerdictCombiner::new();
        let signals = vec![evidence(LayerId::L1, 0.9, 1.0)];
        let verdict = combiner.combine(&signals);
        assert_eq!(verdict.action, VerdictAction::Block);
    }

    #[test]
    fn low_composite_recommends_allow() {
        let combiner = DefaultVerdictCombiner::new();
        let signals = vec![evidence(LayerId::L1, 0.2, 1.0)];
        let verdict = combiner.combine(&signals);
        assert_eq!(verdict.action, VerdictAction::Allow);
    }

    #[test]
    fn empty_signals_returns_allow() {
        let combiner = DefaultVerdictCombiner::new();
        let verdict = combiner.combine(&[]);
        assert_eq!(verdict.action, VerdictAction::Allow);
        assert_eq!(verdict.composite_score, 0.0);
    }
}
