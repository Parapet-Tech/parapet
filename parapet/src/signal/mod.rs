// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// Signal types for the verdict processor.
//
// Layers emit Signals (normalized observations). The VerdictCombiner combines
// them into a composite Verdict. In shadow mode the engine logs the verdict
// but does not enforce it.

pub mod combiner;
pub mod extractor;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Identifies which layer produced a signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerId {
    L1,
    L3,
    L4,
}

impl std::fmt::Display for LayerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayerId::L1 => write!(f, "L1"),
            LayerId::L3 => write!(f, "L3"),
            LayerId::L4 => write!(f, "L4"),
        }
    }
}

/// Whether a signal is an atomic (hard) block or soft evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalKind {
    /// Non-negotiable: match = block. Bypasses the combiner.
    AtomicBlock,
    /// Contributes to composite score.
    Evidence,
}

/// A normalized observation emitted by a layer.
#[derive(Debug, Clone)]
pub struct Signal {
    pub layer: LayerId,
    pub kind: SignalKind,
    pub category: Option<String>,
    /// Strength of the observation, clamped to [0.0, 1.0].
    pub score: f32,
    /// Reliability of the sensor for this observation, clamped to [0.0, 1.0].
    pub confidence: f32,
}

impl Signal {
    /// Construct a signal with clamped score and confidence.
    ///
    /// NaN and -INFINITY clamp to 0.0; INFINITY clamps to 1.0.
    pub fn new(
        layer: LayerId,
        kind: SignalKind,
        category: Option<String>,
        score: f32,
        confidence: f32,
    ) -> Self {
        Self {
            layer,
            kind,
            category,
            score: clamp_f32(score),
            confidence: clamp_f32(confidence),
        }
    }
}

/// Clamp a f32 to [0.0, 1.0], handling NaN and infinities.
fn clamp_f32(v: f32) -> f32 {
    if v.is_nan() || v == f32::NEG_INFINITY {
        0.0
    } else if v == f32::INFINITY {
        1.0
    } else {
        v.clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Verdict types
// ---------------------------------------------------------------------------

/// The combiner's recommended action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerdictAction {
    Allow,
    Block,
}

/// Composite verdict produced by the combiner.
#[derive(Debug, Clone)]
pub struct Verdict {
    pub action: VerdictAction,
    pub composite_score: f32,
    pub contributing: Vec<SignalContribution>,
}

/// What a single signal contributed to the verdict.
#[derive(Debug, Clone)]
pub struct SignalContribution {
    pub layer: LayerId,
    pub category: Option<String>,
    /// Score before dampening.
    pub raw_score: f32,
    /// Post-dampener score. Equal to `raw_score` unless the dampener fired.
    pub weighted_score: f32,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_new_clamps_nan_to_zero() {
        let s = Signal::new(LayerId::L1, SignalKind::Evidence, None, f32::NAN, f32::NAN);
        assert_eq!(s.score, 0.0);
        assert_eq!(s.confidence, 0.0);
    }

    #[test]
    fn signal_new_clamps_neg_infinity_to_zero() {
        let s = Signal::new(LayerId::L1, SignalKind::Evidence, None, f32::NEG_INFINITY, 0.5);
        assert_eq!(s.score, 0.0);
    }

    #[test]
    fn signal_new_clamps_infinity_to_one() {
        let s = Signal::new(LayerId::L1, SignalKind::Evidence, None, f32::INFINITY, 0.5);
        assert_eq!(s.score, 1.0);
    }

    #[test]
    fn signal_new_clamps_negative_to_zero() {
        let s = Signal::new(LayerId::L1, SignalKind::Evidence, None, -0.5, 0.5);
        assert_eq!(s.score, 0.0);
    }

    #[test]
    fn signal_new_clamps_above_one() {
        let s = Signal::new(LayerId::L1, SignalKind::Evidence, None, 1.5, 0.5);
        assert_eq!(s.score, 1.0);
    }

    #[test]
    fn signal_new_preserves_valid_values() {
        let s = Signal::new(LayerId::L1, SignalKind::Evidence, None, 0.7, 0.9);
        assert!((s.score - 0.7).abs() < f32::EPSILON);
        assert!((s.confidence - 0.9).abs() < f32::EPSILON);
    }
}
