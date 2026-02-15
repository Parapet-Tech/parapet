// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

use regex::{Regex, RegexBuilder};

use crate::message::TrustLevel;

use super::error::ConfigError;

/// What happens when a pattern matches.
#[derive(Debug, Clone, PartialEq)]
pub enum PatternAction {
    /// Match triggers a block verdict (default for block_patterns).
    Block,
    /// Match is collected as evidence but never triggers a block on its own.
    Evidence,
}

impl Default for PatternAction {
    fn default() -> Self {
        Self::Block
    }
}

/// Maximum compiled regex size (1 MB). Prevents pathological patterns
/// from consuming excessive memory at startup.
const MAX_REGEX_SIZE: usize = 1024 * 1024;

/// A pre-compiled regex pattern. Wraps `regex::Regex` with the original
/// pattern string preserved for debugging/display.
#[derive(Clone)]
pub struct CompiledPattern {
    pub pattern: String,
    pub regex: Regex,
    /// What to do when this pattern matches. Default: `Block`.
    pub action: PatternAction,
    /// If set, only match content at this trust level (or lower).
    /// `None` means scan all content regardless of trust.
    pub trust_gate: Option<TrustLevel>,
    /// Category label for grouping evidence signals (e.g. "exfil", "roleplay").
    pub category: Option<String>,
    /// Weight for signal scoring. Higher = stronger signal. Default: 1.0.
    pub weight: f64,
    /// Atomic patterns bypass the verdict processor combiner — match = block.
    /// Only structural exploits with zero FP in eval corpus qualify.
    /// Default: false.
    pub atomic: bool,
}

impl CompiledPattern {
    /// Compile a regex pattern with default action (`Block`), no trust gate, no category.
    /// Returns `ConfigError::InvalidRegex` on failure.
    pub fn compile(pattern: &str) -> Result<Self, ConfigError> {
        Self::compile_with(pattern, PatternAction::Block, None, None)
    }

    /// Compile a regex pattern with explicit action, trust gate, and category.
    pub fn compile_with(
        pattern: &str,
        action: PatternAction,
        trust_gate: Option<TrustLevel>,
        category: Option<String>,
    ) -> Result<Self, ConfigError> {
        Self::compile_full(pattern, action, trust_gate, category, 1.0, false)
    }

    /// Compile a regex pattern with all metadata fields.
    pub fn compile_full(
        pattern: &str,
        action: PatternAction,
        trust_gate: Option<TrustLevel>,
        category: Option<String>,
        weight: f64,
        atomic: bool,
    ) -> Result<Self, ConfigError> {
        let regex = RegexBuilder::new(pattern)
            .size_limit(MAX_REGEX_SIZE)
            .build()
            .map_err(|e| ConfigError::InvalidRegex {
                pattern: pattern.to_string(),
                source: e,
            })?;
        Ok(Self {
            pattern: pattern.to_string(),
            regex,
            action,
            trust_gate,
            category,
            weight,
            atomic,
        })
    }

    /// Test whether the pattern matches the given text.
    pub fn is_match(&self, text: &str) -> bool {
        self.regex.is_match(text)
    }
}

impl fmt::Debug for CompiledPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledPattern")
            .field("pattern", &self.pattern)
            .field("action", &self.action)
            .field("trust_gate", &self.trust_gate)
            .field("category", &self.category)
            .field("weight", &self.weight)
            .field("atomic", &self.atomic)
            .finish()
    }
}

impl PartialEq for CompiledPattern {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern
            && self.action == other.action
            && self.trust_gate == other.trust_gate
            && self.category == other.category
            && self.weight == other.weight
            && self.atomic == other.atomic
    }
}
