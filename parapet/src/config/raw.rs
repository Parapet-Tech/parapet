// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// Raw YAML deserialization types (internal)
// These are separate from the public Config structs because:
// 1. serde_yaml needs Deserialize, but our public types contain Regex (not Deserialize)
// 2. We do variable interpolation and regex compilation between raw and public
// 3. Keeps the public API clean

use serde::Deserialize;
use std::collections::HashMap;

/// A block pattern can be a plain regex string or an object with metadata.
///
/// Plain string (backward compatible):
///   - "ignore previous instructions"
///
/// Object format (new):
///   - pattern: "ignore previous instructions"
///     category: instruction_bypass
///     weight: 0.8
///     atomic: false
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum RawBlockPattern {
    /// Plain regex string — defaults: weight=1.0, atomic=false, category=None.
    Simple(String),
    /// Object with explicit metadata fields.
    Full {
        pattern: String,
        category: Option<String>,
        #[serde(default = "default_pattern_weight")]
        weight: f64,
        #[serde(default)]
        atomic: bool,
    },
}

fn default_pattern_weight() -> f64 {
    1.0
}

#[derive(Debug, Deserialize)]
pub struct RawConfig {
    pub parapet: String,
    #[serde(default)]
    pub tools: HashMap<String, RawToolPolicy>,
    #[serde(default)]
    pub block_patterns: Vec<RawBlockPattern>,
    #[serde(default)]
    pub canary_tokens: Vec<String>,
    #[serde(default)]
    pub sensitive_patterns: Vec<String>,
    pub untrusted_content_policy: Option<RawContentPolicy>,
    pub trust: Option<RawTrustConfig>,
    pub engine: Option<RawEngineConfig>,
    pub environment: Option<String>,
    pub layers: Option<RawLayerConfigs>,
    /// If false, skip embedding default block patterns. Default: true.
    pub use_default_block_patterns: Option<bool>,
    /// If false, skip embedding default sensitive patterns. Default: true.
    pub use_default_sensitive_patterns: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct RawToolPolicy {
    pub allowed: bool,
    pub trust: Option<String>,
    #[serde(default)]
    pub constraints: HashMap<String, RawArgumentConstraints>,
    pub result_policy: Option<RawContentPolicy>,
}

#[derive(Debug, Deserialize)]
pub struct RawArgumentConstraints {
    #[serde(rename = "type")]
    pub type_check: Option<String>,
    pub starts_with: Option<String>,
    pub not_contains: Option<Vec<String>>,
    pub matches: Option<String>,
    pub one_of: Option<Vec<String>>,
    pub max_length: Option<usize>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub url_host: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct RawContentPolicy {
    pub max_length: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct RawTrustConfig {
    #[serde(default)]
    pub auto_untrusted_roles: Vec<String>,
    #[serde(default)]
    pub unknown_trust_policy: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
pub struct RawEngineConfig {
    pub on_failure: Option<String>,
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct RawLayerConfigs {
    #[serde(rename = "L0")]
    pub l0: Option<RawLayerConfig>,
    #[serde(rename = "L1")]
    pub l1: Option<RawL1Config>,
    #[serde(rename = "L2a")]
    pub l2a: Option<RawL2aConfig>,
    #[serde(rename = "L3_inbound")]
    pub l3_inbound: Option<RawLayerConfig>,
    #[serde(rename = "L3_outbound")]
    pub l3_outbound: Option<RawLayerConfig>,
    #[serde(rename = "L5a")]
    pub l5a: Option<RawLayerConfig>,
    #[serde(rename = "L4")]
    pub l4: Option<RawL4Config>,
}

#[derive(Debug, Deserialize)]
pub struct RawL2aConfig {
    pub mode: String,
    pub model: String,
    pub model_dir: Option<String>,
    pub pg_threshold: f32,
    pub block_threshold: f32,
    pub heuristic_weight: f32,
    pub fusion_confidence_agreement: f32,
    pub fusion_confidence_pg_only: f32,
    pub fusion_confidence_heuristic_only: f32,
    #[serde(default = "default_l2a_max_segments")]
    pub max_segments: usize,
    #[serde(default = "default_l2a_timeout_ms")]
    pub timeout_ms: u64,
    #[serde(default = "default_l2a_max_concurrent_scans")]
    pub max_concurrent_scans: usize,
}

fn default_l2a_max_segments() -> usize {
    16
}

fn default_l2a_timeout_ms() -> u64 {
    200
}

fn default_l2a_max_concurrent_scans() -> usize {
    4
}

#[derive(Debug, Deserialize)]
pub struct RawLayerConfig {
    pub mode: String,
    pub block_action: Option<String>,
    pub window_chars: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct RawL1Config {
    #[serde(default = "default_l1_mode")]
    pub mode: String,
    #[serde(default = "default_l1_threshold")]
    pub threshold: f64,
}

fn default_l1_mode() -> String {
    "shadow".to_string()
}

fn default_l1_threshold() -> f64 {
    0.5
}

#[derive(Debug, Deserialize)]
pub struct RawL4Config {
    #[serde(default = "default_l4_mode")]
    pub mode: String,
    #[serde(default = "default_risk_threshold")]
    pub risk_threshold: f64,
    #[serde(default = "default_escalation_bonus")]
    pub escalation_bonus: f64,
    #[serde(default = "default_resampling_bonus")]
    pub resampling_bonus: f64,
    #[serde(default = "default_persistence_factor")]
    pub persistence_factor: f64,
    #[serde(default = "default_diversity_factor")]
    pub diversity_factor: f64,
    #[serde(default = "default_min_user_turns")]
    pub min_user_turns: usize,
    #[serde(default)]
    pub cross_turn_patterns: Vec<RawL4PatternCategory>,
    pub use_default_l4_patterns: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct RawL4PatternCategory {
    pub category: String,
    pub weight: f64,
    pub patterns: Vec<String>,
}

fn default_l4_mode() -> String {
    "shadow".to_string()
}

fn default_risk_threshold() -> f64 {
    0.7
}

fn default_escalation_bonus() -> f64 {
    0.2
}

fn default_resampling_bonus() -> f64 {
    0.7
}

fn default_persistence_factor() -> f64 {
    0.45
}

fn default_diversity_factor() -> f64 {
    0.15
}

fn default_min_user_turns() -> usize {
    2
}
