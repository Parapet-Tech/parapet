use std::collections::HashMap;

use crate::message::TrustLevel;

use super::pattern::CompiledPattern;

// ---------------------------------------------------------------------------
// Top-level config (combines policy + runtime)
// ---------------------------------------------------------------------------

/// Top-level parsed and validated parapet config.
#[derive(Debug)]
pub struct Config {
    /// Policy configuration — version-controlled, code-reviewed, contract-hashed.
    pub policy: PolicyConfig,
    /// Runtime configuration — varies by environment, not hashed.
    pub runtime: RuntimeConfig,
    /// SHA256 hash of the policy-relevant YAML: "sha256:{hex}".
    pub contract_hash: String,
}

// Convenience accessors so callers can write `config.version` etc.
// This keeps the refactor non-breaking at the field-access level.
impl Config {
    pub fn version(&self) -> &str {
        &self.policy.version
    }
}

// ---------------------------------------------------------------------------
// Policy config — hashed, version-controlled
// ---------------------------------------------------------------------------

/// Policy-relevant configuration: tools, patterns, trust, layers.
/// Changes here change the contract hash.
#[derive(Debug)]
pub struct PolicyConfig {
    /// Contract version. Always "v1".
    pub version: String,
    /// Tool policies keyed by tool name. "_default" is the fallback.
    pub tools: HashMap<String, ToolConfig>,
    /// Pre-compiled block patterns scanned against all messages.
    pub block_patterns: Vec<CompiledPattern>,
    /// Canary tokens to detect and redact in LLM output.
    pub canary_tokens: Vec<String>,
    /// Pre-compiled patterns for sensitive data (API keys, PII).
    pub sensitive_patterns: Vec<CompiledPattern>,
    /// Policy for untrusted message content.
    pub untrusted_content_policy: ContentPolicy,
    /// Trust assignment rules.
    pub trust: TrustConfig,
    /// Per-layer configuration.
    pub layers: LayerConfigs,
}

// ---------------------------------------------------------------------------
// Runtime config — not hashed
// ---------------------------------------------------------------------------

/// Runtime configuration that varies by environment.
#[derive(Debug)]
pub struct RuntimeConfig {
    /// Engine runtime configuration.
    pub engine: EngineConfig,
    /// Environment label (e.g. "internal", "production").
    pub environment: String,
}

// ---------------------------------------------------------------------------
// Typed config structs
// ---------------------------------------------------------------------------

/// Policy for a single tool (or the _default fallback).
#[derive(Debug)]
pub struct ToolConfig {
    /// Whether this tool is permitted to be called.
    pub allowed: bool,
    /// Trust level override for this tool's result messages.
    pub trust: Option<TrustLevel>,
    /// Per-argument constraints. Key is argument name.
    pub constraints: HashMap<String, ArgumentConstraints>,
    /// Result policy (overrides untrusted_content_policy for this tool).
    pub result_policy: Option<ContentPolicy>,
}

/// Predicate constraints for a single tool argument.
/// All specified predicates must be satisfied (AND semantics).
#[derive(Debug, Default)]
pub struct ArgumentConstraints {
    /// Expected type: "string", "number", "boolean".
    pub type_check: Option<String>,
    /// Value must start with this prefix.
    pub starts_with: Option<String>,
    /// Value must not contain any of these substrings.
    pub not_contains: Option<Vec<String>>,
    /// Value must match this regex (pre-compiled at load time).
    pub matches: Option<CompiledPattern>,
    /// Value must be one of these allowed values.
    pub one_of: Option<Vec<String>>,
    /// Maximum string length.
    pub max_length: Option<usize>,
    /// Minimum numeric value.
    pub min: Option<f64>,
    /// Maximum numeric value.
    pub max: Option<f64>,
    /// URL host must be one of these.
    pub url_host: Option<Vec<String>>,
}

/// Content policy controlling message size limits.
#[derive(Debug, Clone, Default)]
pub struct ContentPolicy {
    /// Maximum content length per message in characters.
    pub max_length: Option<usize>,
}

/// Trust assignment rules.
#[derive(Debug, Default)]
pub struct TrustConfig {
    /// Roles automatically assigned untrusted trust level.
    pub auto_untrusted_roles: Vec<String>,
    /// Base trust assignment mapping (role -> trust level).
    pub unknown_trust_policy: HashMap<String, TrustLevel>,
}

/// Engine runtime configuration.
#[derive(Debug)]
pub struct EngineConfig {
    /// "open" bypasses on failure; "closed" returns error.
    pub on_failure: FailureMode,
    /// Request timeout in milliseconds.
    pub timeout_ms: Option<u64>,
}

/// Engine failure mode.
#[derive(Debug, Clone, PartialEq)]
pub enum FailureMode {
    Open,
    Closed,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            on_failure: FailureMode::Open,
            timeout_ms: None,
        }
    }
}

/// Per-layer configuration.
#[derive(Debug, Default)]
pub struct LayerConfigs {
    pub l0: Option<LayerConfig>,
    pub l3_inbound: Option<LayerConfig>,
    pub l3_outbound: Option<LayerConfig>,
    pub l5a: Option<LayerConfig>,
    pub l4: Option<L4Config>,
}

/// Configuration for a single processing layer.
#[derive(Debug, Clone)]
pub struct LayerConfig {
    /// Processing mode (e.g. "sanitize", "block", "redact").
    pub mode: String,
    /// Action when content is blocked (e.g. "rewrite", "error").
    pub block_action: Option<String>,
    /// Optional window size for streaming redaction (L5a only).
    pub window_chars: Option<usize>,
}

/// L4 operating mode.
#[derive(Debug, Clone, PartialEq)]
pub enum L4Mode {
    /// Log detections but don't block (calibration mode).
    Shadow,
    /// Actively block requests exceeding risk threshold.
    Block,
}

/// Configuration for the L4 multi-turn scanning layer.
#[derive(Debug, Clone)]
pub struct L4Config {
    /// Operating mode: shadow (log-only) or block (enforce).
    pub mode: L4Mode,
    /// Risk score threshold for blocking (0.0 - 1.0).
    pub risk_threshold: f64,
    /// Bonus added to cumulative score when escalation gradient detected.
    pub escalation_bonus: f64,
    /// Bonus added when repetition/resampling detected.
    pub resampling_bonus: f64,
    /// Factor multiplied by match_ratio (matched_turns / total_turns).
    /// Higher values make persistence (same patterns across many turns) weigh more.
    pub persistence_factor: f64,
    /// Bonus per additional distinct category matched beyond the first.
    /// Rewards attack diversity (e.g., instruction_seeding + role_confusion).
    pub diversity_factor: f64,
    /// Minimum number of user turns before L4 activates.
    pub min_user_turns: usize,
    /// Cross-turn pattern categories with compiled regexes.
    pub cross_turn_patterns: Vec<L4PatternCategory>,
}

/// A category of cross-turn patterns with a weight.
#[derive(Debug, Clone)]
pub struct L4PatternCategory {
    /// Category name (e.g., "instruction_seeding", "role_confusion").
    pub category: String,
    /// Weight contributed to per-turn risk score when matched.
    pub weight: f64,
    /// Compiled regex patterns for this category.
    pub patterns: Vec<CompiledPattern>,
}
