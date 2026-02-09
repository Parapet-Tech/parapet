// Raw YAML deserialization types (internal)
// These are separate from the public Config structs because:
// 1. serde_yaml needs Deserialize, but our public types contain Regex (not Deserialize)
// 2. We do variable interpolation and regex compilation between raw and public
// 3. Keeps the public API clean

use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub struct RawConfig {
    pub parapet: String,
    #[serde(default)]
    pub tools: HashMap<String, RawToolPolicy>,
    #[serde(default)]
    pub block_patterns: Vec<String>,
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
pub struct RawLayerConfig {
    pub mode: String,
    pub block_action: Option<String>,
    pub window_chars: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct RawL4Config {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_max_history")]
    pub max_history: usize,
    #[serde(default = "default_session_ttl")]
    pub session_ttl_secs: u64,
    #[serde(default)]
    pub detectors: Vec<RawDetectorConfig>,
}

#[derive(Debug, Deserialize)]
pub struct RawDetectorConfig {
    pub name: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    pub threshold: Option<f64>,
}

fn default_max_history() -> usize {
    50
}

fn default_session_ttl() -> u64 {
    3600
}

fn default_true() -> bool {
    true
}
