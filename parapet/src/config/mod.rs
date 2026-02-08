// Config loader and validator -- defined in M1.2
//
// Loads parapet.yaml, validates structure, resolves variable interpolation,
// compiles regex patterns, and computes a deterministic contract hash.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

use regex::Regex;
use sha2::{Digest, Sha256};

use crate::message::TrustLevel;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// All errors that can occur during config loading and validation.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config source: {0}")]
    IoError(#[from] std::io::Error),

    #[error("failed to parse YAML: {0}")]
    YamlError(#[from] serde_yaml::Error),

    #[error("validation error: {0}")]
    Validation(String),

    #[error("invalid regex pattern \"{pattern}\": {source}")]
    InvalidRegex {
        pattern: String,
        source: regex::Error,
    },

    #[error("undefined variable ${{{name}}} in config (not set in environment)")]
    UndefinedVariable { name: String },
}

// ---------------------------------------------------------------------------
// ConfigSource trait (interface-first, dependency injection)
// ---------------------------------------------------------------------------

/// Abstraction over where config YAML comes from.
///
/// `FileSource` reads from disk; `StringSource` provides content directly
/// (used in tests to avoid file I/O).
pub trait ConfigSource {
    fn load(&self) -> Result<String, ConfigError>;
}

/// Loads config from a file on disk.
pub struct FileSource {
    pub path: PathBuf,
}

impl ConfigSource for FileSource {
    fn load(&self) -> Result<String, ConfigError> {
        Ok(std::fs::read_to_string(&self.path)?)
    }
}

/// Provides config content directly as a string. Used for testing.
pub struct StringSource {
    pub content: String,
}

impl ConfigSource for StringSource {
    fn load(&self) -> Result<String, ConfigError> {
        Ok(self.content.clone())
    }
}

// ---------------------------------------------------------------------------
// Compiled regex wrapper
// ---------------------------------------------------------------------------

/// A pre-compiled regex pattern. Wraps `regex::Regex` with the original
/// pattern string preserved for debugging/display.
#[derive(Clone)]
pub struct CompiledPattern {
    pub pattern: String,
    pub regex: Regex,
}

impl CompiledPattern {
    /// Compile a regex pattern, returning `ConfigError::InvalidRegex` on failure.
    pub fn compile(pattern: &str) -> Result<Self, ConfigError> {
        let regex = Regex::new(pattern).map_err(|e| ConfigError::InvalidRegex {
            pattern: pattern.to_string(),
            source: e,
        })?;
        Ok(Self {
            pattern: pattern.to_string(),
            regex,
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
            .finish()
    }
}

impl PartialEq for CompiledPattern {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern
    }
}

// ---------------------------------------------------------------------------
// Typed config structs
// ---------------------------------------------------------------------------

/// Top-level parsed and validated parapet config.
#[derive(Debug)]
pub struct Config {
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
    /// Engine runtime configuration.
    pub engine: EngineConfig,
    /// Environment label (e.g. "internal", "production").
    pub environment: String,
    /// Per-layer configuration.
    pub layers: LayerConfigs,
    /// SHA256 hash of the raw YAML bytes: "sha256:{hex}".
    pub contract_hash: String,
}

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

// ---------------------------------------------------------------------------
// Raw YAML deserialization types (internal)
// ---------------------------------------------------------------------------
// These are separate from the public Config structs because:
// 1. serde_yaml needs Deserialize, but our public types contain Regex (not Deserialize)
// 2. We do variable interpolation and regex compilation between raw and public
// 3. Keeps the public API clean

mod raw {
    use serde::Deserialize;
    use std::collections::HashMap;

    #[derive(Debug, Deserialize)]
    pub struct RawConfig {
        pub parapet: String,
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
    }

    #[derive(Debug, Deserialize)]
    pub struct RawLayerConfig {
        pub mode: String,
        pub block_action: Option<String>,
        pub window_chars: Option<usize>,
    }
}

// ---------------------------------------------------------------------------
// Variable interpolation
// ---------------------------------------------------------------------------

/// Resolves `${VAR_NAME}` references in a string from environment variables.
/// Returns `ConfigError::UndefinedVariable` if a referenced variable is not set.
fn resolve_variables(input: &str) -> Result<String, ConfigError> {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '$' && chars.peek() == Some(&'{') {
            chars.next(); // consume '{'
            let mut var_name = String::new();
            let mut found_close = false;
            for c in chars.by_ref() {
                if c == '}' {
                    found_close = true;
                    break;
                }
                var_name.push(c);
            }
            if !found_close || var_name.is_empty() {
                // Malformed interpolation -- treat literally
                result.push('$');
                result.push('{');
                result.push_str(&var_name);
                continue;
            }
            let value = std::env::var(&var_name).map_err(|_| ConfigError::UndefinedVariable {
                name: var_name.clone(),
            })?;
            result.push_str(&value);
        } else {
            result.push(ch);
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Config loading and validation
// ---------------------------------------------------------------------------

/// Load and validate a parapet config from the given source.
///
/// Steps:
/// 1. Read raw YAML bytes from source
/// 2. Compute SHA256 contract hash
/// 3. Parse YAML into raw deserialization types
/// 4. Validate required fields and values
/// 5. Resolve variable interpolation in string fields
/// 6. Compile regex patterns (block_patterns, sensitive_patterns, matches constraints)
/// 7. Build typed Config struct
pub fn load_config(source: &dyn ConfigSource) -> Result<Config, ConfigError> {
    let raw_yaml = source.load()?;
    let contract_hash = compute_hash(&raw_yaml);

    let raw: raw::RawConfig = serde_yaml::from_str(&raw_yaml)?;

    // Validate version
    if raw.parapet != "v1" {
        return Err(ConfigError::Validation(format!(
            "unsupported contract version \"{}\", expected \"v1\"",
            raw.parapet
        )));
    }

    // Validate tools: must have at least one entry
    if raw.tools.is_empty() {
        return Err(ConfigError::Validation(
            "\"tools\" must contain at least one tool policy".to_string(),
        ));
    }

    // Build tools
    let mut tools = HashMap::with_capacity(raw.tools.len());
    for (name, raw_tool) in raw.tools {
        let tool_config = build_tool_config(&raw_tool)?;
        tools.insert(name, tool_config);
    }

    // Compile block_patterns
    let block_patterns = raw
        .block_patterns
        .iter()
        .map(|p| CompiledPattern::compile(p))
        .collect::<Result<Vec<_>, _>>()?;

    // Compile sensitive_patterns
    let sensitive_patterns = raw
        .sensitive_patterns
        .iter()
        .map(|p| CompiledPattern::compile(p))
        .collect::<Result<Vec<_>, _>>()?;

    // Content policy
    let untrusted_content_policy = raw
        .untrusted_content_policy
        .map(|cp| ContentPolicy {
            max_length: cp.max_length,
        })
        .unwrap_or_default();

    // Trust config
    let trust = build_trust_config(raw.trust)?;

    // Engine config
    let engine = build_engine_config(raw.engine)?;

    // Layers
    let layers = build_layer_configs(raw.layers);

    Ok(Config {
        version: raw.parapet,
        tools,
        block_patterns,
        canary_tokens: raw.canary_tokens,
        sensitive_patterns,
        untrusted_content_policy,
        trust,
        engine,
        environment: raw.environment.unwrap_or_default(),
        layers,
        contract_hash,
    })
}

fn compute_hash(raw_yaml: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(raw_yaml.as_bytes());
    let hash = hasher.finalize();
    format!("sha256:{:x}", hash)
}

fn build_tool_config(raw: &raw::RawToolPolicy) -> Result<ToolConfig, ConfigError> {
    let trust = match &raw.trust {
        Some(t) => Some(parse_trust_level(t)?),
        None => None,
    };

    let mut constraints = HashMap::with_capacity(raw.constraints.len());
    for (arg_name, raw_constraint) in &raw.constraints {
        constraints.insert(arg_name.clone(), build_argument_constraints(raw_constraint)?);
    }

    let result_policy = raw.result_policy.as_ref().map(|rp| ContentPolicy {
        max_length: rp.max_length,
    });

    Ok(ToolConfig {
        allowed: raw.allowed,
        trust,
        constraints,
        result_policy,
    })
}

fn build_argument_constraints(
    raw: &raw::RawArgumentConstraints,
) -> Result<ArgumentConstraints, ConfigError> {
    // Validate type_check if present
    if let Some(ref tc) = raw.type_check {
        match tc.as_str() {
            "string" | "number" | "boolean" => {}
            other => {
                return Err(ConfigError::Validation(format!(
                    "unknown argument type \"{other}\", expected \"string\", \"number\", or \"boolean\""
                )));
            }
        }
    }

    // Resolve variable interpolation in starts_with
    let starts_with = match &raw.starts_with {
        Some(sw) => Some(resolve_variables(sw)?),
        None => None,
    };

    // Compile matches regex
    let matches = match &raw.matches {
        Some(pattern) => Some(CompiledPattern::compile(pattern)?),
        None => None,
    };

    Ok(ArgumentConstraints {
        type_check: raw.type_check.clone(),
        starts_with,
        not_contains: raw.not_contains.clone(),
        matches,
        one_of: raw.one_of.clone(),
        max_length: raw.max_length,
        min: raw.min,
        max: raw.max,
        url_host: raw.url_host.clone(),
    })
}

fn parse_trust_level(s: &str) -> Result<TrustLevel, ConfigError> {
    match s {
        "trusted" => Ok(TrustLevel::Trusted),
        "untrusted" => Ok(TrustLevel::Untrusted),
        other => Err(ConfigError::Validation(format!(
            "unknown trust level \"{other}\", expected \"trusted\" or \"untrusted\""
        ))),
    }
}

fn build_trust_config(raw: Option<raw::RawTrustConfig>) -> Result<TrustConfig, ConfigError> {
    let raw = match raw {
        Some(r) => r,
        None => return Ok(TrustConfig::default()),
    };

    let mut unknown_trust_policy = HashMap::with_capacity(raw.unknown_trust_policy.len());
    for (role, level_str) in &raw.unknown_trust_policy {
        unknown_trust_policy.insert(role.clone(), parse_trust_level(level_str)?);
    }

    Ok(TrustConfig {
        auto_untrusted_roles: raw.auto_untrusted_roles,
        unknown_trust_policy,
    })
}

fn build_engine_config(raw: Option<raw::RawEngineConfig>) -> Result<EngineConfig, ConfigError> {
    let raw = match raw {
        Some(r) => r,
        None => return Ok(EngineConfig::default()),
    };

    let on_failure = match raw.on_failure.as_deref() {
        Some("open") | None => FailureMode::Open,
        Some("closed") => FailureMode::Closed,
        Some(other) => {
            return Err(ConfigError::Validation(format!(
                "unknown engine on_failure value \"{other}\", expected \"open\" or \"closed\""
            )));
        }
    };

    Ok(EngineConfig {
        on_failure,
        timeout_ms: raw.timeout_ms,
    })
}

fn build_layer_configs(raw: Option<raw::RawLayerConfigs>) -> LayerConfigs {
    let raw = match raw {
        Some(r) => r,
        None => return LayerConfigs::default(),
    };

    LayerConfigs {
        l0: raw.l0.map(|l| LayerConfig {
            mode: l.mode,
            block_action: l.block_action,
            window_chars: l.window_chars,
        }),
        l3_inbound: raw.l3_inbound.map(|l| LayerConfig {
            mode: l.mode,
            block_action: l.block_action,
            window_chars: l.window_chars,
        }),
        l3_outbound: raw.l3_outbound.map(|l| LayerConfig {
            mode: l.mode,
            block_action: l.block_action,
            window_chars: l.window_chars,
        }),
        l5a: raw.l5a.map(|l| LayerConfig {
            mode: l.mode,
            block_action: l.block_action,
            window_chars: l.window_chars,
        }),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    fn with_project_root<F: FnOnce()>(value: &str, f: F) {
        let _guard = ENV_MUTEX.lock().unwrap();
        let previous = std::env::var("PROJECT_ROOT").ok();
        std::env::set_var("PROJECT_ROOT", value);
        f();
        match previous {
            Some(value) => std::env::set_var("PROJECT_ROOT", value),
            None => std::env::remove_var("PROJECT_ROOT"),
        }
    }

    /// The full example config from schema/examples/parapet.yaml, inlined
    /// so tests do not depend on filesystem layout.
    const EXAMPLE_YAML: &str = r#"parapet: v1

tools:
  _default:
    allowed: false
  read_file:
    allowed: true
    trust: untrusted
    constraints:
      path:
        type: string
        starts_with: "${PROJECT_ROOT}"
        not_contains: ["../", "..\\"]
    result_policy:
      max_length: 50000
  write_file:
    allowed: true
    trust: untrusted
    constraints:
      path:
        type: string
        starts_with: "${PROJECT_ROOT}"
        not_contains: ["../", "..\\"]
  internal_lookup:
    allowed: true
    trust: trusted
  exec_command:
    allowed: false

block_patterns:
  - "ignore previous instructions"
  - "ignore all previous"
  - "DAN mode enabled"
  - "ignore all safety guidelines"
  - "you are now [A-Z]+"
  - "jailbreak"

canary_tokens:
  - "{{CANARY_a8f3e9b1}}"

sensitive_patterns:
  - "CEREBRAS_API_KEY"
  - "GROQ_API_KEY"

untrusted_content_policy:
  max_length: 50000

trust:
  auto_untrusted_roles: ["tool"]
  unknown_trust_policy:
    system: trusted
    assistant: trusted
    user: untrusted
    tool: untrusted

engine:
  on_failure: open
  timeout_ms: 5000

environment: "internal"

layers:
  L0: { mode: sanitize }
  L3_inbound: { mode: block }
  L3_outbound: { mode: block, block_action: rewrite }
  L5a: { mode: redact }
"#;

    fn make_source(yaml: &str) -> StringSource {
        StringSource {
            content: yaml.to_string(),
        }
    }

    // ---------------------------------------------------------------
    // 1. Valid config parses into typed struct -- check key fields
    // ---------------------------------------------------------------

    #[test]
    fn valid_config_parses_all_key_fields() {
        // PROJECT_ROOT must be set for variable interpolation
        with_project_root("/home/user/project", || {
            let config = load_config(&make_source(EXAMPLE_YAML)).unwrap();

            assert_eq!(config.version, "v1");
            assert_eq!(config.tools.len(), 5);
            assert!(config.tools.contains_key("_default"));
            assert!(config.tools.contains_key("read_file"));
            assert!(config.tools.contains_key("write_file"));
            assert!(config.tools.contains_key("internal_lookup"));
            assert!(config.tools.contains_key("exec_command"));

            assert_eq!(config.block_patterns.len(), 6);
            assert_eq!(config.canary_tokens, vec!["{{CANARY_a8f3e9b1}}"]);
            assert_eq!(config.sensitive_patterns.len(), 2);
            assert_eq!(config.untrusted_content_policy.max_length, Some(50000));
            assert_eq!(config.trust.auto_untrusted_roles, vec!["tool"]);
            assert_eq!(config.engine.on_failure, FailureMode::Open);
            assert_eq!(config.engine.timeout_ms, Some(5000));
            assert_eq!(config.environment, "internal");

            // Layers
            let l0 = config.layers.l0.as_ref().unwrap();
            assert_eq!(l0.mode, "sanitize");
            let l3_out = config.layers.l3_outbound.as_ref().unwrap();
            assert_eq!(l3_out.mode, "block");
            assert_eq!(l3_out.block_action.as_deref(), Some("rewrite"));
        });
    }

    // ---------------------------------------------------------------
    // 2. Missing tools -> actionable error
    // ---------------------------------------------------------------

    #[test]
    fn missing_tools_produces_actionable_error() {
        let yaml = "parapet: v1\n";
        let err = load_config(&make_source(yaml)).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("tools"),
            "error should mention 'tools': {msg}"
        );
    }

    // ---------------------------------------------------------------
    // 3. Unknown predicate type rejected
    // ---------------------------------------------------------------

    #[test]
    fn unknown_argument_type_rejected() {
        let yaml = r#"
parapet: v1
tools:
  mytool:
    allowed: true
    constraints:
      arg1:
        type: banana
"#;
        let err = load_config(&make_source(yaml)).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("banana"),
            "error should mention the invalid type: {msg}"
        );
        assert!(
            msg.contains("unknown argument type"),
            "error should be descriptive: {msg}"
        );
    }

    // ---------------------------------------------------------------
    // 4. contract_hash is deterministic
    // ---------------------------------------------------------------

    #[test]
    fn contract_hash_is_deterministic() {
        with_project_root("/test", || {
            let config1 = load_config(&make_source(EXAMPLE_YAML)).unwrap();
            let config2 = load_config(&make_source(EXAMPLE_YAML)).unwrap();
            assert_eq!(config1.contract_hash, config2.contract_hash);
            assert!(config1.contract_hash.starts_with("sha256:"));
            assert_eq!(config1.contract_hash.len(), 7 + 64); // "sha256:" + 64 hex chars
        });
    }

    #[test]
    fn different_yaml_produces_different_hash() {
        let yaml_a = "parapet: v1\ntools:\n  _default:\n    allowed: false\n";
        let yaml_b = "parapet: v1\ntools:\n  _default:\n    allowed: true\n";

        let config_a = load_config(&make_source(yaml_a)).unwrap();
        let config_b = load_config(&make_source(yaml_b)).unwrap();
        assert_ne!(config_a.contract_hash, config_b.contract_hash);
    }

    // ---------------------------------------------------------------
    // 5. Variable interpolation: PROJECT_ROOT resolved from env
    // ---------------------------------------------------------------

    #[test]
    fn variable_interpolation_resolves_project_root() {
        with_project_root("/workspace/myapp", || {
            let yaml = r#"
  parapet: v1
  tools:
    read_file:
      allowed: true
      constraints:
        path:
          type: string
          starts_with: "${PROJECT_ROOT}"
  "#;
            let config = load_config(&make_source(yaml)).unwrap();
            let path_constraint = &config.tools["read_file"].constraints["path"];
            assert_eq!(
                path_constraint.starts_with.as_deref(),
                Some("/workspace/myapp")
            );
        });
    }

    // ---------------------------------------------------------------
    // 6. Undefined variable -> fail at load time with clear error
    // ---------------------------------------------------------------

    #[test]
    fn undefined_variable_fails_with_clear_error() {
        // Make sure the variable definitely does not exist
        std::env::remove_var("PARAPET_TEST_UNDEFINED_12345");

        let yaml = r#"
parapet: v1
tools:
  read_file:
    allowed: true
    constraints:
      path:
        starts_with: "${PARAPET_TEST_UNDEFINED_12345}"
"#;
        let err = load_config(&make_source(yaml)).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("PARAPET_TEST_UNDEFINED_12345"),
            "error should name the missing variable: {msg}"
        );
        assert!(
            msg.contains("undefined variable"),
            "error should say undefined: {msg}"
        );
    }

    // ---------------------------------------------------------------
    // 7. matches predicate with invalid regex -> fail at load time
    // ---------------------------------------------------------------

    #[test]
    fn invalid_regex_in_matches_fails_at_load_time() {
        let yaml = r#"
parapet: v1
tools:
  mytool:
    allowed: true
    constraints:
      arg:
        matches: "[invalid regex("
"#;
        let err = load_config(&make_source(yaml)).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("invalid regex"),
            "error should mention invalid regex: {msg}"
        );
    }

    // ---------------------------------------------------------------
    // 8. _default tool config parsed correctly
    // ---------------------------------------------------------------

    #[test]
    fn default_tool_config_parsed() {
        let yaml = r#"
parapet: v1
tools:
  _default:
    allowed: false
"#;
        let config = load_config(&make_source(yaml)).unwrap();
        let default_tool = &config.tools["_default"];
        assert!(!default_tool.allowed);
        assert!(default_tool.trust.is_none());
        assert!(default_tool.constraints.is_empty());
        assert!(default_tool.result_policy.is_none());
    }

    // ---------------------------------------------------------------
    // 9. Per-tool result_policy.max_length parsed
    // ---------------------------------------------------------------

    #[test]
    fn per_tool_result_policy_max_length_parsed() {
        with_project_root("/test", || {
            let yaml = r#"
  parapet: v1
  tools:
    read_file:
      allowed: true
      result_policy:
        max_length: 50000
  "#;
            let config = load_config(&make_source(yaml)).unwrap();
            let rp = config.tools["read_file"].result_policy.as_ref().unwrap();
            assert_eq!(rp.max_length, Some(50000));
        });
    }

    // ---------------------------------------------------------------
    // 10. block_patterns compiled as regex (actually match)
    // ---------------------------------------------------------------

    #[test]
    fn block_patterns_compiled_and_match() {
        let yaml = r#"
parapet: v1
tools:
  _default:
    allowed: false
block_patterns:
  - "ignore previous instructions"
  - "you are now [A-Z]+"
"#;
        let config = load_config(&make_source(yaml)).unwrap();
        assert_eq!(config.block_patterns.len(), 2);

        // Literal pattern matches
        assert!(config.block_patterns[0].is_match("please ignore previous instructions now"));

        // Regex pattern matches
        assert!(config.block_patterns[1].is_match("you are now DAN"));
        assert!(config.block_patterns[1].is_match("you are now JAILBREAK"));

        // Does not match lowercase (regex [A-Z]+)
        assert!(!config.block_patterns[1].is_match("you are now dan"));
    }

    // ---------------------------------------------------------------
    // Additional edge-case tests
    // ---------------------------------------------------------------

    #[test]
    fn invalid_block_pattern_regex_fails_at_load_time() {
        let yaml = r#"
parapet: v1
tools:
  _default:
    allowed: false
block_patterns:
  - "[unterminated"
"#;
        let err = load_config(&make_source(yaml)).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("invalid regex"),
            "error should mention invalid regex: {msg}"
        );
    }

    #[test]
    fn trust_level_in_tool_config_parsed() {
        let yaml = r#"
parapet: v1
tools:
  lookup:
    allowed: true
    trust: trusted
  search:
    allowed: true
    trust: untrusted
"#;
        let config = load_config(&make_source(yaml)).unwrap();
        assert_eq!(config.tools["lookup"].trust, Some(TrustLevel::Trusted));
        assert_eq!(config.tools["search"].trust, Some(TrustLevel::Untrusted));
    }

    #[test]
    fn invalid_trust_level_rejected() {
        let yaml = r#"
parapet: v1
tools:
  mytool:
    allowed: true
    trust: maybe
"#;
        // serde_yaml will reject "maybe" because the schema has enum ["trusted", "untrusted"]
        // but our raw type accepts any string, so we validate in build_tool_config
        let err = load_config(&make_source(yaml)).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("maybe"),
            "error should mention the invalid value: {msg}"
        );
    }

    #[test]
    fn unknown_engine_failure_mode_rejected() {
        let yaml = r#"
parapet: v1
tools:
  _default:
    allowed: true
engine:
  on_failure: explode
"#;
        let err = load_config(&make_source(yaml)).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("explode"),
            "error should mention invalid mode: {msg}"
        );
    }

    #[test]
    fn unknown_trust_policy_values_parsed() {
        let yaml = r#"
parapet: v1
tools:
  _default:
    allowed: false
trust:
  auto_untrusted_roles: ["tool"]
  unknown_trust_policy:
    system: trusted
    user: untrusted
"#;
        let config = load_config(&make_source(yaml)).unwrap();
        assert_eq!(
            config.trust.unknown_trust_policy["system"],
            TrustLevel::Trusted
        );
        assert_eq!(
            config.trust.unknown_trust_policy["user"],
            TrustLevel::Untrusted
        );
    }

    #[test]
    fn not_contains_constraint_parsed() {
        let yaml = r#"
parapet: v1
tools:
  write_file:
    allowed: true
    constraints:
      path:
        not_contains: ["../", "..\\"]
"#;
        let config = load_config(&make_source(yaml)).unwrap();
        let nc = config.tools["write_file"].constraints["path"]
            .not_contains
            .as_ref()
            .unwrap();
        assert_eq!(nc, &vec!["../".to_string(), "..\\".to_string()]);
    }

    #[test]
    fn config_source_string_source_works() {
        let source = StringSource {
            content: "parapet: v1\ntools:\n  _default:\n    allowed: false\n".to_string(),
        };
        let yaml = source.load().unwrap();
        assert!(yaml.contains("parapet: v1"));
    }

    #[test]
    fn unsupported_version_rejected() {
        let yaml = "parapet: v2\ntools:\n  _default:\n    allowed: false\n";
        let err = load_config(&make_source(yaml)).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("v2"),
            "error should mention invalid version: {msg}"
        );
    }

    #[test]
    fn sensitive_patterns_compiled_as_regex() {
        let yaml = r#"
parapet: v1
tools:
  _default:
    allowed: false
sensitive_patterns:
  - "API_KEY_[A-Za-z0-9]+"
"#;
        let config = load_config(&make_source(yaml)).unwrap();
        assert_eq!(config.sensitive_patterns.len(), 1);
        assert!(config.sensitive_patterns[0].is_match("found API_KEY_abc123 in output"));
        assert!(!config.sensitive_patterns[0].is_match("no key here"));
    }

    #[test]
    fn empty_optional_sections_use_defaults() {
        let yaml = "parapet: v1\ntools:\n  _default:\n    allowed: false\n";
        let config = load_config(&make_source(yaml)).unwrap();

        assert!(config.block_patterns.is_empty());
        assert!(config.canary_tokens.is_empty());
        assert!(config.sensitive_patterns.is_empty());
        assert_eq!(config.untrusted_content_policy.max_length, None);
        assert!(config.trust.auto_untrusted_roles.is_empty());
        assert!(config.trust.unknown_trust_policy.is_empty());
        assert_eq!(config.engine.on_failure, FailureMode::Open);
        assert_eq!(config.engine.timeout_ms, None);
        assert_eq!(config.environment, "");
        assert!(config.layers.l0.is_none());
        assert!(config.layers.l3_inbound.is_none());
        assert!(config.layers.l3_outbound.is_none());
        assert!(config.layers.l5a.is_none());
    }

    #[test]
    fn matches_constraint_with_valid_regex_compiles() {
        let yaml = r#"
parapet: v1
tools:
  mytool:
    allowed: true
    constraints:
      arg:
        matches: "^[a-z]+$"
"#;
        let config = load_config(&make_source(yaml)).unwrap();
        let m = config.tools["mytool"].constraints["arg"]
            .matches
            .as_ref()
            .unwrap();
        assert!(m.is_match("hello"));
        assert!(!m.is_match("Hello123"));
    }

    #[test]
    fn one_of_constraint_parsed() {
        let yaml = r#"
parapet: v1
tools:
  mytool:
    allowed: true
    constraints:
      format:
        one_of: ["json", "yaml", "toml"]
"#;
        let config = load_config(&make_source(yaml)).unwrap();
        let one_of = config.tools["mytool"].constraints["format"]
            .one_of
            .as_ref()
            .unwrap();
        assert_eq!(one_of, &vec!["json", "yaml", "toml"]);
    }

    #[test]
    fn numeric_constraints_parsed() {
        let yaml = r#"
parapet: v1
tools:
  mytool:
    allowed: true
    constraints:
      count:
        type: number
        min: 1.0
        max: 100.0
"#;
        let config = load_config(&make_source(yaml)).unwrap();
        let c = &config.tools["mytool"].constraints["count"];
        assert_eq!(c.type_check.as_deref(), Some("number"));
        assert_eq!(c.min, Some(1.0));
        assert_eq!(c.max, Some(100.0));
    }

    #[test]
    fn url_host_constraint_parsed() {
        let yaml = r#"
parapet: v1
tools:
  fetch:
    allowed: true
    constraints:
      url:
        url_host: ["example.com", "api.example.com"]
"#;
        let config = load_config(&make_source(yaml)).unwrap();
        let hosts = config.tools["fetch"].constraints["url"]
            .url_host
            .as_ref()
            .unwrap();
        assert_eq!(hosts, &vec!["example.com", "api.example.com"]);
    }

    #[test]
    fn max_length_constraint_parsed() {
        let yaml = r#"
parapet: v1
tools:
  mytool:
    allowed: true
    constraints:
      text:
        max_length: 1024
"#;
        let config = load_config(&make_source(yaml)).unwrap();
        assert_eq!(
            config.tools["mytool"].constraints["text"].max_length,
            Some(1024)
        );
    }

    #[test]
    fn multiple_variables_in_one_string() {
        std::env::set_var("PARAPET_TEST_A", "hello");
        std::env::set_var("PARAPET_TEST_B", "world");

        let result = resolve_variables("${PARAPET_TEST_A}/${PARAPET_TEST_B}").unwrap();
        assert_eq!(result, "hello/world");
    }

    #[test]
    fn string_without_variables_unchanged() {
        let result = resolve_variables("no variables here").unwrap();
        assert_eq!(result, "no variables here");
    }

    #[test]
    fn layer_l5a_parsed() {
        let yaml = r#"
parapet: v1
tools:
  _default:
    allowed: false
layers:
  L5a: { mode: redact, window_chars: 2048 }
"#;
        let config = load_config(&make_source(yaml)).unwrap();
        let l5a = config.layers.l5a.as_ref().unwrap();
        assert_eq!(l5a.mode, "redact");
        assert!(l5a.block_action.is_none());
        assert_eq!(l5a.window_chars, Some(2048));
    }
}
