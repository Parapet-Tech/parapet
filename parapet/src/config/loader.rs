use std::collections::HashMap;

use sha2::{Digest, Sha256};

use crate::message::TrustLevel;

use super::defaults::default_block_patterns;
use super::error::ConfigError;
use super::interpolation::resolve_variables;
use super::pattern::CompiledPattern;
use super::raw;
use super::source::ConfigSource;
use super::types::*;

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

    // Compile block_patterns: defaults first, then user patterns
    let mut block_patterns = if raw.use_default_block_patterns != Some(false) {
        default_block_patterns()
    } else {
        Vec::new()
    };

    let user_patterns = raw
        .block_patterns
        .iter()
        .map(|p| CompiledPattern::compile(p))
        .collect::<Result<Vec<_>, _>>()?;
    block_patterns.extend(user_patterns);

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
        policy: PolicyConfig {
            version: raw.parapet,
            tools,
            block_patterns,
            canary_tokens: raw.canary_tokens,
            sensitive_patterns,
            untrusted_content_policy,
            trust,
            layers,
        },
        runtime: RuntimeConfig {
            engine,
            environment: raw.environment.unwrap_or_default(),
        },
        contract_hash,
    })
}

pub fn compute_hash(raw_yaml: &str) -> String {
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
