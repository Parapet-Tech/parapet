// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use sha2::{Digest, Sha256};

use crate::message::TrustLevel;

use super::defaults::{
    default_block_patterns, default_evidence_patterns, default_l4_patterns,
    default_layer_configs, default_sensitive_patterns,
};
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
        .map(|p| match p {
            raw::RawBlockPattern::Simple(s) => CompiledPattern::compile(s),
            raw::RawBlockPattern::Full { pattern, category, weight, atomic } => {
                CompiledPattern::compile_full(
                    pattern,
                    super::PatternAction::Block,
                    None,
                    category.clone(),
                    *weight,
                    *atomic,
                )
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    block_patterns.extend(user_patterns);

    // Append evidence patterns (action: Evidence, trust-gated to untrusted).
    // These are scanned alongside block patterns but never trigger block verdicts.
    // Gated by the same flag as block patterns — if defaults are disabled, evidence is too.
    if raw.use_default_block_patterns != Some(false) {
        block_patterns.extend(default_evidence_patterns());
    }

    // Compile sensitive_patterns: defaults first, then user patterns
    let mut sensitive_patterns = if raw.use_default_sensitive_patterns != Some(false) {
        default_sensitive_patterns()
    } else {
        Vec::new()
    };
    let user_sensitive = raw
        .sensitive_patterns
        .iter()
        .map(|p| CompiledPattern::compile(p))
        .collect::<Result<Vec<_>, _>>()?;
    sensitive_patterns.extend(user_sensitive);

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
    let layers = build_layer_configs(raw.layers)?;

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

fn build_layer_configs(raw: Option<raw::RawLayerConfigs>) -> Result<LayerConfigs, ConfigError> {
    let raw = match raw {
        Some(r) => r,
        None => return Ok(default_layer_configs()),
    };

    let l1 = raw
        .l1
        .map(|l1| {
            let mode = match l1.mode.as_str() {
                "shadow" => L1Mode::Shadow,
                "block" => L1Mode::Block,
                other => {
                    return Err(ConfigError::Validation(format!(
                        "unknown L1 mode \"{other}\", expected \"shadow\" or \"block\""
                    )))
                }
            };
            Ok(L1Config {
                mode,
                threshold: l1.threshold,
            })
        })
        .transpose()?;

    let l2a = raw
        .l2a
        .map(|l2a| {
            let mode = match l2a.mode.as_str() {
                "shadow" => L2aMode::Shadow,
                "block" => L2aMode::Block,
                other => {
                    return Err(ConfigError::Validation(format!(
                        "unknown L2a mode \"{other}\", expected \"shadow\" or \"block\""
                    )))
                }
            };

            // Validate model name against known models.
            if !crate::layers::l2a_model::KNOWN_MODELS.contains(&l2a.model.as_str()) {
                return Err(ConfigError::Validation(format!(
                    "unknown L2a model \"{}\", expected one of: {}",
                    l2a.model,
                    crate::layers::l2a_model::KNOWN_MODELS.join(", "),
                )));
            }

            // Validate all numeric fields with actionable error messages.
            fn validate_unit(name: &str, v: f32) -> Result<(), ConfigError> {
                if !(0.0..=1.0).contains(&v) {
                    return Err(ConfigError::Validation(format!(
                        "L2a {name} must be in [0.0, 1.0], got {v}"
                    )));
                }
                Ok(())
            }
            validate_unit("pg_threshold", l2a.pg_threshold)?;
            validate_unit("block_threshold", l2a.block_threshold)?;
            validate_unit("heuristic_weight", l2a.heuristic_weight)?;
            validate_unit("fusion_confidence_agreement", l2a.fusion_confidence_agreement)?;
            validate_unit("fusion_confidence_pg_only", l2a.fusion_confidence_pg_only)?;
            validate_unit("fusion_confidence_heuristic_only", l2a.fusion_confidence_heuristic_only)?;

            if l2a.max_segments < 1 {
                return Err(ConfigError::Validation(
                    "L2a max_segments must be >= 1".into(),
                ));
            }
            if l2a.timeout_ms == 0 {
                return Err(ConfigError::Validation(
                    "L2a timeout_ms must be > 0".into(),
                ));
            }
            if l2a.max_concurrent_scans < 1 {
                return Err(ConfigError::Validation(
                    "L2a max_concurrent_scans must be >= 1".into(),
                ));
            }

            Ok(L2aConfig {
                mode,
                model: l2a.model,
                model_dir: l2a.model_dir,
                pg_threshold: l2a.pg_threshold,
                block_threshold: l2a.block_threshold,
                heuristic_weight: l2a.heuristic_weight,
                fusion_confidence_agreement: l2a.fusion_confidence_agreement,
                fusion_confidence_pg_only: l2a.fusion_confidence_pg_only,
                fusion_confidence_heuristic_only: l2a.fusion_confidence_heuristic_only,
                max_segments: l2a.max_segments,
                timeout_ms: l2a.timeout_ms,
                max_concurrent_scans: l2a.max_concurrent_scans,
            })
        })
        .transpose()?;

    let l4 = raw
        .l4
        .map(|l4| {
            let mode = match l4.mode.as_str() {
                "shadow" => L4Mode::Shadow,
                "block" => L4Mode::Block,
                other => {
                    return Err(ConfigError::Validation(format!(
                        "unknown L4 mode \"{other}\", expected \"shadow\" or \"block\""
                    )))
                }
            };
            let mut patterns = if l4.use_default_l4_patterns != Some(false) {
                default_l4_patterns()
            } else {
                Vec::new()
            };
            for raw_cat in l4.cross_turn_patterns {
                let compiled = raw_cat
                    .patterns
                    .iter()
                    .map(|p| CompiledPattern::compile(p))
                    .collect::<Result<Vec<_>, _>>()?;
                patterns.push(L4PatternCategory {
                    category: raw_cat.category,
                    weight: raw_cat.weight,
                    patterns: compiled,
                });
            }
            Ok(L4Config {
                mode,
                risk_threshold: l4.risk_threshold,
                escalation_bonus: l4.escalation_bonus,
                resampling_bonus: l4.resampling_bonus,
                persistence_factor: l4.persistence_factor,
                diversity_factor: l4.diversity_factor,
                min_user_turns: l4.min_user_turns,
                cross_turn_patterns: patterns,
            })
        })
        .transpose()?;

    Ok(LayerConfigs {
        l0: raw.l0.map(|l| LayerConfig {
            mode: l.mode,
            block_action: l.block_action,
            window_chars: l.window_chars,
        }),
        l1,
        l2a,
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
        l4,
    })
}
