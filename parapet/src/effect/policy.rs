// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Resolved policy, build identity and the production complete-collection
//! decision function (Routing contract 50747a52, section 3 step 4; draft 12
//! sections 5.5, 9 and 9.1).

use super::interfaces::{
    BuildIdentityProviderV0, PolicyResolveErrorV0, ResolvedPolicyProviderV0, ResolvedPolicyV0,
};
use crate::config::Config;
use crate::normalize::normalize_for_comparison;
use crate::receipt::{
    canonical_policy_bytes, code_digest, policy_digest, BlockAction, CodeError, CodeIdentity,
    ConstraintName, DecisionFunction, DecisionOutcome, DecisionOutput, DecisionReason,
    FailedConstraint, PolicyBytesError, PolicySchema, PresentUnavailable, ResolvedActionPolicyV0,
    ResolvedConstraints, ResolvedL3Outbound, ResolvedTool, TypeCheck,
};
use regex::Regex;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

/// Section 4.2 `ResolvedPolicyProviderV0` over the live resolved `Config`.
pub struct ConfigPolicyProviderV0 {
    config: Arc<Config>,
}

impl ConfigPolicyProviderV0 {
    /// Read-only view of the live resolved configuration.
    pub fn new(config: Arc<Config>) -> Self {
        Self { config }
    }
}

/// Draft 12 section 9.1: the exact `ResolvedActionPolicyV0` of a `Config`.
pub fn resolve_action_policy(
    config: &Config,
) -> Result<ResolvedActionPolicyV0, PolicyResolveErrorV0> {
    let mut tools = BTreeMap::new();
    for (name, tool) in &config.policy.tools {
        let mut constraints = BTreeMap::new();
        for (argument, c) in &tool.constraints {
            let type_check = match c.type_check.as_deref() {
                None => None,
                Some("string") => Some(TypeCheck::String),
                Some("number") => Some(TypeCheck::Number),
                Some("boolean") => Some(TypeCheck::Boolean),
                Some(_) => return Err(PolicyResolveErrorV0::TypeCheckUnrepresentable),
            };
            constraints.insert(
                argument.clone(),
                ResolvedConstraints {
                    type_check,
                    starts_with: c.starts_with.clone(),
                    not_contains: c.not_contains.clone(),
                    matches: c.matches.as_ref().map(|p| p.pattern.clone()),
                    one_of: c.one_of.clone(),
                    max_length: c.max_length.map(|n| n as u64),
                    min: c.min,
                    max: c.max,
                    url_host: c.url_host.clone(),
                },
            );
        }
        tools.insert(
            name.clone(),
            ResolvedTool {
                allowed: tool.allowed,
                constraints,
            },
        );
    }
    let layer = config.policy.layers.l3_outbound.as_ref();
    let configured = layer.and_then(|l| l.block_action.clone());
    Ok(ResolvedActionPolicyV0 {
        schema_version: PolicySchema::V0,
        policy_version: config.policy.version.clone(),
        l3_outbound: ResolvedL3Outbound {
            present: layer.is_some(),
            mode: layer.map(|l| l.mode.clone()),
            block_action_effective: if configured.as_deref() == Some("error") {
                BlockAction::Error
            } else {
                BlockAction::Rewrite
            },
            block_action_configured: configured,
        },
        tools,
    })
}

impl ResolvedPolicyProviderV0 for ConfigPolicyProviderV0 {
    fn resolve(&self) -> Result<ResolvedPolicyV0, PolicyResolveErrorV0> {
        let policy = resolve_action_policy(&self.config)?;
        let bytes = canonical_policy_bytes(&policy).map_err(|e| match e {
            PolicyBytesError::NonFinite => PolicyResolveErrorV0::NonFiniteBound,
            PolicyBytesError::Serialization(_) => PolicyResolveErrorV0::Serialization,
        })?;
        Ok(ResolvedPolicyV0 {
            digest: policy_digest(&bytes),
            policy,
            bytes,
            contract_hash: self.config.contract_hash.clone(),
        })
    }
}

/// Whether the resolved policy enforces outbound tool calls (draft 12 section 9 field 8).
pub fn policy_enforced(policy: &ResolvedActionPolicyV0) -> bool {
    policy.l3_outbound.present && policy.l3_outbound.mode.as_deref() == Some("block")
}

/// Section 4.2 `BuildIdentityProviderV0`: injected values only.
#[derive(Debug, Clone)]
pub struct InjectedBuildIdentityV0 {
    identity: PresentUnavailable<CodeIdentity, CodeError>,
}

impl InjectedBuildIdentityV0 {
    /// Crate version plus the tagged digest of the supplied executable bytes.
    pub fn from_executable_bytes(code_version: &str, executable: &[u8]) -> Self {
        Self {
            identity: PresentUnavailable::Present(CodeIdentity {
                code_version: code_version.to_owned(),
                code_digest: code_digest(executable),
            }),
        }
    }

    /// The exact draft-12 unavailable shape; there is no ambient fallback.
    pub fn unavailable() -> Self {
        Self {
            identity: PresentUnavailable::Unavailable(CodeError::NoBuildDigest),
        }
    }

    /// Read the given executable path; any read failure is `no_build_digest`.
    pub fn from_executable_path(code_version: &str, path: &std::path::Path) -> Self {
        match std::fs::read(path) {
            Ok(bytes) => Self::from_executable_bytes(code_version, &bytes),
            Err(_) => Self::unavailable(),
        }
    }
}

impl BuildIdentityProviderV0 for InjectedBuildIdentityV0 {
    fn identity(&self) -> PresentUnavailable<CodeIdentity, CodeError> {
        self.identity.clone()
    }
}

/// Section 4.2 `CompleteConstraintDecisionV0`: the production `DecisionFunction`.
///
/// Predicate semantics equal the landed DSL evaluator (normalized strings,
/// raw-length `max_length`, numeric `min`/`max`, exact `url_host`) but every
/// configured argument and predicate is evaluated without short-circuit.
#[derive(Default)]
pub struct CompleteConstraintDecisionV0 {
    patterns: Mutex<BTreeMap<String, Option<Regex>>>,
}

impl CompleteConstraintDecisionV0 {
    /// A decision function with an empty pattern cache.
    pub fn new() -> Self {
        Self::default()
    }

    fn pattern_matches(&self, pattern: &str, value: &str) -> bool {
        let mut cache = self.patterns.lock().expect("pattern cache");
        let compiled = cache.entry(pattern.to_owned()).or_insert_with(|| {
            crate::config::CompiledPattern::compile(pattern)
                .ok()
                .map(|p| p.regex)
        });
        // A source pattern that does not compile cannot be satisfied.
        compiled.as_ref().is_some_and(|r| r.is_match(value))
    }

    fn predicates(
        &self,
        value: &serde_json::Value,
        c: &ResolvedConstraints,
    ) -> Vec<ConstraintName> {
        let mut failed = Vec::new();
        if let Some(expected) = c.type_check {
            let actual = match value {
                serde_json::Value::String(_) => Some(TypeCheck::String),
                serde_json::Value::Number(_) => Some(TypeCheck::Number),
                serde_json::Value::Bool(_) => Some(TypeCheck::Boolean),
                _ => None,
            };
            if actual != Some(expected) {
                failed.push(ConstraintName::TypeCheck);
            }
        }
        let raw = value.as_str();
        let normalized = raw.map(normalize_for_comparison);
        let normalized = normalized.as_deref();
        if let Some(prefix) = &c.starts_with {
            if !normalized.is_some_and(|s| s.starts_with(prefix.as_str())) {
                failed.push(ConstraintName::StartsWith);
            }
        }
        if let Some(forbidden) = &c.not_contains {
            if !normalized.is_some_and(|s| !forbidden.iter().any(|f| s.contains(f.as_str()))) {
                failed.push(ConstraintName::NotContains);
            }
        }
        if let Some(pattern) = &c.matches {
            if !normalized.is_some_and(|s| self.pattern_matches(pattern, s)) {
                failed.push(ConstraintName::Matches);
            }
        }
        if let Some(allowed) = &c.one_of {
            if !normalized.is_some_and(|s| allowed.iter().any(|a| a == s)) {
                failed.push(ConstraintName::OneOf);
            }
        }
        if let Some(max) = c.max_length {
            if !raw.is_some_and(|s| s.chars().count() as u64 <= max) {
                failed.push(ConstraintName::MaxLength);
            }
        }
        if let Some(min) = c.min {
            if !value.as_f64().is_some_and(|n| n >= min) {
                failed.push(ConstraintName::Min);
            }
        }
        if let Some(max) = c.max {
            if !value.as_f64().is_some_and(|n| n <= max) {
                failed.push(ConstraintName::Max);
            }
        }
        if let Some(hosts) = &c.url_host {
            let host_ok = normalized
                .and_then(|s| url::Url::parse(s).ok())
                .and_then(|u| u.host_str().map(|h| hosts.iter().any(|a| a == h)))
                .unwrap_or(false);
            if !host_ok {
                failed.push(ConstraintName::UrlHost);
            }
        }
        failed
    }
}

fn constraint_str(name: ConstraintName) -> String {
    serde_json::to_value(name)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

/// Draft 12 section 9 field 12 order: null first, then argument bytes, then constraint.
pub fn order_failed_constraints(mut failures: Vec<FailedConstraint>) -> Vec<FailedConstraint> {
    failures.sort_by(|a, b| {
        let key = |f: &FailedConstraint| {
            (
                f.argument.clone().map(String::into_bytes),
                constraint_str(f.constraint),
            )
        };
        key(a).cmp(&key(b))
    });
    failures.dedup();
    failures
}

/// Draft 12 section 9 field 11: reason derived from the failure set.
pub fn derived_reason(failures: &[FailedConstraint]) -> DecisionReason {
    match failures {
        [] => DecisionReason::Allowed,
        [only] => match only.constraint {
            ConstraintName::ToolAllowed => DecisionReason::ToolNotAllowed,
            ConstraintName::ArgumentPresent => DecisionReason::ArgumentMissing,
            ConstraintName::TypeCheck => DecisionReason::TypeCheck,
            ConstraintName::StartsWith => DecisionReason::StartsWith,
            ConstraintName::NotContains => DecisionReason::NotContains,
            ConstraintName::Matches => DecisionReason::Matches,
            ConstraintName::OneOf => DecisionReason::OneOf,
            ConstraintName::MaxLength => DecisionReason::MaxLength,
            ConstraintName::Min => DecisionReason::Min,
            ConstraintName::Max => DecisionReason::Max,
            ConstraintName::UrlHost => DecisionReason::UrlHost,
        },
        _ => DecisionReason::Multiple,
    }
}

impl DecisionFunction for CompleteConstraintDecisionV0 {
    fn decide(
        &self,
        policy: &ResolvedActionPolicyV0,
        tool_name: &str,
        arguments: &serde_json::Value,
    ) -> DecisionOutput {
        let tool = policy
            .tools
            .get(tool_name)
            .or_else(|| policy.tools.get("_default"));
        let mut failures = Vec::new();
        match tool {
            None => {}
            Some(tool) if !tool.allowed => failures.push(FailedConstraint {
                argument: None,
                constraint: ConstraintName::ToolAllowed,
            }),
            Some(tool) => {
                for (argument, constraints) in &tool.constraints {
                    match arguments.get(argument) {
                        None | Some(serde_json::Value::Null) => failures.push(FailedConstraint {
                            argument: Some(argument.clone()),
                            constraint: ConstraintName::ArgumentPresent,
                        }),
                        Some(value) => {
                            failures.extend(self.predicates(value, constraints).into_iter().map(
                                |constraint| FailedConstraint {
                                    argument: Some(argument.clone()),
                                    constraint,
                                },
                            ))
                        }
                    }
                }
            }
        }
        let failed_constraints = order_failed_constraints(failures);
        DecisionOutput {
            decision: if failed_constraints.is_empty() {
                DecisionOutcome::Allow
            } else {
                DecisionOutcome::Block
            },
            reason_code: derived_reason(&failed_constraints),
            failed_constraints,
        }
    }
}
