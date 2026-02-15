// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// Config loader and validator -- defined in M1.2
//
// Loads parapet.yaml, validates structure, resolves variable interpolation,
// compiles regex patterns, and computes a deterministic contract hash.

mod defaults;
mod error;
mod interpolation;
mod loader;
mod pattern;
pub(crate) mod raw;
mod source;
mod types;

pub use defaults::{
    default_block_patterns, default_evidence_patterns, default_l4_patterns, default_layer_configs,
    default_sensitive_patterns,
};
pub use error::ConfigError;
pub use loader::{compute_hash, load_config};
pub use pattern::{CompiledPattern, PatternAction};
pub use source::{ConfigSource, FileSource, StringSource};
pub use types::{
    ArgumentConstraints, Config, ContentPolicy, EngineConfig, FailureMode, L1Config, L1Mode,
    L4Config, L4Mode, L4PatternCategory, LayerConfig, LayerConfigs, PolicyConfig, RuntimeConfig,
    ToolConfig, TrustConfig,
};

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

            assert_eq!(config.policy.version, "v1");
            assert_eq!(config.policy.tools.len(), 5);
            assert!(config.policy.tools.contains_key("_default"));
            assert!(config.policy.tools.contains_key("read_file"));
            assert!(config.policy.tools.contains_key("write_file"));
            assert!(config.policy.tools.contains_key("internal_lookup"));
            assert!(config.policy.tools.contains_key("exec_command"));

            // Default block patterns + evidence patterns + 6 user patterns
            let default_count = super::defaults::default_block_patterns().len();
            let evidence_count = super::defaults::default_evidence_patterns().len();
            assert_eq!(config.policy.block_patterns.len(), default_count + 6 + evidence_count);
            assert_eq!(config.policy.canary_tokens, vec!["{{CANARY_a8f3e9b1}}"]);
            let default_sensitive_count = super::defaults::default_sensitive_patterns().len();
            assert_eq!(config.policy.sensitive_patterns.len(), default_sensitive_count + 2);
            assert_eq!(config.policy.untrusted_content_policy.max_length, Some(50000));
            assert_eq!(config.policy.trust.auto_untrusted_roles, vec!["tool"]);
            assert_eq!(config.runtime.engine.on_failure, FailureMode::Open);
            assert_eq!(config.runtime.engine.timeout_ms, Some(5000));
            assert_eq!(config.runtime.environment, "internal");

            // Layers
            let l0 = config.policy.layers.l0.as_ref().unwrap();
            assert_eq!(l0.mode, "sanitize");
            let l3_out = config.policy.layers.l3_outbound.as_ref().unwrap();
            assert_eq!(l3_out.mode, "block");
            assert_eq!(l3_out.block_action.as_deref(), Some("rewrite"));
        });
    }

    // ---------------------------------------------------------------
    // 2. Minimal config is valid (parapet: v1 is all you need)
    // ---------------------------------------------------------------

    #[test]
    fn minimal_config_is_valid() {
        let yaml = "parapet: v1\n";
        let config = load_config(&make_source(yaml)).unwrap();

        // Tools: empty map, allow-all behavior
        assert!(config.policy.tools.is_empty());

        // Block patterns: defaults + evidence active
        let default_count = super::defaults::default_block_patterns().len();
        let evidence_count = super::defaults::default_evidence_patterns().len();
        assert!(default_count >= 60);
        assert_eq!(config.policy.block_patterns.len(), default_count + evidence_count);

        // Sensitive patterns: defaults active
        let default_sensitive_count = super::defaults::default_sensitive_patterns().len();
        assert!(default_sensitive_count >= 10);
        assert_eq!(config.policy.sensitive_patterns.len(), default_sensitive_count);

        // All 5 layers active
        let l0 = config.policy.layers.l0.as_ref().unwrap();
        assert_eq!(l0.mode, "sanitize");
        let l3i = config.policy.layers.l3_inbound.as_ref().unwrap();
        assert_eq!(l3i.mode, "block");
        let l3o = config.policy.layers.l3_outbound.as_ref().unwrap();
        assert_eq!(l3o.mode, "block");
        let l5a = config.policy.layers.l5a.as_ref().unwrap();
        assert_eq!(l5a.mode, "redact");
        let l4 = config.policy.layers.l4.as_ref().unwrap();
        assert_eq!(l4.mode, L4Mode::Shadow);
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
            let path_constraint = &config.policy.tools["read_file"].constraints["path"];
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
        let default_tool = &config.policy.tools["_default"];
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
            let rp = config.policy.tools["read_file"].result_policy.as_ref().unwrap();
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
        let default_count = super::defaults::default_block_patterns().len();
        let evidence_count = super::defaults::default_evidence_patterns().len();
        assert_eq!(config.policy.block_patterns.len(), default_count + 2 + evidence_count);

        // User patterns come after defaults, before evidence
        let user_start = default_count;

        // Literal pattern matches
        assert!(config.policy.block_patterns[user_start].is_match("please ignore previous instructions now"));

        // Regex pattern matches
        assert!(config.policy.block_patterns[user_start + 1].is_match("you are now DAN"));
        assert!(config.policy.block_patterns[user_start + 1].is_match("you are now JAILBREAK"));

        // Does not match lowercase (regex [A-Z]+)
        assert!(!config.policy.block_patterns[user_start + 1].is_match("you are now dan"));
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
        assert_eq!(config.policy.tools["lookup"].trust, Some(crate::message::TrustLevel::Trusted));
        assert_eq!(config.policy.tools["search"].trust, Some(crate::message::TrustLevel::Untrusted));
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
            config.policy.trust.unknown_trust_policy["system"],
            crate::message::TrustLevel::Trusted
        );
        assert_eq!(
            config.policy.trust.unknown_trust_policy["user"],
            crate::message::TrustLevel::Untrusted
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
        let nc = config.policy.tools["write_file"].constraints["path"]
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
        let default_sensitive_count = super::defaults::default_sensitive_patterns().len();
        assert_eq!(config.policy.sensitive_patterns.len(), default_sensitive_count + 1);
        // User pattern is last
        let user_pattern = config.policy.sensitive_patterns.last().unwrap();
        assert!(user_pattern.is_match("found API_KEY_abc123 in output"));
        assert!(!user_pattern.is_match("no key here"));
    }

    #[test]
    fn empty_optional_sections_use_defaults() {
        let yaml = "parapet: v1\ntools:\n  _default:\n    allowed: false\n";
        let config = load_config(&make_source(yaml)).unwrap();

        // Default block + evidence patterns are active even with no user patterns
        let default_count = super::defaults::default_block_patterns().len();
        let evidence_count = super::defaults::default_evidence_patterns().len();
        assert_eq!(config.policy.block_patterns.len(), default_count + evidence_count);
        assert!(config.policy.canary_tokens.is_empty());
        // Default sensitive patterns are active
        let default_sensitive_count = super::defaults::default_sensitive_patterns().len();
        assert_eq!(config.policy.sensitive_patterns.len(), default_sensitive_count);
        assert_eq!(config.policy.untrusted_content_policy.max_length, None);
        assert!(config.policy.trust.auto_untrusted_roles.is_empty());
        assert!(config.policy.trust.unknown_trust_policy.is_empty());
        assert_eq!(config.runtime.engine.on_failure, FailureMode::Open);
        assert_eq!(config.runtime.engine.timeout_ms, None);
        assert_eq!(config.runtime.environment, "");
        // All layers active by default
        assert_eq!(config.policy.layers.l0.as_ref().unwrap().mode, "sanitize");
        assert_eq!(config.policy.layers.l3_inbound.as_ref().unwrap().mode, "block");
        assert_eq!(config.policy.layers.l3_outbound.as_ref().unwrap().mode, "block");
        assert_eq!(config.policy.layers.l5a.as_ref().unwrap().mode, "redact");
        assert_eq!(config.policy.layers.l4.as_ref().unwrap().mode, L4Mode::Shadow);
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
        let m = config.policy.tools["mytool"].constraints["arg"]
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
        let one_of = config.policy.tools["mytool"].constraints["format"]
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
        let c = &config.policy.tools["mytool"].constraints["count"];
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
        let hosts = config.policy.tools["fetch"].constraints["url"]
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
            config.policy.tools["mytool"].constraints["text"].max_length,
            Some(1024)
        );
    }

    #[test]
    fn multiple_variables_in_one_string() {
        std::env::set_var("PARAPET_TEST_A", "hello");
        std::env::set_var("PARAPET_TEST_B", "world");

        let result = interpolation::resolve_variables("${PARAPET_TEST_A}/${PARAPET_TEST_B}").unwrap();
        assert_eq!(result, "hello/world");
    }

    #[test]
    fn string_without_variables_unchanged() {
        let result = interpolation::resolve_variables("no variables here").unwrap();
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
        let l5a = config.policy.layers.l5a.as_ref().unwrap();
        assert_eq!(l5a.mode, "redact");
        assert!(l5a.block_action.is_none());
        assert_eq!(l5a.window_chars, Some(2048));
    }

    // ---------------------------------------------------------------
    // Default block patterns tests
    // ---------------------------------------------------------------

    #[test]
    fn use_default_sensitive_patterns_false_disables_defaults() {
        let yaml = r#"
parapet: v1
tools:
  _default:
    allowed: false
use_default_sensitive_patterns: false
sensitive_patterns:
  - "MY_CUSTOM_SECRET_[0-9]+"
"#;
        let config = load_config(&make_source(yaml)).unwrap();
        assert_eq!(config.policy.sensitive_patterns.len(), 1);
        assert_eq!(config.policy.sensitive_patterns[0].pattern, "MY_CUSTOM_SECRET_[0-9]+");
    }

    #[test]
    fn use_default_sensitive_patterns_false_no_user_patterns_gives_empty() {
        let yaml = r#"
parapet: v1
tools:
  _default:
    allowed: false
use_default_sensitive_patterns: false
"#;
        let config = load_config(&make_source(yaml)).unwrap();
        assert!(config.policy.sensitive_patterns.is_empty());
    }

    #[test]
    fn explicit_layers_override_defaults() {
        let yaml = r#"
parapet: v1
tools:
  _default:
    allowed: false
layers:
  L0: { mode: sanitize }
  L5a: { mode: redact, window_chars: 4096 }
"#;
        let config = load_config(&make_source(yaml)).unwrap();
        // Explicitly specified layers are present
        assert_eq!(config.policy.layers.l0.as_ref().unwrap().mode, "sanitize");
        assert_eq!(config.policy.layers.l5a.as_ref().unwrap().mode, "redact");
        assert_eq!(config.policy.layers.l5a.as_ref().unwrap().window_chars, Some(4096));
        // Unspecified layers are None (user provided layers block, so only their entries apply)
        assert!(config.policy.layers.l3_inbound.is_none());
        assert!(config.policy.layers.l3_outbound.is_none());
    }

    #[test]
    fn use_default_block_patterns_false_skips_defaults() {
        let yaml = r#"
parapet: v1
tools:
  _default:
    allowed: false
use_default_block_patterns: false
block_patterns:
  - "custom pattern only"
"#;
        let config = load_config(&make_source(yaml)).unwrap();
        assert_eq!(config.policy.block_patterns.len(), 1);
        assert_eq!(config.policy.block_patterns[0].pattern, "custom pattern only");
    }

    #[test]
    fn use_default_block_patterns_false_no_user_patterns_gives_empty() {
        let yaml = r#"
parapet: v1
tools:
  _default:
    allowed: false
use_default_block_patterns: false
"#;
        let config = load_config(&make_source(yaml)).unwrap();
        assert!(config.policy.block_patterns.is_empty());
    }

    #[test]
    fn default_patterns_prepended_before_user_patterns() {
        let yaml = r#"
parapet: v1
tools:
  _default:
    allowed: false
block_patterns:
  - "my custom attack"
"#;
        let config = load_config(&make_source(yaml)).unwrap();
        let default_count = super::defaults::default_block_patterns().len();
        let evidence_count = super::defaults::default_evidence_patterns().len();
        assert_eq!(config.policy.block_patterns.len(), default_count + 1 + evidence_count);
        // User pattern comes after defaults, before evidence
        assert_eq!(
            config.policy.block_patterns[default_count].pattern,
            "my custom attack"
        );
    }
}
