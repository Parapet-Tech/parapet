// Role-based trust assignment — defined in M1.7
//
// Assigns `TrustLevel` to each `Message` based on:
// 1. `unknown_trust_policy`: role -> trust level mapping
// 2. `auto_untrusted_roles`: override matching roles to Untrusted
// 3. Per-tool `trust` config: override for specific tool messages
// 4. Safe default: Untrusted if no policy matches

use crate::config::Config;
use crate::message::{Message, Role, TrustLevel};

/// Maps a `Role` enum variant to its lowercase config string.
fn role_to_str(role: &Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

/// Trait for assigning trust levels to messages.
///
/// Implementations inspect each message's role, tool name, and the
/// config's trust rules to determine the appropriate `TrustLevel`.
pub trait TrustAssigner: Send + Sync {
    /// Assign trust levels to all messages in the slice, mutating each
    /// message's `trust` field in place.
    fn assign_trust(&self, messages: &mut [Message], config: &Config);
}

/// Default implementation: role-based trust with per-tool overrides.
pub struct RoleTrustAssigner;

impl TrustAssigner for RoleTrustAssigner {
    fn assign_trust(&self, messages: &mut [Message], config: &Config) {
        for msg in messages.iter_mut() {
            let role_str = role_to_str(&msg.role);

            // Step 1: Check unknown_trust_policy for a base trust level.
            // If no entry exists, default to Untrusted (safe default).
            let mut trust = config
                .policy
                .trust
                .unknown_trust_policy
                .get(role_str)
                .cloned()
                .unwrap_or(TrustLevel::Untrusted);

            // Step 2: If the role is in auto_untrusted_roles, override to Untrusted.
            if config
                .policy
                .trust
                .auto_untrusted_roles
                .iter()
                .any(|r| r == role_str)
            {
                trust = TrustLevel::Untrusted;
            }

            // Step 3: For Tool messages with a tool_name, check per-tool trust override.
            if msg.role == Role::Tool {
                if let Some(tool_name) = &msg.tool_name {
                    if let Some(tool_config) = config.policy.tools.get(tool_name) {
                        if let Some(ref tool_trust) = tool_config.trust {
                            trust = tool_trust.clone();
                        }
                    }
                }
            }

            msg.trust = trust;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        Config, ContentPolicy, EngineConfig, LayerConfigs, PolicyConfig, RuntimeConfig, ToolConfig,
        TrustConfig,
    };
    use crate::message::{Message, Role, TrustLevel};
    use std::collections::HashMap;

    /// Build a minimal Config with the standard trust rules from the spec:
    /// - unknown_trust_policy: system->trusted, assistant->trusted, user->untrusted, tool->untrusted
    /// - auto_untrusted_roles: ["tool"]
    /// - tools: internal_lookup (trust: trusted), read_file (trust: untrusted)
    fn make_test_config() -> Config {
        let mut unknown_trust_policy = HashMap::new();
        unknown_trust_policy.insert("system".to_string(), TrustLevel::Trusted);
        unknown_trust_policy.insert("assistant".to_string(), TrustLevel::Trusted);
        unknown_trust_policy.insert("user".to_string(), TrustLevel::Untrusted);
        unknown_trust_policy.insert("tool".to_string(), TrustLevel::Untrusted);

        let mut tools = HashMap::new();
        tools.insert(
            "internal_lookup".to_string(),
            ToolConfig {
                allowed: true,
                trust: Some(TrustLevel::Trusted),
                constraints: HashMap::new(),
                result_policy: None,
            },
        );
        tools.insert(
            "read_file".to_string(),
            ToolConfig {
                allowed: true,
                trust: Some(TrustLevel::Untrusted),
                constraints: HashMap::new(),
                result_policy: None,
            },
        );
        tools.insert(
            "_default".to_string(),
            ToolConfig {
                allowed: false,
                trust: None,
                constraints: HashMap::new(),
                result_policy: None,
            },
        );

        Config {
            policy: PolicyConfig {
                version: "v1".to_string(),
                tools,
                block_patterns: Vec::new(),
                canary_tokens: Vec::new(),
                sensitive_patterns: Vec::new(),
                untrusted_content_policy: ContentPolicy::default(),
                trust: TrustConfig {
                    auto_untrusted_roles: vec!["tool".to_string()],
                    unknown_trust_policy,
                },
                layers: LayerConfigs::default(),
            },
            runtime: RuntimeConfig {
                engine: EngineConfig::default(),
                environment: String::new(),
            },
            contract_hash: String::new(),
        }
    }

    fn make_tool_message(tool_name: &str) -> Message {
        Message {
            role: Role::Tool,
            content: "tool result".to_string(),
            tool_calls: Vec::new(),
            tool_call_id: Some("call_1".to_string()),
            tool_name: Some(tool_name.to_string()),
            trust: TrustLevel::Trusted, // default; will be overridden
        }
    }

    // ---------------------------------------------------------------
    // 1. role: user -> Untrusted (from unknown_trust_policy)
    // ---------------------------------------------------------------

    #[test]
    fn user_role_assigned_untrusted_from_policy() {
        let config = make_test_config();
        let mut messages = vec![Message::new(Role::User, "hello")];

        RoleTrustAssigner.assign_trust(&mut messages, &config);

        assert_eq!(messages[0].trust, TrustLevel::Untrusted);
    }

    // ---------------------------------------------------------------
    // 2. role: tool -> Untrusted (from auto_untrusted_roles)
    // ---------------------------------------------------------------

    #[test]
    fn tool_role_assigned_untrusted_from_auto_untrusted() {
        let config = make_test_config();
        let mut messages = vec![Message::new(Role::Tool, "result")];

        RoleTrustAssigner.assign_trust(&mut messages, &config);

        assert_eq!(messages[0].trust, TrustLevel::Untrusted);
    }

    // ---------------------------------------------------------------
    // 3. role: system -> Trusted
    // ---------------------------------------------------------------

    #[test]
    fn system_role_assigned_trusted() {
        let config = make_test_config();
        let mut messages = vec![Message::new(Role::System, "system prompt")];

        RoleTrustAssigner.assign_trust(&mut messages, &config);

        assert_eq!(messages[0].trust, TrustLevel::Trusted);
    }

    // ---------------------------------------------------------------
    // 4. role: assistant -> Trusted
    // ---------------------------------------------------------------

    #[test]
    fn assistant_role_assigned_trusted() {
        let config = make_test_config();
        let mut messages = vec![Message::new(Role::Assistant, "response")];

        RoleTrustAssigner.assign_trust(&mut messages, &config);

        assert_eq!(messages[0].trust, TrustLevel::Trusted);
    }

    // ---------------------------------------------------------------
    // 5. Tool message from internal_lookup with trust: trusted -> Trusted
    // ---------------------------------------------------------------

    #[test]
    fn tool_with_per_tool_trusted_override() {
        let config = make_test_config();
        let mut messages = vec![make_tool_message("internal_lookup")];

        RoleTrustAssigner.assign_trust(&mut messages, &config);

        assert_eq!(messages[0].trust, TrustLevel::Trusted);
    }

    // ---------------------------------------------------------------
    // 6. Tool message from read_file with trust: untrusted -> Untrusted
    // ---------------------------------------------------------------

    #[test]
    fn tool_with_per_tool_untrusted_confirms_default() {
        let config = make_test_config();
        let mut messages = vec![make_tool_message("read_file")];

        RoleTrustAssigner.assign_trust(&mut messages, &config);

        assert_eq!(messages[0].trust, TrustLevel::Untrusted);
    }

    // ---------------------------------------------------------------
    // 7. Tool message from unlisted tool -> Untrusted (role-based default)
    // ---------------------------------------------------------------

    #[test]
    fn tool_from_unlisted_tool_falls_through_to_role_default() {
        let config = make_test_config();
        let mut messages = vec![make_tool_message("some_unknown_tool")];

        RoleTrustAssigner.assign_trust(&mut messages, &config);

        assert_eq!(messages[0].trust, TrustLevel::Untrusted);
    }

    // ---------------------------------------------------------------
    // 8. Unknown role not in unknown_trust_policy -> Untrusted (safe default)
    // ---------------------------------------------------------------

    #[test]
    fn role_not_in_policy_defaults_to_untrusted() {
        // Build a config with an empty unknown_trust_policy so no role matches
        let config = Config {
            policy: PolicyConfig {
                version: "v1".to_string(),
                tools: HashMap::new(),
                block_patterns: Vec::new(),
                canary_tokens: Vec::new(),
                sensitive_patterns: Vec::new(),
                untrusted_content_policy: ContentPolicy::default(),
                trust: TrustConfig {
                    auto_untrusted_roles: Vec::new(),
                    unknown_trust_policy: HashMap::new(),
                },
                layers: LayerConfigs::default(),
            },
            runtime: RuntimeConfig {
                engine: EngineConfig::default(),
                environment: String::new(),
            },
            contract_hash: String::new(),
        };

        let mut messages = vec![Message::new(Role::User, "hello")];
        RoleTrustAssigner.assign_trust(&mut messages, &config);

        assert_eq!(messages[0].trust, TrustLevel::Untrusted);
    }

    // ---------------------------------------------------------------
    // 9. Multiple messages: trust assigned correctly to each
    // ---------------------------------------------------------------

    #[test]
    fn multiple_messages_each_assigned_correctly() {
        let config = make_test_config();
        let mut messages = vec![
            Message::new(Role::System, "system prompt"),
            Message::new(Role::User, "user input"),
            Message::new(Role::Assistant, "assistant response"),
            make_tool_message("internal_lookup"),
            make_tool_message("read_file"),
        ];

        RoleTrustAssigner.assign_trust(&mut messages, &config);

        assert_eq!(messages[0].trust, TrustLevel::Trusted, "system");
        assert_eq!(messages[1].trust, TrustLevel::Untrusted, "user");
        assert_eq!(messages[2].trust, TrustLevel::Trusted, "assistant");
        assert_eq!(
            messages[3].trust,
            TrustLevel::Trusted,
            "tool: internal_lookup"
        );
        assert_eq!(
            messages[4].trust,
            TrustLevel::Untrusted,
            "tool: read_file"
        );
    }

    // ---------------------------------------------------------------
    // 10. Per-tool override takes precedence over auto_untrusted_roles
    // ---------------------------------------------------------------

    #[test]
    fn per_tool_override_takes_precedence_over_auto_untrusted() {
        let config = make_test_config();

        // "tool" is in auto_untrusted_roles, but internal_lookup has trust: trusted
        let mut messages = vec![make_tool_message("internal_lookup")];

        RoleTrustAssigner.assign_trust(&mut messages, &config);

        // Per-tool override should win over auto_untrusted_roles
        assert_eq!(messages[0].trust, TrustLevel::Trusted);
    }

    // ---------------------------------------------------------------
    // Extra: Tool message with no tool_name uses role-based default
    // ---------------------------------------------------------------

    #[test]
    fn tool_message_without_tool_name_uses_role_default() {
        let config = make_test_config();
        let mut messages = vec![Message::new(Role::Tool, "result")];
        // Message::new sets tool_name to None

        RoleTrustAssigner.assign_trust(&mut messages, &config);

        assert_eq!(messages[0].trust, TrustLevel::Untrusted);
    }

    // ---------------------------------------------------------------
    // Extra: role_to_str covers all Role variants
    // ---------------------------------------------------------------

    #[test]
    fn role_to_str_maps_all_variants() {
        assert_eq!(role_to_str(&Role::System), "system");
        assert_eq!(role_to_str(&Role::User), "user");
        assert_eq!(role_to_str(&Role::Assistant), "assistant");
        assert_eq!(role_to_str(&Role::Tool), "tool");
    }
}
