// Internal message representation — defined in M1.4a
//
// These are the canonical types that ALL layers operate on.
// Provider adapters produce `Vec<Message>`, and every downstream
// bead (trust assignment, constraint evaluation, normalization,
// streaming) consumes them.

use serde::{Deserialize, Serialize};

/// The role of a message participant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Whether content should be treated as trusted or untrusted.
///
/// Provider adapters default to `Trusted`. Role-based trust
/// assignment (M1.7) overrides this based on role and per-tool config.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrustLevel {
    Trusted,
    Untrusted,
}

/// A tool call requested by the assistant.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Parsed JSON arguments (not a raw string).
    pub arguments: serde_json::Value,
}

/// A single message in the conversation.
///
/// All layers operate on `Vec<Message>`. The struct is intentionally
/// flat: provider adapters populate it, trust assignment sets `trust`,
/// and constraint/normalization layers read from it.
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: Role,
    /// Text content (after JSON parsing, before L0 normalization).
    pub content: String,
    /// Tool calls requested by the assistant. Empty for non-assistant messages.
    pub tool_calls: Vec<ToolCall>,
    /// For tool result messages: the ID of the tool call this responds to.
    pub tool_call_id: Option<String>,
    /// For tool result messages: maps to per-tool config in parapet.yaml.
    pub tool_name: Option<String>,
    /// Assigned by role-based trust (M1.7). Defaults to `Trusted`.
    pub trust: TrustLevel,
}

impl Message {
    /// Create a new message with sensible defaults.
    ///
    /// - `trust` defaults to `TrustLevel::Trusted`
    /// - `tool_calls` defaults to empty
    /// - `tool_call_id` and `tool_name` default to `None`
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_name: None,
            trust: TrustLevel::Trusted,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---------------------------------------------------------------
    // 1. Message with each Role variant constructs correctly
    // ---------------------------------------------------------------

    #[test]
    fn message_with_system_role() {
        let msg = Message::new(Role::System, "You are helpful.");
        assert_eq!(msg.role, Role::System);
        assert_eq!(msg.content, "You are helpful.");
    }

    #[test]
    fn message_with_user_role() {
        let msg = Message::new(Role::User, "Hello");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content, "Hello");
    }

    #[test]
    fn message_with_assistant_role() {
        let msg = Message::new(Role::Assistant, "Hi there");
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.content, "Hi there");
    }

    #[test]
    fn message_with_tool_role() {
        let msg = Message::new(Role::Tool, "{\"result\": 42}");
        assert_eq!(msg.role, Role::Tool);
        assert_eq!(msg.content, "{\"result\": 42}");
    }

    // ---------------------------------------------------------------
    // 2. Default trust is Trusted
    // ---------------------------------------------------------------

    #[test]
    fn default_trust_is_trusted() {
        let msg = Message::new(Role::User, "test");
        assert_eq!(msg.trust, TrustLevel::Trusted);
    }

    // ---------------------------------------------------------------
    // 3. ToolCall.arguments accepts arbitrary JSON
    // ---------------------------------------------------------------

    #[test]
    fn tool_call_with_nested_object_arguments() {
        let tc = ToolCall {
            id: "call_1".to_string(),
            name: "read_file".to_string(),
            arguments: json!({
                "path": "/tmp/foo.txt",
                "options": {
                    "encoding": "utf-8",
                    "line_count": 100
                }
            }),
        };
        assert_eq!(tc.arguments["path"], "/tmp/foo.txt");
        assert_eq!(tc.arguments["options"]["encoding"], "utf-8");
        assert_eq!(tc.arguments["options"]["line_count"], 100);
    }

    #[test]
    fn tool_call_with_array_arguments() {
        let tc = ToolCall {
            id: "call_2".to_string(),
            name: "batch_process".to_string(),
            arguments: json!({"items": [1, 2, 3, "four"]}),
        };
        assert_eq!(tc.arguments["items"][0], 1);
        assert_eq!(tc.arguments["items"][3], "four");
    }

    #[test]
    fn tool_call_with_string_argument() {
        let tc = ToolCall {
            id: "call_3".to_string(),
            name: "echo".to_string(),
            arguments: json!({"text": "hello world"}),
        };
        assert_eq!(tc.arguments["text"], "hello world");
    }

    #[test]
    fn tool_call_with_numeric_arguments() {
        let tc = ToolCall {
            id: "call_4".to_string(),
            name: "calculate".to_string(),
            arguments: json!({"a": 3.14, "b": -42, "c": 0}),
        };
        assert_eq!(tc.arguments["a"], 3.14);
        assert_eq!(tc.arguments["b"], -42);
        assert_eq!(tc.arguments["c"], 0);
    }

    // ---------------------------------------------------------------
    // 4. Message with empty tool_calls and None optional fields
    // ---------------------------------------------------------------

    #[test]
    fn message_defaults_have_empty_tool_calls_and_none_fields() {
        let msg = Message::new(Role::User, "test");
        assert!(msg.tool_calls.is_empty());
        assert_eq!(msg.tool_call_id, None);
        assert_eq!(msg.tool_name, None);
    }

    // ---------------------------------------------------------------
    // 5. Message::new() returns correct defaults
    // ---------------------------------------------------------------

    #[test]
    fn message_new_returns_all_correct_defaults() {
        let msg = Message::new(Role::Assistant, "response");
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.content, "response");
        assert_eq!(msg.trust, TrustLevel::Trusted);
        assert!(msg.tool_calls.is_empty());
        assert_eq!(msg.tool_call_id, None);
        assert_eq!(msg.tool_name, None);
    }

    // ---------------------------------------------------------------
    // 6. Role and TrustLevel implement Clone and PartialEq
    // ---------------------------------------------------------------

    #[test]
    fn role_clone_and_partial_eq() {
        let role = Role::User;
        let cloned = role.clone();
        assert_eq!(role, cloned);

        // Different variants are not equal
        assert_ne!(Role::User, Role::Assistant);
        assert_ne!(Role::System, Role::Tool);
    }

    #[test]
    fn trust_level_clone_and_partial_eq() {
        let trust = TrustLevel::Trusted;
        let cloned = trust.clone();
        assert_eq!(trust, cloned);

        assert_ne!(TrustLevel::Trusted, TrustLevel::Untrusted);
    }

    // ---------------------------------------------------------------
    // 7. ToolCall with empty arguments {} works
    // ---------------------------------------------------------------

    #[test]
    fn tool_call_with_empty_arguments() {
        let tc = ToolCall {
            id: "call_5".to_string(),
            name: "no_args_tool".to_string(),
            arguments: json!({}),
        };
        assert_eq!(tc.arguments, json!({}));
        assert!(tc.arguments.as_object().unwrap().is_empty());
    }

    // ---------------------------------------------------------------
    // Additional edge-case coverage
    // ---------------------------------------------------------------

    #[test]
    fn message_with_empty_content() {
        let msg = Message::new(Role::User, "");
        assert_eq!(msg.content, "");
    }

    #[test]
    fn message_with_tool_call_fields_set() {
        let msg = Message {
            role: Role::Tool,
            content: "file contents here".to_string(),
            tool_calls: Vec::new(),
            tool_call_id: Some("call_abc".to_string()),
            tool_name: Some("read_file".to_string()),
            trust: TrustLevel::Untrusted,
        };
        assert_eq!(msg.role, Role::Tool);
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_abc"));
        assert_eq!(msg.tool_name.as_deref(), Some("read_file"));
        assert_eq!(msg.trust, TrustLevel::Untrusted);
    }

    #[test]
    fn message_with_tool_calls_populated() {
        let mut msg = Message::new(Role::Assistant, "");
        msg.tool_calls.push(ToolCall {
            id: "call_1".to_string(),
            name: "read_file".to_string(),
            arguments: json!({"path": "/tmp/test.txt"}),
        });
        msg.tool_calls.push(ToolCall {
            id: "call_2".to_string(),
            name: "write_file".to_string(),
            arguments: json!({"path": "/tmp/out.txt", "content": "data"}),
        });
        assert_eq!(msg.tool_calls.len(), 2);
        assert_eq!(msg.tool_calls[0].name, "read_file");
        assert_eq!(msg.tool_calls[1].name, "write_file");
    }

    #[test]
    fn message_clone_produces_independent_copy() {
        let original = Message {
            role: Role::Assistant,
            content: "original".to_string(),
            tool_calls: vec![ToolCall {
                id: "tc1".to_string(),
                name: "tool".to_string(),
                arguments: json!({"key": "value"}),
            }],
            tool_call_id: None,
            tool_name: None,
            trust: TrustLevel::Trusted,
        };
        let mut cloned = original.clone();
        cloned.content = "modified".to_string();
        cloned.trust = TrustLevel::Untrusted;

        // Original is unchanged
        assert_eq!(original.content, "original");
        assert_eq!(original.trust, TrustLevel::Trusted);
        // Clone has the new values
        assert_eq!(cloned.content, "modified");
        assert_eq!(cloned.trust, TrustLevel::Untrusted);
    }

    #[test]
    fn role_serialization_roundtrip() {
        let role = Role::Assistant;
        let json_str = serde_json::to_string(&role).unwrap();
        let deserialized: Role = serde_json::from_str(&json_str).unwrap();
        assert_eq!(role, deserialized);
    }

    #[test]
    fn trust_level_serialization_roundtrip() {
        let trust = TrustLevel::Untrusted;
        let json_str = serde_json::to_string(&trust).unwrap();
        let deserialized: TrustLevel = serde_json::from_str(&json_str).unwrap();
        assert_eq!(trust, deserialized);
    }
}
