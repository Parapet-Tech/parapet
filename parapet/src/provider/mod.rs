// Provider adapters (OpenAI + Anthropic) -- defined in M1.4b
//
// Responsibilities:
// - Parse OpenAI request JSON (messages[]) into Vec<Message>
// - Parse Anthropic request JSON (messages[] + top-level system) into Vec<Message>
// - Extract tool calls from both formats into Vec<ToolCall>
// - Serialize error responses in provider-specific format
// - Handle edge cases: empty messages, null arguments, content arrays, etc.

use crate::message::{Message, Role, ToolCall, TrustLevel};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during provider-specific parsing or serialization.
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("invalid JSON: {0}")]
    InvalidJson(String),

    #[error("missing required field: {0}")]
    MissingField(String),

    #[error("invalid role: {0}")]
    InvalidRole(String),

    #[error("invalid message format: {0}")]
    InvalidFormat(String),
}

// ---------------------------------------------------------------------------
// Trait: ProviderAdapter
// ---------------------------------------------------------------------------

/// Which LLM provider format this adapter handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderType {
    OpenAi,
    Anthropic,
}

/// Adapter that converts between provider-specific JSON and internal Message types.
///
/// Implementations must be Send + Sync so they can be shared across request
/// handlers via Arc.
pub trait ProviderAdapter: Send + Sync {
    /// Parse a raw request body into a list of internal messages.
    fn parse_request(&self, body: &[u8]) -> Result<Vec<Message>, ProviderError>;

    /// Serialize an error string into a provider-specific JSON error response body.
    fn serialize_error(&self, error: &str, status: u16) -> Vec<u8>;

    /// Return which provider this adapter handles.
    fn provider_type(&self) -> ProviderType;
}

// ---------------------------------------------------------------------------
// OpenAI adapter
// ---------------------------------------------------------------------------

/// Parses OpenAI chat completion request format.
///
/// Expected input shape:
/// ```json
/// {
///   "messages": [
///     {"role": "system", "content": "..."},
///     {"role": "user", "content": "..."},
///     {"role": "assistant", "content": null, "tool_calls": [...]},
///     {"role": "tool", "tool_call_id": "call_1", "content": "..."}
///   ]
/// }
/// ```
pub struct OpenAiAdapter;

impl OpenAiAdapter {
    pub fn new() -> Self {
        Self
    }

    /// Map an OpenAI role string to the internal Role enum.
    fn parse_role(role: &str) -> Result<Role, ProviderError> {
        match role {
            "system" => Ok(Role::System),
            "user" => Ok(Role::User),
            "assistant" => Ok(Role::Assistant),
            "tool" => Ok(Role::Tool),
            other => Err(ProviderError::InvalidRole(other.to_string())),
        }
    }

    /// Parse OpenAI tool_calls array into Vec<ToolCall>.
    ///
    /// OpenAI format: arguments is a JSON string that needs to be parsed.
    fn parse_tool_calls(
        tool_calls: &[serde_json::Value],
    ) -> Result<Vec<ToolCall>, ProviderError> {
        let mut result = Vec::with_capacity(tool_calls.len());

        for tc in tool_calls {
            let id = tc
                .get("id")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ProviderError::MissingField("tool_calls[].id".to_string()))?
                .to_string();

            let function = tc
                .get("function")
                .ok_or_else(|| ProviderError::MissingField("tool_calls[].function".to_string()))?;

            let name = function
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    ProviderError::MissingField("tool_calls[].function.name".to_string())
                })?
                .to_string();

            let arguments = Self::parse_openai_arguments(function.get("arguments"))?;

            result.push(ToolCall {
                id,
                name,
                arguments,
            });
        }

        Ok(result)
    }

    /// Parse the OpenAI arguments field.
    ///
    /// OpenAI sends arguments as a JSON string (e.g., `"{\"path\": \"/tmp\"}"`).
    /// We also handle:
    /// - null -> empty object
    /// - already an object -> use as-is
    /// - absent -> empty object
    fn parse_openai_arguments(
        value: Option<&serde_json::Value>,
    ) -> Result<serde_json::Value, ProviderError> {
        match value {
            None | Some(serde_json::Value::Null) => Ok(serde_json::json!({})),
            Some(serde_json::Value::String(s)) => {
                if s.is_empty() {
                    return Ok(serde_json::json!({}));
                }
                serde_json::from_str(s).map_err(|e| {
                    ProviderError::InvalidFormat(format!(
                        "tool_calls[].function.arguments is not valid JSON: {e}"
                    ))
                })
            }
            Some(serde_json::Value::Object(_)) => Ok(value.unwrap().clone()),
            Some(other) => Err(ProviderError::InvalidFormat(format!(
                "tool_calls[].function.arguments has unexpected type: {}",
                other
            ))),
        }
    }
}

impl ProviderAdapter for OpenAiAdapter {
    fn parse_request(&self, body: &[u8]) -> Result<Vec<Message>, ProviderError> {
        let root: serde_json::Value = serde_json::from_slice(body)
            .map_err(|e| ProviderError::InvalidJson(e.to_string()))?;

        let messages_val = root
            .get("messages")
            .ok_or_else(|| ProviderError::MissingField("messages".to_string()))?;

        let messages_arr = messages_val
            .as_array()
            .ok_or_else(|| ProviderError::InvalidFormat("messages is not an array".to_string()))?;

        let mut result = Vec::with_capacity(messages_arr.len());

        for msg_val in messages_arr {
            let role_str = msg_val
                .get("role")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ProviderError::MissingField("messages[].role".to_string()))?;

            let role = Self::parse_role(role_str)?;

            // Content can be null (for assistant messages with tool_calls)
            let content = match msg_val.get("content") {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(serde_json::Value::Null) | None => String::new(),
                Some(other) => other.to_string(),
            };

            // Parse tool_calls if present (assistant messages)
            let tool_calls = match msg_val.get("tool_calls") {
                Some(serde_json::Value::Array(arr)) => Self::parse_tool_calls(arr)?,
                Some(serde_json::Value::Null) | None => Vec::new(),
                Some(_) => {
                    return Err(ProviderError::InvalidFormat(
                        "tool_calls is not an array".to_string(),
                    ))
                }
            };

            // Parse tool_call_id for tool result messages
            let tool_call_id = msg_val
                .get("tool_call_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            // OpenAI tool result messages do not carry tool_name in the wire format,
            // but we can try to extract "name" if present (some API versions include it).
            let tool_name = msg_val
                .get("name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            result.push(Message {
                role,
                content,
                tool_calls,
                tool_call_id,
                tool_name,
                trust: TrustLevel::Trusted,
            });
        }

        Ok(result)
    }

    fn serialize_error(&self, error: &str, status: u16) -> Vec<u8> {
        let error_type = if status >= 500 {
            "server_error"
        } else if status >= 400 {
            "invalid_request_error"
        } else {
            "error"
        };

        let response = serde_json::json!({
            "error": {
                "message": error,
                "type": error_type,
                "param": null,
                "code": null
            }
        });

        serde_json::to_vec(&response).unwrap_or_else(|_| {
            format!(r#"{{"error":{{"message":"{}","type":"server_error"}}}}"#, error).into_bytes()
        })
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::OpenAi
    }
}

// ---------------------------------------------------------------------------
// Anthropic adapter
// ---------------------------------------------------------------------------

/// Parses Anthropic messages API request format.
///
/// Expected input shape:
/// ```json
/// {
///   "system": "You are helpful",
///   "messages": [
///     {"role": "user", "content": "Hello"},
///     {"role": "assistant", "content": [
///       {"type": "text", "text": "Let me check"},
///       {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "/tmp"}}
///     ]},
///     {"role": "user", "content": [
///       {"type": "tool_result", "tool_use_id": "toolu_1", "content": "file contents"}
///     ]}
///   ]
/// }
/// ```
pub struct AnthropicAdapter;

impl AnthropicAdapter {
    pub fn new() -> Self {
        Self
    }

    /// Map an Anthropic role string to the internal Role enum.
    fn parse_role(role: &str) -> Result<Role, ProviderError> {
        match role {
            "user" => Ok(Role::User),
            "assistant" => Ok(Role::Assistant),
            other => Err(ProviderError::InvalidRole(other.to_string())),
        }
    }

    /// Parse Anthropic content field, which can be a string or array of typed blocks.
    ///
    /// Returns (text_content, tool_calls, tool_results) where:
    /// - text_content is concatenated text from text blocks
    /// - tool_calls are extracted from tool_use blocks
    /// - tool_results are extracted from tool_result blocks
    fn parse_content(
        content: &serde_json::Value,
    ) -> Result<(String, Vec<ToolCall>, Vec<ToolResultInfo>), ProviderError> {
        match content {
            serde_json::Value::String(s) => Ok((s.clone(), Vec::new(), Vec::new())),
            serde_json::Value::Array(blocks) => {
                let mut text_parts = Vec::new();
                let mut tool_calls = Vec::new();
                let mut tool_results = Vec::new();

                for block in blocks {
                    let block_type = block
                        .get("type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");

                    match block_type {
                        "text" => {
                            if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                                text_parts.push(text.to_string());
                            }
                        }
                        "tool_use" => {
                            let id = block
                                .get("id")
                                .and_then(|v| v.as_str())
                                .ok_or_else(|| {
                                    ProviderError::MissingField(
                                        "content[].id (tool_use)".to_string(),
                                    )
                                })?
                                .to_string();

                            let name = block
                                .get("name")
                                .and_then(|v| v.as_str())
                                .ok_or_else(|| {
                                    ProviderError::MissingField(
                                        "content[].name (tool_use)".to_string(),
                                    )
                                })?
                                .to_string();

                            // Anthropic input is already a JSON object, not a string.
                            // Handle null as empty object.
                            let input = match block.get("input") {
                                Some(serde_json::Value::Null) | None => serde_json::json!({}),
                                Some(val) => val.clone(),
                            };

                            tool_calls.push(ToolCall {
                                id,
                                name,
                                arguments: input,
                            });
                        }
                        "tool_result" => {
                            let tool_use_id = block
                                .get("tool_use_id")
                                .and_then(|v| v.as_str())
                                .ok_or_else(|| {
                                    ProviderError::MissingField(
                                        "content[].tool_use_id (tool_result)".to_string(),
                                    )
                                })?
                                .to_string();

                            // tool_result content can be a string or array of blocks
                            let result_content =
                                match block.get("content") {
                                    Some(serde_json::Value::String(s)) => s.clone(),
                                    Some(serde_json::Value::Array(arr)) => {
                                        // Extract text from inner blocks
                                        arr.iter()
                                            .filter_map(|b| {
                                                if b.get("type").and_then(|v| v.as_str())
                                                    == Some("text")
                                                {
                                                    b.get("text")
                                                        .and_then(|v| v.as_str())
                                                        .map(|s| s.to_string())
                                                } else {
                                                    None
                                                }
                                            })
                                            .collect::<Vec<_>>()
                                            .join("")
                                    }
                                    Some(serde_json::Value::Null) | None => String::new(),
                                    Some(other) => other.to_string(),
                                };

                            tool_results.push(ToolResultInfo {
                                tool_use_id,
                                content: result_content,
                            });
                        }
                        _ => {
                            // Non-text, non-tool blocks: preserved but not scanned.
                            // We do not extract text from image blocks, etc.
                        }
                    }
                }

                Ok((text_parts.join(""), tool_calls, tool_results))
            }
            serde_json::Value::Null => Ok((String::new(), Vec::new(), Vec::new())),
            _ => Err(ProviderError::InvalidFormat(
                "content must be a string or array".to_string(),
            )),
        }
    }
}

/// Internal helper: extracted tool result information from Anthropic content blocks.
struct ToolResultInfo {
    tool_use_id: String,
    content: String,
}

impl ProviderAdapter for AnthropicAdapter {
    fn parse_request(&self, body: &[u8]) -> Result<Vec<Message>, ProviderError> {
        let root: serde_json::Value = serde_json::from_slice(body)
            .map_err(|e| ProviderError::InvalidJson(e.to_string()))?;

        let mut result = Vec::new();

        // Handle top-level system field
        if let Some(system_val) = root.get("system") {
            let system_content = match system_val {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Null => String::new(),
                other => other.to_string(),
            };
            if !system_content.is_empty() {
                result.push(Message::new(Role::System, system_content));
            }
        }

        let messages_val = root
            .get("messages")
            .ok_or_else(|| ProviderError::MissingField("messages".to_string()))?;

        let messages_arr = messages_val
            .as_array()
            .ok_or_else(|| ProviderError::InvalidFormat("messages is not an array".to_string()))?;

        for msg_val in messages_arr {
            let role_str = msg_val
                .get("role")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ProviderError::MissingField("messages[].role".to_string()))?;

            let role = Self::parse_role(role_str)?;

            let content_val = msg_val
                .get("content")
                .ok_or_else(|| ProviderError::MissingField("messages[].content".to_string()))?;

            let (text_content, tool_calls, tool_results) = Self::parse_content(content_val)?;

            // If there are tool_result blocks, each becomes a separate Tool message
            if !tool_results.is_empty() {
                for tr in &tool_results {
                    result.push(Message {
                        role: Role::Tool,
                        content: tr.content.clone(),
                        tool_calls: Vec::new(),
                        tool_call_id: Some(tr.tool_use_id.clone()),
                        tool_name: None,
                        trust: TrustLevel::Trusted,
                    });
                }
                // If there was also text content alongside tool_results, include it
                if !text_content.is_empty() {
                    result.push(Message {
                        role: role.clone(),
                        content: text_content,
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                        tool_name: None,
                        trust: TrustLevel::Trusted,
                    });
                }
            } else {
                result.push(Message {
                    role,
                    content: text_content,
                    tool_calls,
                    tool_call_id: None,
                    tool_name: None,
                    trust: TrustLevel::Trusted,
                });
            }
        }

        Ok(result)
    }

    fn serialize_error(&self, error: &str, status: u16) -> Vec<u8> {
        let error_type = if status >= 500 {
            "api_error"
        } else if status == 429 {
            "rate_limit_error"
        } else if status >= 400 {
            "invalid_request_error"
        } else {
            "error"
        };

        let response = serde_json::json!({
            "type": "error",
            "error": {
                "type": error_type,
                "message": error
            }
        });

        serde_json::to_vec(&response).unwrap_or_else(|_| {
            format!(
                r#"{{"type":"error","error":{{"type":"api_error","message":"{}"}}}}"#,
                error
            )
            .into_bytes()
        })
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::Anthropic
    }
}

// ---------------------------------------------------------------------------
// Factory function
// ---------------------------------------------------------------------------

/// Create the appropriate provider adapter for the given provider type.
pub fn adapter_for(provider: ProviderType) -> Box<dyn ProviderAdapter> {
    match provider {
        ProviderType::OpenAi => Box::new(OpenAiAdapter::new()),
        ProviderType::Anthropic => Box::new(AnthropicAdapter::new()),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---------------------------------------------------------------
    // Test 1: OpenAI messages[] -> Vec<Message> with correct roles and content
    // ---------------------------------------------------------------

    #[test]
    fn openai_messages_parsed_with_correct_roles_and_content() {
        let body = json!({
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]
        });

        let adapter = OpenAiAdapter::new();
        let messages = adapter.parse_request(body.to_string().as_bytes()).unwrap();

        assert_eq!(messages.len(), 3);

        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].content, "You are helpful");

        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[1].content, "Hello");

        assert_eq!(messages[2].role, Role::Assistant);
        assert_eq!(messages[2].content, "Hi there");
    }

    // ---------------------------------------------------------------
    // Test 2: Anthropic messages[] -> Vec<Message> with correct roles and content
    // ---------------------------------------------------------------

    #[test]
    fn anthropic_messages_parsed_with_correct_roles_and_content() {
        let body = json!({
            "system": "You are helpful",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]
        });

        let adapter = AnthropicAdapter::new();
        let messages = adapter.parse_request(body.to_string().as_bytes()).unwrap();

        assert_eq!(messages.len(), 3);

        // System message extracted from top-level field
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].content, "You are helpful");

        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[1].content, "Hello");

        assert_eq!(messages[2].role, Role::Assistant);
        assert_eq!(messages[2].content, "Hi there");
    }

    // ---------------------------------------------------------------
    // Test 3: OpenAI tool_calls[].function.{name, arguments} -> Vec<ToolCall>
    //         (arguments parsed from JSON string)
    // ---------------------------------------------------------------

    #[test]
    fn openai_tool_calls_parsed_with_arguments_from_string() {
        let body = json!({
            "messages": [
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": "{\"path\": \"/tmp/foo\"}"
                            }
                        }
                    ]
                }
            ]
        });

        let adapter = OpenAiAdapter::new();
        let messages = adapter.parse_request(body.to_string().as_bytes()).unwrap();

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::Assistant);
        assert_eq!(messages[0].content, "");
        assert_eq!(messages[0].tool_calls.len(), 1);

        let tc = &messages[0].tool_calls[0];
        assert_eq!(tc.id, "call_1");
        assert_eq!(tc.name, "read_file");
        assert_eq!(tc.arguments, json!({"path": "/tmp/foo"}));
    }

    // ---------------------------------------------------------------
    // Test 4: Anthropic content[].{type: "tool_use", name, input} -> Vec<ToolCall>
    //         (input as-is, already a JSON object)
    // ---------------------------------------------------------------

    #[test]
    fn anthropic_tool_use_blocks_parsed_as_tool_calls() {
        let body = json!({
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "read_file",
                            "input": {"path": "/tmp/foo"}
                        }
                    ]
                }
            ]
        });

        let adapter = AnthropicAdapter::new();
        let messages = adapter.parse_request(body.to_string().as_bytes()).unwrap();

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::Assistant);
        assert_eq!(messages[0].content, "Let me check");
        assert_eq!(messages[0].tool_calls.len(), 1);

        let tc = &messages[0].tool_calls[0];
        assert_eq!(tc.id, "toolu_1");
        assert_eq!(tc.name, "read_file");
        assert_eq!(tc.arguments, json!({"path": "/tmp/foo"}));
    }

    // ---------------------------------------------------------------
    // Test 5: Error response matches OpenAI JSON format
    // ---------------------------------------------------------------

    #[test]
    fn openai_error_response_format() {
        let adapter = OpenAiAdapter::new();
        let error_bytes = adapter.serialize_error("something went wrong", 400);
        let parsed: serde_json::Value = serde_json::from_slice(&error_bytes).unwrap();

        assert_eq!(parsed["error"]["message"], "something went wrong");
        assert_eq!(parsed["error"]["type"], "invalid_request_error");
        assert!(parsed["error"]["param"].is_null());
        assert!(parsed["error"]["code"].is_null());
    }

    #[test]
    fn openai_error_response_5xx_type() {
        let adapter = OpenAiAdapter::new();
        let error_bytes = adapter.serialize_error("internal error", 500);
        let parsed: serde_json::Value = serde_json::from_slice(&error_bytes).unwrap();

        assert_eq!(parsed["error"]["type"], "server_error");
    }

    // ---------------------------------------------------------------
    // Test 6: Error response matches Anthropic JSON format
    // ---------------------------------------------------------------

    #[test]
    fn anthropic_error_response_format() {
        let adapter = AnthropicAdapter::new();
        let error_bytes = adapter.serialize_error("something went wrong", 400);
        let parsed: serde_json::Value = serde_json::from_slice(&error_bytes).unwrap();

        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["error"]["message"], "something went wrong");
        assert_eq!(parsed["error"]["type"], "invalid_request_error");
    }

    #[test]
    fn anthropic_error_response_5xx_type() {
        let adapter = AnthropicAdapter::new();
        let error_bytes = adapter.serialize_error("internal error", 500);
        let parsed: serde_json::Value = serde_json::from_slice(&error_bytes).unwrap();

        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["error"]["type"], "api_error");
    }

    // ---------------------------------------------------------------
    // Test 7: System prompt extraction: OpenAI messages[0] if role=system
    // ---------------------------------------------------------------

    #[test]
    fn openai_system_prompt_extraction() {
        let body = json!({
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hi"}
            ]
        });

        let adapter = OpenAiAdapter::new();
        let messages = adapter.parse_request(body.to_string().as_bytes()).unwrap();

        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].content, "You are a helpful assistant");
    }

    // ---------------------------------------------------------------
    // Test 8: System prompt extraction: Anthropic top-level system field
    // ---------------------------------------------------------------

    #[test]
    fn anthropic_system_prompt_from_top_level_field() {
        let body = json!({
            "system": "You are a helpful assistant",
            "messages": [
                {"role": "user", "content": "Hi"}
            ]
        });

        let adapter = AnthropicAdapter::new();
        let messages = adapter.parse_request(body.to_string().as_bytes()).unwrap();

        // System message should be first
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].content, "You are a helpful assistant");
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[1].content, "Hi");
    }

    #[test]
    fn anthropic_no_system_field_produces_no_system_message() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "Hi"}
            ]
        });

        let adapter = AnthropicAdapter::new();
        let messages = adapter.parse_request(body.to_string().as_bytes()).unwrap();

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::User);
    }

    // ---------------------------------------------------------------
    // Test 9: Anthropic content as array of typed blocks -> text extracted,
    //         non-text preserved but not scanned
    // ---------------------------------------------------------------

    #[test]
    fn anthropic_content_array_extracts_text_ignores_non_text() {
        let body = json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Check this image"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBOR..."
                            }
                        }
                    ]
                }
            ]
        });

        let adapter = AnthropicAdapter::new();
        let messages = adapter.parse_request(body.to_string().as_bytes()).unwrap();

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "Check this image");
        // Non-text blocks are silently ignored (not extracted into content)
    }

    #[test]
    fn anthropic_content_array_multiple_text_blocks_concatenated() {
        let body = json!({
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "First part. "},
                        {"type": "text", "text": "Second part."}
                    ]
                }
            ]
        });

        let adapter = AnthropicAdapter::new();
        let messages = adapter.parse_request(body.to_string().as_bytes()).unwrap();

        assert_eq!(messages[0].content, "First part. Second part.");
    }

    // ---------------------------------------------------------------
    // Test 10: Tool call with empty arguments {} -> parsed, no error
    // ---------------------------------------------------------------

    #[test]
    fn openai_tool_call_with_empty_arguments() {
        let body = json!({
            "messages": [
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "no_args_tool",
                                "arguments": "{}"
                            }
                        }
                    ]
                }
            ]
        });

        let adapter = OpenAiAdapter::new();
        let messages = adapter.parse_request(body.to_string().as_bytes()).unwrap();

        assert_eq!(messages[0].tool_calls[0].arguments, json!({}));
    }

    #[test]
    fn anthropic_tool_use_with_empty_input() {
        let body = json!({
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "no_args_tool",
                            "input": {}
                        }
                    ]
                }
            ]
        });

        let adapter = AnthropicAdapter::new();
        let messages = adapter.parse_request(body.to_string().as_bytes()).unwrap();

        assert_eq!(messages[0].tool_calls[0].arguments, json!({}));
    }

    // ---------------------------------------------------------------
    // Test 11: Tool call with null arguments -> parsed as empty object
    // ---------------------------------------------------------------

    #[test]
    fn openai_tool_call_with_null_arguments() {
        let body = json!({
            "messages": [
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "no_args_tool",
                                "arguments": null
                            }
                        }
                    ]
                }
            ]
        });

        let adapter = OpenAiAdapter::new();
        let messages = adapter.parse_request(body.to_string().as_bytes()).unwrap();

        assert_eq!(messages[0].tool_calls[0].arguments, json!({}));
    }

    #[test]
    fn anthropic_tool_use_with_null_input() {
        let body = json!({
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "no_args_tool",
                            "input": null
                        }
                    ]
                }
            ]
        });

        let adapter = AnthropicAdapter::new();
        let messages = adapter.parse_request(body.to_string().as_bytes()).unwrap();

        assert_eq!(messages[0].tool_calls[0].arguments, json!({}));
    }

    // ---------------------------------------------------------------
    // Test 12: Tool call with arguments as JSON string (not object)
    //          -> parsed correctly
    // ---------------------------------------------------------------

    #[test]
    fn openai_tool_call_arguments_as_json_string_parsed() {
        let body = json!({
            "messages": [
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": "{\"query\": \"rust programming\", \"limit\": 10}"
                            }
                        }
                    ]
                }
            ]
        });

        let adapter = OpenAiAdapter::new();
        let messages = adapter.parse_request(body.to_string().as_bytes()).unwrap();

        let tc = &messages[0].tool_calls[0];
        assert_eq!(tc.arguments["query"], "rust programming");
        assert_eq!(tc.arguments["limit"], 10);
    }

    // ---------------------------------------------------------------
    // Test 13: Messages array with zero messages -> empty Vec<Message>,
    //          no error
    // ---------------------------------------------------------------

    #[test]
    fn openai_empty_messages_array() {
        let body = json!({
            "messages": []
        });

        let adapter = OpenAiAdapter::new();
        let messages = adapter.parse_request(body.to_string().as_bytes()).unwrap();

        assert!(messages.is_empty());
    }

    #[test]
    fn anthropic_empty_messages_array() {
        let body = json!({
            "messages": []
        });

        let adapter = AnthropicAdapter::new();
        let messages = adapter.parse_request(body.to_string().as_bytes()).unwrap();

        assert!(messages.is_empty());
    }

    #[test]
    fn anthropic_empty_messages_with_system() {
        let body = json!({
            "system": "You are helpful",
            "messages": []
        });

        let adapter = AnthropicAdapter::new();
        let messages = adapter.parse_request(body.to_string().as_bytes()).unwrap();

        // System message is still extracted even with empty messages array
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].content, "You are helpful");
    }

    // ---------------------------------------------------------------
    // Test 14: Anthropic tool_result block -> parsed as Tool role message
    //          with tool_call_id and content
    // ---------------------------------------------------------------

    #[test]
    fn anthropic_tool_result_parsed_as_tool_message() {
        let body = json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "file contents here"
                        }
                    ]
                }
            ]
        });

        let adapter = AnthropicAdapter::new();
        let messages = adapter.parse_request(body.to_string().as_bytes()).unwrap();

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::Tool);
        assert_eq!(messages[0].content, "file contents here");
        assert_eq!(messages[0].tool_call_id, Some("toolu_1".to_string()));
    }

    // ---------------------------------------------------------------
    // Additional: all messages get TrustLevel::Trusted by default
    // ---------------------------------------------------------------

    #[test]
    fn openai_all_messages_default_to_trusted() {
        let body = json!({
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "usr"},
                {"role": "assistant", "content": "ast"},
                {"role": "tool", "tool_call_id": "call_1", "content": "tool result"}
            ]
        });

        let adapter = OpenAiAdapter::new();
        let messages = adapter.parse_request(body.to_string().as_bytes()).unwrap();

        for msg in &messages {
            assert_eq!(msg.trust, TrustLevel::Trusted);
        }
    }

    #[test]
    fn anthropic_all_messages_default_to_trusted() {
        let body = json!({
            "system": "sys",
            "messages": [
                {"role": "user", "content": "usr"},
                {"role": "assistant", "content": "ast"}
            ]
        });

        let adapter = AnthropicAdapter::new();
        let messages = adapter.parse_request(body.to_string().as_bytes()).unwrap();

        for msg in &messages {
            assert_eq!(msg.trust, TrustLevel::Trusted);
        }
    }

    // ---------------------------------------------------------------
    // Additional: OpenAI tool result messages
    // ---------------------------------------------------------------

    #[test]
    fn openai_tool_result_message_parsed() {
        let body = json!({
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "file contents"
                }
            ]
        });

        let adapter = OpenAiAdapter::new();
        let messages = adapter.parse_request(body.to_string().as_bytes()).unwrap();

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::Tool);
        assert_eq!(messages[0].content, "file contents");
        assert_eq!(messages[0].tool_call_id, Some("call_1".to_string()));
    }

    // ---------------------------------------------------------------
    // Additional: Provider type identification
    // ---------------------------------------------------------------

    #[test]
    fn provider_type_identification() {
        let openai = OpenAiAdapter::new();
        assert_eq!(openai.provider_type(), ProviderType::OpenAi);

        let anthropic = AnthropicAdapter::new();
        assert_eq!(anthropic.provider_type(), ProviderType::Anthropic);
    }

    // ---------------------------------------------------------------
    // Additional: Factory function
    // ---------------------------------------------------------------

    #[test]
    fn adapter_for_creates_correct_type() {
        let openai = adapter_for(ProviderType::OpenAi);
        assert_eq!(openai.provider_type(), ProviderType::OpenAi);

        let anthropic = adapter_for(ProviderType::Anthropic);
        assert_eq!(anthropic.provider_type(), ProviderType::Anthropic);
    }

    // ---------------------------------------------------------------
    // Additional: Invalid JSON body
    // ---------------------------------------------------------------

    #[test]
    fn openai_invalid_json_returns_error() {
        let adapter = OpenAiAdapter::new();
        let result = adapter.parse_request(b"not json at all");
        assert!(result.is_err());
        match result.unwrap_err() {
            ProviderError::InvalidJson(_) => {}
            other => panic!("expected InvalidJson, got: {other:?}"),
        }
    }

    #[test]
    fn anthropic_invalid_json_returns_error() {
        let adapter = AnthropicAdapter::new();
        let result = adapter.parse_request(b"not json at all");
        assert!(result.is_err());
        match result.unwrap_err() {
            ProviderError::InvalidJson(_) => {}
            other => panic!("expected InvalidJson, got: {other:?}"),
        }
    }

    // ---------------------------------------------------------------
    // Additional: Missing messages field
    // ---------------------------------------------------------------

    #[test]
    fn openai_missing_messages_field_returns_error() {
        let body = json!({"model": "gpt-4o"});
        let adapter = OpenAiAdapter::new();
        let result = adapter.parse_request(body.to_string().as_bytes());
        assert!(result.is_err());
        match result.unwrap_err() {
            ProviderError::MissingField(field) => assert_eq!(field, "messages"),
            other => panic!("expected MissingField, got: {other:?}"),
        }
    }

    #[test]
    fn anthropic_missing_messages_field_returns_error() {
        let body = json!({"system": "hello"});
        let adapter = AnthropicAdapter::new();
        let result = adapter.parse_request(body.to_string().as_bytes());
        assert!(result.is_err());
        match result.unwrap_err() {
            ProviderError::MissingField(field) => assert_eq!(field, "messages"),
            other => panic!("expected MissingField, got: {other:?}"),
        }
    }

    // ---------------------------------------------------------------
    // Additional: OpenAI invalid role
    // ---------------------------------------------------------------

    #[test]
    fn openai_invalid_role_returns_error() {
        let body = json!({
            "messages": [
                {"role": "unknown_role", "content": "hi"}
            ]
        });
        let adapter = OpenAiAdapter::new();
        let result = adapter.parse_request(body.to_string().as_bytes());
        assert!(result.is_err());
        match result.unwrap_err() {
            ProviderError::InvalidRole(role) => assert_eq!(role, "unknown_role"),
            other => panic!("expected InvalidRole, got: {other:?}"),
        }
    }

    // ---------------------------------------------------------------
    // Additional: Multiple tool calls in single assistant message
    // ---------------------------------------------------------------

    #[test]
    fn openai_multiple_tool_calls_in_one_message() {
        let body = json!({
            "messages": [
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": "{\"path\": \"/tmp/a\"}"
                            }
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": "{\"path\": \"/tmp/b\", \"content\": \"data\"}"
                            }
                        }
                    ]
                }
            ]
        });

        let adapter = OpenAiAdapter::new();
        let messages = adapter.parse_request(body.to_string().as_bytes()).unwrap();

        assert_eq!(messages[0].tool_calls.len(), 2);
        assert_eq!(messages[0].tool_calls[0].name, "read_file");
        assert_eq!(messages[0].tool_calls[1].name, "write_file");
    }

    #[test]
    fn anthropic_multiple_tool_use_blocks_in_one_message() {
        let body = json!({
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I will do both"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "read_file",
                            "input": {"path": "/tmp/a"}
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_2",
                            "name": "write_file",
                            "input": {"path": "/tmp/b", "content": "data"}
                        }
                    ]
                }
            ]
        });

        let adapter = AnthropicAdapter::new();
        let messages = adapter.parse_request(body.to_string().as_bytes()).unwrap();

        assert_eq!(messages[0].tool_calls.len(), 2);
        assert_eq!(messages[0].tool_calls[0].name, "read_file");
        assert_eq!(messages[0].tool_calls[1].name, "write_file");
        assert_eq!(messages[0].content, "I will do both");
    }

    // ---------------------------------------------------------------
    // Additional: Full conversation round-trip (OpenAI)
    // ---------------------------------------------------------------

    #[test]
    fn openai_full_conversation_with_tool_use() {
        let body = json!({
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Read /tmp/foo"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": "{\"path\": \"/tmp/foo\"}"
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "file contents"
                }
            ]
        });

        let adapter = OpenAiAdapter::new();
        let messages = adapter.parse_request(body.to_string().as_bytes()).unwrap();

        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[2].role, Role::Assistant);
        assert_eq!(messages[2].tool_calls.len(), 1);
        assert_eq!(messages[3].role, Role::Tool);
        assert_eq!(messages[3].tool_call_id, Some("call_1".to_string()));
        assert_eq!(messages[3].content, "file contents");
    }

    // ---------------------------------------------------------------
    // Additional: Full conversation round-trip (Anthropic)
    // ---------------------------------------------------------------

    #[test]
    fn anthropic_full_conversation_with_tool_use() {
        let body = json!({
            "system": "You are helpful",
            "messages": [
                {"role": "user", "content": "Read /tmp/foo"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me read that"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "read_file",
                            "input": {"path": "/tmp/foo"}
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "file contents"
                        }
                    ]
                }
            ]
        });

        let adapter = AnthropicAdapter::new();
        let messages = adapter.parse_request(body.to_string().as_bytes()).unwrap();

        // system + user + assistant + tool_result (as Tool message)
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].content, "You are helpful");
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[1].content, "Read /tmp/foo");
        assert_eq!(messages[2].role, Role::Assistant);
        assert_eq!(messages[2].content, "Let me read that");
        assert_eq!(messages[2].tool_calls.len(), 1);
        assert_eq!(messages[3].role, Role::Tool);
        assert_eq!(messages[3].tool_call_id, Some("toolu_1".to_string()));
        assert_eq!(messages[3].content, "file contents");
    }
}
