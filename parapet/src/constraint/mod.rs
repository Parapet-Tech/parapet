// Constraint DSL evaluator -- defined in M1.9
//
// Evaluates tool calls in LLM responses against the contract defined
// in parapet.yaml. Each tool call is checked against its ToolConfig
// (or the _default fallback). Argument constraints use AND semantics:
// all specified predicates on an argument must pass.

use crate::config::{ArgumentConstraints, Config, ToolConfig};
use crate::message::ToolCall;

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

/// Verdict for a single tool call after constraint evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolCallVerdict {
    Allow,
    Block { tool_name: String, reason: String },
}

/// Evaluates tool calls against a Config's tool policies and constraints.
pub trait ConstraintEvaluator: Send + Sync {
    fn evaluate_tool_calls(
        &self,
        tool_calls: &[ToolCall],
        config: &Config,
    ) -> Vec<ToolCallVerdict>;
}

// ---------------------------------------------------------------------------
// DSL implementation
// ---------------------------------------------------------------------------

/// Constraint evaluator that applies the 9-predicate DSL from parapet.yaml.
pub struct DslConstraintEvaluator;

impl DslConstraintEvaluator {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DslConstraintEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintEvaluator for DslConstraintEvaluator {
    fn evaluate_tool_calls(
        &self,
        tool_calls: &[ToolCall],
        config: &Config,
    ) -> Vec<ToolCallVerdict> {
        tool_calls
            .iter()
            .map(|tc| evaluate_single(tc, config))
            .collect()
    }
}

/// Evaluate a single tool call against the config.
fn evaluate_single(tool_call: &ToolCall, config: &Config) -> ToolCallVerdict {
    let tool_config = config
        .policy
        .tools
        .get(&tool_call.name)
        .or_else(|| config.policy.tools.get("_default"));

    let tool_config = match tool_config {
        Some(tc) => tc,
        // No config and no _default -> allow (no policy means no restriction)
        None => return ToolCallVerdict::Allow,
    };

    if !tool_config.allowed {
        return ToolCallVerdict::Block {
            tool_name: tool_call.name.clone(),
            reason: "tool not allowed".to_string(),
        };
    }

    // Tool is allowed -- evaluate argument constraints
    evaluate_constraints(tool_call, tool_config)
}

/// Evaluate all argument constraints for an allowed tool call.
fn evaluate_constraints(tool_call: &ToolCall, tool_config: &ToolConfig) -> ToolCallVerdict {
    let args = &tool_call.arguments;

    for (arg_name, constraints) in &tool_config.constraints {
        // Look up the argument value in the tool call's JSON arguments
        let arg_value = args.get(arg_name);

        match arg_value {
            None | Some(serde_json::Value::Null) => {
                // Constraint on an argument that is absent or null -> blocked
                return ToolCallVerdict::Block {
                    tool_name: tool_call.name.clone(),
                    reason: format!("required argument \"{}\" is missing", arg_name),
                };
            }
            Some(value) => {
                if let Some(reason) = check_predicates(arg_name, value, constraints) {
                    return ToolCallVerdict::Block {
                        tool_name: tool_call.name.clone(),
                        reason,
                    };
                }
            }
        }
    }

    ToolCallVerdict::Allow
}

/// Check all 9 predicates for a single argument value. Returns `Some(reason)`
/// if any predicate fails, `None` if all pass.
fn check_predicates(
    arg_name: &str,
    value: &serde_json::Value,
    constraints: &ArgumentConstraints,
) -> Option<String> {
    // 1. type check
    if let Some(ref expected_type) = constraints.type_check {
        let actual_type = json_type_name(value);
        if actual_type != expected_type.as_str() {
            return Some(format!(
                "argument \"{}\" expected type \"{}\", got \"{}\"",
                arg_name, expected_type, actual_type
            ));
        }
    }

    // For string predicates, extract the string value
    let string_value = value.as_str();

    // 2. starts_with
    if let Some(ref prefix) = constraints.starts_with {
        match string_value {
            Some(s) if s.starts_with(prefix.as_str()) => {}
            Some(s) => {
                return Some(format!(
                    "argument \"{}\" value \"{}\" does not start with \"{}\"",
                    arg_name, s, prefix
                ));
            }
            None => {
                return Some(format!(
                    "argument \"{}\" expected string for starts_with check",
                    arg_name
                ));
            }
        }
    }

    // 3. not_contains
    if let Some(ref forbidden) = constraints.not_contains {
        match string_value {
            Some(s) => {
                for substring in forbidden {
                    if s.contains(substring.as_str()) {
                        return Some(format!(
                            "argument \"{}\" contains forbidden substring \"{}\"",
                            arg_name, substring
                        ));
                    }
                }
            }
            None => {
                return Some(format!(
                    "argument \"{}\" expected string for not_contains check",
                    arg_name
                ));
            }
        }
    }

    // 4. matches (pre-compiled regex)
    if let Some(ref pattern) = constraints.matches {
        match string_value {
            Some(s) if pattern.is_match(s) => {}
            Some(s) => {
                return Some(format!(
                    "argument \"{}\" value \"{}\" does not match pattern \"{}\"",
                    arg_name, s, pattern.pattern
                ));
            }
            None => {
                return Some(format!(
                    "argument \"{}\" expected string for matches check",
                    arg_name
                ));
            }
        }
    }

    // 5. one_of
    if let Some(ref allowed) = constraints.one_of {
        match string_value {
            Some(s) if allowed.iter().any(|a| a == s) => {}
            Some(s) => {
                return Some(format!(
                    "argument \"{}\" value \"{}\" not in allowed list",
                    arg_name, s
                ));
            }
            None => {
                return Some(format!(
                    "argument \"{}\" expected string for one_of check",
                    arg_name
                ));
            }
        }
    }

    // 6. max_length
    if let Some(max_len) = constraints.max_length {
        match string_value {
            Some(s) if s.len() <= max_len => {}
            Some(s) => {
                return Some(format!(
                    "argument \"{}\" length {} exceeds maximum {}",
                    arg_name,
                    s.len(),
                    max_len
                ));
            }
            None => {
                return Some(format!(
                    "argument \"{}\" expected string for max_length check",
                    arg_name
                ));
            }
        }
    }

    // 7. min (numeric)
    if let Some(min_val) = constraints.min {
        match value.as_f64() {
            Some(n) if n >= min_val => {}
            Some(n) => {
                return Some(format!(
                    "argument \"{}\" value {} is less than minimum {}",
                    arg_name, n, min_val
                ));
            }
            None => {
                return Some(format!(
                    "argument \"{}\" expected number for min check",
                    arg_name
                ));
            }
        }
    }

    // 8. max (numeric)
    if let Some(max_val) = constraints.max {
        match value.as_f64() {
            Some(n) if n <= max_val => {}
            Some(n) => {
                return Some(format!(
                    "argument \"{}\" value {} exceeds maximum {}",
                    arg_name, n, max_val
                ));
            }
            None => {
                return Some(format!(
                    "argument \"{}\" expected number for max check",
                    arg_name
                ));
            }
        }
    }

    // 9. url_host (parse as URL, exact host match)
    if let Some(ref allowed_hosts) = constraints.url_host {
        match string_value {
            Some(s) => match url::Url::parse(s) {
                Ok(parsed) => match parsed.host_str() {
                    Some(host) if allowed_hosts.iter().any(|h| h == host) => {}
                    Some(host) => {
                        return Some(format!(
                            "argument \"{}\" URL host \"{}\" not in allowed hosts",
                            arg_name, host
                        ));
                    }
                    None => {
                        return Some(format!(
                            "argument \"{}\" URL has no host",
                            arg_name
                        ));
                    }
                },
                Err(_) => {
                    return Some(format!(
                        "argument \"{}\" value is not a valid URL",
                        arg_name
                    ));
                }
            },
            None => {
                return Some(format!(
                    "argument \"{}\" expected string for url_host check",
                    arg_name
                ));
            }
        }
    }

    None
}

/// Map a serde_json::Value to the type name used in constraint configs.
fn json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::String(_) => "string",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
        serde_json::Value::Null => "null",
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;
    use serde_json::json;
    use std::collections::HashMap;

    // ---------------------------------------------------------------
    // Test helpers
    // ---------------------------------------------------------------

    /// Build a minimal Config with the given tools map.
    fn config_with_tools(tools: HashMap<String, ToolConfig>) -> Config {
        Config {
            policy: PolicyConfig {
                version: "v1".to_string(),
                tools,
                block_patterns: Vec::new(),
                canary_tokens: Vec::new(),
                sensitive_patterns: Vec::new(),
                untrusted_content_policy: ContentPolicy::default(),
                trust: TrustConfig::default(),
                layers: LayerConfigs::default(),
            },
            runtime: RuntimeConfig {
                engine: EngineConfig::default(),
                environment: String::new(),
            },
            contract_hash: "sha256:test".to_string(),
        }
    }

    fn tool_blocked() -> ToolConfig {
        ToolConfig {
            allowed: false,
            trust: None,
            constraints: HashMap::new(),
            result_policy: None,
        }
    }

    fn tool_allowed() -> ToolConfig {
        ToolConfig {
            allowed: true,
            trust: None,
            constraints: HashMap::new(),
            result_policy: None,
        }
    }

    fn tool_allowed_with_constraints(constraints: HashMap<String, ArgumentConstraints>) -> ToolConfig {
        ToolConfig {
            allowed: true,
            trust: None,
            constraints,
            result_policy: None,
        }
    }

    fn make_tool_call(name: &str, arguments: serde_json::Value) -> ToolCall {
        ToolCall {
            id: format!("call_{}", name),
            name: name.to_string(),
            arguments,
        }
    }

    fn evaluator() -> DslConstraintEvaluator {
        DslConstraintEvaluator::new()
    }

    // ---------------------------------------------------------------
    // 1. _default: allowed: false blocks unknown tool
    // ---------------------------------------------------------------

    #[test]
    fn default_blocked_rejects_unknown_tool() {
        let mut tools = HashMap::new();
        tools.insert("_default".to_string(), tool_blocked());
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("unknown_tool", json!({}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts.len(), 1);
        match &verdicts[0] {
            ToolCallVerdict::Block { tool_name, reason } => {
                assert_eq!(tool_name, "unknown_tool");
                assert!(reason.contains("not allowed"), "reason: {}", reason);
            }
            ToolCallVerdict::Allow => panic!("expected Block, got Allow"),
        }
    }

    // ---------------------------------------------------------------
    // 2. allowed: true with no constraints passes
    // ---------------------------------------------------------------

    #[test]
    fn allowed_tool_with_no_constraints_passes() {
        let mut tools = HashMap::new();
        tools.insert("my_tool".to_string(), tool_allowed());
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("my_tool", json!({"any": "arg"}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts, vec![ToolCallVerdict::Allow]);
    }

    // ---------------------------------------------------------------
    // 3. starts_with pass and fail
    // ---------------------------------------------------------------

    #[test]
    fn starts_with_pass() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "path".to_string(),
            ArgumentConstraints {
                starts_with: Some("/home/user/".to_string()),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("read_file".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("read_file", json!({"path": "/home/user/file.txt"}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts, vec![ToolCallVerdict::Allow]);
    }

    #[test]
    fn starts_with_fail() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "path".to_string(),
            ArgumentConstraints {
                starts_with: Some("/home/user/".to_string()),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("read_file".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("read_file", json!({"path": "/etc/passwd"}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts.len(), 1);
        match &verdicts[0] {
            ToolCallVerdict::Block { tool_name, reason } => {
                assert_eq!(tool_name, "read_file");
                assert!(reason.contains("does not start with"), "reason: {}", reason);
            }
            ToolCallVerdict::Allow => panic!("expected Block"),
        }
    }

    // ---------------------------------------------------------------
    // 4. not_contains pass and fail
    // ---------------------------------------------------------------

    #[test]
    fn not_contains_pass() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "path".to_string(),
            ArgumentConstraints {
                not_contains: Some(vec!["../".to_string(), "..\\".to_string()]),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("read_file".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("read_file", json!({"path": "/home/user/file.txt"}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts, vec![ToolCallVerdict::Allow]);
    }

    #[test]
    fn not_contains_fail() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "path".to_string(),
            ArgumentConstraints {
                not_contains: Some(vec!["../".to_string(), "..\\".to_string()]),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("read_file".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("read_file", json!({"path": "/home/user/../etc/passwd"}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts.len(), 1);
        match &verdicts[0] {
            ToolCallVerdict::Block { tool_name, reason } => {
                assert_eq!(tool_name, "read_file");
                assert!(reason.contains("forbidden substring"), "reason: {}", reason);
            }
            ToolCallVerdict::Allow => panic!("expected Block"),
        }
    }

    // ---------------------------------------------------------------
    // 5. url_host pass, fail, and evil subdomain
    // ---------------------------------------------------------------

    #[test]
    fn url_host_pass() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "url".to_string(),
            ArgumentConstraints {
                url_host: Some(vec!["example.com".to_string(), "api.example.com".to_string()]),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("fetch".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("fetch", json!({"url": "https://example.com/path"}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts, vec![ToolCallVerdict::Allow]);
    }

    #[test]
    fn url_host_fail() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "url".to_string(),
            ArgumentConstraints {
                url_host: Some(vec!["example.com".to_string()]),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("fetch".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("fetch", json!({"url": "https://evil.com/steal"}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts.len(), 1);
        match &verdicts[0] {
            ToolCallVerdict::Block { tool_name, reason } => {
                assert_eq!(tool_name, "fetch");
                assert!(reason.contains("not in allowed hosts"), "reason: {}", reason);
            }
            ToolCallVerdict::Allow => panic!("expected Block"),
        }
    }

    #[test]
    fn url_host_evil_subdomain_blocked() {
        // a.com.evil.com should NOT match allowed host "a.com"
        let mut constraints = HashMap::new();
        constraints.insert(
            "url".to_string(),
            ArgumentConstraints {
                url_host: Some(vec!["a.com".to_string()]),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("fetch".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("fetch", json!({"url": "https://a.com.evil.com/path"}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts.len(), 1);
        match &verdicts[0] {
            ToolCallVerdict::Block { tool_name, reason } => {
                assert_eq!(tool_name, "fetch");
                assert!(reason.contains("not in allowed hosts"), "reason: {}", reason);
            }
            ToolCallVerdict::Allow => panic!("expected Block -- evil subdomain should not match"),
        }
    }

    // ---------------------------------------------------------------
    // 6. url_host with non-URL argument -> blocked
    // ---------------------------------------------------------------

    #[test]
    fn url_host_non_url_blocked() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "url".to_string(),
            ArgumentConstraints {
                url_host: Some(vec!["example.com".to_string()]),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("fetch".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("fetch", json!({"url": "not a url at all"}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts.len(), 1);
        match &verdicts[0] {
            ToolCallVerdict::Block { tool_name, reason } => {
                assert_eq!(tool_name, "fetch");
                assert!(reason.contains("not a valid URL"), "reason: {}", reason);
            }
            ToolCallVerdict::Allow => panic!("expected Block"),
        }
    }

    // ---------------------------------------------------------------
    // 7. matches regex pass and fail
    // ---------------------------------------------------------------

    #[test]
    fn matches_regex_pass() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "name".to_string(),
            ArgumentConstraints {
                matches: Some(CompiledPattern::compile("^[a-z_]+$").unwrap()),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("create".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("create", json!({"name": "hello_world"}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts, vec![ToolCallVerdict::Allow]);
    }

    #[test]
    fn matches_regex_fail() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "name".to_string(),
            ArgumentConstraints {
                matches: Some(CompiledPattern::compile("^[a-z_]+$").unwrap()),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("create".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("create", json!({"name": "Hello World 123!"}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts.len(), 1);
        match &verdicts[0] {
            ToolCallVerdict::Block { tool_name, reason } => {
                assert_eq!(tool_name, "create");
                assert!(reason.contains("does not match pattern"), "reason: {}", reason);
            }
            ToolCallVerdict::Allow => panic!("expected Block"),
        }
    }

    // ---------------------------------------------------------------
    // 8. one_of pass and fail
    // ---------------------------------------------------------------

    #[test]
    fn one_of_pass() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "format".to_string(),
            ArgumentConstraints {
                one_of: Some(vec!["json".to_string(), "yaml".to_string(), "toml".to_string()]),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("export".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("export", json!({"format": "yaml"}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts, vec![ToolCallVerdict::Allow]);
    }

    #[test]
    fn one_of_fail() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "format".to_string(),
            ArgumentConstraints {
                one_of: Some(vec!["json".to_string(), "yaml".to_string()]),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("export".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("export", json!({"format": "xml"}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts.len(), 1);
        match &verdicts[0] {
            ToolCallVerdict::Block { tool_name, reason } => {
                assert_eq!(tool_name, "export");
                assert!(reason.contains("not in allowed list"), "reason: {}", reason);
            }
            ToolCallVerdict::Allow => panic!("expected Block"),
        }
    }

    // ---------------------------------------------------------------
    // 9. max_length pass and fail
    // ---------------------------------------------------------------

    #[test]
    fn max_length_pass() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "content".to_string(),
            ArgumentConstraints {
                max_length: Some(10),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("write".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("write", json!({"content": "short"}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts, vec![ToolCallVerdict::Allow]);
    }

    #[test]
    fn max_length_fail() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "content".to_string(),
            ArgumentConstraints {
                max_length: Some(5),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("write".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("write", json!({"content": "this is way too long"}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts.len(), 1);
        match &verdicts[0] {
            ToolCallVerdict::Block { tool_name, reason } => {
                assert_eq!(tool_name, "write");
                assert!(reason.contains("exceeds maximum"), "reason: {}", reason);
            }
            ToolCallVerdict::Allow => panic!("expected Block"),
        }
    }

    // ---------------------------------------------------------------
    // 10. min/max numeric bounds pass and fail
    // ---------------------------------------------------------------

    #[test]
    fn min_max_numeric_pass() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "count".to_string(),
            ArgumentConstraints {
                min: Some(1.0),
                max: Some(100.0),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("batch".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("batch", json!({"count": 50}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts, vec![ToolCallVerdict::Allow]);
    }

    #[test]
    fn min_numeric_fail() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "count".to_string(),
            ArgumentConstraints {
                min: Some(1.0),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("batch".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("batch", json!({"count": 0}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts.len(), 1);
        match &verdicts[0] {
            ToolCallVerdict::Block { tool_name, reason } => {
                assert_eq!(tool_name, "batch");
                assert!(reason.contains("less than minimum"), "reason: {}", reason);
            }
            ToolCallVerdict::Allow => panic!("expected Block"),
        }
    }

    #[test]
    fn max_numeric_fail() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "count".to_string(),
            ArgumentConstraints {
                max: Some(100.0),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("batch".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("batch", json!({"count": 200}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts.len(), 1);
        match &verdicts[0] {
            ToolCallVerdict::Block { tool_name, reason } => {
                assert_eq!(tool_name, "batch");
                assert!(reason.contains("exceeds maximum"), "reason: {}", reason);
            }
            ToolCallVerdict::Allow => panic!("expected Block"),
        }
    }

    // ---------------------------------------------------------------
    // 11. type check: string, number, boolean -- pass and fail
    // ---------------------------------------------------------------

    #[test]
    fn type_check_string_pass() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "name".to_string(),
            ArgumentConstraints {
                type_check: Some("string".to_string()),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("greet".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("greet", json!({"name": "Alice"}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts, vec![ToolCallVerdict::Allow]);
    }

    #[test]
    fn type_check_number_pass() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "count".to_string(),
            ArgumentConstraints {
                type_check: Some("number".to_string()),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("process".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("process", json!({"count": 42}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts, vec![ToolCallVerdict::Allow]);
    }

    #[test]
    fn type_check_boolean_pass() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "verbose".to_string(),
            ArgumentConstraints {
                type_check: Some("boolean".to_string()),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("run".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("run", json!({"verbose": true}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts, vec![ToolCallVerdict::Allow]);
    }

    #[test]
    fn type_check_fail_number_when_string_expected() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "name".to_string(),
            ArgumentConstraints {
                type_check: Some("string".to_string()),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("greet".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("greet", json!({"name": 42}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts.len(), 1);
        match &verdicts[0] {
            ToolCallVerdict::Block { tool_name, reason } => {
                assert_eq!(tool_name, "greet");
                assert!(reason.contains("expected type \"string\""), "reason: {}", reason);
                assert!(reason.contains("got \"number\""), "reason: {}", reason);
            }
            ToolCallVerdict::Allow => panic!("expected Block"),
        }
    }

    // ---------------------------------------------------------------
    // 12. Multiple tool calls: one blocked, others pass
    // ---------------------------------------------------------------

    #[test]
    fn multiple_tool_calls_mixed_verdicts() {
        let mut tools = HashMap::new();
        tools.insert("_default".to_string(), tool_blocked());
        tools.insert("allowed_tool".to_string(), tool_allowed());
        let config = config_with_tools(tools);

        let calls = vec![
            make_tool_call("allowed_tool", json!({})),
            make_tool_call("blocked_tool", json!({})),
            make_tool_call("allowed_tool", json!({"extra": "arg"})),
        ];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts.len(), 3);
        assert_eq!(verdicts[0], ToolCallVerdict::Allow);
        match &verdicts[1] {
            ToolCallVerdict::Block { tool_name, .. } => assert_eq!(tool_name, "blocked_tool"),
            ToolCallVerdict::Allow => panic!("expected Block for blocked_tool"),
        }
        assert_eq!(verdicts[2], ToolCallVerdict::Allow);
    }

    // ---------------------------------------------------------------
    // 13. Tool call with argument absent from constraints -> passes
    // ---------------------------------------------------------------

    #[test]
    fn extra_argument_not_in_constraints_passes() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "path".to_string(),
            ArgumentConstraints {
                type_check: Some("string".to_string()),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("read_file".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        // "extra_arg" is not mentioned in constraints -- should be ignored
        let calls = vec![make_tool_call(
            "read_file",
            json!({"path": "/tmp/file.txt", "extra_arg": "whatever"}),
        )];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts, vec![ToolCallVerdict::Allow]);
    }

    // ---------------------------------------------------------------
    // 14. Constraint on argument absent from tool call -> blocked
    // ---------------------------------------------------------------

    #[test]
    fn missing_required_argument_blocked() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "path".to_string(),
            ArgumentConstraints {
                type_check: Some("string".to_string()),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("read_file".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        // Tool call has no "path" argument
        let calls = vec![make_tool_call("read_file", json!({"something_else": "value"}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts.len(), 1);
        match &verdicts[0] {
            ToolCallVerdict::Block { tool_name, reason } => {
                assert_eq!(tool_name, "read_file");
                assert!(reason.contains("missing"), "reason: {}", reason);
                assert!(reason.contains("path"), "reason: {}", reason);
            }
            ToolCallVerdict::Allow => panic!("expected Block"),
        }
    }

    // ---------------------------------------------------------------
    // 15. Tool call argument is null when constraint expects string -> blocked
    // ---------------------------------------------------------------

    #[test]
    fn null_argument_with_constraint_blocked() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "path".to_string(),
            ArgumentConstraints {
                type_check: Some("string".to_string()),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("read_file".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("read_file", json!({"path": null}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts.len(), 1);
        match &verdicts[0] {
            ToolCallVerdict::Block { tool_name, reason } => {
                assert_eq!(tool_name, "read_file");
                assert!(reason.contains("missing"), "reason: {}", reason);
            }
            ToolCallVerdict::Allow => panic!("expected Block"),
        }
    }

    // ---------------------------------------------------------------
    // Boundary: min/max at exact boundary values
    // ---------------------------------------------------------------

    #[test]
    fn min_at_exact_boundary_passes() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "count".to_string(),
            ArgumentConstraints {
                min: Some(1.0),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("batch".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("batch", json!({"count": 1}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts, vec![ToolCallVerdict::Allow]);
    }

    #[test]
    fn max_at_exact_boundary_passes() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "count".to_string(),
            ArgumentConstraints {
                max: Some(100.0),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("batch".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("batch", json!({"count": 100}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts, vec![ToolCallVerdict::Allow]);
    }

    #[test]
    fn max_length_at_exact_boundary_passes() {
        let mut constraints = HashMap::new();
        constraints.insert(
            "text".to_string(),
            ArgumentConstraints {
                max_length: Some(5),
                ..Default::default()
            },
        );
        let mut tools = HashMap::new();
        tools.insert("echo".to_string(), tool_allowed_with_constraints(constraints));
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("echo", json!({"text": "abcde"}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts, vec![ToolCallVerdict::Allow]);
    }

    // ---------------------------------------------------------------
    // No config and no _default -> allow
    // ---------------------------------------------------------------

    #[test]
    fn no_config_no_default_allows() {
        let mut tools = HashMap::new();
        tools.insert("some_other_tool".to_string(), tool_allowed());
        let config = config_with_tools(tools);

        let calls = vec![make_tool_call("unknown_tool", json!({}))];
        let verdicts = evaluator().evaluate_tool_calls(&calls, &config);

        assert_eq!(verdicts, vec![ToolCallVerdict::Allow]);
    }

    // ---------------------------------------------------------------
    // Empty tool calls vec -> empty verdicts
    // ---------------------------------------------------------------

    #[test]
    fn empty_tool_calls_returns_empty_verdicts() {
        let config = config_with_tools(HashMap::new());
        let verdicts = evaluator().evaluate_tool_calls(&[], &config);
        assert!(verdicts.is_empty());
    }
}
