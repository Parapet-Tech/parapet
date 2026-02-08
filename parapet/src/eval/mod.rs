// Eval harness — runs attack/benign samples through the pipeline,
// reports per-layer TP/FP/FN/TN/precision/recall/F1.
//
// Key insight: we're testing the *firewall*, not the LLM.
// A MockHttpSender returns canned responses; the eval checks
// whether parapet correctly blocks/allows/redacts.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use axum::http::{HeaderMap, Method, StatusCode};
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::constraint::DslConstraintEvaluator;
use crate::engine::{
    DefaultProviderRegistry, EngineDeps, EngineUpstreamClient, HttpBody, HttpError, HttpRequest,
    HttpResponse, HttpSender, UpstreamResolver,
};
use crate::layers::l3_inbound::DefaultInboundScanner;
use crate::layers::l5a::L5aScanner;
use crate::normalize::L0Normalizer;
use crate::proxy::{Provider, ProxyRequest, UpstreamClient};
use crate::trust::RoleTrustAssigner;

// ---------------------------------------------------------------------------
// Dataset types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct EvalCase {
    pub id: String,
    pub layer: String,
    pub label: String,
    pub description: String,
    pub content: String,
    #[serde(default)]
    pub mock_tool_calls: Vec<MockToolCall>,
    #[serde(default)]
    pub mock_content: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MockToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

// ---------------------------------------------------------------------------
// Eval result and report
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct EvalResult {
    pub case_id: String,
    pub layer: String,
    pub label: String,
    pub expected: String,
    pub actual: String,
    pub correct: bool,
    pub detail: String,
}

#[derive(Debug, Default, Serialize)]
pub struct LayerMetrics {
    pub layer: String,
    pub tp: usize,
    pub fp: usize,
    pub fn_count: usize,
    pub tn: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub total: usize,
}

#[derive(Debug, Serialize)]
pub struct EvalReport {
    pub layers: Vec<LayerMetrics>,
    pub results: Vec<EvalResult>,
    pub total_cases: usize,
    pub total_correct: usize,
    pub accuracy: f64,
}

// ---------------------------------------------------------------------------
// Mock HTTP sender
// ---------------------------------------------------------------------------

pub struct MockHttpSender {
    response_body: Mutex<Bytes>,
}

impl MockHttpSender {
    pub fn new() -> Self {
        let default = serde_json::json!({
            "choices": [{"message": {"content": "ok", "tool_calls": []}}]
        });
        Self {
            response_body: Mutex::new(Bytes::from(serde_json::to_vec(&default).unwrap())),
        }
    }

    pub fn set_response(&self, body: serde_json::Value) {
        let bytes = serde_json::to_vec(&body).unwrap();
        *self.response_body.lock().unwrap() = Bytes::from(bytes);
    }
}

#[async_trait]
impl HttpSender for MockHttpSender {
    async fn send(&self, _request: HttpRequest) -> Result<HttpResponse, HttpError> {
        let body = self.response_body.lock().unwrap().clone();
        Ok(HttpResponse {
            status: StatusCode::OK,
            headers: HeaderMap::new(),
            body: HttpBody::Full(body),
        })
    }
}

// ---------------------------------------------------------------------------
// Mock resolver (never actually connects)
// ---------------------------------------------------------------------------

struct EvalResolver;

impl UpstreamResolver for EvalResolver {
    fn base_url(&self, _provider: Provider) -> String {
        "https://eval.local".to_string()
    }
}

// ---------------------------------------------------------------------------
// Dataset loading
// ---------------------------------------------------------------------------

pub fn load_dataset(dir: &Path) -> Result<Vec<EvalCase>, String> {
    let mut cases = Vec::new();

    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("failed to read dataset directory {}: {e}", dir.display()))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("failed to read directory entry: {e}"))?;
        let path = entry.path();

        let ext = path.extension().and_then(|e| e.to_str());
        if ext != Some("yaml") && ext != Some("yml") {
            continue;
        }

        // Skip config files
        if let Some(name) = path.file_name().and_then(|f| f.to_str()) {
            if name.starts_with("eval_config") {
                continue;
            }
        }

        let content = std::fs::read_to_string(&path)
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;

        let file_cases: Vec<EvalCase> = serde_yaml::from_str(&content)
            .map_err(|e| format!("failed to parse {}: {e}", path.display()))?;

        cases.extend(file_cases);
    }

    cases.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(cases)
}

// ---------------------------------------------------------------------------
// Build engine with mock HTTP sender
// ---------------------------------------------------------------------------

pub fn build_eval_engine(config: Arc<Config>) -> (EngineUpstreamClient, Arc<MockHttpSender>) {
    let mock = Arc::new(MockHttpSender::new());

    let deps = EngineDeps {
        config,
        http: mock.clone(),
        resolver: Arc::new(EvalResolver),
        registry: Arc::new(DefaultProviderRegistry),
        normalizer: Arc::new(L0Normalizer::new()),
        trust_assigner: Arc::new(RoleTrustAssigner),
        inbound_scanner: Arc::new(DefaultInboundScanner::new()),
        constraint_evaluator: Arc::new(DslConstraintEvaluator::new()),
        output_scanner: Arc::new(L5aScanner),
    };

    (EngineUpstreamClient::new_with(deps), mock)
}

// ---------------------------------------------------------------------------
// Run eval
// ---------------------------------------------------------------------------

pub async fn run_eval(
    cases: &[EvalCase],
    engine: &EngineUpstreamClient,
    mock: &MockHttpSender,
) -> Vec<EvalResult> {
    let mut results = Vec::with_capacity(cases.len());

    for case in cases {
        // Set mock response for this case
        mock.set_response(build_mock_response(case));

        // Build proxy request
        let request = build_proxy_request(case);

        // Run through engine
        let response = engine.forward(Provider::OpenAi, request).await;

        // Collect response data
        let (status, body) = match response {
            Ok(resp) => {
                let status = resp.status;
                let body_bytes = axum::body::to_bytes(resp.body, 10 * 1024 * 1024)
                    .await
                    .unwrap_or_default();
                (status, String::from_utf8_lossy(&body_bytes).to_string())
            }
            Err(e) => {
                results.push(EvalResult {
                    case_id: case.id.clone(),
                    layer: case.layer.clone(),
                    label: case.label.clone(),
                    expected: expected_action(&case.label),
                    actual: "error".to_string(),
                    correct: false,
                    detail: format!("engine error: {e}"),
                });
                continue;
            }
        };

        // Determine verdict
        let (actual, detail) = determine_verdict(case, status, &body);
        let expected = expected_action(&case.label);
        let correct = actual == expected;

        results.push(EvalResult {
            case_id: case.id.clone(),
            layer: case.layer.clone(),
            label: case.label.clone(),
            expected,
            actual,
            correct,
            detail,
        });
    }

    results
}

// ---------------------------------------------------------------------------
// Compute metrics
// ---------------------------------------------------------------------------

pub fn compute_metrics(results: &[EvalResult]) -> EvalReport {
    let mut by_layer: HashMap<String, (usize, usize, usize, usize)> = HashMap::new();
    let total_cases = results.len();
    let total_correct = results.iter().filter(|r| r.correct).count();

    for r in results {
        let entry = by_layer.entry(r.layer.clone()).or_default();
        match (r.label.as_str(), r.actual.as_str()) {
            ("malicious", "blocked") => entry.0 += 1, // TP
            ("benign", "blocked") => entry.1 += 1,    // FP
            ("malicious", "allowed") => entry.2 += 1,  // FN
            ("benign", "allowed") => entry.3 += 1,    // TN
            _ => {} // error cases
        }
    }

    let mut layers: Vec<LayerMetrics> = by_layer
        .into_iter()
        .map(|(layer, (tp, fp, fn_count, tn))| {
            let precision = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            };
            let recall = if tp + fn_count > 0 {
                tp as f64 / (tp + fn_count) as f64
            } else {
                0.0
            };
            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };
            LayerMetrics {
                layer,
                tp,
                fp,
                fn_count,
                tn,
                precision,
                recall,
                f1,
                total: tp + fp + fn_count + tn,
            }
        })
        .collect();

    layers.sort_by(|a, b| a.layer.cmp(&b.layer));

    let accuracy = if total_cases > 0 {
        total_correct as f64 / total_cases as f64
    } else {
        0.0
    };

    EvalReport {
        layers,
        results: Vec::new(), // populated by caller if needed
        total_cases,
        total_correct,
        accuracy,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_mock_response(case: &EvalCase) -> serde_json::Value {
    if !case.mock_tool_calls.is_empty() {
        let tool_calls: Vec<serde_json::Value> = case
            .mock_tool_calls
            .iter()
            .map(|tc| {
                serde_json::json!({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                })
            })
            .collect();

        serde_json::json!({
            "choices": [{"message": {"content": null, "tool_calls": tool_calls}}]
        })
    } else if let Some(ref content) = case.mock_content {
        serde_json::json!({
            "choices": [{"message": {"content": content, "tool_calls": []}}]
        })
    } else {
        serde_json::json!({
            "choices": [{"message": {"content": "ok", "tool_calls": []}}]
        })
    }
}

fn build_proxy_request(case: &EvalCase) -> ProxyRequest {
    let body = serde_json::json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": case.content}]
    });

    let body_bytes = serde_json::to_vec(&body).unwrap();
    let uri: axum::http::Uri = "/v1/chat/completions".parse().unwrap();

    let mut headers = HeaderMap::new();
    headers.insert("content-type", "application/json".parse().unwrap());

    ProxyRequest {
        method: Method::POST,
        uri,
        headers,
        body: Bytes::from(body_bytes),
    }
}

fn expected_action(label: &str) -> String {
    match label {
        "malicious" => "blocked".to_string(),
        "benign" => "allowed".to_string(),
        other => format!("unknown:{other}"),
    }
}

fn determine_verdict(case: &EvalCase, status: StatusCode, body: &str) -> (String, String) {
    match case.layer.as_str() {
        "l3_inbound" => {
            let fired = status == StatusCode::FORBIDDEN;
            let actual = if fired { "blocked" } else { "allowed" };
            (actual.to_string(), format!("status={}", status.as_u16()))
        }
        "l3_outbound" => {
            // In rewrite mode, blocked tool calls are removed and refusal text injected.
            // Check if refusal text is present (contains "blocked" in the message content).
            let fired = body.contains("blocked");
            let actual = if fired { "blocked" } else { "allowed" };
            let detail = if fired {
                "tool call rewritten/blocked".to_string()
            } else {
                "tool call passed through".to_string()
            };
            (actual.to_string(), detail)
        }
        "l5a" => {
            let fired = body.contains("[REDACTED]");
            let actual = if fired { "blocked" } else { "allowed" };
            let detail = if fired {
                "sensitive content redacted".to_string()
            } else {
                "content unchanged".to_string()
            };
            (actual.to_string(), detail)
        }
        other => ("allowed".to_string(), format!("unknown layer: {other}")),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{self, StringSource};

    const EVAL_YAML: &str = r#"parapet: v1

tools:
  _default:
    allowed: false
  read_file:
    allowed: true
    constraints:
      path:
        type: string
        starts_with: "/safe/"
        not_contains:
          - "../"

block_patterns:
  - "(?i)eval test block pattern"

canary_tokens:
  - "{{CANARY_EVAL}}"

sensitive_patterns:
  - "SECRET_KEY=[A-Za-z0-9]+"

engine:
  on_failure: closed
  timeout_ms: 5000

environment: "eval"

layers:
  L0: { mode: sanitize }
  L3_inbound: { mode: block }
  L3_outbound: { mode: block, block_action: rewrite }
  L5a: { mode: redact }
"#;

    fn eval_config() -> Arc<Config> {
        let source = StringSource {
            content: EVAL_YAML.to_string(),
        };
        Arc::new(config::load_config(&source).expect("eval config should parse"))
    }

    #[tokio::test]
    async fn l3_inbound_malicious_detected() {
        let config = eval_config();
        let (engine, mock) = build_eval_engine(config);

        let case = EvalCase {
            id: "test-001".to_string(),
            layer: "l3_inbound".to_string(),
            label: "malicious".to_string(),
            description: "test".to_string(),
            content: "eval test block pattern triggered".to_string(),
            mock_tool_calls: Vec::new(),
            mock_content: None,
        };

        let results = run_eval(&[case], &engine, &mock).await;
        assert_eq!(results.len(), 1);
        assert!(results[0].correct, "malicious l3_inbound should be blocked");
        assert_eq!(results[0].actual, "blocked");
    }

    #[tokio::test]
    async fn l3_inbound_benign_allowed() {
        let config = eval_config();
        let (engine, mock) = build_eval_engine(config);

        let case = EvalCase {
            id: "test-002".to_string(),
            layer: "l3_inbound".to_string(),
            label: "benign".to_string(),
            description: "test".to_string(),
            content: "How do I write a for loop in Python?".to_string(),
            mock_tool_calls: Vec::new(),
            mock_content: None,
        };

        let results = run_eval(&[case], &engine, &mock).await;
        assert_eq!(results.len(), 1);
        assert!(results[0].correct, "benign l3_inbound should be allowed");
        assert_eq!(results[0].actual, "allowed");
    }

    #[tokio::test]
    async fn l3_outbound_blocked_tool_detected() {
        let config = eval_config();
        let (engine, mock) = build_eval_engine(config);

        let case = EvalCase {
            id: "test-003".to_string(),
            layer: "l3_outbound".to_string(),
            label: "malicious".to_string(),
            description: "test".to_string(),
            content: "hello".to_string(),
            mock_tool_calls: vec![MockToolCall {
                id: "call_1".to_string(),
                name: "exec_command".to_string(),
                arguments: r#"{"cmd":"rm -rf /"}"#.to_string(),
            }],
            mock_content: None,
        };

        let results = run_eval(&[case], &engine, &mock).await;
        assert_eq!(results.len(), 1);
        assert!(
            results[0].correct,
            "blocked tool should be detected: actual={}, detail={}",
            results[0].actual,
            results[0].detail
        );
    }

    #[tokio::test]
    async fn l3_outbound_allowed_tool_passes() {
        let config = eval_config();
        let (engine, mock) = build_eval_engine(config);

        let case = EvalCase {
            id: "test-004".to_string(),
            layer: "l3_outbound".to_string(),
            label: "benign".to_string(),
            description: "test".to_string(),
            content: "hello".to_string(),
            mock_tool_calls: vec![MockToolCall {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
                arguments: r#"{"path":"/safe/readme.txt"}"#.to_string(),
            }],
            mock_content: None,
        };

        let results = run_eval(&[case], &engine, &mock).await;
        assert_eq!(results.len(), 1);
        assert!(
            results[0].correct,
            "allowed tool should pass: actual={}, detail={}",
            results[0].actual,
            results[0].detail
        );
    }

    #[tokio::test]
    async fn l5a_canary_redacted() {
        let config = eval_config();
        let (engine, mock) = build_eval_engine(config);

        let case = EvalCase {
            id: "test-005".to_string(),
            layer: "l5a".to_string(),
            label: "malicious".to_string(),
            description: "test".to_string(),
            content: "hello".to_string(),
            mock_tool_calls: Vec::new(),
            mock_content: Some("Here is the secret: {{CANARY_EVAL}} leaked".to_string()),
        };

        let results = run_eval(&[case], &engine, &mock).await;
        assert_eq!(results.len(), 1);
        assert!(
            results[0].correct,
            "canary token should be redacted: actual={}, detail={}",
            results[0].actual,
            results[0].detail
        );
    }

    #[tokio::test]
    async fn l5a_clean_content_passes() {
        let config = eval_config();
        let (engine, mock) = build_eval_engine(config);

        let case = EvalCase {
            id: "test-006".to_string(),
            layer: "l5a".to_string(),
            label: "benign".to_string(),
            description: "test".to_string(),
            content: "hello".to_string(),
            mock_tool_calls: Vec::new(),
            mock_content: Some("The result is 42.".to_string()),
        };

        let results = run_eval(&[case], &engine, &mock).await;
        assert_eq!(results.len(), 1);
        assert!(
            results[0].correct,
            "clean content should pass: actual={}, detail={}",
            results[0].actual,
            results[0].detail
        );
    }

    #[test]
    fn compute_metrics_basic() {
        let results = vec![
            EvalResult {
                case_id: "1".into(),
                layer: "l3_inbound".into(),
                label: "malicious".into(),
                expected: "blocked".into(),
                actual: "blocked".into(),
                correct: true,
                detail: String::new(),
            },
            EvalResult {
                case_id: "2".into(),
                layer: "l3_inbound".into(),
                label: "benign".into(),
                expected: "allowed".into(),
                actual: "allowed".into(),
                correct: true,
                detail: String::new(),
            },
            EvalResult {
                case_id: "3".into(),
                layer: "l3_inbound".into(),
                label: "malicious".into(),
                expected: "blocked".into(),
                actual: "allowed".into(),
                correct: false,
                detail: String::new(),
            },
        ];

        let report = compute_metrics(&results);
        assert_eq!(report.total_cases, 3);
        assert_eq!(report.total_correct, 2);
        assert_eq!(report.layers.len(), 1);

        let l3 = &report.layers[0];
        assert_eq!(l3.tp, 1);
        assert_eq!(l3.fp, 0);
        assert_eq!(l3.fn_count, 1);
        assert_eq!(l3.tn, 1);
        assert!((l3.precision - 1.0).abs() < f64::EPSILON);
        assert!((l3.recall - 0.5).abs() < f64::EPSILON);
    }
}
