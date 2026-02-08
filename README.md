# Parapet

A transparent LLM proxy firewall. Sits between your app and the LLM provider, enforcing tool constraints, blocking prompt injection patterns, and redacting sensitive output.

## What it does

Point any LLM client at `localhost:9800` instead of the provider API. Parapet intercepts requests and responses, applying security layers defined in `parapet.yaml`:

- **Block unauthorized tools** -- LLM tries `exec_command`? Blocked. Only allowlisted tools pass through.
- **Enforce argument constraints** -- `read_file(path="../../etc/passwd")` blocked by `not_contains` + `starts_with` predicates.
- **Catch prompt injection** -- "ignore previous instructions" in user messages blocked before reaching the LLM.
- **Scan trusted content too** -- RAG-poisoned system prompts with jailbreak patterns also caught.
- **Redact leaked secrets** -- API keys or canary tokens in LLM output scrubbed to `[REDACTED]`.
- **Streaming-safe** -- Text flows through immediately. Tool calls buffered, validated, then released or blocked.

## Security layers

```
Request in
  -> Parse (OpenAI / Anthropic format)
  -> Trust assignment (role-based + per-tool overrides)
  -> L0 normalize (NFKC, HTML strip, zero-width removal)
  -> L3-inbound (block patterns on ALL messages + length policy on untrusted)
  -> Forward to LLM provider
  -> Parse response
  -> L3-outbound (tool call validation via 9-predicate constraint DSL)
  -> L5a (canary token + sensitive pattern redaction)
  -> Return response
```

## Quick start

### Zero-SDK mode

No code changes. Just point your client at the proxy:

```bash
# Start the engine
parapet-engine --config parapet.yaml --port 9800

# Use any LLM client -- just change the base URL
export OPENAI_API_BASE=http://127.0.0.1:9800
```

### Python SDK mode

```python
import parapet

parapet.init("parapet.yaml")

with parapet.session(user_id="u_1", role="admin"):
    # All httpx-based LLM calls now route through parapet
    response = client.chat.completions.create(...)
```

The SDK starts the engine as a sidecar, patches httpx transparently, and adds W3C Baggage headers for session tracking.

## Example config

```yaml
parapet: v1

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
  exec_command:
    allowed: false

block_patterns:
  - "ignore previous instructions"
  - "DAN mode enabled"
  - "you are now [A-Z]+"

canary_tokens:
  - "{{CANARY_a8f3e9b1}}"

sensitive_patterns:
  - "CEREBRAS_API_KEY"

untrusted_content_policy:
  max_length: 50000

engine:
  on_failure: open
  timeout_ms: 5000
```

## Constraint predicates (v1)

`type`, `starts_with`, `not_contains`, `matches`, `one_of`, `max_length`, `min`, `max`, `url_host`

## Failover behavior

- **Engine down** (connection refused) -- failopen, request goes directly to provider, warning logged.
- **Engine slow** (timeout) -- failclosed, error returned.

## Building

```bash
# Rust engine
cd parapet && cargo build && cargo test

# Python SDK
cd parapet-py && pip install -e ".[dev]" && pytest tests/ -v
```

