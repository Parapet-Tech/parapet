# Parapet

> Active research project. Parapet is not stable or production-ready.

Parapet is a local LLM proxy firewall for prompt-injection defense. It sits between your app and model provider, applies layered checks to requests and responses, and can run either as a transparent local proxy or as an SDK-managed sidecar.

**[parapet.tech](https://parapet.tech)** | **[GitHub](https://github.com/Parapet-Tech/parapet)**

## What It Is

Parapet is built around a simple idea: put an inspectable security layer in front of model calls instead of burying safety logic inside application code.

Today the project includes:

- a Rust engine
- a Python SDK
- a TypeScript SDK
- a data curation package for classifier corpora
- an experiment runner for reproducible training and evaluation

If you only need the high-level model, the request path is:

```text
L0 normalize
-> L1 lightweight classifier
-> L2a optional payload scanner
-> L3_inbound pattern scanning
-> L4 multi-turn risk scoring
-> upstream model
-> L3_outbound tool constraints
-> L5a output redaction
```

For the canonical layer definitions, see [strategy/layers.md](strategy/layers.md).

## Quick Start

### 1. Install

```bash
# Python SDK
pip install parapet

# TypeScript SDK
npm install @parapet-tech/parapet

# Or build the Rust engine directly
cd parapet && cargo build --release
```

### 2. Write a policy

```yaml
parapet: v1
```

That enables the default protective stack. You add config to customize behavior, not to turn the core layers on.

Example with tool policy and custom patterns:

```yaml
parapet: v1

tools:
  _default:
    allowed: false
  read_file:
    allowed: true
    constraints:
      path:
        type: string
        starts_with: "${PROJECT_ROOT}"
        not_contains: ["../", "..\\"]

canary_tokens:
  - "{{CANARY_a8f3e9b1}}"

sensitive_patterns:
  - "CEREBRAS_API_KEY"
```

### 3. Run it

Python SDK:

```python
import parapet

parapet.init("parapet.yaml")

with parapet.session(user_id="u_1", role="admin"):
    response = client.chat.completions.create(...)
```

Proxy-only mode:

```bash
parapet-engine --config parapet.yaml --port 9800
```

Then point your OpenAI-compatible client at `http://127.0.0.1:9800`.

### 4. Verify behavior

Prompt injection attempts should block. Ordinary traffic should pass through.

## What It Catches

| Threat | Layer | Action |
|--------|-------|--------|
| Encoding tricks (Unicode, zero-width, HTML entities) | L0 normalize | Strip before scanning |
| Prompt injection (broad) | L1 classifier | Block (403) |
| Injection in data payloads (tool results, RAG docs) | L2a scanner | Block (403) |
| Prompt injection patterns (DAN, jailbreaks, extraction) | L3 inbound | Block (403) |
| Multi-turn attacks (seeding, role confusion, escalation) | L4 multi-turn | Block |
| Unauthorized tool calls | L3 outbound | Block |
| Dangerous tool arguments (path traversal, shell injection) | L3 outbound | Block |
| API keys / secrets in LLM output | L5a redact | Replace with `[REDACTED]` |
| System prompt leakage | L5a canary | Detect via canary tokens |

## Deployment Modes

### SDK sidecar

The SDK starts and talks to a local engine process for you.

Pros:

- easiest integration
- no app-level proxy wiring

Tradeoff:

- if the sidecar is unavailable, your fail-open or fail-closed behavior matters a lot

### Local proxy

Run `parapet-engine` directly and route client traffic through it.

Pros:

- clearest security boundary
- easiest mode to reason about operationally

If you need hard enforcement, prefer the proxy path plus network rules that force model traffic through Parapet.

## Failure Behavior

Parapet supports both fail-open and fail-closed behavior depending on where the failure occurs and how you configure the engine.

The practical takeaway:

- engine unavailable in sidecar mode can bypass scanning if you intentionally allow fail-open behavior
- timeout and policy behavior should be treated as deployment decisions, not defaults to ignore

If you are evaluating Parapet seriously, read the engine and SDK behavior before assuming traffic is always blocked on failure.

## Repo Map

The repository has a few distinct parts:

- [parapet/](parapet) - Rust engine and eval binary
- [parapet-py/](parapet-py) - Python SDK
- [parapet-ts/](parapet-ts) - TypeScript SDK
- [parapet-data/](parapet-data) - corpus staging and curation
- [parapet-runner/](parapet-runner) - experiment orchestration
- [strategy/](strategy) - layer docs and research direction

## Data And Experiments

Use these entry points:

- [parapet-data/README.md](parapet-data/README.md) - how source data is staged and curated into train, val, and holdout splits
- [parapet-runner/README.md](parapet-runner/README.md) - how training and evaluation runs are orchestrated from curated manifests
- [schema/README.md](schema/README.md) - schema layout, staged artifacts, and dataset contract notes

In short:

- `parapet-data` builds reproducible datasets
- `parapet-runner` trains, calibrates, evaluates, and records experiment receipts

## Integrations

LiteLLM chaining is documented separately in [use/litellm.md](use/litellm.md).

## Research Notes

Parapet includes ongoing work on multi-turn attack detection and prompt-injection classifiers.

Current paper:

- [publications/mirror/paper.pdf](publications/mirror/paper.pdf)

The broader research and layer notes live under [strategy/](strategy).

## Building From Source

```bash
# Rust engine
cd parapet && cargo build --release && cargo test

# With optional L2a support
cd parapet && cargo build --features l2a --release

# Python SDK
cd parapet-py && pip install -e ".[dev]" && pytest tests/ -v

# TypeScript SDK
cd parapet-ts && npm install && npm test && npm run build
```

## License

Apache 2.0
