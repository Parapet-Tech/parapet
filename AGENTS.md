# Agent Instructions

## Project Structure

```
parapet/
  parapet/                  # Rust engine (cargo workspace root)
    src/
      main.rs               # HTTP proxy entrypoint (axum)
      message.rs            # Internal message representation (Message, Role, TrustLevel, ToolCall)
      provider/             # OpenAI + Anthropic adapters (parse/serialize)
      trust.rs              # Role-based trust assignment + per-tool overrides
      layers/               # L0, L3-inbound, L3-outbound, L5a
      constraint/           # DSL evaluator (9 predicates)
      normalize/            # NFKC, HTML strip, encoding hygiene
      stream/               # SSE passthrough, tool call buffering
      config/               # parapet.yaml schema, validation, contract_hash
      bin/                  # CLI tools (parapet-eval, parapet-fetch, pg2-diag)
    tests/                  # Integration tests (full proxy round-trips)
  parapet-py/               # Python SDK (thin: init, session)
    parapet/
      __init__.py           # init(), session()
      transport.py          # httpx patch
      sidecar.py            # Engine process lifecycle (start, stop, orphan detection)
      header.py             # W3C Baggage serialization
    tests/
  parapet-ts/               # TypeScript SDK (preview)
    src/
      transport.ts          # Fetch wrapper, URL rewrite + failover
      header.ts             # W3C Baggage serialization
      trust.ts              # Trust metadata helpers
    test/                   # Vitest coverage for transport/header/context/trust
  strategy/                 # Layer docs, configs, and eval snapshots
  scripts/                  # Dataset fetch/convert/analyze/training scripts
  schema/                   # Config schema + eval datasets/configs/results
    parapet.schema.json
    examples/
    eval/
```

## Key Files

| File | Purpose |
|------|---------|
| `parapet/src/main.rs` | HTTP proxy entrypoint (axum, `127.0.0.1:9800`) |
| `parapet/src/message.rs` | `Message`, `Role`, `TrustLevel`, `ToolCall` types |
| `parapet/src/provider/mod.rs` | OpenAI + Anthropic request/response parsing |
| `parapet/src/config/mod.rs` | `parapet.yaml` loader, validator, `contract_hash` |
| `parapet/src/bin/parapet_eval.rs` | Evaluation harness CLI (`parapet-eval`) |
| `parapet/src/bin/pg2_diag.rs` | Prompt Guard 2 diagnostics/benchmark helper |
| `parapet/src/trust.rs` | Role-based trust assignment + per-tool overrides |
| `parapet/src/normalize/mod.rs` | L0: NFKC, HTML strip, zero-width removal |
| `parapet/src/layers/l3_inbound.rs` | Block patterns (all messages) + content policy (untrusted) |
| `parapet/src/constraint/mod.rs` | L3-outbound: 9-predicate DSL evaluator |
| `parapet/src/layers/l5a.rs` | Output pattern scanning + redaction |
| `parapet/src/stream/mod.rs` | SSE passthrough + tool call buffering |
| `parapet/src/proxy.rs` | Request forwarding + provider detection |
| `parapet-py/parapet/__init__.py` | `init()`, `session()` context manager |
| `parapet-py/parapet/transport.py` | httpx monkey-patch, failopen/failclosed |
| `parapet-py/parapet/sidecar.py` | Engine subprocess lifecycle, PID file, heartbeat |
| `parapet-py/parapet/header.py` | W3C Baggage serialization |
| `parapet-ts/src/transport.ts` | TypeScript fetch wrapper for OpenAI/fetch clients |
| `strategy/layers.md` | Canonical layer pipeline guide + links to per-layer docs |
| `scripts/train_l1.py` | Retrains L1 classifier and regenerates `l1_weights.rs` |
| `scripts/fetch_*.py` | Pulls open-source eval/training datasets |
| `schema/eval/eval_config.yaml` | Default eval harness config |
| `schema/parapet.schema.json` | JSON Schema for config validation |

## Building & Running

```bash
# Rust engine
cd parapet && cargo build
cargo test

# Optional: enable L2a data payload scanning
cd parapet && cargo build --features l2a --release

# Eval harness (JSON output)
cd parapet && cargo run --bin parapet-eval -- \
  --config ../schema/eval/eval_config.yaml \
  --dataset ../schema/eval/ \
  --json

# Python SDK
cd parapet-py
pip install -e ".[dev]"
pytest tests/ -v

# TypeScript SDK (preview)
cd parapet-ts
npm install
npm test
npm run build
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | OpenAI-compatible proxy |
| `POST /v1/messages` | Anthropic-compatible proxy |
| `GET /v1/heartbeat` | SDK heartbeat (sidecar watchdog) |

## Issue Tracking

**br (beads_rust)** is a lightweight, git-integrated issue tracker. Run `br help` to get started.

## Quick Reference

```bash
br ready              # Find available work
br show <id>          # View issue details
br update <id> --status in_progress  # Claim work
br close <id>         # Complete work
br sync               # Sync with git
```

---

## Code Style

### MANDATORY: Interface-First Design

**NEVER write a concrete type without defining its interface first.**

**Why?** Interfaces enable:
- Testing with test doubles
- Swapping implementations
- Dependency inversion
- Clear contracts

## DEPENDENCY INJECTION (MANDATORY)

### The Rule: ALL Dependencies MUST Be Injected

**NO direct instantiation of dependencies in type constructors.**

### Constructor Rules

1. **NO side effects** - Constructors assign dependencies only
2. **NO business logic** - Move logic to explicit methods
3. **NO I/O operations** - No file reads, no network calls
4. **NO auto-registration** - Explicit is better than implicit
5. **Provide defaults with factory functions** - `NewFoo()` creates with defaults, `NewFooWith(deps)` for injection

---

## SECURITY

### Input Validation

**Validate at system boundaries, trust internal code.**

### Sensitive Data

- **NEVER** log API keys, tokens, or credentials
- Use the `Redactor` interface to mask sensitive data before storage/display
- Environment variables for secrets, not config files

---

## LOGGING

Use structured logging with consistent levels:
- **ERROR**: Something failed that affects functionality
- **WARN**: Something unexpected that the system handled
- **INFO**: Significant state changes (startup, shutdown, connections)
- **DEBUG**: Detailed diagnostic info (disabled in production)

Include context in log messages: request IDs, user identifiers, operation names.

---

## TESTS

### Test Quality Checklist

Before writing ANY test, answer these questions:

**Does this test prove a feature works?**
- If NO -> Don't write it

**Would this test fail if the feature breaks?**
- If NO -> Don't write it

**Can I refactor internals without breaking this test?**
- If NO -> You're testing implementation, not behavior

**Does this test cover edge cases?**
- Empty inputs?
- Boundary values?
- Error conditions?
- Invalid data?

### TEST ISOLATION (CRITICAL)

**NEVER MAKE REAL API CALLS IN TESTS. EVER.**
**TEST BEHAVIOR THROUGH INTERFACE-BASED TEST DOUBLES, NOT REAL APIS**

---

## Philosophy

## Rule #1: Work Is Not Done Until User Confirms

**NEVER claim work is complete, fixed, or working until the user has tested and confirmed.**

- Don't say "Fixed!" or "Done!" after making changes
- Don't assume code works because it compiles/passes tests
- Always ask the user to verify before moving on
- If the user reports a problem, the fix isn't done yet

## Boy Scout Rule

Always leave code better than you found it. Every touch should improve:
- Clarity
- Simplicity
- Correctness

If you can't make it better, at least don't make it worse.

## Think Like an Architect

Before writing ANY code:
1. Design interfaces first - what's the contract?
2. Consider dependencies - what does this need? What needs this?
3. Plan for testing - how will this be verified?
4. Follow SOLID - single responsibility, open/closed, etc.
5. Write tests that prove it works - behavior, not implementation
6. THEN implement

## Never Be a Lazy Coder

- Don't mock everything - test real behavior
- Don't write tests that prove nothing - "it returns something" is useless
- Don't create god classes - split responsibilities
- Don't hard-code dependencies - inject them
- Don't skip edge cases - they're where bugs live
- Don't write code without designing first - thinking is not optional

## SOLID Principles

**Single Responsibility**: Each class/function does ONE thing. If you're writing "and" in the description, split it.

**Open/Closed**: Open for extension, closed for modification. Use strategies/plugins, not if/else chains.

**Liskov Substitution**: Subtypes must honor the base contract completely. No surprises.

**Interface Segregation**: Don't force clients to depend on methods they don't use. Small, focused interfaces.

**Dependency Inversion**: Depend on abstractions, not concretions. Inject dependencies, don't instantiate them.

## Core Principle: Delete More Than You Add

Before writing ANY fix, ask:
1. Why does this problem exist?
2. Is there code that should be deleted instead of code to add?
3. Am I fixing a symptom or the root cause?