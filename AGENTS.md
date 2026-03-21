# Agent Instructions

## Project Structure

```
parapet/
  parapet-data/             # Mirror specs, curation, verified-sync, adjudication plumbing
  parapet-runner/           # Training/eval runner, manifests, reproducible experiment outputs
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
  scripts/                  # Analysis/training scripts
    sources/                # Dataset fetch/source-ingest scripts
  schema/                   # Config schema + eval datasets/configs/results
    parapet.schema.json
    examples/
    eval/
      staging/              # Mechanically staged training candidates
      verified/             # Ledger-applied staged projections for provenance-checked training/review
```

## Public Repo Boundary

`parapet/` is the public GitHub repository.

- Never commit private/raw/generated corpora.
- Treat all TheWall-derived data as local-only unless explicitly approved for publication.
- Keep local-only derived artifacts out of git:
  - adjudication ledgers and review exports
  - local verified projections and sync receipts
  - curated outputs under `parapet-data/curated*`
  - runner outputs under `parapet-runner/runs/`
- Version reproducible public assets only: code, docs, specs, manifests, and publishable source YAMLs.
- Prefer committing reproducible scripts/configs/docs instead of data files.
- Before commit/push, run: `python scripts/check_no_data_commit.py`.

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
| `parapet/src/layers/l1_weights.rs` | Auto-generated default L1 classifier weights (feature count varies by run) |
| `schema/eval/l1_holdout.yaml` | Auto-generated L1 holdout eval set |
| `scripts/train_l1.py` | Retrains L1 classifier and regenerates `l1_weights.rs` + `l1_holdout.yaml` |
| `scripts/train_l1_specialist.py` | Current curated mirror/generalist training path used for source-locked L1 runs |
| `scripts/sources/fetch_*.py` | Pulls open-source eval/training datasets |
| `parapet-data/generate_spec.py` | Expands compact mirror specs into full curation specs |
| `parapet-data/parapet_data/ledger.py` | Hash-based adjudication actions applied during verified-sync/curation |
| `parapet-data/parapet_data/verified.py` | Materializes `schema/eval/verified/` from staged sources + ledger |
| `schema/eval/eval_config.yaml` | Default eval harness config |
| `schema/parapet.schema.json` | JSON Schema for config validation |

## Diagnostic Automation

Prefer automation-first diagnostics. For drift, integrity, reproducibility, or
curation-quality issues, extend scripts and guardrails before adding prose
runbooks to `AGENTS.md`.

Canonical entrypoints:

| Entry Point | Purpose |
|------------|---------|
| `parapet-data/scripts/check_golden_contract.py` | Fail closed on source/language composition drift against a trusted curated baseline |
| `parapet-data/scripts/diagnose_drift.py` | Compare sampler/backfill behavior when curated outputs diverge |
| `parapet-data/scripts/audit_curated.py` | Audit curated corpora and export reviewable issues |
| `parapet-data/scripts/extract_golden_contract.py` | Extract a golden contract from a trusted curated artifact |
| `python -m parapet_data verified-sync ...` | Materialize and inspect staged-source projections before curation |

Diagnostic completion standard:

- A diagnosis is not complete until it is verified by a concrete artifact:
  guardrail result, semantic hash, manifest diff, source-mix report, or a
  minimal reproducer.
- If the required automation does not exist yet, add or improve the script,
  guardrail, or skill instead of encoding a manual checklist here.

## L1 Classifier (Current Work)

Current default direction:
- Treat the cleaned mirror generalist as the default L1 path.
- Keep ensemble work experimental and opt-in only unless the user explicitly asks for it.
- Keep `l2a` opt-in/off by default.

### Architecture
L1 is a char n-gram SVM (`CountVectorizer` + `LinearSVC`) compiled to Rust `phf_map` weights for very low-latency inference. Runtime scoring uses:

1. raw text pass
2. squashed alphanumeric-only pass
3. max(raw, squashed)

### Training Pipeline
Run from `parapet/`:

1. Build or update mirror specs in `parapet-data/` when changing corpus composition
2. Use `parapet-data` to curate source-locked corpora, optionally with `--ledger` and `verified-sync`
3. `scripts/train_l1_specialist.py` - train/eval/codegen
4. `cargo build && cargo test -- l1` - compile and validate generated weights

### Key Flags
- `--apply-l0-transform` - align train preprocessing with runtime L0
- `--squash-augment` - augment train set with squashed variants
- `--analyzer char_wb` - default analyzer for L1 (do not switch casually)

### Current Data Notes
- Attack training set: `schema/eval/training/attacks51042.yaml` (~51k)
- Current source-locked cleaned mirror baseline lives in `parapet-data/mirror_v4.compact.yaml` and `parapet-data/mirror_spec_v4_19k.yaml`
- Mirror-based curation via `parapet-data/` is the active workflow for L1 corpus work
- Adjudication ledgers are local-only; keep the machinery versioned, not the live ledger
- Benign composition target:
  - categories: instructions/chat/creative/code/hard-negatives/knowledge/system-prompts
  - languages: EN 75, RU 10, ZH 8, AR 7
  - lengths: short 25, medium 40, long 25, structured 10

### Known Issues
- `pi-cleaned-v2` negative class has attack contamination
- whitespace artifacts can leak into n-gram features
- LinearSVC scaling past ~60k-150k total samples is compute-sensitive
- older 96.2% F1 numbers were invalid due to eval contamination

### Clean Holdout Baselines
| Config | F1 | Recall | Precision |
|--------|----|--------|-----------|
| 51k attack + 25k benign (clean) | 0.937 | 0.945 | 0.929 |
| 51k attack + 51k benign (1:1) | 0.931 | 0.911 | 0.952 |

### What Does Not Work Well for L1
- kernel SVM (support vector cost too high for L1 latency budget)
- neural models at L1 gate latency target
- RL framing for one-shot binary classification
- scaling data volume without curation quality controls

## Experiment Conventions
- Training experiments: `schema/eval/t2/`
- Strategy docs: `strategy/`
- Run scripts from `parapet/` so relative paths resolve correctly
- Always review top features and FP/FN outputs after training
- For large YAML/JSONL/curated corpora, prefer `rg`, targeted sampling,
  manifests, and script outputs over full-file parsing; load the entire dataset
  only when required.
- If a required command is blocked by sandbox/permissions/timeouts, stop and ask the user to run it or approve escalation; do not change the experiment definition as a workaround

## Codex CLI Subagents

Use `codex exec` for explicit subagent-style batch work instead of silently
doing the task in the main thread.

- Default pattern for parallel batch annotation/review:
  run one `codex exec` worker per batch and write the worker's final response to
  a file with `-o`.
- If the user asked for a cheaper/lighter worker, use a lower-tier model such
  as `gpt-5.1` instead of the default top-tier model.
- The local config currently defaults to `model_reasoning_effort = "xhigh"`.
  Override that for `gpt-5.1` runs with:
  `-c model_reasoning_effort='medium'`
- If `codex exec` fails because the sandbox blocks network access, rerun the
  exact command with escalation instead of falling back to in-thread manual
  work.
- Keep the prompt narrow: point the worker at the task file, schema/rubric, and
  one batch file; require machine-readable output only.

Example:

```bash
cd parapet
codex exec \
  -m gpt-5.1 \
  -c model_reasoning_effort='medium' \
  -s read-only \
  -o parapet-runner/runs/.../batch_audit.jsonl \
  "Read CODEX_TASK.md, the audit rubric, and one batch file. Output JSONL only."
```

## Building & Running

```bash
# Rust engine
cd parapet && cargo build
cargo test

# Optional: enable L2a data payload scanning
cd parapet && cargo build --features l2a --release

# Eval harness (JSON output)
cd parapet && cargo run --features eval --bin parapet-eval -- \
  --config ../schema/eval/eval_config.yaml \
  --dataset ../schema/eval/ \
  --json \
  --output ../schema/eval/results/output.json

# L1 retrain (baseline path)
python scripts/train_l1.py --data-dir schema/eval
cd parapet && cargo build && cargo test -- l1

# L1 retrain (current curated mirror path)
python scripts/train_l1_specialist.py --help

# Python SDK
cd parapet-py
pip install -e ".[dev]"
pytest tests/ -v

# TypeScript SDK (preview)
cd parapet-ts
npm install
npm test
npm run build

# E2E tests (requires API key in .env + release engine binary)
cd parapet && cargo build --release
cd parapet-py && pytest tests/test_e2e.py -v
cd parapet-ts && E2E=1 npm run test:e2e
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | OpenAI-compatible proxy |
| `POST /openai/v1/chat/completions` | OpenAI-compatible proxy (Groq path) |
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
