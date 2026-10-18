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
-> L1 deterministic pattern gate
-> L2 lightweight lexical classifier
-> L3 orthogonal sensors / router
-> L4 multi-turn risk scoring
-> upstream model
-> L5 outbound tool constraints
-> L6 output redaction
```

Implementation names are still in transition. For the canonical target taxonomy
and current implementation mapping, see [strategy/layers.md](strategy/layers.md).

## Quick Start

### 1. Supported Local First-Use Journey (CLI and Locally Built Wheel)

The currently proven first-use environment is **macOS arm64 with Python 3.10 or newer** using a **locally built wheel** (`parapet-0.1.0-*.whl` built locally from `parapet-py` with Maturin). Parapet is not published to PyPI or npm, the protected runtime, evidence store, control socket, and file-effect lifecycle are POSIX-only (Windows protected-runtime support is not supported or claimed), and deterministic offline fixture checks do not constitute real-model validation.

Build the local wheel from `parapet-py` (replace `<wheel-out-dir>` with a disposable directory outside the checkout):

```bash
cd parapet-py
maturin build --release --locked --out <wheel-out-dir>
```

#### Ordinary first use (default)

Create a fresh disposable Python 3.10+ virtual environment outside the checkout, install the pinned runtime dependency `httpx==0.28.1`, then install the locally built wheel with `--no-deps` and activate the environment:

```bash
python3 -m venv <venv>
<venv>/bin/python -m pip install httpx==0.28.1
<venv>/bin/python -m pip install --no-deps <path-to-parapet-wheel>
source <venv>/bin/activate
```

Resolving `httpx==0.28.1` and its dependencies in this default path may use the network. In contrast, `<path-to-parapet-wheel>` is the local Section 11.2 wheel artifact built above; it is not a network package name and is not fetched from or published to PyPI.

#### Maintainer frozen offline verification

This variant is executable only on a host with the pre-populated cache and pinned inputs from Section 2.2 of the implementation brief. In both installation paths, `<path-to-parapet-wheel>` is replaced by the local wheel artifact produced in Section 11.2 (never a network package name). Automated dispatch acceptance executes only this second path and never gains network authority from the ordinary-user documentation:

```bash
python3 -m venv <venv>
uv pip install --offline --python <venv>/bin/python httpx==0.28.1
uv pip install --offline --python <venv>/bin/python --no-deps <path-to-parapet-wheel>
source <venv>/bin/activate
```

#### Verify `site-packages` and installed binaries

Verify that `parapet` imports from the isolated environment's `site-packages` (never the repository checkout) and that both compiled binaries are present on `PATH`:

```bash
python -c "import parapet; print(parapet.__file__)"
command -v parapet
command -v parapet-engine
```

The wheel installs only the Python `parapet` package plus the `parapet` and `parapet-engine` binaries; `parapet-fetch`, `parapet-scope-audit`, `pg2-diag`, and `parapet-eval` are intentionally absent.

#### First installed CLI checks (`info` then bounded `scan`)

Run the installed CLI in this order:

```bash
parapet info --json
printf 'hello parapet' | parapet scan --stdin --json
```

Default `parapet scan --json` output is bounded summary metadata (model identity, digests, score, and detection outcome without raw input or matched feature substrings/spans). By contrast, `--record-json` is a separately warned sensitive evidence export that writes exact canonical `l1-detection-record/v0` bytes (including matched feature substrings and character spans) and must never be used on secret or regulated input without appropriate custody.

#### Protected runtime and evidence lifecycle

Run the protected lifecycle in this exact order (`init` -> `run` -> `inspect` -> `restore` -> `disable` -> post-disable `inspect` -> optional explicit `evidence release` -> post-release `inspect`). Because `parapet run` starts a long-lived engine process, keep it running in one terminal and run the control and inspection commands from a second terminal:

```bash
# Terminal 1: initialize disposable state and start the long-lived engine
parapet init --state-dir <state-dir> --effect-root <new-effect-root> --json
parapet run --state-dir <state-dir> --config <config> --effect-root <new-effect-root> --file-effect-profile v0 --max-content-bytes 4096 --json

# Terminal 2: inspect -> restore -> disable -> post-disable inspect -> optional evidence release -> post-release inspect
parapet inspect --state-dir <state-dir> --run-id <run-id> --json
parapet restore --state-dir <state-dir> --json
parapet disable --state-dir <state-dir> --json
# protected evidence: retained (use 'parapet evidence release' to release store objects)
parapet inspect --state-dir <state-dir> --run-id <run-id> --json
parapet evidence release --state-dir <state-dir> --run-id <run-id> --json
parapet inspect --state-dir <state-dir> --run-id <run-id> --json
```

The effect-enabled engine process is control-long-lived (it keeps serving until `parapet disable` stops it), but its effect runtime is one-shot: it accepts exactly one non-streaming request, rejects any streaming request with HTTP 400 `effect_streaming_unsupported` (which does not consume the run), and rejects every request after the first non-streaming one with HTTP 409 `effect_run_consumed`. Start a new `parapet run` with a new `<new-effect-root>` for each effect run.

Run-creating commands (`parapet run`, `parapet accept`, and `parapet-engine`) never accept a caller-supplied run ID. The engine generates the only run identity; `<run-id>` above is that generated 32-character lowercase hex value. `parapet run` reports it when it exits after `parapet disable`: `"run_id"` in JSON output and `run_id: <run-id>` in human output (`"run_id": null` and `run_id: not_applicable` for a run without `--effect-root`). Use that CLI-reported value for the post-disable `inspect` and `evidence release` steps. While the engine is still live, the same identity is witnessed by the `<new-effect-root>/.parapet-run-<run-id>.sentinel` file that the effect-enabled engine creates at startup; it always names the identity that `parapet run` later reports.

`parapet disable` never performs `restore` itself: it is an observational shutdown gate that refuses to stop an owned effect-enabled engine until file cleanup is already `clear` (after `parapet restore`) or `not_applicable`. Stopping the engine always retains protected receipt evidence (`protected evidence: retained (use 'parapet evidence release' to release store objects)`) so post-restore and post-disable `parapet inspect` continue to succeed until you optionally run explicit `parapet evidence release`, after which `parapet inspect` reports `reader: object_released`.

#### Rust CLI acceptance (disposable offline fixture)

This is **Rust CLI acceptance**: the installed Rust `parapet` CLI is the only requester, sending one request through a loopback engine/proxy to the deterministic offline fixture. It **does not exercise Python or TypeScript**; the Python and TypeScript SDKs are verified separately by the installed SDK commands below. Supply a disposable `<state-dir>` and a disposable `<new-effect-root>` that does not exist beforehand:

```bash
parapet accept --state-dir <state-dir> --config <config> --scenario allow_verified_write --effect-root <new-effect-root> --json
```

`upstream: deterministic offline fixture (no real model contacted)`

Both `<state-dir>` and `<new-effect-root>` are disposable local directories, `<new-effect-root>` must not exist beforehand, and the command contacts only the deterministic local fixture (`"real_model_contacted": false`). `<config>` must be the exact bytes of `parapet/tests/fixtures/cli_v0/accept_enforced_v0.yaml` (`shadow_parse_error_l3_off` uses `parapet/tests/fixtures/cli_v0/accept_shadow_l3_off_v0.yaml`); any other config exits 3 with `accept_config_mismatch` before creating state. The JSON output reports `"requester": "rust_cli"`, the installed CLI `requester_version`, the engine-generated `run_id`, and the accepted `config_contract_hash`; human output prints `requester: Rust CLI <version>`.

#### Installed Python SDK verification

This runs the Python SDK from the locally built wheel installed in `<ROLE>/python-installed-venv` (the role's disposable installed venv), from the role's neutral working directory `<ROLE>/neutral-cwd` outside the checkout `<LANE>`, with `PYTHONPATH` unset and the installed `parapet-engine` and `parapet` binaries bound explicitly:

```bash
cd <ROLE>/neutral-cwd
env -u PYTHONPATH PARAPET_NO_REAL_MODEL=1 PARAPET_EXPECTED_ENV_ROOT=<ROLE>/python-installed-venv PARAPET_ENGINE_BIN=<ROLE>/python-installed-venv/bin/parapet-engine PARAPET_CLI_BIN=<ROLE>/python-installed-venv/bin/parapet PYTHONDONTWRITEBYTECODE=1 <ROLE>/python-installed-venv/bin/python -m pytest -q --import-mode=importlib -p no:cacheprovider <LANE>/parapet-py/tests/test_e2e.py -k installed_python_sdk_is_actual_loopback_requester_rev9
```

The test requires the real `Path.cwd()` to equal `<ROLE>/neutral-cwd`, records the installed `parapet` distribution version and `parapet.__file__`, and requires the module below the installed venv's `site-packages` and outside `<LANE>`; no checkout Python package is imported. It starts the installed engine on a bound loopback endpoint, records that endpoint and the engine owner PID, and makes an actual request through the installed SDK transport. Its substitution witness is an in-process recorder under the installed SDK's base HTTP transport: the request must be recorded exactly once, from the installed `parapet/transport.py`, with the SDK's `x-parapet-original-host` header, and the caller must receive that recorded response. A request made by Rust `reqwest` or by the TypeScript SDK never reaches the recorder and fails the test. Every non-loopback destination is refused, and the retained `PARAPET_PY_INSTALLED_REQUESTER_RECORD` reports `"real_model_contacted": false`.

#### Installed TypeScript SDK verification

The TypeScript SDK is a separate npm package (`@parapet-tech/parapet`) in its own Node environment; it is not a Python-wheel payload. After building, packing, and installing that package into the disposable consumer `<ROLE>/ts-install/consumer`, run from the same neutral working directory against the wheel-installed engine (`<NODE>` is the Node binary; Vitest comes from the role's dev-dependency copy):

```bash
cd <ROLE>/neutral-cwd
E2E=1 PARAPET_NO_REAL_MODEL=1 PARAPET_ENGINE_PATH=<ROLE>/python-installed-venv/bin/parapet-engine PARAPET_CLI_BIN=<ROLE>/python-installed-venv/bin/parapet PARAPET_TS_INSTALLED_ENTRY=<ROLE>/ts-install/consumer/node_modules/@parapet-tech/parapet/dist/index.js PARAPET_TS_CHECKOUT_ROOT=<LANE>/parapet-ts PARAPET_TS_NEUTRAL_CWD=<ROLE>/neutral-cwd <NODE> <ROLE>/tool-cache/ts-seed/node_modules/vitest/vitest.mjs run --root <LANE>/parapet-ts --config <LANE>/parapet-ts/vitest.config.ts test/e2e.test.ts -t "installed TypeScript SDK is actual loopback requester Rev9"
```

The test requires `process.cwd()` to equal `<ROLE>/neutral-cwd`, records the installed package name, version, and resolved entry (inside the consumer and outside `<LANE>/parapet-ts`), records the bound loopback engine endpoint and the actual request/response, and retains the `PARAPET_TS_INSTALLED_REQUESTER_RECORD` substitution witness proving the installed entry, not Rust `reqwest` or Python, made the request with `"real_model_contacted": false`. The same command with `PARAPET_TS_INSTALLED_ENTRY=<LANE>/parapet-ts/src/index.ts` must exit nonzero with `installed_entry_inside_checkout` before starting the engine.

A missing runtime, package, binary, or other installed input is a hard error in every SDK verification command; there is no client fallback to Rust, the other SDK, or a checkout source tree.

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
| Prompt injection patterns (DAN, jailbreaks, extraction) | L1 pattern gate | Block (403) |
| Prompt injection (broad lexical signal) | L2 lexical classifier | Block (403) |
| Obfuscation, structure, entropy, sizing anomalies | L3 orthogonal sensors | Block or route by policy |
| Multi-turn attacks (seeding, role confusion, escalation) | L4 multi-turn | Block |
| Unauthorized tool calls | L5 outbound constraints | Block |
| Dangerous tool arguments (path traversal, shell injection) | L5 outbound constraints | Block |
| API keys / secrets in LLM output | L6 redaction | Replace with `[REDACTED]` |
| System prompt leakage | L6 canary | Detect via canary tokens |

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

Parapet includes ongoing work on multi-turn attack detection, lexical
classifiers, and deterministic orthogonal sensors.

Current public papers:

- [Amplify, Don't Create: Temporal Accumulation for Slow-Burn Prompt Injection](docs/publications/amplify/paper.pdf)
- [Peak + Accumulation: A Proxy-Level Scoring Formula for Multi-Turn LLM Attack Detection](docs/publications/multiturn/paper.pdf)
- [The Mirror Design Pattern: Strict Data Geometry over Model Scale for Prompt Injection Detection](docs/publications/mirror/paper.pdf)

The broader research and layer notes live under [strategy/](strategy).

For the current detection direction, including the closure of the inline
semantic-transformer hot-path slot, see
[strategy/current_direction.md](strategy/current_direction.md).

## Building From Source

```bash
# Rust engine
cd parapet && cargo build --release && cargo test

# With optional legacy L2a support
cd parapet && cargo build --features l2a --release

# Python SDK
cd parapet-py && pip install -e ".[dev]" && pytest tests/ -v

# TypeScript SDK
cd parapet-ts && npm install && npm test && npm run build
```

## License

Apache 2.0
