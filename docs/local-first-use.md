# Local First-Use and Verification Guide

This guide contains the detailed installation, protected-runtime, acceptance,
and SDK verification procedures. For the shortest installation path, start with
the [README Quick Start](../README.md#quick-start).

## Supported environment

The currently proven first-use environment is **macOS arm64 with Python 3.10 or
newer** using a **locally built wheel** (`parapet-0.1.0-*.whl` built locally
from `parapet-py` with Maturin). Parapet is not published to PyPI or npm. The
protected runtime, evidence store, control socket, and file-effect lifecycle
are POSIX-only; Windows protected-runtime support is not supported or claimed.
Deterministic offline fixture checks do not constitute real-model validation.

## Build the local wheel

Replace `<wheel-out-dir>` with a disposable directory outside the checkout:

```bash
cd parapet-py
maturin build --release --locked --out <wheel-out-dir>
```

## Ordinary first use

Create a fresh disposable Python 3.10+ virtual environment outside the
checkout, install the pinned runtime dependency `httpx==0.28.1`, then install
the locally built wheel with `--no-deps` and activate the environment:

```bash
python3 -m venv <venv>
<venv>/bin/python -m pip install httpx==0.28.1
<venv>/bin/python -m pip install --no-deps <path-to-parapet-wheel>
source <venv>/bin/activate
```

Resolving `httpx==0.28.1` and its dependencies in this default path may use the
network. `<path-to-parapet-wheel>` is the local wheel artifact built above; it
is not a network package name and is not fetched from or published to PyPI.

## Maintainer frozen offline verification

This variant is executable only on a host with the pre-populated cache and
pinned inputs required by the maintainer acceptance environment. Automated
acceptance uses this offline path and does not gain network authority from the
ordinary-user documentation:

```bash
python3 -m venv <venv>
uv pip install --offline --python <venv>/bin/python httpx==0.28.1
uv pip install --offline --python <venv>/bin/python --no-deps <path-to-parapet-wheel>
source <venv>/bin/activate
```

In both installation paths, replace `<path-to-parapet-wheel>` with the local
wheel artifact built above, never a network package name.

## Verify `site-packages` and installed binaries

Verify that `parapet` imports from the isolated environment's `site-packages`,
never the repository checkout, and that both compiled binaries are on `PATH`:

```bash
python -c "import parapet; print(parapet.__file__)"
command -v parapet
command -v parapet-engine
```

The wheel installs only the Python `parapet` package plus the `parapet` and
`parapet-engine` binaries. `parapet-fetch`, `parapet-scope-audit`, `pg2-diag`,
and `parapet-eval` are intentionally absent.

## First installed CLI checks

Run `info` and then a bounded `scan`:

```bash
parapet info --json
printf 'hello parapet' | parapet scan --stdin --json
```

Default `parapet scan --json` output is bounded summary metadata: model
identity, digests, score, and detection outcome without raw input or matched
feature substrings or spans. By contrast, `--record-json` is a separately
warned sensitive evidence export. It writes exact canonical
`l1-detection-record/v0` bytes, including matched feature substrings and
character spans, and must never be used on secret or regulated input without
appropriate custody.

Human `parapet scan` output (without `--json`) puts the verdict first and prints
the same fields in the same order whether the input is clean or flagged:

```text
verdict: clean
detection_outcome: threshold_not_breached
effective_raw: <score>
threshold: <threshold>
mode: shadow|block
record_schema_version: l1-detection-record/v0
record_digest: <64-hex-sha256>
model_id: <model_id>
model_digest: <64-hex-sha256>
input_digest: <64-hex-sha256>
input_len_bytes: <bytes>
attribution_status: present|absent_by_contract|errored
feature_count_matched: <count>|none
```

A flagged input prints `verdict: flagged` and
`detection_outcome: threshold_breached` on the first two lines. The verdict
comes from the raw score comparison `effective_raw >= threshold`. Human
`effective_raw` and `threshold` are rounded to at most 8 decimal places; when
both round to the same digits, the score line adds
`(raw_relation: effective_raw < threshold)`, `==`, or `>` to show the raw
relation. Both verdicts exit 0. `--json` and `--record-json` bytes are
unchanged and keep the raw numbers.

## Protected runtime and evidence lifecycle

Run the protected lifecycle in this exact order:

`init` -> `run` -> `inspect` -> `restore` -> `disable` -> post-disable
`inspect` -> optional explicit `evidence release` -> post-release `inspect`.

Because `parapet run` starts a long-lived engine process, keep it running in
one terminal and run the control and inspection commands from a second
terminal:

```bash
# Terminal 1: initialize disposable state and start the long-lived engine
parapet init --state-dir <state-dir> --effect-root <new-effect-root> --json
parapet run --state-dir <state-dir> --config <config> --effect-root <new-effect-root> --file-effect-profile v0 --max-content-bytes 4096 --json

# Terminal 2: inspect, restore, disable, inspect, release, and inspect again
parapet inspect --state-dir <state-dir> --run-id <run-id> --json
parapet restore --state-dir <state-dir> --json
parapet disable --state-dir <state-dir> --json
# protected evidence: retained (use 'parapet evidence release' to release store objects)
parapet inspect --state-dir <state-dir> --run-id <run-id> --json
parapet evidence release --state-dir <state-dir> --run-id <run-id> --json
parapet inspect --state-dir <state-dir> --run-id <run-id> --json
```

The effect-enabled engine process is control-long-lived: it keeps serving until
`parapet disable` stops it. Its effect runtime is one-shot: it accepts exactly
one non-streaming request, rejects any streaming request with HTTP 400
`effect_streaming_unsupported` (which does not consume the run), and rejects
every request after the first non-streaming one with HTTP 409
`effect_run_consumed`. Start a new `parapet run` with a new
`<new-effect-root>` for each effect run.

Run-creating commands (`parapet run`, `parapet accept`, and `parapet-engine`)
never accept a caller-supplied run ID. The engine generates the only run
identity; `<run-id>` above is that generated 32-character lowercase hex value.

In human mode (without `--json`), `parapet run` prints a startup status to
stdout once the engine is ready and before it starts serving, so you learn the
port, run ID, and mode without reading files from the state directory:

```text
status: ready
port: <port>
run_id: <run-id>
mode: protected
next: send one request, then parapet restore --state-dir <state-dir>
```

A run without `--effect-root` prints `run_id: not_applicable` and
`mode: proxy`. The startup status never contains the control token or any
path. When the engine exits after `parapet disable`, human output starts with
`status: stopped` and repeats the port and run ID. In JSON mode, stdout stays
the single `parapet-cli-run/v0` line printed at exit (`"run_id"`, or
`"run_id": null` without `--effect-root`). Use the reported run ID for
`inspect` and `evidence release`.

`parapet disable` never performs `restore` itself. It is an observational
shutdown gate that refuses to stop an owned effect-enabled engine until file
cleanup is already `clear` (after `parapet restore`) or `not_applicable`.
Stopping the engine retains protected receipt evidence, so post-restore and
post-disable `parapet inspect` continue to succeed until you optionally run
`parapet evidence release`. After release, `parapet inspect` reports
`projection: object_released`.

### Reading lifecycle output

Human `restore`, `disable`, `inspect`, and `evidence release` output reports
three separate things on separate lines:

- `original decision:` the receipt summary the engine recorded, such as
  `allowed; effect verified` or `blocked before execution`. Cleanup never
  changes it. It reads `not recorded` before the engine has recorded a request,
  `not settled (...)` while the run is still open, and
  `not available (object_released)` after `evidence release`.
- `file cleanup:` the current state of the file effect in the effect root, as
  reported by the live owning engine (`clear`, `not run (restore required)`,
  `indeterminate`, and so on). `inspect` and `evidence release` print
  `file cleanup: not recorded in protected evidence` because only the live
  engine tracks it.
- `evidence cleanup:` the state of the protected receipt evidence
  (`not started`, `clear` after release, and so on).

When the next valid command is known, the last line names it with a
`<state-dir>` placeholder, for example
`next: parapet disable --state-dir <state-dir>` after a clear restore or
`next: parapet restore --state-dir <state-dir>` when `disable` refuses because
file cleanup has not run. An `indeterminate` file cleanup (for example,
undeclared files in the effect root) prints
`next: remove undeclared files from the effect root, then parapet restore --state-dir <state-dir>`.
Human output does not echo paths or the control token. JSON output for these
commands is unchanged, including the `a6_booking_required` field of
`parapet-cli-restore/v0`.

## Stock OpenAI-compatible clients with a local loopback server

An unmodified OpenAI-compatible client can use Parapet in ordinary
whole-message mode in front of a local OpenAI-compatible model server. No SDK,
provenance profile, or Parapet-specific request header is required.

```bash
parapet run --state-dir <state-dir> --config <config> --json \
  --upstream openai=http://127.0.0.1:<server-port>/v1 \
  --allow-loopback-http-upstream
```

Point the client's base URL at `http://127.0.0.1:<parapet-port>/v1` and keep
the server's model name and API key as the client normally sends them. Parapet
accepts `POST /v1/chat/completions`, forwards it to
`<upstream-base>/chat/completions`, and returns JSON or SSE streaming responses.
Client headers, including `Authorization` and application headers, are
forwarded unchanged except `Host`, `x-parapet-original-host`,
`x-parapet-provenance`, and `x-guard-trust*`; streaming requests also force
`Accept-Encoding: identity` upstream. A request blocked by the configured
layers returns HTTP 403 with `x-parapet-blocked-by` and is never sent upstream.

Prerequisites:

- The upstream must be a literal loopback IP (`127.0.0.0/8` or `[::1]`), passed
  through `--upstream` or `PARAPET_OPENAI_BASE_URL` together with
  `--allow-loopback-http-upstream`. `localhost`, LAN addresses, and plain-HTTP
  URLs without the opt-in flag fail closed with `upstream_config_invalid`.
  The ambient `OPENAI_BASE_URL` fallback never enables plain HTTP.
- The upstream base URL includes the server's API prefix (usually `/v1`);
  Parapet appends only `/chat/completions`.
- `GET /v1/models` is not proxied (HTTP 501). Configure the model name in the
  client directly instead of relying on model discovery.

Regression coverage: `parapet/tests/stock_openai_loopback_v0.rs`.

## Rust CLI acceptance with the offline fixture

This is **Rust CLI acceptance**: the installed Rust `parapet` CLI is the only
requester, sending one request through a loopback engine/proxy to the
deterministic offline fixture. It **does not exercise Python or TypeScript**;
the SDKs are verified separately below.

Supply a disposable `<state-dir>` and a disposable `<new-effect-root>` that
does not exist beforehand:

```bash
parapet accept --state-dir <state-dir> --config <config> --scenario allow_verified_write --effect-root <new-effect-root> --json
```

The expected upstream is `deterministic offline fixture (no real model
contacted)`.

Both directories are disposable. The command contacts only the deterministic
local fixture (`"real_model_contacted": false`). `<config>` must contain the
exact bytes of `parapet/tests/fixtures/cli_v0/accept_enforced_v0.yaml`;
`shadow_parse_error_l3_off` uses
`parapet/tests/fixtures/cli_v0/accept_shadow_l3_off_v0.yaml`. Any other config
exits 3 with `accept_config_mismatch` before creating state. JSON output reports
`"requester": "rust_cli"`, the installed CLI `requester_version`, the
engine-generated `run_id`, and the accepted `config_contract_hash`. Human
output prints `requester: Rust CLI <version>`.

For the full multi-journey deterministic offline workflow—covering `parapet
demo export` and the complete lifecycle across benign writes, unauthorized
blocks, injection-looking benign allows, verification mismatch, and undeclared
residue recovery from a neutral working directory—see the
[Deterministic Offline Demo Guide](../parapet/demo/v0/README.md).

## Installed Python SDK verification

This runs the Python SDK from the locally built wheel installed in
`<ROLE>/python-installed-venv`, from the role's neutral working directory
`<ROLE>/neutral-cwd` outside the checkout `<LANE>`, with `PYTHONPATH` unset and
the installed `parapet-engine` and `parapet` binaries bound explicitly:

```bash
cd <ROLE>/neutral-cwd
env -u PYTHONPATH PARAPET_NO_REAL_MODEL=1 PARAPET_EXPECTED_ENV_ROOT=<ROLE>/python-installed-venv PARAPET_ENGINE_BIN=<ROLE>/python-installed-venv/bin/parapet-engine PARAPET_CLI_BIN=<ROLE>/python-installed-venv/bin/parapet PYTHONDONTWRITEBYTECODE=1 <ROLE>/python-installed-venv/bin/python -m pytest -q --import-mode=importlib -p no:cacheprovider <LANE>/parapet-py/tests/test_e2e.py -k installed_python_sdk_is_actual_loopback_requester_rev9
```

The test requires the real `Path.cwd()` to equal `<ROLE>/neutral-cwd`, records
the installed `parapet` distribution version and `parapet.__file__`, and
requires the module below the installed venv's `site-packages` and outside
`<LANE>`. No checkout Python package is imported. It starts the installed
engine on a bound loopback endpoint, records that endpoint and the engine owner
PID, and makes an actual request through the installed SDK transport.

Its substitution witness is an in-process recorder under the installed SDK's
base HTTP transport. The request must be recorded exactly once, from the
installed `parapet/transport.py`, with the SDK's `x-parapet-original-host`
header, and the caller must receive that recorded response. A request made by
Rust `reqwest` or by the TypeScript SDK never reaches the recorder and fails
the test. Every non-loopback destination is refused, and the retained
`PARAPET_PY_INSTALLED_REQUESTER_RECORD` reports `"real_model_contacted": false`.

## Installed TypeScript SDK verification

The TypeScript SDK is a separate npm package (`@parapet-tech/parapet`) in its
own Node environment; it is not a Python-wheel payload. After building,
packing, and installing that package into the disposable consumer
`<ROLE>/ts-install/consumer`, run from the same neutral working directory
against the wheel-installed engine. `<NODE>` is the Node binary; Vitest comes
from the role's dev-dependency copy:

```bash
cd <ROLE>/neutral-cwd
E2E=1 PARAPET_NO_REAL_MODEL=1 PARAPET_ENGINE_PATH=<ROLE>/python-installed-venv/bin/parapet-engine PARAPET_CLI_BIN=<ROLE>/python-installed-venv/bin/parapet PARAPET_TS_INSTALLED_ENTRY=<ROLE>/ts-install/consumer/node_modules/@parapet-tech/parapet/dist/index.js PARAPET_TS_CHECKOUT_ROOT=<LANE>/parapet-ts PARAPET_TS_NEUTRAL_CWD=<ROLE>/neutral-cwd <NODE> <ROLE>/tool-cache/ts-seed/node_modules/vitest/vitest.mjs run --root <LANE>/parapet-ts --config <LANE>/parapet-ts/vitest.config.ts test/e2e.test.ts -t "installed TypeScript SDK is actual loopback requester Rev9"
```

The test requires `process.cwd()` to equal `<ROLE>/neutral-cwd`, records the
installed package name, version, and resolved entry (inside the consumer and
outside `<LANE>/parapet-ts`), records the bound loopback engine endpoint and
the actual request/response, and retains the
`PARAPET_TS_INSTALLED_REQUESTER_RECORD` substitution witness proving the
installed entry—not Rust `reqwest` or Python—made the request with
`"real_model_contacted": false`.

The same command with
`PARAPET_TS_INSTALLED_ENTRY=<LANE>/parapet-ts/src/index.ts` must exit nonzero
with `installed_entry_inside_checkout` before starting the engine.

A missing runtime, package, binary, or other installed input is a hard error in
every SDK verification command. There is no client fallback to Rust, the other
SDK, or a checkout source tree.
