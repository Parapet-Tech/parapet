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
`parapet run` reports it when it exits after `parapet disable`: `"run_id"` in
JSON output and `run_id: <run-id>` in human output (`"run_id": null` and
`run_id: not_applicable` for a run without `--effect-root`). Use that
CLI-reported value for post-disable `inspect` and `evidence release` steps.
While the engine is live, the same identity is witnessed by the
`<new-effect-root>/.parapet-run-<run-id>.sentinel` file created at startup.

`parapet disable` never performs `restore` itself. It is an observational
shutdown gate that refuses to stop an owned effect-enabled engine until file
cleanup is already `clear` (after `parapet restore`) or `not_applicable`.
Stopping the engine retains protected receipt evidence, so post-restore and
post-disable `parapet inspect` continue to succeed until you optionally run
`parapet evidence release`. After release, `parapet inspect` reports
`reader: object_released`.

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
