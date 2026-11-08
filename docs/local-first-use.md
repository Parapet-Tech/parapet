# Local First-Use and Verification Guide

This guide contains the detailed installation, protected-runtime, acceptance,
and SDK verification procedures. For the shortest installation path, start with
the [README Quick Start](../README.md#quick-start).

## Supported environment

The currently proven first-use environment is **macOS arm64 with Python 3.10 or
newer** using the wheel shipped in the unpacked evaluator at
`dist/parapet-0.1.0-py3-none-macosx_11_0_arm64.whl`. Start all evaluator
commands from the unpacked `parapet-v0.1.0-macos-arm64-evaluator` root; no
source checkout or local build is required. Parapet is not published to PyPI
or npm.

The protected runtime, evidence store, control socket, and file-effect
lifecycle are POSIX-only; Windows protected-runtime support is not supported
or claimed. Deterministic offline fixture checks do not constitute real-model
validation.

## Install the evaluator wheel

Create a disposable environment, install the exact pinned runtime dependency,
then install the reviewed wheel without resolving dependencies again:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install httpx==0.28.1
.venv/bin/python -m pip install --no-deps \
  dist/parapet-0.1.0-py3-none-macosx_11_0_arm64.whl
```

The `httpx==0.28.1` command normally resolves from the configured package index
and may use the network. For an offline install, first supply that exact
dependency and its transitive dependencies from an approved local cache; only
then use `--no-deps` for the Parapet wheel as shown. The wheel path is a local
file in the evaluator, never a network package name.

## Verify `site-packages` and installed binaries

Verify that `parapet` imports from the isolated environment's `site-packages`
and that both compiled programs are present:

```bash
.venv/bin/python -c "import parapet; print(parapet.__file__)"
test -x .venv/bin/parapet
test -x .venv/bin/parapet-engine
```

The wheel installs only the Python `parapet` package plus the `parapet` and
`parapet-engine` binaries. `parapet-fetch`, `parapet-scope-audit`, `pg2-diag`,
and `parapet-eval` are intentionally absent.

## First installed CLI checks

Run `info` and then a bounded `scan`:

```bash
.venv/bin/parapet info --json
printf 'hello parapet' | .venv/bin/parapet scan --stdin --json
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

Activate the evaluator environment and define every path before starting the
engine. Run these setup commands from the unpacked evaluator root:

```bash
. .venv/bin/activate
EVALUATOR_ROOT=$(pwd -P)
WORK=$(mktemp -d /tmp/parapet.XXXXXX)
printf 'WORK=%s\n' "$WORK"
STATE_DIR="$WORK/state"
EFFECT_ROOT="$WORK/effects"
CONFIG="$EVALUATOR_ROOT/parapet/demo/v0/config_demo_v0.yaml"
RUN_JSON="$WORK/run.json"
parapet init --state-dir "$STATE_DIR" --effect-root "$EFFECT_ROOT" --json
```

`WORK` is a short disposable root under `/tmp`, so the derived
`$STATE_DIR/control.sock` pathname does not grow with the evaluator's unpack
path. `STATE_DIR` is disposable engine state, `EFFECT_ROOT` is the new one-run
file effect root, `CONFIG` is packaged in the evaluator, and `RUN_JSON` is used
only by the JSON startup variant below. Choose either the human or JSON startup
variant. In both variants, **Terminal 1 is the engine terminal**: the
long-lived `parapet run` process stays there until `parapet disable` stops it.

Use Terminal 2 for the request and lifecycle commands. From the evaluator root
in Terminal 2, activate the same environment, copy the exact `WORK=/tmp/...`
value printed by Terminal 1, and recreate the derived values. Do not run
`mktemp` or `init` again in Terminal 2:

```bash
. .venv/bin/activate
EVALUATOR_ROOT=$(pwd -P)
WORK=/tmp/parapet.ABC123  # replace with Terminal 1's exact printed WORK value
STATE_DIR="$WORK/state"
EFFECT_ROOT="$WORK/effects"
CONFIG="$EVALUATOR_ROOT/parapet/demo/v0/config_demo_v0.yaml"
RUN_JSON="$WORK/run.json"
```

Both startup variants use the packaged witness helpers during the shared
lifecycle below. From
[Neutral Working Directory, Asset Export, and Witness Helpers](../parapet/demo/v0/README.md#1-neutral-working-directory-asset-export-and-witness-helpers),
copy the `parapet_run_id` and `parapet_released` functions into Terminal 2
without weakening their checks before choosing either startup variant.

### Human startup and discovery

In Terminal 1, omit `--json`. Readiness is the `status: ready` block printed
before the engine starts serving:

```bash
parapet run --state-dir "$STATE_DIR" --config "$CONFIG" \
  --effect-root "$EFFECT_ROOT" --file-effect-profile v0 \
  --max-content-bytes 4096 --offline-fixture demo_benign_write_v0 --port 0
```

The ready block supplies both values needed in Terminal 2:

```text
status: ready
port: <port>
run_id: <run-id>
mode: protected
next: send one request, then parapet restore --state-dir <state-dir>
```

Here `<port>` means the decimal port printed on the `port:` line, `<run-id>`
means the 32-character lowercase hexadecimal value printed on the `run_id:`
line, and `<state-dir>` in the hint means the already defined `STATE_DIR`.
Before using either value, assign the exact printed values in Terminal 2:

```bash
PORT=49152  # replace 49152 with the printed port
RUN_ID=0123456789abcdef0123456789abcdef  # replace with the printed run_id
```

The startup block never contains the control token or a path. After disable,
Terminal 1 prints `status: stopped` and repeats the same port and run ID.

### JSON startup and discovery

For this alternative, use a fresh `WORK`, `STATE_DIR`, and `EFFECT_ROOT` from
the setup block. In Terminal 1, `--ready-file ready` writes the selected port
to `$STATE_DIR/ready`; JSON stdout is redirected and remains empty until the
engine exits:

```bash
parapet run --state-dir "$STATE_DIR" --config "$CONFIG" \
  --effect-root "$EFFECT_ROOT" --file-effect-profile v0 \
  --max-content-bytes 4096 --offline-fixture demo_benign_write_v0 \
  --port 0 --ready-file ready --json > "$RUN_JSON"
```

In Terminal 2, wait for readiness, then read the port. Discover the run ID with
the copied `parapet_run_id` helper as shown:

```bash
until [ -s "$STATE_DIR/ready" ]; do sleep 0.1; done
PORT=$(cat "$STATE_DIR/ready")
RUN_ID=$(parapet_run_id "$EFFECT_ROOT")
```

The sentinel is the engine-generated identity; run-creating commands never
accept a caller-supplied run ID. After the common lifecycle below makes
Terminal 1 exit, `RUN_JSON` contains the sole `parapet-cli-run/v0` object.

### Request, lifecycle, and cleanup

Only after `PORT` and `RUN_ID` have been discovered, send the packaged
deterministic request from Terminal 2:

```bash
curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'content-type: application/json' \
  -H 'x-parapet-original-host: api.openai.com' \
  --data-binary \
  @"$EVALUATOR_ROOT/parapet/demo/v0/request_benign_write.json"
```

The effect-enabled engine is control-long-lived but its effect runtime is
one-shot. It accepts one non-streaming request, rejects streaming with HTTP 400
`effect_streaming_unsupported` without consuming the run, and rejects later
requests with HTTP 409 `effect_run_consumed`.

Keep the lifecycle order exact: inspect, restore, disable, post-disable
inspect, optional evidence release, then post-release inspect.

```bash
parapet inspect --state-dir "$STATE_DIR" --run-id "$RUN_ID" --json
parapet restore --state-dir "$STATE_DIR" --json
parapet disable --state-dir "$STATE_DIR" --json
parapet inspect --state-dir "$STATE_DIR" --run-id "$RUN_ID" --json

# Optional pair: release retained evidence, then confirm released reader state.
parapet evidence release --state-dir "$STATE_DIR" --run-id "$RUN_ID" --json
POST_RELEASE_CODE=0
parapet inspect --state-dir "$STATE_DIR" --run-id "$RUN_ID" --json \
  > "$WORK/post-release.json" || POST_RELEASE_CODE=$?
parapet_released evaluator "$POST_RELEASE_CODE" "$WORK/post-release.json"
```

For the JSON variant, now confirm that the terminal run ID matches the value
discovered from the sentinel:

```bash
python3 -c 'import json,sys
value=json.load(open(sys.argv[1]))
assert value["schema"] == "parapet-cli-run/v0"
assert value["run_id"] == sys.argv[2]' "$RUN_JSON" "$RUN_ID"
```

`parapet disable` never restores files itself. It refuses to stop an
effect-enabled engine until `parapet restore` reports cleanup as `clear` or
`not_applicable`. Protected evidence remains inspectable after disable until
the optional release; post-release inspect exits 4 with
`projection_code: "object_released"`.

After Terminal 1 has exited, require an empty effect root and remove only the
disposable paths created above:

```bash
test ! -e "$EFFECT_ROOT" || \
  test -z "$(find "$EFFECT_ROOT" -mindepth 1 -print -quit)"
rm -rf "$WORK"
deactivate
rm -rf .venv
```

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

## Maintainer verification from the tagged source archive

The evaluator asset intentionally omits the full Rust, Python, and TypeScript
test trees. Those tests are available in the
[tagged `v0.1.0` source archive](https://github.com/Parapet-Tech/parapet/archive/refs/tags/v0.1.0.tar.gz)
and are not part of the 15-minute evaluator journey. Download and unpack that
archive, change to its root, and define the source and disposable verification
paths before using the source-only commands below:

```bash
SOURCE_ROOT=$(pwd -P)
VERIFY_ROOT=$(mktemp -d)
PY_ENV="$VERIFY_ROOT/python-installed-venv"
NEUTRAL_CWD="$VERIFY_ROOT/neutral-cwd"
TS_CONSUMER="$VERIFY_ROOT/ts-install/consumer"
mkdir -p "$NEUTRAL_CWD" "$TS_CONSUMER"
```

### Maintainer wheel build

Build from the tagged source tree, not from the evaluator asset:

```bash
mkdir "$VERIFY_ROOT/dist"
(cd "$SOURCE_ROOT/parapet-py" && \
  maturin build --release --locked --out "$VERIFY_ROOT/dist")
WHEEL="$VERIFY_ROOT/dist/parapet-0.1.0-py3-none-macosx_11_0_arm64.whl"
test -f "$WHEEL"
python3 -m venv "$PY_ENV"
"$PY_ENV/bin/python" -m pip install httpx==0.28.1 pytest
"$PY_ENV/bin/python" -m pip install --no-deps "$WHEEL"
```

### Rust CLI acceptance with the offline fixture

This is Rust CLI acceptance, not Python or TypeScript SDK verification. Both
configuration files named here live only in the tagged source archive:

```bash
ACCEPT_STATE="$VERIFY_ROOT/accept-state"
ACCEPT_EFFECT="$VERIFY_ROOT/accept-effect"
ACCEPT_CONFIG="$SOURCE_ROOT/parapet/tests/fixtures/cli_v0/accept_enforced_v0.yaml"
"$PY_ENV/bin/parapet" accept --state-dir "$ACCEPT_STATE" \
  --config "$ACCEPT_CONFIG" --scenario allow_verified_write \
  --effect-root "$ACCEPT_EFFECT" --json
```

The `shadow_parse_error_l3_off` scenario instead uses
`$SOURCE_ROOT/parapet/tests/fixtures/cli_v0/accept_shadow_l3_off_v0.yaml`.
The expected upstream is the deterministic offline fixture, with
`"real_model_contacted": false`. Any other config exits 3 with
`accept_config_mismatch` before creating state.

The evaluator does include the complete multi-journey
[Deterministic Offline Demo Guide](../parapet/demo/v0/README.md), which uses
packaged assets and requires no source tree.

### Installed Python SDK verification

Run the installed SDK from the neutral directory, with the tagged source test
path supplied explicitly and `PYTHONPATH` unset:

```bash
cd "$NEUTRAL_CWD"
env -u PYTHONPATH \
  PARAPET_NO_REAL_MODEL=1 \
  PARAPET_EXPECTED_ENV_ROOT="$PY_ENV" \
  PARAPET_ENGINE_BIN="$PY_ENV/bin/parapet-engine" \
  PARAPET_CLI_BIN="$PY_ENV/bin/parapet" \
  PYTHONDONTWRITEBYTECODE=1 \
  "$PY_ENV/bin/python" -m pytest -q --import-mode=importlib \
  -p no:cacheprovider "$SOURCE_ROOT/parapet-py/tests/test_e2e.py" \
  -k installed_python_sdk_is_actual_loopback_requester_rev9
```

The test requires imports from the installed environment's `site-packages`,
outside `SOURCE_ROOT`; it starts the installed engine, makes one actual
loopback request through the installed SDK transport, refuses non-loopback
destinations, and records `"real_model_contacted": false`.

### Installed TypeScript SDK verification

The TypeScript SDK is a separate npm package, not a Python-wheel payload. Build,
pack, and install `@parapet-tech/parapet` from `$SOURCE_ROOT/parapet-ts` into
`TS_CONSUMER`. Then define its installed entry and the tagged source test tools:

```bash
NODE=$(command -v node)
VITEST="$SOURCE_ROOT/parapet-ts/node_modules/vitest/vitest.mjs"
TS_ENTRY="$TS_CONSUMER/node_modules/@parapet-tech/parapet/dist/index.js"
test -f "$NODE"
test -f "$VITEST"
test -f "$TS_ENTRY"
cd "$NEUTRAL_CWD"
E2E=1 \
PARAPET_NO_REAL_MODEL=1 \
PARAPET_ENGINE_PATH="$PY_ENV/bin/parapet-engine" \
PARAPET_CLI_BIN="$PY_ENV/bin/parapet" \
PARAPET_TS_INSTALLED_ENTRY="$TS_ENTRY" \
PARAPET_TS_CHECKOUT_ROOT="$SOURCE_ROOT/parapet-ts" \
PARAPET_TS_NEUTRAL_CWD="$NEUTRAL_CWD" \
"$NODE" "$VITEST" run --root "$SOURCE_ROOT/parapet-ts" \
  --config "$SOURCE_ROOT/parapet-ts/vitest.config.ts" test/e2e.test.ts \
  -t "installed TypeScript SDK is actual loopback requester Rev9"
```

The test requires the resolved package entry inside `TS_CONSUMER` and outside
the tagged source tree, then proves that installed TypeScript—not Rust or
Python—made the loopback request with `"real_model_contacted": false`. A
missing runtime, package, binary, or test input is a hard error; there is no
fallback to another client.

After maintainer verification, remove only its disposable root:

```bash
rm -rf "$VERIFY_ROOT"
```
