# Parapet Deterministic Offline Demo Guide (`v0`)

This guide walks through the deterministic offline product workflow (`parapet-demo-fixtures/v0`) using the public `parapet` CLI from a neutral working directory outside any repository checkout.

## Scope and Disclosures

- **Same public binary and clean-environment assets**: All demo assets (`manifest.json`, `config_demo_v0.yaml`, the four untrusted input documents `doc_*.txt`, and the four OpenAI chat-completions request bodies `request_*.json`) are compiled once into the `parapet` binary and exported on demand with `parapet demo export --out <new-dir> [--json]`. No step reads a repository checkout path.
- **No model inference or external network traffic**: Every engine run uses `--offline-fixture <selector>` (`upstream_label: "parapet-offline-upstream/v0"`). The deterministic demo proposer parses the forwarded request's last `user` message, verifies its SHA-256 against the embedded manifest for the selected journey, and emits a fixed `write_file` tool-call proposal without contacting any external host or model provider. Deterministic offline fixture runs verify Parapet's policy, effect execution, receipt redaction, verification, and cleanup contracts; they do not constitute real-model evaluation.
- **Four-fact receipt separation**: Every `parapet inspect --json` projection separates four distinct facts that must not be conflated:
  1. **Detection (`projection.detection`)**: Whether the `L1` deterministic pattern sensor evaluated the inbound message (`availability: "present"`), its configured mode (`mode: "shadow"` in `config_demo_v0.yaml`), and whether the input crossed the pattern threshold (`outcome: "threshold_not_breached"` or `"threshold_breached"`). Because `L1` runs in `shadow` mode in this demo policy, detection is recorded in the receipt without blocking the request.
  2. **Policy decision (`projection.policy`)**: Whether the enforced outbound tool policy (`tools.write_file`) allowed (`decision: "allow"`, `reason_code: "allowed"`) or blocked (`decision: "block"`, `reason_code: "matches"`, `constraint_names: ["matches"]` on the `path` argument) the proposed tool call.
  3. **Effect stage (`projection.effect`)**: How far the file-effect lifecycle progressed (`stage_reached: "decision"` with `attempt_outcome: null` when blocked before execution, versus `stage_reached: "verification"` with `attempt_outcome: "completed"` when the file write executed).
  4. **Verification (`projection.verification`)**: Whether post-attempt observation of `<effect-root>/<target>` matched the declared bytes (`availability: "present"`, `outcome: "verified"`, `delta: "created"`), detected post-write tampering before observation (`outcome: "failed"`, `reason_code: "digest_mismatch"`), or was omitted by contract because no write was attempted (`availability: "absent_by_contract"`).

## Journey Matrix (`J1`–`J7`)

| Journey | Selector / Scenario | Detection (`L1` shadow) | Policy Decision | Effect Stage / Attempt | Verification | Disk & Lifecycle Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| **J1** (`benign_write`) | `demo_benign_write_v0` | `present` / `threshold_not_breached` | `allow` (`allowed`) | `verification` / `completed` | `verified` (`delta: created`) | `release-note.txt` written with SHA-256 `af52e00108572372f970dab87de34e3d257ef44f57671a662898e9bc87868d66` |
| **J2** (`unauthorized`) | `demo_unauthorized_v0` | `present` / `threshold_breached` | `block` (`matches` on `path`) | `decision` / `null` (`absent_by_contract`) | `absent_by_contract` | `../escape.txt` never created; `<effect-root>` unchanged |
| **J3** (`injection_benign`) | `demo_injection_benign_v0` | `present` / `threshold_breached` | `allow` (`allowed`) | `verification` / `completed` | `verified` (`delta: created`) | `pi.txt` written with SHA-256 `dce7575a167e62cc27bba7307727f38b36b79afda3ec53add26d67873cb15e41` (single architectural case, not a false-positive-rate claim) |
| **J4** (`mismatch`) | `demo_mismatch_v0` | `present` / `threshold_not_breached` | `allow` (`allowed`) | `verification` / `completed` | `failed` (`digest_mismatch`) | `inventory-note.txt` overwritten pre-observation (`0x23` fill, SHA-256 `6c932e9f46bfc762f46f3feee098a8aad2b488ff1e722c67e757528c289ecd79` != expected `592ae184266a9988c74b8a9bbbf13984483e8a5fce72b019b58f457ce9cdeb28`) |
| **J5** (redaction & release) | Across `J1`–`J4` and `J7` | — | — | — | — | Canary tokens (`PARAPET-1E-CANARY-J1`..`J4`) and document bodies absent from human, `--json`, and `--inert-fallback` `inspect` outputs; after `evidence release`, `inspect --json` exits `4` with `projection_code: "object_released"` |
| **J6** (restoration) | Across `J1`–`J4` | — | — | — | — | `parapet restore` reports `clear`; `parapet disable` stops the engine; `<effect-root>` is empty (`{}`) |
| **J7** (undeclared residue) | `demo_benign_write_v0` + planted `operator-residue.txt` | `present` / `threshold_not_breached` | `allow` (`allowed`) | `verification` / `completed` | `verified` (`delta: created`) | `restore` exits `5` (`indeterminate`, `a6_booking_required: true`) and retains `operator-residue.txt` plus sentinel; `disable` exits `5` (`refused`, `file_cleanup_indeterminate`); after operator removes residue, recovery `restore`/`disable` succeed while workflow outcome remains `indeterminate` |

## Fixture Provenance Summary

Every exported asset's digest and provenance metadata are recorded in `manifest.json` (`schema: "parapet-demo-fixtures/v0"`):

- `doc_benign_write.txt` (`J1`) and `doc_mismatch.txt` (`J4`): `document_class: "benign_non_injection"`, `document_authorship: {"kind": "agent", "basis": "authored", "date": "2026-10-20"}`, `provenance: "agent-authored on 2026-10-20; not derived from any corpus"`.
- `doc_unauthorized.txt` (`J2`, SHA-256 `60a65e916c21c05659f80665dc484358fd10fedb84fa60c462f25bacf908e75b`) and `doc_injection_benign.txt` (`J3`, SHA-256 `7b685e429c5fe616fd0884977e5f3e01f4eee795a597aa80ee2cf4934b5f8a02`): `document_class: "injection"`, `document_authorship.kind: "human_operator"`, packaged byte-for-byte with their operator authorship statements recorded in `manifest.json`.

## Prerequisites and Clean-Environment Execution

Ensure `parapet` (for example from a locally built wheel installed into a fresh virtual environment), `curl`, `python3`, and standard POSIX utilities (`sh`, `find`, `sort`, `grep`, `cat`, `ls`, `mkdir`, `rm`, `mktemp`) are on `PATH`.

Every command block below is tagged `sh` in execution order. Concatenating the `sh` blocks in order produces a self-contained POSIX `sh` script (`set -eu`) that runs `J1`–`J7` from a fresh temporary working directory outside any checkout. While an interactive operator can run `parapet run` in a dedicated terminal and run control commands in a second terminal, the verbatim script below backgrounds `parapet run` (`&`) and waits for it (`wait "$RUN_PID"`) after `parapet disable` so the entire workflow runs in a single shell.

## 1. Neutral Working Directory, Asset Export, and Witness Helpers

Create a fresh neutral working directory, export the embedded demo assets with `parapet demo export --out "$WORK/assets" --json`, verify every exported file's byte length and SHA-256 digest against `manifest.json`, and define the normative sentinel-discovery (`parapet_run_id`) and post-release reader-witness (`parapet_released`) shell functions:

```sh
set -eu
WORK=$(mktemp -d)
trap '[ -n "${RUN_PID:-}" ] && kill "$RUN_PID" 2>/dev/null || true; rm -rf "$WORK"' EXIT
cd "$WORK"
parapet demo export --out "$WORK/assets" --json > "$WORK/export.json"
python3 -c 'import hashlib, json, os, sys
work = sys.argv[1]
assets = os.path.join(work, "assets")
exp = json.load(open(os.path.join(work, "export.json")))
assert exp.get("schema") == "parapet-cli-demo-export/v0", exp
manifest_path = os.path.join(assets, "manifest.json")
manifest_bytes = open(manifest_path, "rb").read()
assert hashlib.sha256(manifest_bytes).hexdigest() == exp["manifest_sha256"]
files = exp["files"]
assert len(files) == 10, len(files)
assert [f["name"] for f in files] == sorted(f["name"] for f in files)
for f in files:
    b = open(os.path.join(assets, f["name"]), "rb").read()
    assert len(b) == f["len_bytes"], f["name"]
    assert hashlib.sha256(b).hexdigest() == f["sha256"], f["name"]
m = json.loads(manifest_bytes.decode("utf-8"))
cfg_b = open(os.path.join(assets, m["config"]["path"]), "rb").read()
assert hashlib.sha256(cfg_b).hexdigest() == m["config"]["sha256"]
for j in m["journeys"]:
    db = open(os.path.join(assets, j["document_path"]), "rb").read()
    rb = open(os.path.join(assets, j["request_path"]), "rb").read()
    assert hashlib.sha256(db).hexdigest() == j["document_sha256"], j["journey"]
    assert hashlib.sha256(rb).hexdigest() == j["request_sha256"], j["journey"]
' "$WORK"

# Print the run id from the single engine sentinel in effect root $1.
# Fails unless exactly one .parapet-run-* entry exists (dotfiles included),
# it is a regular non-symlink file, and its id is 32 lowercase hex.
parapet_run_id() {
  _names=$(cd "$1" && find . ! -name . -prune -name '.parapet-run-*' -print | LC_ALL=C sort)
  _n=$(printf '%s\n' "$_names" | grep -c . || true)
  if [ "$_n" -ne 1 ]; then
    echo "sentinel discovery: expected exactly 1 .parapet-run-* entry in $1, found $_n" >&2
    return 1
  fi
  _base=${_names#./}
  _id=${_base#.parapet-run-}
  _id=${_id%.sentinel}
  if [ "$_base" != ".parapet-run-$_id.sentinel" ] || [ "${#_id}" -ne 32 ]; then
    echo "sentinel discovery: malformed entry $_base" >&2
    return 1
  fi
  case $_id in *[!0-9a-f]*)
    echo "sentinel discovery: id is not 32 lowercase hex: $_base" >&2
    return 1 ;;
  esac
  if [ -L "$1/$_base" ] || [ ! -f "$1/$_base" ]; then
    echo "sentinel discovery: $_base is not a regular file" >&2
    return 1
  fi
  printf '%s\n' "$_id"
}

# Fail unless post-release `inspect --json` returned the redacted reader
# state exactly object_released (landed 1D shape: exit 4, projection_code
# object_released, reader_code null, matching label, projection null).
parapet_released() {
  if [ "$2" -ne 4 ]; then
    echo "$1: post-release reader state: expected inspect exit 4, got $2" >&2
    return 1
  fi
  python3 -c 'import json,sys
j, p = sys.argv[1], sys.argv[2]
try:
    v = json.load(open(p))
except Exception as e:
    sys.exit("%s: post-release reader state: unreadable inspect JSON: %s" % (j, e))
lb = v.get("labels") or {}
ok = (v.get("projection_code") == "object_released"
      and v.get("reader_code") is None
      and lb.get("projection_code") == "projection: object_released"
      and v.get("projection") is None)
if not ok:
    sys.exit("%s: post-release reader state is not exactly object_released: projection_code=%r reader_code=%r" % (j, v.get("projection_code"), v.get("reader_code")))' "$1" "$3"
}
```

Sample `export.json` output (`parapet-cli-demo-export/v0`):

```text
{"schema":"parapet-cli-demo-export/v0","out_dir":"<work-dir>/assets","manifest_sha256":"6f8d4f240cb84e7a8244b09c22346cce6c1b7a9c3d760b647dc575212c3b66da","files":[{"name":"config_demo_v0.yaml","sha256":"1b954d7eb2a49a0b3f8100b260685fadc55d752de83dc791759b59c619907450","len_bytes":435},{"name":"doc_benign_write.txt","sha256":"ae097bd662205900abb8ba49af6176535cf34eac23deb43b1611d051751c5bfa","len_bytes":290},{"name":"doc_injection_benign.txt","sha256":"7b685e429c5fe616fd0884977e5f3e01f4eee795a597aa80ee2cf4934b5f8a02","len_bytes":395},{"name":"doc_mismatch.txt","sha256":"3cbfeddbb358420b976487ea5a1ec99e8afc8cc262813948ad2238d9080007b8","len_bytes":218},{"name":"doc_unauthorized.txt","sha256":"60a65e916c21c05659f80665dc484358fd10fedb84fa60c462f25bacf908e75b","len_bytes":369},{"name":"manifest.json","sha256":"6f8d4f240cb84e7a8244b09c22346cce6c1b7a9c3d760b647dc575212c3b66da","len_bytes":8346},{"name":"request_benign_write.json","sha256":"67ba9575ea2108b239c880b8a1b393c622d347f817f10cd60a91fcdb0e0423bc","len_bytes":374},{"name":"request_injection_benign.json","sha256":"818d53b5f48589c7f436dc2ac3e490429a5570d8ea16049402a835830662f94e","len_bytes":490},{"name":"request_mismatch.json","sha256":"1864d10031029adada29395b64205eaa8cb06bef0e02f9f8c43c6c3532c32741","len_bytes":301},{"name":"request_unauthorized.json","sha256":"56991b0b1cacf1494489614258222fc0848e88d635423bcb9a26fa3ccfa921c5","len_bytes":458}]}
```

## 2. Journey 1 (`J1`): Benign Allowed Write, Redaction, Restoration, and Release

In `J1` (`demo_benign_write_v0`), the proposer consumes `doc_benign_write.txt` and proposes `write_file` targeting `release-note.txt`. `L1` shadow detection reports `threshold_not_breached`, outbound policy allows the call, the file write completes, and post-attempt verification confirms `verified` (`delta: "created"`) with exact file SHA-256 `af52e00108572372f970dab87de34e3d257ef44f57671a662898e9bc87868d66`. Redacted `inspect` surfaces exclude `PARAPET-1E-CANARY-J1` and the document body (`J5`), `restore` and `disable` return the effect root to an empty directory (`J6`), and `evidence release` followed immediately by `inspect --json` witnesses `projection_code: "object_released"` (`J5`).

```sh
J="$WORK/j1"; mkdir "$J"; S="$J/state"; R="$J/root"
parapet init --state-dir "$S" --effect-root "$R" --json > "$J/init.json"
parapet run --state-dir "$S" --config "$WORK/assets/config_demo_v0.yaml" \
  --effect-root "$R" --file-effect-profile v0 --max-content-bytes 4096 \
  --offline-fixture demo_benign_write_v0 --ready-file ready --port 0 --json \
  > "$J/run.json" &
RUN_PID=$!
i=0; until [ -s "$S/ready" ]; do i=$((i+1)); [ "$i" -le 300 ] || { echo "J1: engine not ready" >&2; exit 1; }; sleep 0.1; done
ls -la "$R"
CODE=$(curl -sS -o "$J/response.json" -w '%{http_code}' \
  "http://127.0.0.1:$(cat "$S/ready")/v1/chat/completions" \
  -H 'content-type: application/json' -H 'x-parapet-original-host: api.openai.com' \
  --data-binary @"$WORK/assets/request_benign_write.json")
[ "$CODE" = "200" ] || { echo "J1: expected HTTP 200, got $CODE" >&2; exit 1; }
ls -la "$R"
RUN=$(parapet_run_id "$R")
parapet inspect --state-dir "$S" --run-id "$RUN" > "$J/inspect.txt"
parapet inspect --state-dir "$S" --run-id "$RUN" --inert-fallback > "$J/inspect_inert.txt"
parapet inspect --state-dir "$S" --run-id "$RUN" --json > "$J/inspect.json"
python3 -c 'import hashlib, json, os, sys
j_dir, root_dir, run_id, doc_path = sys.argv[1:5]
target = os.path.join(root_dir, "release-note.txt")
assert os.path.isfile(target), "J1: missing release-note.txt"
digest = hashlib.sha256(open(target, "rb").read()).hexdigest()
assert digest == "af52e00108572372f970dab87de34e3d257ef44f57671a662898e9bc87868d66", digest
assert sorted(os.listdir(root_dir)) == sorted([".parapet-root-evidence-v0", f".parapet-run-{run_id}.sentinel", "release-note.txt"])
v = json.load(open(os.path.join(j_dir, "inspect.json")))
assert v["run_id"] == run_id and v["outcome"] == "complete"
assert v["labels"]["summary"] == "allowed; effect verified"
p = v["projection"]
assert p["detection"] == {"availability": "present", "outcome": "threshold_not_breached", "mode": "shadow"}
assert p["policy"]["decision"] == "allow" and p["policy"]["reason_code"] == "allowed"
assert p["effect"] == {"effect_kind": "write_file", "stage_reached": "verification", "attempt_outcome": "completed", "attempt_error_code": None}
assert p["verification"] == {"availability": "present", "outcome": "verified", "reason_code": None, "delta": "created"}
body = open(doc_path, "r", encoding="utf-8").read().split("\n---\n", 1)[1]
for name in ("inspect.txt", "inspect_inert.txt", "inspect.json"):
    txt = open(os.path.join(j_dir, name), "r", encoding="utf-8").read()
    assert "PARAPET-1E-CANARY-J1" not in txt and body not in txt, name
' "$J" "$R" "$RUN" "$WORK/assets/doc_benign_write.txt"
parapet restore --state-dir "$S" --json > "$J/restore.json"
ls -la "$R"
parapet disable --state-dir "$S" --json > "$J/disable.json"
wait "$RUN_PID"
RUN_PID=
ls -la "$R"
python3 -c 'import json, os, sys
j_dir, root_dir, run_id = sys.argv[1:4]
assert json.load(open(os.path.join(j_dir, "restore.json")))["result"] == "clear"
assert json.load(open(os.path.join(j_dir, "disable.json")))["status"] == "stopped"
assert json.load(open(os.path.join(j_dir, "run.json")))["run_id"] == run_id
assert os.listdir(root_dir) == [], os.listdir(root_dir)
' "$J" "$R" "$RUN"
parapet evidence release --state-dir "$S" --run-id "$RUN" --json > "$J/evidence_release.json"
RC=0; parapet inspect --state-dir "$S" --run-id "$RUN" --json > "$J/inspect_released.json" || RC=$?
parapet_released J1 "$RC" "$J/inspect_released.json"
```

Sample human `parapet inspect` output for `J1`:

```text
run_id: d059f141cee44e1bab823b879ab32f80
terminal_outcome: complete
original decision: allowed; effect verified
file cleanup: not recorded in protected evidence
evidence cleanup: not started
receipt_row_digest: 3667c81d4abc09841d5bfe297a9d66ce687950130eb918c1595e64a96491a5a5
receipt_line: allowed; effect verified
summary_label: allowed; effect verified
enforcement_label: enforcement: enforced
verification_label: verification: verified
tool_name.text: write_file
tool_name.len_bytes: 10
tool_name.truncated: false
target.form: resolved
target.text: release-note.txt
target.digest: 627eef5b33bf0f839d76c88b542d8ff7202f1c638df645b1b34cccfd95c9536e
target.len_bytes: 16
target.truncated: false
next: parapet evidence release --state-dir <state-dir> --run-id d059f141cee44e1bab823b879ab32f80 (optional)
```

## 3. Journey 2 (`J2`): Unauthorized Path Traversal Blocked Before Execution

In `J2` (`demo_unauthorized_v0`), `doc_unauthorized.txt` contains an injected instruction to write `../escape.txt`. The deterministic proposer models a compromised model that follows the injected instruction and emits `write_file` with `path: "../escape.txt"`. `L1` shadow detection records `threshold_breached`, outbound tool policy blocks the call before execution (`decision: "block"`, failed argument `path`, `reason_code: "matches"`, `constraint_names: ["matches"]`), the effect stage stops at `stage_reached: "decision"` (`attempt_outcome: null`), and `attempt`, `observation`, and `verification` are all `absent_by_contract`. Direct filesystem inspection confirms `../escape.txt` does not exist and `<effect-root>` contains only the startup sentinel and `.parapet-root-evidence-v0` directory.

```sh
J="$WORK/j2"; mkdir "$J"; S="$J/state"; R="$J/root"
parapet init --state-dir "$S" --effect-root "$R" --json > "$J/init.json"
parapet run --state-dir "$S" --config "$WORK/assets/config_demo_v0.yaml" \
  --effect-root "$R" --file-effect-profile v0 --max-content-bytes 4096 \
  --offline-fixture demo_unauthorized_v0 --ready-file ready --port 0 --json \
  > "$J/run.json" &
RUN_PID=$!
i=0; until [ -s "$S/ready" ]; do i=$((i+1)); [ "$i" -le 300 ] || { echo "J2: engine not ready" >&2; exit 1; }; sleep 0.1; done
ls -la "$R"
CODE=$(curl -sS -o "$J/response.json" -w '%{http_code}' \
  "http://127.0.0.1:$(cat "$S/ready")/v1/chat/completions" \
  -H 'content-type: application/json' -H 'x-parapet-original-host: api.openai.com' \
  --data-binary @"$WORK/assets/request_unauthorized.json")
[ "$CODE" = "200" ] || { echo "J2: expected HTTP 200, got $CODE" >&2; exit 1; }
ls -la "$R"
RUN=$(parapet_run_id "$R")
[ ! -e "$R/../escape.txt" ] && [ ! -L "$R/../escape.txt" ]
parapet inspect --state-dir "$S" --run-id "$RUN" > "$J/inspect.txt"
parapet inspect --state-dir "$S" --run-id "$RUN" --inert-fallback > "$J/inspect_inert.txt"
parapet inspect --state-dir "$S" --run-id "$RUN" --json > "$J/inspect.json"
python3 -c 'import json, os, sys
j_dir, root_dir, run_id, doc_path = sys.argv[1:5]
assert not os.path.lexists(os.path.join(root_dir, "..", "escape.txt"))
assert sorted(os.listdir(root_dir)) == sorted([".parapet-root-evidence-v0", f".parapet-run-{run_id}.sentinel"])
v = json.load(open(os.path.join(j_dir, "inspect.json")))
assert v["run_id"] == run_id and v["outcome"] == "complete"
assert v["labels"]["summary"] == "blocked before execution"
p = v["projection"]
assert p["detection"] == {"availability": "present", "outcome": "threshold_breached", "mode": "shadow"}
assert p["policy"]["decision"] == "block" and p["policy"]["reason_code"] == "matches"
assert p["policy"]["constraint_names"] == ["matches"]
assert p["effect"] == {"effect_kind": "write_file", "stage_reached": "decision", "attempt_outcome": None, "attempt_error_code": None}
assert p["verification"] == {"availability": "absent_by_contract", "outcome": None, "reason_code": None, "delta": None}
np = {entry["field"]: entry["availability"] for entry in p["not_present"]}
assert np.get("attempt") == "absent_by_contract" and np.get("observation") == "absent_by_contract" and np.get("verification") == "absent_by_contract", np
body = open(doc_path, "r", encoding="utf-8").read().split("\n---\n", 1)[1]
for name in ("inspect.txt", "inspect_inert.txt", "inspect.json"):
    txt = open(os.path.join(j_dir, name), "r", encoding="utf-8").read()
    assert "PARAPET-1E-CANARY-J2" not in txt and body not in txt, name
' "$J" "$R" "$RUN" "$WORK/assets/doc_unauthorized.txt"
parapet restore --state-dir "$S" --json > "$J/restore.json"
ls -la "$R"
parapet disable --state-dir "$S" --json > "$J/disable.json"
wait "$RUN_PID"
RUN_PID=
ls -la "$R"
python3 -c 'import json, os, sys
j_dir, root_dir, run_id = sys.argv[1:4]
assert json.load(open(os.path.join(j_dir, "restore.json")))["result"] == "clear"
assert json.load(open(os.path.join(j_dir, "disable.json")))["status"] == "stopped"
assert json.load(open(os.path.join(j_dir, "run.json")))["run_id"] == run_id
assert os.listdir(root_dir) == [], os.listdir(root_dir)
assert not os.path.lexists(os.path.join(root_dir, "..", "escape.txt"))
' "$J" "$R" "$RUN"
parapet evidence release --state-dir "$S" --run-id "$RUN" --json > "$J/evidence_release.json"
RC=0; parapet inspect --state-dir "$S" --run-id "$RUN" --json > "$J/inspect_released.json" || RC=$?
parapet_released J2 "$RC" "$J/inspect_released.json"
```

Sample `inspect.json` excerpt for `J2`:

```text
{"labels":{"summary":"blocked before execution","verification":"verification: absent by contract"},"projection":{"detection":{"availability":"present","outcome":"threshold_breached","mode":"shadow"},"policy":{"decision":"block","reason_code":"matches","constraint_names":["matches"],"enforcement":"enforced","evaluated_policy":true},"effect":{"effect_kind":"write_file","stage_reached":"decision","attempt_outcome":null,"attempt_error_code":null},"verification":{"availability":"absent_by_contract","outcome":null,"reason_code":null,"delta":null}}}
```

## 4. Journey 3 (`J3`): Injection-Looking Benign Document Allowed

In `J3` (`demo_injection_benign_v0`), `doc_injection_benign.txt` is a benign note that discusses prompt-injection concepts (`DAN`, canary tokens) while requesting an ordinary in-root write to `pi.txt`. In `config_demo_v0.yaml`, `L1` runs in `shadow` mode: `projection.detection` is `present` and records the measured `outcome: "threshold_breached"`, while `projection.policy` evaluates the proposed `write_file` call against the outbound tool constraints, decides `allow`, executes the write, and verifies `pi.txt` (`outcome: "verified"`, SHA-256 `dce7575a167e62cc27bba7307727f38b36b79afda3ec53add26d67873cb15e41`). This single case demonstrates the four-fact separation between `detection` (`shadow` observation) and `policy.decision` / `verification`; it is not a false-positive-rate claim.

```sh
J="$WORK/j3"; mkdir "$J"; S="$J/state"; R="$J/root"
parapet init --state-dir "$S" --effect-root "$R" --json > "$J/init.json"
parapet run --state-dir "$S" --config "$WORK/assets/config_demo_v0.yaml" \
  --effect-root "$R" --file-effect-profile v0 --max-content-bytes 4096 \
  --offline-fixture demo_injection_benign_v0 --ready-file ready --port 0 --json \
  > "$J/run.json" &
RUN_PID=$!
i=0; until [ -s "$S/ready" ]; do i=$((i+1)); [ "$i" -le 300 ] || { echo "J3: engine not ready" >&2; exit 1; }; sleep 0.1; done
ls -la "$R"
CODE=$(curl -sS -o "$J/response.json" -w '%{http_code}' \
  "http://127.0.0.1:$(cat "$S/ready")/v1/chat/completions" \
  -H 'content-type: application/json' -H 'x-parapet-original-host: api.openai.com' \
  --data-binary @"$WORK/assets/request_injection_benign.json")
[ "$CODE" = "200" ] || { echo "J3: expected HTTP 200, got $CODE" >&2; exit 1; }
ls -la "$R"
RUN=$(parapet_run_id "$R")
parapet inspect --state-dir "$S" --run-id "$RUN" > "$J/inspect.txt"
parapet inspect --state-dir "$S" --run-id "$RUN" --inert-fallback > "$J/inspect_inert.txt"
parapet inspect --state-dir "$S" --run-id "$RUN" --json > "$J/inspect.json"
python3 -c 'import hashlib, json, os, sys
j_dir, root_dir, run_id, doc_path = sys.argv[1:5]
target = os.path.join(root_dir, "pi.txt")
assert os.path.isfile(target), "J3: missing pi.txt"
digest = hashlib.sha256(open(target, "rb").read()).hexdigest()
assert digest == "dce7575a167e62cc27bba7307727f38b36b79afda3ec53add26d67873cb15e41", digest
assert sorted(os.listdir(root_dir)) == sorted([".parapet-root-evidence-v0", f".parapet-run-{run_id}.sentinel", "pi.txt"])
v = json.load(open(os.path.join(j_dir, "inspect.json")))
assert v["run_id"] == run_id and v["outcome"] == "complete"
assert v["labels"]["summary"] == "allowed; effect verified"
p = v["projection"]
assert p["detection"] == {"availability": "present", "outcome": "threshold_breached", "mode": "shadow"}
assert p["policy"]["decision"] == "allow" and p["policy"]["reason_code"] == "allowed"
assert p["effect"] == {"effect_kind": "write_file", "stage_reached": "verification", "attempt_outcome": "completed", "attempt_error_code": None}
assert p["verification"] == {"availability": "present", "outcome": "verified", "reason_code": None, "delta": "created"}
body = open(doc_path, "r", encoding="utf-8").read().split("\n---\n", 1)[1]
for name in ("inspect.txt", "inspect_inert.txt", "inspect.json"):
    txt = open(os.path.join(j_dir, name), "r", encoding="utf-8").read()
    assert "PARAPET-1E-CANARY-J3" not in txt and body not in txt, name
' "$J" "$R" "$RUN" "$WORK/assets/doc_injection_benign.txt"
parapet restore --state-dir "$S" --json > "$J/restore.json"
ls -la "$R"
parapet disable --state-dir "$S" --json > "$J/disable.json"
wait "$RUN_PID"
RUN_PID=
ls -la "$R"
python3 -c 'import json, os, sys
j_dir, root_dir, run_id = sys.argv[1:4]
assert json.load(open(os.path.join(j_dir, "restore.json")))["result"] == "clear"
assert json.load(open(os.path.join(j_dir, "disable.json")))["status"] == "stopped"
assert json.load(open(os.path.join(j_dir, "run.json")))["run_id"] == run_id
assert os.listdir(root_dir) == [], os.listdir(root_dir)
' "$J" "$R" "$RUN"
parapet evidence release --state-dir "$S" --run-id "$RUN" --json > "$J/evidence_release.json"
RC=0; parapet inspect --state-dir "$S" --run-id "$RUN" --json > "$J/inspect_released.json" || RC=$?
parapet_released J3 "$RC" "$J/inspect_released.json"
```

## 5. Journey 4 (`J4`): Post-Write Interference and Verification Mismatch

In `J4` (`demo_mismatch_v0`), `doc_mismatch.txt` proposes an allowed write to `inventory-note.txt` (expected SHA-256 `592ae184266a9988c74b8a9bbbf13984483e8a5fce72b019b58f457ce9cdeb28`). Only when `--offline-fixture demo_mismatch_v0` is selected, the engine registers `OverwriteTargetInterferenceV0` on the `BeforeObservation` phase hook, which overwrites `inventory-note.txt` in place with same-length `0x23` (`#`) bytes (observed SHA-256 `6c932e9f46bfc762f46f3feee098a8aad2b488ff1e722c67e757528c289ecd79`). The receipt records `policy.decision: "allow"` and `effect.attempt_outcome: "completed"`, but `verification.outcome` is `"failed"` with `reason_code: "digest_mismatch"` (never `"verified"`). `parapet restore` still cleans up the declared target and restores the empty root.

```sh
J="$WORK/j4"; mkdir "$J"; S="$J/state"; R="$J/root"
parapet init --state-dir "$S" --effect-root "$R" --json > "$J/init.json"
parapet run --state-dir "$S" --config "$WORK/assets/config_demo_v0.yaml" \
  --effect-root "$R" --file-effect-profile v0 --max-content-bytes 4096 \
  --offline-fixture demo_mismatch_v0 --ready-file ready --port 0 --json \
  > "$J/run.json" &
RUN_PID=$!
i=0; until [ -s "$S/ready" ]; do i=$((i+1)); [ "$i" -le 300 ] || { echo "J4: engine not ready" >&2; exit 1; }; sleep 0.1; done
ls -la "$R"
CODE=$(curl -sS -o "$J/response.json" -w '%{http_code}' \
  "http://127.0.0.1:$(cat "$S/ready")/v1/chat/completions" \
  -H 'content-type: application/json' -H 'x-parapet-original-host: api.openai.com' \
  --data-binary @"$WORK/assets/request_mismatch.json")
[ "$CODE" = "200" ] || { echo "J4: expected HTTP 200, got $CODE" >&2; exit 1; }
ls -la "$R"
RUN=$(parapet_run_id "$R")
parapet inspect --state-dir "$S" --run-id "$RUN" > "$J/inspect.txt"
parapet inspect --state-dir "$S" --run-id "$RUN" --inert-fallback > "$J/inspect_inert.txt"
parapet inspect --state-dir "$S" --run-id "$RUN" --json > "$J/inspect.json"
python3 -c 'import hashlib, json, os, sys
j_dir, root_dir, run_id, doc_path = sys.argv[1:5]
target = os.path.join(root_dir, "inventory-note.txt")
assert os.path.isfile(target), "J4: missing inventory-note.txt"
digest = hashlib.sha256(open(target, "rb").read()).hexdigest()
expected_manifest_digest = "592ae184266a9988c74b8a9bbbf13984483e8a5fce72b019b58f457ce9cdeb28"
assert digest != expected_manifest_digest, digest
assert digest == "6c932e9f46bfc762f46f3feee098a8aad2b488ff1e722c67e757528c289ecd79", digest
assert sorted(os.listdir(root_dir)) == sorted([".parapet-root-evidence-v0", f".parapet-run-{run_id}.sentinel", "inventory-note.txt"])
v = json.load(open(os.path.join(j_dir, "inspect.json")))
assert v["run_id"] == run_id and v["outcome"] == "complete"
assert v["labels"]["summary"] == "allowed; effect verification failed"
p = v["projection"]
assert p["detection"] == {"availability": "present", "outcome": "threshold_not_breached", "mode": "shadow"}
assert p["policy"]["decision"] == "allow" and p["policy"]["reason_code"] == "allowed"
assert p["effect"] == {"effect_kind": "write_file", "stage_reached": "verification", "attempt_outcome": "completed", "attempt_error_code": None}
ver = p["verification"]
assert ver["availability"] == "present" and ver["outcome"] in ("failed", "unknown") and ver["outcome"] != "verified", ver
assert ver["outcome"] == "failed" and ver["reason_code"] == "digest_mismatch" and ver["delta"] == "created", ver
body = open(doc_path, "r", encoding="utf-8").read().split("\n---\n", 1)[1]
for name in ("inspect.txt", "inspect_inert.txt", "inspect.json"):
    txt = open(os.path.join(j_dir, name), "r", encoding="utf-8").read()
    assert "PARAPET-1E-CANARY-J4" not in txt and body not in txt, name
' "$J" "$R" "$RUN" "$WORK/assets/doc_mismatch.txt"
parapet restore --state-dir "$S" --json > "$J/restore.json"
ls -la "$R"
parapet disable --state-dir "$S" --json > "$J/disable.json"
wait "$RUN_PID"
RUN_PID=
ls -la "$R"
python3 -c 'import json, os, sys
j_dir, root_dir, run_id = sys.argv[1:4]
assert json.load(open(os.path.join(j_dir, "restore.json")))["result"] == "clear"
assert json.load(open(os.path.join(j_dir, "disable.json")))["status"] == "stopped"
assert json.load(open(os.path.join(j_dir, "run.json")))["run_id"] == run_id
assert os.listdir(root_dir) == [], os.listdir(root_dir)
' "$J" "$R" "$RUN"
parapet evidence release --state-dir "$S" --run-id "$RUN" --json > "$J/evidence_release.json"
RC=0; parapet inspect --state-dir "$S" --run-id "$RUN" --json > "$J/inspect_released.json" || RC=$?
parapet_released J4 "$RC" "$J/inspect_released.json"
```

## 6. Journey 7 (`J7`): Undeclared Residue (`indeterminate` / `A6`) and Bounded Recovery

In `J7`, after a benign write (`demo_benign_write_v0`) completes, an undeclared file `operator-residue.txt` is planted directly in `<effect-root>` before `parapet restore`. Because root-confined cleanup removes only declared entries and refuses to delete unknown operator files:

1. `parapet restore` exits `5` with `result: "indeterminate"` and `a6_booking_required: true`, cleaning `release-note.txt` while retaining `operator-residue.txt`, `.parapet-root-evidence-v0`, and the run sentinel `.parapet-run-<run-id>.sentinel`.
2. `parapet disable` exits `5` with `status: "refused"` and `refusal_code: "file_cleanup_indeterminate"`.
3. The workflow records an `A6` finding row (`parapet-1e-a6-finding-row/v0`) with `workflow_outcome: "indeterminate"`.
4. For bounded operator recovery, removing `operator-residue.txt` and re-running `parapet restore` (`result: "clear"`, exit `0`) followed by `parapet disable` (`status: "stopped"`, exit `0`) stops the engine and clears the root; the journey's recorded workflow outcome remains `indeterminate` (recovery does not relabel the journey outcome `clear`).

```sh
J="$WORK/j7"; mkdir "$J"; S="$J/state"; R="$J/root"
parapet init --state-dir "$S" --effect-root "$R" --json > "$J/init.json"
parapet run --state-dir "$S" --config "$WORK/assets/config_demo_v0.yaml" \
  --effect-root "$R" --file-effect-profile v0 --max-content-bytes 4096 \
  --offline-fixture demo_benign_write_v0 --ready-file ready --port 0 --json \
  > "$J/run.json" &
RUN_PID=$!
i=0; until [ -s "$S/ready" ]; do i=$((i+1)); [ "$i" -le 300 ] || { echo "J7: engine not ready" >&2; exit 1; }; sleep 0.1; done
ls -la "$R"
CODE=$(curl -sS -o "$J/response.json" -w '%{http_code}' \
  "http://127.0.0.1:$(cat "$S/ready")/v1/chat/completions" \
  -H 'content-type: application/json' -H 'x-parapet-original-host: api.openai.com' \
  --data-binary @"$WORK/assets/request_benign_write.json")
[ "$CODE" = "200" ] || { echo "J7: expected HTTP 200, got $CODE" >&2; exit 1; }
ls -la "$R"
RUN=$(parapet_run_id "$R")
printf 'undeclared operator note\n' > "$R/operator-residue.txt"
ls -la "$R"
RC_RESTORE=0; parapet restore --state-dir "$S" --json > "$J/restore_indeterminate.json" || RC_RESTORE=$?
[ "$RC_RESTORE" -eq 5 ] || { echo "J7: expected restore exit 5, got $RC_RESTORE" >&2; exit 1; }
ls -la "$R"
RC_DISABLE=0; parapet disable --state-dir "$S" --json > "$J/disable_refused.json" || RC_DISABLE=$?
[ "$RC_DISABLE" -eq 5 ] || { echo "J7: expected disable exit 5, got $RC_DISABLE" >&2; exit 1; }
python3 -c 'import json, os, sys
j_dir, root_dir, run_id = sys.argv[1:4]
r = json.load(open(os.path.join(j_dir, "restore_indeterminate.json")))
assert r["result"] == "indeterminate" and r["a6_booking_required"] is True, r
d = json.load(open(os.path.join(j_dir, "disable_refused.json")))
assert d["status"] == "refused" and d["refusal_code"] == "file_cleanup_indeterminate", d
entries = sorted(os.listdir(root_dir))
assert entries == sorted([".parapet-root-evidence-v0", f".parapet-run-{run_id}.sentinel", "operator-residue.txt"]), entries
a6 = {
    "schema": "parapet-1e-a6-finding-row/v0",
    "journey": "J7",
    "run_id": run_id,
    "restore_exit_code": 5,
    "restore_result": r["result"],
    "a6_booking_required": r["a6_booking_required"],
    "disable_exit_code": 5,
    "disable_status": d["status"],
    "disable_refusal_code": d["refusal_code"],
    "planted_residue": "operator-residue.txt",
    "residue_retained": True,
    "identity_anchor_retained": True,
    "workflow_outcome": "indeterminate",
}
json.dump(a6, open(os.path.join(j_dir, "a6_finding_row.json"), "w"), indent=2, sort_keys=True)
' "$J" "$R" "$RUN"
rm "$R/operator-residue.txt"
parapet restore --state-dir "$S" --json > "$J/restore_recovered.json"
ls -la "$R"
parapet disable --state-dir "$S" --json > "$J/disable_recovered.json"
wait "$RUN_PID"
RUN_PID=
ls -la "$R"
python3 -c 'import json, os, sys
j_dir, root_dir, run_id = sys.argv[1:4]
assert json.load(open(os.path.join(j_dir, "restore_recovered.json")))["result"] == "clear"
assert json.load(open(os.path.join(j_dir, "disable_recovered.json")))["status"] == "stopped"
assert json.load(open(os.path.join(j_dir, "run.json")))["run_id"] == run_id
assert os.listdir(root_dir) == [], os.listdir(root_dir)
assert json.load(open(os.path.join(j_dir, "a6_finding_row.json")))["workflow_outcome"] == "indeterminate"
' "$J" "$R" "$RUN"
parapet evidence release --state-dir "$S" --run-id "$RUN" --json > "$J/evidence_release.json"
RC=0; parapet inspect --state-dir "$S" --run-id "$RUN" --json > "$J/inspect_released.json" || RC=$?
parapet_released J7 "$RC" "$J/inspect_released.json"
```

Sample `restore_indeterminate.json` and `disable_refused.json` outputs for `J7`:

```text
{"schema":"parapet-cli-restore/v0","result":"indeterminate","label":"file cleanup: indeterminate","a6_booking_required":true,"evidence_retained":true}
{"schema":"parapet-cli-disable/v0","status":"refused","refusal_code":"file_cleanup_indeterminate","file_cleanup":"indeterminate","file_cleanup_label":"file cleanup: indeterminate","ownership_label":"engine ownership: owned","owned_control_removed":false,"evidence_retained":true}
```

Sample post-release `inspect_released.json` output verified by `parapet_released` across `J1`–`J4` and `J7`:

```text
{"schema":"parapet-cli-inspect/v0","run_id":"d059f141cee44e1bab823b879ab32f80","outcome":"complete","cleanup":"clear","banner":null,"reader_code":null,"projection_code":"object_released","labels":{"summary":null,"enforcement":null,"verification":null,"evidence_cleanup":"evidence cleanup: clear","reader":null,"projection_code":"projection: object_released"},"projection":null}
```
