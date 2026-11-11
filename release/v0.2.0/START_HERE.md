# Parapet v0.2.0 Evaluator: Start Here

This package is the shortest supported first-use path for Parapet v0.2.0 on
macOS arm64 with Python 3.10 or newer. It is a deterministic local fixture: it
does not contact a model provider, exercise a real model, or establish
production readiness. It is not an independent usability validation.

Full Python and TypeScript SDK verification is available from the tagged
`v0.2.0` source archive; those suites are not part of this 15-minute evaluator
journey.

## 1. Verify the package

From the unpacked evaluator root, verify every packaged file before running it:

```sh
shasum -a 256 -c EVALUATOR_SHA256SUMS
```

Stop if any entry is missing, unexpected, or fails verification.

## 2. Install the reviewed wheel

Create a disposable virtual environment. Ordinary installation of
`httpx==0.28.1` may contact the configured package index. If that dependency is
already available in an approved local cache, install it offline instead.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install httpx==0.28.1
.venv/bin/python -m pip install --no-deps \
  dist/parapet-0.2.0-py3-none-macosx_11_0_arm64.whl
.venv/bin/parapet --version
.venv/bin/parapet info --json
printf 'hello parapet' | .venv/bin/parapet scan --stdin --json
```

Both `.venv/bin/parapet` and `.venv/bin/parapet-engine` must exist. The clean
scan is a local installation check, not a security benchmark. Both CLI identity
outputs above must report `0.2.0`.

Bind the installed engine script to the wheel RECORD:

```sh
.venv/bin/python - <<'PY'
import base64, csv, hashlib, importlib.metadata, pathlib, sys

record = importlib.metadata.distribution("parapet").read_text("RECORD")
assert record is not None
rows = [row for row in csv.reader(record.splitlines()) if pathlib.PurePosixPath(row[0]).name == "parapet-engine"]
assert len(rows) == 1 and rows[0][1].startswith("sha256=") and rows[0][2].isdigit()
engine = pathlib.Path(sys.executable).parent / "parapet-engine"
observed = base64.urlsafe_b64encode(hashlib.sha256(engine.read_bytes()).digest()).rstrip(b"=").decode()
assert rows[0][1] == f"sha256={observed}" and int(rows[0][2]) == engine.stat().st_size
PY
```

`parapet-engine --version` is not implemented in this mechanics-only release;
that missing flag is an accepted, recorded residual rather than a skipped
check. No engine lifecycle receipt exposes `code_version` because the packaged
CLI does not provide a build-digest input. The engine version is therefore
bound only through the RECORD digest of the exact same-package build, while
`parapet --version` and `parapet info --json` provide the package-level v0.2.0
identity. Record that limitation explicitly; do not substitute a raw binary
search, which can match unrelated dependency versions.

## 3. Run the deterministic allowed and blocked cases

Activate the environment and follow the packaged
[`parapet/demo/v0/README.md`](parapet/demo/v0/README.md). Run at least:

- J1, which permits one bounded `write_file` to `release-note.txt`, then proves
  the exact file digest on disk;
- J2, which blocks the proposed `../escape.txt` traversal before execution and
  proves the escaped file does not exist.

The guide uses a deterministic offline upstream fixture. No real model or
external network request is part of these cases. Compare the redacted
`parapet inspect` projection with the direct filesystem witnesses; neither is
a substitute for the other.

## 4. Verify lifecycle and cleanup

For each protected run, keep the order shown in the demo guide:

1. inspect the generated run identifier;
2. restore the file effect;
3. disable the owning engine;
4. inspect the retained protected evidence;
5. optionally run `parapet evidence release`, then confirm the released reader
   state;
6. verify the declared effect root is empty and no engine process or listener
   remains.

Remove only the disposable evaluator environment and paths you created:

```sh
deactivate 2>/dev/null || true
rm -rf .venv
```

Record the outcome, elapsed time, wheel digest, deviations, and cleanup result
in [`RESULT_TEMPLATE.md`](RESULT_TEMPLATE.md). A failed checksum, unexpected
file, leftover process/listener, or non-empty declared effect root is a failed
run, not a result to waive.
