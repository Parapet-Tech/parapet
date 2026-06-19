# Parapet Repro Contracts

This directory holds the public reproducibility inventory and the small schema
surface needed to run or audit published result artifacts.

The v0 shape is intentionally thin:

- `index.json` names artifacts, owners, release-boundary profiles, primitive
  chains, expected slots, and required primitive-integrity guards.
- `schemas/inventory.v0.schema.json` validates the public inventory.
- `schemas/plan.v0.schema.json` validates a resolved immutable run plan.
- `schemas/primitive-guard.v0.schema.json` validates reusable guard specs under
  `guards/`.

Result-receipt envelope validation is intentionally deferred until the
sensor/l1-owned receipt schema is relocated into the public `repro/` tree. The
inventory and runner should reference that schema; routing-owned glue must not
hand-copy the verification-rung field set.

The v0 runner still owns referential checks that JSON Schema cannot express
portably: artifact owners must exist in `owners`, `paper_owner` must exist when
present, input keys must match the selected primitive specs, output slots must
match the resolved primitive aliases, and receipt gates must enforce
`resample_unit == independence_unit` through the sensor/l1-owned receipt schema.

Generated run output belongs under ignored local paths such as `repro/out/`.
Only public-safe frozen receipts should be committed under `repro/receipts/`
after operator approval.

## Validation POC

`validate_contract.py` is a minimal, stdlib-only validation POC for this
contract slice. It is not the runner: it does not resolve or execute primitive
chains, fetch data, emit metrics, or read receipts. It checks the cheap
structural invariants the runner design owns before any chain resolves:

- every committed `repro/*.json` parses and each schema declares its expected
  `schema_version` const;
- artifact `owner` and `paper_owner` exist in `owners` (referential, not
  expressible in JSON Schema);
- every `artifact_id` is unique, because `resolve` uses it as the primary key;
- each `expected_outputs` key is `<alias>.<slot>` and `<alias>` is a declared
  `chain[].as` alias;
- release-boundary profiles carry their conditionally-required justification
  fields;
- comparative / superiority / noninferiority claims set `requires_ci: true`;
- inventory paths reject absolute paths, traversal segments, and blocked local
  data/run roots aligned with `scripts/check_no_data_commit.py`, even when no
  JSON Schema validator is installed.

The release-boundary and comparative-CI invariants are also encoded in
`schemas/inventory.v0.schema.json`, and path safety is encoded in the schema's
`repoPath` definition; they are re-checked in stdlib so the runner fails closed
even without a JSON Schema validator. When `jsonschema` is installed the script
additionally validates the schema files themselves and validates `index.json`
against the inventory schema as bonus checks. Synthetic fixtures (no private
data, no receipts) exercise each invariant since the live inventory currently
has zero artifacts.

Run it:

```bash
python3 repro/validate_contract.py             # live tree + synthetic self-test
python3 repro/validate_contract.py --self-test  # synthetic fixtures only
python3 repro/validate_contract.py --live        # committed tree only
```

It exits 0 on success and nonzero on any failure, so CI can invoke it directly.
The test functions are also pytest-collectable once pytest is wired in
(`pytest repro/validate_contract.py`). Receipt / verification-rung validation
and `resample_unit == independence_unit` stay deferred until the
sensor/l1-owned receipt schema is relocated into `repro/`.
