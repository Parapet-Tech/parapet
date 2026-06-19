#!/usr/bin/env python3
"""Minimal executable validation POC for the repro contract slice.

Scope (v0, intentionally thin):

  * Parse every committed JSON file under ``repro/`` and confirm the three
    schema files declare their expected ``schema_version`` / ``const`` anchors.
  * Validate the committed ``repro/index.json`` against the runner-owned
    referential invariants that JSON Schema cannot express portably.
  * Self-test those invariant checks against synthetic fixtures so the contract
    is exercised even though the live inventory currently has zero artifacts.

This is NOT the runner. It does not execute primitive chains, fetch data, emit
metrics, or touch receipts. It checks the cheap, structural invariants the
``repro_inventory_runner_design.md`` assigns to the runner before any chain is
resolved:

  * owner refs        - ``owner`` (and ``paper_owner`` if present) exist in
                        ``owners``.
  * output slot shape - every ``expected_outputs`` key is ``<alias>.<slot>`` and
                        ``<alias>`` is a declared ``chain[].as`` alias.
  * artifact ids      - every artifact id is unique, because ``resolve`` uses it
                        as the primary key.
  * release boundary  - the conditionally-required justification fields are
                        present for delayed/withheld/non-runnable profiles.
  * comparative CI    - comparative / superiority / noninferiority claims set
                        ``requires_ci: true``.
  * path safety       - inventory paths do not point at absolute paths,
                        traversal segments, or blocked local data/run roots.

Release-boundary and comparative-CI invariants are also encoded in
``schemas/inventory.v0.schema.json`` via ``if/then``; path safety is encoded in
the schema's ``repoPath`` definition. They are re-checked here in pure stdlib on
purpose: the runner must fail closed even in an environment that has no JSON
Schema validator installed. When ``jsonschema`` is importable this script
additionally runs the full schema validation as a bonus check.

Deliberately deferred (return as findings, not enforced here):

  * Result-receipt / verification-rung field validation - waits on the
    sensor/l1-owned receipt schema being relocated into ``repro/``.
  * ``resample_unit == independence_unit`` - same dependency.
  * Input-key-vs-primitive-spec checks - waits on ``repro/primitives/*`` specs.
  * Guard-file existence - waits on ``repro/guards/*`` being committed.

Usage:

    python3 repro/validate_contract.py            # validate live tree + self-test
    python3 repro/validate_contract.py --self-test # only the synthetic fixtures
    python3 repro/validate_contract.py --live      # only the committed tree

Exits 0 when every check passes, nonzero otherwise.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPRO_DIR = Path(__file__).resolve().parent
INDEX_PATH = REPRO_DIR / "index.json"
SCHEMA_DIR = REPRO_DIR / "schemas"

SCHEMA_FILES = {
    "inventory.v0.schema.json": "parapet-repro-index/v0",
    "plan.v0.schema.json": "parapet-repro-plan/v0",
    "primitive-guard.v0.schema.json": "parapet-primitive-guard/v0",
}

# Closed vocabulary of release-boundary profiles and the extra fields each
# profile makes mandatory. Mirrors the runner design's release-boundary table.
PROFILE_REQUIRED_FIELDS = {
    "public": (),
    "delayed_public": ("operator_justification", "planned_public_condition"),
    "withheld": ("operator_justification", "withheld_boundary_ref"),
    "non_runnable_public_claim": ("operator_justification",),
}

# Claim verbs that may not stand on a point estimate alone.
CI_REQUIRED_CLAIM_TYPES = {"comparative", "superiority", "noninferiority"}

OUTPUT_SLOT_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
BLOCKED_PATH_PREFIXES = (
    "data",
    "runs",
    "models",
    "adjudication",
    "TheWall",
    "curated",
    "schema/eval/staging",
    "schema/eval/benign",
    "schema/eval/malicious",
    "schema/eval/training",
    "schema/eval/challenges",
    "schema/eval/heuristic_staged",
    "parapet-runner/runs",
    "parapet-data/curated",
)
BLOCKED_PATH_STARTS = ("parapet-data/curated_",)


# --------------------------------------------------------------------------
# Runner-owned referential invariants (pure stdlib).
# Each returns a list of human-readable error strings; empty means pass.
# --------------------------------------------------------------------------


def check_owner_refs(inventory: dict) -> list[str]:
    errors: list[str] = []
    owners = inventory.get("owners", {})
    for artifact in inventory.get("artifacts", []):
        aid = artifact.get("artifact_id", "<missing artifact_id>")
        owner = artifact.get("owner")
        if owner not in owners:
            errors.append(f"{aid}: owner '{owner}' is not declared in owners")
        paper_owner = artifact.get("paper_owner")
        if paper_owner is not None and paper_owner not in owners:
            errors.append(
                f"{aid}: paper_owner '{paper_owner}' is not declared in owners"
            )
    return errors


def check_output_slot_aliases(inventory: dict) -> list[str]:
    errors: list[str] = []
    for artifact in inventory.get("artifacts", []):
        aid = artifact.get("artifact_id", "<missing artifact_id>")
        aliases = {step.get("as") for step in artifact.get("chain", [])}
        for slot_key in artifact.get("expected_outputs", {}):
            if not OUTPUT_SLOT_RE.match(slot_key):
                errors.append(
                    f"{aid}: output slot '{slot_key}' is not '<alias>.<slot>' shape"
                )
                continue
            alias = slot_key.split(".", 1)[0]
            if alias not in aliases:
                errors.append(
                    f"{aid}: output slot '{slot_key}' references alias '{alias}' "
                    f"absent from chain aliases {sorted(a for a in aliases if a)}"
                )
    return errors


def check_artifact_id_uniqueness(inventory: dict) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for artifact in inventory.get("artifacts", []):
        aid = artifact.get("artifact_id", "<missing artifact_id>")
        if aid in seen:
            errors.append(f"{aid}: duplicate artifact_id")
        seen.add(aid)
    return errors


def check_release_boundary(inventory: dict) -> list[str]:
    errors: list[str] = []
    for artifact in inventory.get("artifacts", []):
        aid = artifact.get("artifact_id", "<missing artifact_id>")
        boundary = artifact.get("release_boundary", {})
        profile = boundary.get("profile")
        if profile not in PROFILE_REQUIRED_FIELDS:
            errors.append(f"{aid}: unknown release-boundary profile '{profile}'")
            continue
        for field in PROFILE_REQUIRED_FIELDS[profile]:
            value = boundary.get(field)
            if not (isinstance(value, str) and value.strip()):
                errors.append(
                    f"{aid}: profile '{profile}' requires non-empty '{field}'"
                )
    return errors


def check_comparative_claims_ci(inventory: dict) -> list[str]:
    errors: list[str] = []
    for artifact in inventory.get("artifacts", []):
        aid = artifact.get("artifact_id", "<missing artifact_id>")
        for claim in artifact.get("declared_claims", []):
            ctype = claim.get("claim_type")
            cid = claim.get("claim_id", "<missing claim_id>")
            if ctype in CI_REQUIRED_CLAIM_TYPES and claim.get("requires_ci") is not True:
                errors.append(
                    f"{aid}: claim '{cid}' is '{ctype}' but requires_ci is not true"
                )
    return errors


def _path_safety_errors(label: str, path: object) -> list[str]:
    if path is None:
        return []
    if not isinstance(path, str):
        return [f"{label}: path value must be a string"]

    errors: list[str] = []
    normalized = path.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if normalized.startswith("/"):
        errors.append(f"{label}: absolute path '{path}' is not public-safe")
    if ".." in parts:
        errors.append(f"{label}: traversal path '{path}' is not public-safe")
    for prefix in BLOCKED_PATH_PREFIXES:
        if normalized == prefix or normalized.startswith(f"{prefix}/"):
            errors.append(f"{label}: blocked private/output path '{path}'")
            break
    else:
        for prefix in BLOCKED_PATH_STARTS:
            if normalized.startswith(prefix):
                errors.append(f"{label}: blocked private/output path '{path}'")
                break
    return errors


def check_repo_paths(inventory: dict) -> list[str]:
    errors: list[str] = []

    for key, path in inventory.get("defaults", {}).items():
        errors.extend(_path_safety_errors(f"defaults.{key}", path))

    for artifact in inventory.get("artifacts", []):
        aid = artifact.get("artifact_id", "<missing artifact_id>")
        for key, path in artifact.get("inputs", {}).items():
            errors.extend(_path_safety_errors(f"{aid}: inputs.{key}", path))
        for key, path in artifact.get("expected_outputs", {}).items():
            errors.extend(
                _path_safety_errors(f"{aid}: expected_outputs.{key}", path)
            )
        boundary = artifact.get("release_boundary", {})
        if "withheld_boundary_ref" in boundary:
            errors.extend(
                _path_safety_errors(
                    f"{aid}: release_boundary.withheld_boundary_ref",
                    boundary.get("withheld_boundary_ref"),
                )
            )
        for guard in artifact.get("required_guards", []):
            guard_id = guard.get("guard_id", "<missing guard_id>")
            errors.extend(
                _path_safety_errors(
                    f"{aid}: required_guards.{guard_id}.guard_ref",
                    guard.get("guard_ref"),
                )
            )
    return errors


INVARIANTS = (
    ("owner_refs", check_owner_refs),
    ("output_slot_aliases", check_output_slot_aliases),
    ("artifact_id_uniqueness", check_artifact_id_uniqueness),
    ("release_boundary", check_release_boundary),
    ("comparative_claims_ci", check_comparative_claims_ci),
    ("repo_paths", check_repo_paths),
)


def validate_inventory(inventory: dict) -> list[str]:
    """Run all runner-owned referential invariants; return flat error list."""
    errors: list[str] = []
    for _name, check in INVARIANTS:
        errors.extend(check(inventory))
    return errors


# --------------------------------------------------------------------------
# Live-tree checks.
# --------------------------------------------------------------------------


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def check_live_tree() -> list[str]:
    errors: list[str] = []

    # 1. Every committed JSON file parses.
    for json_path in sorted(REPRO_DIR.rglob("*.json")):
        try:
            load_json(json_path)
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"{json_path.relative_to(REPRO_DIR)}: JSON parse failed: {exc}")

    # 2. Each schema declares its expected version anchor.
    for filename, expected_version in SCHEMA_FILES.items():
        schema_path = SCHEMA_DIR / filename
        if not schema_path.exists():
            errors.append(f"schemas/{filename}: missing")
            continue
        try:
            schema = load_json(schema_path)
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"schemas/{filename}: JSON parse failed: {exc}")
            continue
        const = schema.get("properties", {}).get("schema_version", {}).get("const")
        if const != expected_version:
            errors.append(
                f"schemas/{filename}: schema_version const '{const}' != "
                f"expected '{expected_version}'"
            )

    # 3. Committed inventory passes the referential invariants.
    try:
        inventory = load_json(INDEX_PATH)
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(f"index.json: load failed: {exc}")
        return errors
    errors.extend(f"index.json: {e}" for e in validate_inventory(inventory))

    # 4. Bonus: full JSON Schema validation when jsonschema is importable.
    try:
        import jsonschema  # type: ignore
    except ImportError:
        pass
    else:
        try:
            schema = load_json(SCHEMA_DIR / "inventory.v0.schema.json")
            for schema_path in sorted(SCHEMA_DIR.glob("*.json")):
                jsonschema.Draft202012Validator.check_schema(load_json(schema_path))
            jsonschema.validate(instance=inventory, schema=schema)
        except jsonschema.SchemaError as exc:  # type: ignore[attr-defined]
            errors.append(f"schema validation failed: {exc.message}")
        except jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
            errors.append(f"index.json: jsonschema validation failed: {exc.message}")
        except Exception as exc:  # noqa: BLE001 - schema authoring error surfaces here
            errors.append(f"inventory schema: jsonschema check errored: {exc}")

    return errors


# --------------------------------------------------------------------------
# Synthetic fixtures (no private data, no receipts).
# --------------------------------------------------------------------------


def _valid_inventory() -> dict:
    return {
        "schema_version": "parapet-repro-index/v0",
        "inventory_id": "synthetic-fixture",
        "owners": {
            "sensor": {"kind": "lane", "maintainer": "sensor-architect"},
            "operator": {"kind": "operator", "maintainer": "operator"},
        },
        "artifacts": [
            {
                "artifact_id": "synthetic.table1.row_auc",
                "title": "Synthetic AUC row",
                "artifact_type": "paper_table_row",
                "owner": "sensor",
                "paper_owner": "sensor",
                "release_boundary": {"profile": "public"},
                "chain": [
                    {"primitive": "run.command/v0", "as": "run_eval"},
                    {"primitive": "emit.metrics/v0", "as": "emit_metric"},
                ],
                "inputs": {"config": "schema/eval/example_config.yaml"},
                "expected_outputs": {
                    "emit_metric.result_receipt": "repro/receipts/synthetic/result.json"
                },
                "required_guards": [],
                "upstream_artifacts": [],
                "declared_claims": [
                    {
                        "claim_id": "auc-noninferiority",
                        "claim_type": "noninferiority",
                        "requires_ci": True,
                    }
                ],
            }
        ],
    }


# (label, mutation, the invariant name that must catch it)
def _fixture_cases():
    cases = []

    def mutate(fn):
        inv = _valid_inventory()
        fn(inv["artifacts"][0], inv)
        return inv

    cases.append(
        (
            "owner not in owners",
            mutate(lambda a, _inv: a.__setitem__("owner", "ghost")),
            "owner_refs",
        )
    )
    cases.append(
        (
            "paper_owner not in owners",
            mutate(lambda a, _inv: a.__setitem__("paper_owner", "ghost")),
            "owner_refs",
        )
    )
    cases.append(
        (
            "output slot alias not in chain",
            mutate(
                lambda a, _inv: a.__setitem__(
                    "expected_outputs", {"missing_alias.result_receipt": "repro/x.json"}
                )
            ),
            "output_slot_aliases",
        )
    )
    cases.append(
        (
            "output slot wrong shape",
            mutate(
                lambda a, _inv: a.__setitem__(
                    "expected_outputs", {"emit_metric": "repro/x.json"}
                )
            ),
            "output_slot_aliases",
        )
    )
    cases.append(
        (
            "duplicate artifact_id",
            mutate(lambda a, inv: inv["artifacts"].append(dict(a))),
            "artifact_id_uniqueness",
        )
    )
    cases.append(
        (
            "delayed_public missing planned_public_condition",
            mutate(
                lambda a, _inv: a.__setitem__(
                    "release_boundary",
                    {"profile": "delayed_public", "operator_justification": "ops note"},
                )
            ),
            "release_boundary",
        )
    )
    cases.append(
        (
            "withheld missing withheld_boundary_ref",
            mutate(
                lambda a, _inv: a.__setitem__(
                    "release_boundary",
                    {"profile": "withheld", "operator_justification": "ops note"},
                )
            ),
            "release_boundary",
        )
    )
    cases.append(
        (
            "non_runnable_public_claim missing operator_justification",
            mutate(
                lambda a, _inv: a.__setitem__(
                    "release_boundary", {"profile": "non_runnable_public_claim"}
                )
            ),
            "release_boundary",
        )
    )
    cases.append(
        (
            "comparative claim without CI",
            mutate(
                lambda a, _inv: a["declared_claims"][0].update(
                    {"claim_type": "comparative", "requires_ci": False}
                )
            ),
            "comparative_claims_ci",
        )
    )
    cases.append(
        (
            "absolute input path",
            mutate(lambda a, _inv: a.__setitem__("inputs", {"config": "/etc/passwd"})),
            "repo_paths",
        )
    )
    cases.append(
        (
            "traversal input path",
            mutate(
                lambda a, _inv: a.__setitem__(
                    "inputs", {"config": "../../private/keys.json"}
                )
            ),
            "repo_paths",
        )
    )
    cases.append(
        (
            "blocked private input path",
            mutate(
                lambda a, _inv: a.__setitem__(
                    "inputs", {"data_spec": "data/thewall/raw.jsonl"}
                )
            ),
            "repo_paths",
        )
    )
    cases.append(
        (
            "blocked TheWall input path",
            mutate(
                lambda a, _inv: a.__setitem__(
                    "inputs", {"data_spec": "TheWall/raw.jsonl"}
                )
            ),
            "repo_paths",
        )
    )
    cases.append(
        (
            "blocked schema eval corpus path",
            mutate(
                lambda a, _inv: a.__setitem__(
                    "inputs", {"data_spec": "schema/eval/malicious/x.jsonl"}
                )
            ),
            "repo_paths",
        )
    )
    cases.append(
        (
            "blocked parapet-data curated underscore path",
            mutate(
                lambda a, _inv: a.__setitem__(
                    "inputs", {"data_spec": "parapet-data/curated_v2/x.jsonl"}
                )
            ),
            "repo_paths",
        )
    )
    cases.append(
        (
            "blocked expected output path",
            mutate(
                lambda a, _inv: a.__setitem__(
                    "expected_outputs",
                    {"emit_metric.result_receipt": "parapet-runner/runs/leak.json"},
                )
            ),
            "repo_paths",
        )
    )
    cases.append(
        (
            "blocked defaults path",
            mutate(
                lambda _a, inv: inv.__setitem__(
                    "defaults", {"receipt_schema": "models/private/schema.json"}
                )
            ),
            "repo_paths",
        )
    )
    cases.append(
        (
            "blocked withheld boundary ref",
            mutate(
                lambda a, _inv: a.__setitem__(
                    "release_boundary",
                    {
                        "profile": "withheld",
                        "operator_justification": "ops note",
                        "withheld_boundary_ref": "adjudication/private.md",
                    },
                )
            ),
            "repo_paths",
        )
    )
    cases.append(
        (
            "blocked guard ref",
            mutate(
                lambda a, _inv: a.__setitem__(
                    "required_guards",
                    [
                        {
                            "guard_id": "synthetic.guard",
                            "guard_ref": "runs/private_guard.json",
                            "required_receipt": "primitive_guard.synthetic.guard",
                        }
                    ],
                )
            ),
            "repo_paths",
        )
    )
    return cases


# --------------------------------------------------------------------------
# Test functions (pytest-collectable AND runnable under plain python3).
# --------------------------------------------------------------------------


def test_valid_fixture_passes():
    assert validate_inventory(_valid_inventory()) == []


def test_each_invalid_fixture_is_caught():
    invariant_map = dict(INVARIANTS)
    for label, inventory, expected_invariant in _fixture_cases():
        # The whole-inventory validator must flag it.
        assert validate_inventory(inventory), f"expected failure not caught: {label}"
        # The specifically responsible invariant must be the one that fires.
        check = invariant_map[expected_invariant]
        assert check(inventory), (
            f"'{label}' should be caught by '{expected_invariant}'"
        )


def test_committed_tree_is_clean():
    assert check_live_tree() == []


def _run_self_tests() -> list[str]:
    failures: list[str] = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as exc:
                failures.append(f"{name}: {exc}")
    return failures


def main(argv: list[str]) -> int:
    run_live = "--self-test" not in argv
    run_self = "--live" not in argv

    errors: list[str] = []

    if run_live:
        live_errors = check_live_tree()
        if live_errors:
            print("LIVE TREE: FAIL")
            for err in live_errors:
                print(f"  - {err}")
        else:
            print("LIVE TREE: ok (parse + schema anchors + referential invariants)")
        errors.extend(live_errors)

    if run_self:
        self_failures = _run_self_tests()
        if self_failures:
            print("SELF-TEST: FAIL")
            for err in self_failures:
                print(f"  - {err}")
        else:
            print("SELF-TEST: ok (synthetic valid + invalid fixtures)")
        errors.extend(self_failures)

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
