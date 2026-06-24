#!/usr/bin/env python3
"""Minimal executable validation POC for the repro contract slice.

Scope (v0, intentionally thin):

  * Parse every committed JSON file under ``repro/`` and confirm the three
    schema files declare their expected ``schema_version`` / ``const`` anchors.
  * Validate the committed ``repro/index.json`` and any frozen
    ``repro/receipts/**/*.json`` receipts against the runner-owned invariants
    that JSON Schema cannot express portably.
  * Self-test those invariant checks against synthetic fixtures so the contract
    is exercised even though the live inventory currently has zero artifacts.

This is NOT the runner. It does not execute primitive chains, fetch data, or emit
metrics. It checks the cheap, structural invariants the
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
  * receipt envelope  - frozen result/verification receipts carry the DoD-6
                        verification-rung fields and public-safe references.

Release-boundary and comparative-CI invariants are also encoded in
``schemas/inventory.v0.schema.json`` via ``if/then``; path safety is encoded in
the schema's ``repoPath`` definition. They are re-checked here in pure stdlib on
purpose: the runner must fail closed even in an environment that has no JSON
Schema validator installed. When ``jsonschema`` is importable this script
additionally runs the full schema validation as a bonus check.

Deliberately deferred (return as findings, not enforced here):

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
RECEIPTS_DIR = REPRO_DIR / "receipts"

SCHEMA_FILES = {
    "inventory.v0.schema.json": "parapet-repro-index/v0",
    "plan.v0.schema.json": "parapet-repro-plan/v0",
    "primitive-guard.v0.schema.json": "parapet-primitive-guard/v0",
    "receipt.v0.schema.json": "parapet-repro-receipt/v0",
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
RUNG2 = "rung2_algorithmic"

OUTPUT_SLOT_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
WINDOWS_DRIVE_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:/")
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
    if WINDOWS_DRIVE_ABSOLUTE_RE.match(normalized):
        errors.append(f"{label}: absolute path '{path}' is not public-safe")
    if "." in parts or ".." in parts:
        errors.append(f"{label}: dot-segment path '{path}' is not public-safe")
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


def check_receipt_envelope(receipt: dict) -> list[str]:
    """Presence-check the public repro-receipt / verification-rung envelope."""
    errors: list[str] = []
    rid = receipt.get("receipt_id", "<missing receipt_id>")

    if receipt.get("schema_version") != "parapet-repro-receipt/v0":
        errors.append(f"{rid}: schema_version must be parapet-repro-receipt/v0")
    if not (isinstance(receipt.get("receipt_id"), str) and receipt["receipt_id"]):
        errors.append(f"{rid}: receipt_id is required")
    if not (isinstance(receipt.get("artifact_id"), str) and receipt["artifact_id"]):
        errors.append(f"{rid}: artifact_id is required")
    if receipt.get("receipt_type") not in {"result", "verification"}:
        errors.append(f"{rid}: receipt_type is invalid")

    producer = receipt.get("producer")
    if not isinstance(producer, dict):
        errors.append(f"{rid}: missing producer object")
    else:
        for field in ("owner", "generated_at"):
            value = producer.get(field)
            if not (isinstance(value, str) and value.strip()):
                errors.append(f"{rid}: producer.{field} must be non-empty")

    verification = receipt.get("verification")
    if not isinstance(verification, dict):
        errors.append(f"{rid}: missing verification object")
        return errors

    for field in (
        "independence_rung",
        "residual_named",
        "independence_unit",
        "n_independent_units",
        "resample_unit",
        "criterion_provenance",
        "ci_present_where_materially_exceeds",
    ):
        if field not in verification:
            errors.append(f"{rid}: verification missing '{field}'")

    for field in ("residual_named", "independence_unit", "resample_unit"):
        value = verification.get(field)
        if not (isinstance(value, str) and value.strip()):
            errors.append(f"{rid}: verification.{field} must be non-empty")
    n_independent_units = verification.get("n_independent_units")
    if not (isinstance(n_independent_units, int) and n_independent_units >= 1):
        errors.append(f"{rid}: verification.n_independent_units must be >= 1")

    if verification.get("resample_unit") != verification.get("independence_unit"):
        errors.append(f"{rid}: resample_unit must equal independence_unit")

    criterion = verification.get("criterion_provenance")
    if not isinstance(criterion, dict):
        errors.append(f"{rid}: criterion_provenance must be an object")
    else:
        kind = criterion.get("kind")
        evidence = criterion.get("evidence")
        if kind not in {"predating_timestamp", "post_hoc_exploratory"}:
            errors.append(f"{rid}: criterion_provenance.kind is invalid")
        if not (isinstance(evidence, str) and evidence.strip()):
            errors.append(f"{rid}: criterion_provenance.evidence must be non-empty")

    if verification.get("ci_present_where_materially_exceeds") is not True:
        errors.append(f"{rid}: ci_present_where_materially_exceeds must be true")

    rung = verification.get("independence_rung")
    if rung not in {"rung1_data_path", RUNG2}:
        errors.append(f"{rid}: independence_rung is invalid")
    if rung == RUNG2:
        audit = verification.get("disambiguation_audit")
        if not isinstance(audit, list) or not audit:
            errors.append(f"{rid}: rung2_algorithmic requires disambiguation_audit")
        else:
            for index, item in enumerate(audit):
                if not isinstance(item, dict):
                    errors.append(f"{rid}: disambiguation_audit[{index}] must be object")
                    continue
                ambiguity = item.get("ambiguity")
                if not (isinstance(ambiguity, str) and ambiguity.strip()):
                    errors.append(
                        f"{rid}: disambiguation_audit[{index}].ambiguity is required"
                    )
                if item.get("status") not in {
                    "exercised_by_data",
                    "exercised_by_fixture",
                    "not_yet_validated",
                }:
                    errors.append(
                        f"{rid}: disambiguation_audit[{index}].status is invalid"
                    )

    per_unit = receipt.get("per_unit_outputs")
    if not isinstance(per_unit, dict):
        errors.append(f"{rid}: missing per_unit_outputs object")
    else:
        errors.extend(
            _path_safety_errors(f"{rid}: per_unit_outputs.path", per_unit.get("path"))
        )
        per_unit_sha = per_unit.get("sha256")
        if not (
            isinstance(per_unit_sha, str)
            and re.match(r"^[a-f0-9]{64}$", per_unit_sha)
        ):
            errors.append(f"{rid}: per_unit_outputs.sha256 must be a sha256")

    provenance = receipt.get("provenance")
    if not isinstance(provenance, dict):
        errors.append(f"{rid}: missing provenance object")
    else:
        detector = provenance.get("detector_of_record")
        if not (isinstance(detector, str) and detector.strip()):
            errors.append(f"{rid}: provenance.detector_of_record must be non-empty")
        input_artifact_shas = provenance.get("input_artifact_shas")
        if not isinstance(input_artifact_shas, dict) or not input_artifact_shas:
            errors.append(f"{rid}: provenance.input_artifact_shas is required")
        else:
            for name, sha in input_artifact_shas.items():
                if not (
                    isinstance(name, str)
                    and isinstance(sha, str)
                    and re.match(r"^[a-f0-9]{64}$", sha)
                ):
                    errors.append(
                        f"{rid}: provenance.input_artifact_shas has invalid entry"
                    )
                    break
        operating_point = provenance.get("operating_point_lock")
        if not isinstance(operating_point, dict) or not operating_point:
            errors.append(f"{rid}: provenance.operating_point_lock is required")
        model = provenance.get("model")
        if model is not None and not isinstance(model, dict):
            errors.append(f"{rid}: provenance.model must be an object when present")
        elif isinstance(model, dict) and not (
            isinstance(model.get("model_id"), str) and model["model_id"]
        ):
            errors.append(f"{rid}: provenance.model.model_id is required")
        code_ref = provenance.get("code_ref")
        if not isinstance(code_ref, dict) or not re.match(
            r"^[a-f0-9]{7,64}$", str(code_ref.get("git_sha", ""))
        ):
            errors.append(f"{rid}: provenance.code_ref.git_sha is required")
        preregistration = provenance.get("pre_registration")
        provider = provenance.get("provider")
        for label, path in (
            (
                "provenance.code_ref.repo_path",
                code_ref.get("repo_path") if isinstance(code_ref, dict) else None,
            ),
            (
                "provenance.model.receipt_ref",
                model.get("receipt_ref") if isinstance(model, dict) else None,
            ),
            (
                "provenance.provider.receipt_ref",
                provider.get("receipt_ref") if isinstance(provider, dict) else None,
            ),
            (
                "provenance.pre_registration.ref",
                preregistration.get("ref")
                if isinstance(preregistration, dict)
                else None,
            ),
        ):
            errors.extend(_path_safety_errors(f"{rid}: {label}", path))

    for claim in receipt.get("claims", []):
        ctype = claim.get("claim_type") if isinstance(claim, dict) else None
        if ctype in CI_REQUIRED_CLAIM_TYPES and claim.get("ci_present") is not True:
            cid = claim.get("claim_id", "<missing claim_id>")
            errors.append(
                f"{rid}: claim '{cid}' is '{ctype}' but ci_present is not true"
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

    # 4. Any frozen public receipts pass the stdlib envelope checks.
    receipt_paths = []
    if RECEIPTS_DIR.exists():
        receipt_paths = sorted(RECEIPTS_DIR.rglob("*.json"))
    for receipt_path in receipt_paths:
        try:
            receipt = load_json(receipt_path)
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(
                f"{receipt_path.relative_to(REPRO_DIR)}: JSON parse failed: {exc}"
            )
            continue
        prefix = f"{receipt_path.relative_to(REPRO_DIR)}"
        errors.extend(f"{prefix}: {e}" for e in check_receipt_envelope(receipt))

    # 5. Bonus: full JSON Schema validation when jsonschema is importable.
    try:
        import jsonschema  # type: ignore
    except ImportError:
        pass
    else:
        try:
            inventory_schema = load_json(SCHEMA_DIR / "inventory.v0.schema.json")
            receipt_schema = load_json(SCHEMA_DIR / "receipt.v0.schema.json")
            for schema_path in sorted(SCHEMA_DIR.glob("*.json")):
                jsonschema.Draft202012Validator.check_schema(load_json(schema_path))
            jsonschema.validate(instance=inventory, schema=inventory_schema)
            for receipt_path in receipt_paths:
                jsonschema.validate(
                    instance=load_json(receipt_path), schema=receipt_schema
                )
        except jsonschema.SchemaError as exc:  # type: ignore[attr-defined]
            errors.append(f"schema validation failed: {exc.message}")
        except jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
            errors.append(f"jsonschema validation failed: {exc.message}")
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


def _inventory_with_input_path(path: str) -> dict:
    inv = _valid_inventory()
    inv["artifacts"][0]["inputs"] = {"candidate": path}
    return inv


def _schema_accepts_inventory(inventory: dict) -> bool | None:
    try:
        import jsonschema  # type: ignore
    except ImportError:
        return None

    schema = load_json(SCHEMA_DIR / "inventory.v0.schema.json")
    try:
        jsonschema.validate(instance=inventory, schema=schema)
    except jsonschema.ValidationError:  # type: ignore[attr-defined]
        return False
    return True


def _valid_receipt() -> dict:
    sha = "0" * 64
    return {
        "schema_version": "parapet-repro-receipt/v0",
        "receipt_id": "synthetic.receipt",
        "artifact_id": "synthetic.table1.row_auc",
        "receipt_type": "verification",
        "producer": {
            "owner": "sensor",
            "generated_at": "2026-06-21T00:00:00Z",
            "tool": "synthetic fixture",
        },
        "provenance": {
            "code_ref": {
                "git_sha": "ab8b607",
                "repo_path": "parapet/src/lib.rs",
            },
            "detector_of_record": "synthetic-detector",
            "model": {"model_id": "synthetic-model", "sha256": sha},
            "input_artifact_shas": {"inputs": sha},
            "operating_point_lock": {"threshold": 0.5},
            "pre_registration": {
                "timestamp": "2026-06-20T00:00:00Z",
                "sha256": sha,
                "ref": "repro/receipts/synthetic/pre_registration.json",
            },
        },
        "per_unit_outputs": {
            "path": "repro/receipts/synthetic/per_unit_outputs.jsonl",
            "sha256": sha,
            "media_type": "application/jsonl",
            "row_count": 2,
        },
        "verification": {
            "independence_rung": "rung1_data_path",
            "residual_named": "scoring-layer trust",
            "independence_unit": "task",
            "n_independent_units": 2,
            "resample_unit": "task",
            "criterion_provenance": {
                "kind": "predating_timestamp",
                "evidence": "synthetic pre-registration fixture",
            },
            "ci_present_where_materially_exceeds": True,
        },
        "metrics": [{"metric_id": "auc", "value": 0.5}],
        "claims": [
            {
                "claim_id": "auc_noninferiority",
                "claim_type": "noninferiority",
                "ci_present": True,
            }
        ],
    }


def _receipt_cases():
    cases = []

    def mutate(fn):
        receipt = _valid_receipt()
        fn(receipt)
        return receipt

    cases.append(
        (
            "missing verification",
            mutate(lambda r: r.pop("verification")),
        )
    )
    cases.append(
        (
            "missing receipt version",
            mutate(lambda r: r.__setitem__("schema_version", "wrong")),
        )
    )
    cases.append(
        (
            "missing producer owner",
            mutate(lambda r: r["producer"].pop("owner")),
        )
    )
    cases.append(
        (
            "missing detector of record",
            mutate(lambda r: r["provenance"].pop("detector_of_record")),
        )
    )
    cases.append(
        (
            "missing operating point lock",
            mutate(lambda r: r["provenance"].pop("operating_point_lock")),
        )
    )
    cases.append(
        (
            "bad model provenance",
            mutate(lambda r: r["provenance"].__setitem__("model", "not-a-model")),
        )
    )
    cases.append(
        (
            "bad input artifact hash",
            mutate(
                lambda r: r["provenance"].__setitem__(
                    "input_artifact_shas", {"inputs": "not-a-sha"}
                )
            ),
        )
    )
    cases.append(
        (
            "resample finer than independence unit",
            mutate(lambda r: r["verification"].__setitem__("resample_unit", "run")),
        )
    )
    cases.append(
        (
            "missing independent unit count",
            mutate(lambda r: r["verification"].pop("n_independent_units")),
        )
    )
    cases.append(
        (
            "missing criterion evidence",
            mutate(lambda r: r["verification"]["criterion_provenance"].pop("evidence")),
        )
    )
    cases.append(
        (
            "missing CI presence flag",
            mutate(
                lambda r: r["verification"].__setitem__(
                    "ci_present_where_materially_exceeds", False
                )
            ),
        )
    )
    cases.append(
        (
            "rung2 without disambiguation audit",
            mutate(
                lambda r: r["verification"].__setitem__(
                    "independence_rung", RUNG2
                )
            ),
        )
    )
    cases.append(
        (
            "blocked per-unit path",
            mutate(
                lambda r: r["per_unit_outputs"].__setitem__(
                    "path", "parapet-runner/runs/private.jsonl"
                )
            ),
        )
    )
    cases.append(
        (
            "comparative claim missing CI",
            mutate(
                lambda r: r["claims"][0].update(
                    {"claim_type": "comparative", "ci_present": False}
                )
            ),
        )
    )
    return cases


def _schema_accepts_receipt(receipt: dict) -> bool | None:
    try:
        import jsonschema  # type: ignore
    except ImportError:
        return None

    schema = load_json(SCHEMA_DIR / "receipt.v0.schema.json")
    try:
        jsonschema.validate(instance=receipt, schema=schema)
    except jsonschema.ValidationError:  # type: ignore[attr-defined]
        return False
    return True


# --------------------------------------------------------------------------
# Test functions (pytest-collectable AND runnable under plain python3).
# --------------------------------------------------------------------------


def test_valid_fixture_passes():
    assert validate_inventory(_valid_inventory()) == []


def test_valid_receipt_fixture_passes():
    assert check_receipt_envelope(_valid_receipt()) == []


def test_model_provenance_is_optional_for_non_model_detectors():
    receipt = _valid_receipt()
    receipt["provenance"].pop("model")
    assert check_receipt_envelope(receipt) == []


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


def test_each_invalid_receipt_fixture_is_caught():
    for label, receipt in _receipt_cases():
        assert check_receipt_envelope(receipt), (
            f"expected receipt failure not caught: {label}"
        )


def test_rung2_receipt_requires_disambiguation_audit_entries():
    receipt = _valid_receipt()
    receipt["verification"]["independence_rung"] = RUNG2
    receipt["verification"]["disambiguation_audit"] = [
        {"ambiguity": "tie handling", "status": "exercised_by_fixture"}
    ]
    assert check_receipt_envelope(receipt) == []


def test_repo_path_blocks_exact_roots_and_children():
    blocked_paths = [
        "data",
        "data/raw.jsonl",
        "runs",
        "runs/latest.json",
        "models",
        "models/private.bin",
        "adjudication",
        "adjudication/private.md",
        "TheWall",
        "TheWall/raw.jsonl",
        "curated",
        "curated/local.jsonl",
        "schema/eval/staging",
        "schema/eval/staging/source.jsonl",
        "schema/eval/benign",
        "schema/eval/benign/source.jsonl",
        "schema/eval/malicious",
        "schema/eval/malicious/source.jsonl",
        "schema/eval/training",
        "schema/eval/training/source.jsonl",
        "schema/eval/challenges",
        "schema/eval/challenges/source.jsonl",
        "schema/eval/heuristic_staged",
        "schema/eval/heuristic_staged/source.jsonl",
        "parapet-runner/runs",
        "parapet-runner/runs/latest.json",
        "parapet-data/curated",
        "parapet-data/curated/source.jsonl",
        "parapet-data/curated_v2/source.jsonl",
    ]

    for path in blocked_paths:
        errors = check_repo_paths(_inventory_with_input_path(path))
        assert errors, f"expected blocked repo path: {path}"


def test_repo_path_allows_sibling_prefixes():
    allowed_paths = [
        "database.json",
        "runstats.json",
        "model_card.json",
        "adjudication_notes.md",
        "TheWallaby/fixture.json",
        "curated-notes/summary.md",
        "schema/eval/maliciousness.json",
        "schema/eval/staging_notes.json",
        "parapet-runner/runstats.json",
        "parapet-data/curated.json",
        "parapet-data/curated-notes/summary.md",
    ]

    for path in allowed_paths:
        errors = check_repo_paths(_inventory_with_input_path(path))
        assert errors == [], f"expected allowed repo path {path}, got {errors}"


def test_repo_path_normalizes_backslash_separators():
    blocked_paths = [
        "\\absolute\\path.json",
        "..\\private\\keys.json",
        "data\\raw.jsonl",
        "C:/Users/example/TheWall/raw.jsonl",
        "C:\\Users\\example\\TheWall\\raw.jsonl",
        "./data/raw.jsonl",
        ".\\data\\raw.jsonl",
        "././TheWall/secret.jsonl",
        "./parapet-data/curated_v2/source.jsonl",
        "schema\\eval\\malicious\\source.jsonl",
        "parapet-runner\\runs\\latest.json",
        "parapet-data\\curated_v2\\source.jsonl",
    ]

    for path in blocked_paths:
        errors = check_repo_paths(_inventory_with_input_path(path))
        assert errors, f"expected blocked backslash repo path: {path}"


def test_schema_repo_path_matches_stdlib_path_safety_when_jsonschema_available():
    checks = {
        "data": False,
        "data/raw.jsonl": False,
        "data\\raw.jsonl": False,
        "C:/Users/example/TheWall/raw.jsonl": False,
        "C:\\Users\\example\\TheWall\\raw.jsonl": False,
        "./data/raw.jsonl": False,
        ".\\data\\raw.jsonl": False,
        "././TheWall/secret.jsonl": False,
        "./parapet-data/curated_v2/source.jsonl": False,
        "..\\private\\keys.json": False,
        "schema/eval/malicious": False,
        "schema\\eval\\malicious\\source.jsonl": False,
        "parapet-data/curated_v2/source.jsonl": False,
        "parapet-data\\curated_v2\\source.jsonl": False,
        "database.json": True,
        "runstats.json": True,
        "model_card.json": True,
        "schema/eval/maliciousness.json": True,
        "parapet-data/curated.json": True,
    }

    for path, expected in checks.items():
        accepted = _schema_accepts_inventory(_inventory_with_input_path(path))
        if accepted is None:
            continue
        assert accepted is expected, (
            f"schema acceptance for {path!r} should be {expected}, got {accepted}"
        )


def test_receipt_schema_matches_stdlib_presence_gate_when_jsonschema_available():
    valid = _schema_accepts_receipt(_valid_receipt())
    if valid is None:
        return
    assert valid is True

    stdlib_only = {"resample finer than independence unit"}
    for label, receipt in _receipt_cases():
        if label in stdlib_only:
            continue
        accepted = _schema_accepts_receipt(receipt)
        assert accepted is False, (
            f"schema should reject invalid receipt fixture {label!r}"
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
