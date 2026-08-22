"""Deterministic Phase-A reference traversal cells.

The sweep layer supplies durability; this module supplies the Phase-A fold and
its JSON wire contract.  Scores are transported as integer sufficient
statistics and converted to binary64 only by :func:`winner_compare`.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .sweep import CellResult, CellSpec, canonical_digest, normalize_json

GRAM_SET_BOUND = 2**25
FLOAT_PROBES = (
    "4503599627370497/9007199254740993==0.5",
    "3/10==0.3",
)


def require_binary64() -> dict[str, Any]:
    """Fail closed unless the result-affecting Python float contract is met."""
    info = sys.float_info
    invariants = {
        "radix": info.radix,
        "mant_dig": info.mant_dig,
        "max_exp": info.max_exp,
        "min_exp": info.min_exp,
    }
    expected = {"radix": 2, "mant_dig": 53, "max_exp": 1024, "min_exp": -1021}
    if invariants != expected:
        raise RuntimeError(f"binary64 contract failed: {invariants!r} != {expected!r}")
    if 4503599627370497 / 9007199254740993 != 0.5:
        raise RuntimeError(f"binary64 division probe failed: {FLOAT_PROBES[0]}")
    if 3 / 10 != 0.3:
        raise RuntimeError(f"binary64 division probe failed: {FLOAT_PROBES[1]}")
    return {"invariants": invariants, "probes": list(FLOAT_PROBES)}


def runtime_revision() -> dict[str, Any]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "unicode_version": unicodedata.unidata_version,
        "float_contract": require_binary64(),
    }


# Check before any cell can execute or fixture fold can run.
RUNTIME_REV = runtime_revision()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ChunkRecord(_StrictModel):
    name: str
    row_count: int = Field(ge=0)
    content_digest: str = Field(min_length=1)
    parent_file_digest: str | None
    start_row: int | None
    end_row: int | None
    ordered_record_digest: str | None

    @model_validator(mode="after")
    def validate_row_span(self) -> "ChunkRecord":
        span = (self.start_row, self.end_row, self.parent_file_digest, self.ordered_record_digest)
        if any(value is not None for value in span) != all(value is not None for value in span):
            raise ValueError("row-span fields must be all present or all absent")
        if self.start_row is not None:
            assert self.end_row is not None
            if self.start_row < 0 or self.start_row >= self.end_row:
                raise ValueError("row span must satisfy 0 <= start_row < end_row")
            if self.row_count != self.end_row - self.start_row:
                raise ValueError("row_count does not match row span")
        return self


class CodeRevisionEntry(_StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CellParams(_StrictModel):
    tier: str
    label: str
    contributes_c1: bool
    chunk: list[ChunkRecord]
    candidate_universe_digest: str = Field(min_length=1)
    code_rev: str = Field(min_length=1)
    code_rev_manifest: list[CodeRevisionEntry] = Field(min_length=1)
    plan_rev: str = Field(min_length=1)
    runtime_rev: dict[str, Any]

    @model_validator(mode="after")
    def validate_runtime(self) -> "CellParams":
        if self.runtime_rev != RUNTIME_REV:
            raise ValueError("runtime_rev does not match the live enforced contract")
        manifest = [entry.model_dump(mode="json") for entry in self.code_rev_manifest]
        if canonical_digest(manifest) != self.code_rev:
            raise ValueError("code_rev does not match code_rev_manifest")
        return self


class C1Partial(_StrictModel):
    candidate_id: str
    reference_file: str
    locator: str
    containment_num: int = Field(ge=0)
    containment_den: int = Field(gt=0)
    jaccard_num: int = Field(ge=0)
    jaccard_den: int = Field(gt=0)
    reference: dict[str, Any]

    @model_validator(mode="after")
    def validate_wire_domain(self) -> "C1Partial":
        if self.containment_num > self.containment_den:
            raise ValueError("containment_num must not exceed containment_den")
        if self.jaccard_num > self.jaccard_den:
            raise ValueError("jaccard_num must not exceed jaccard_den")
        if self.containment_num != self.jaccard_num:
            raise ValueError("c1 containment_num must equal jaccard_num")
        if self.containment_den > self.jaccard_den:
            raise ValueError("c1 containment_den must not exceed jaccard_den")
        if self.containment_num < 1:
            raise ValueError("c1 intersection must be at least 1")
        if self.containment_num / self.containment_den < .3:
            raise ValueError("c1 containment score must satisfy the .3 admission gate")
        if self.containment_den > GRAM_SET_BOUND:
            raise ValueError(f"containment_den exceeds B={GRAM_SET_BOUND}")
        if self.jaccard_den > 2 * GRAM_SET_BOUND - 1:
            raise ValueError(f"jaccard_den exceeds 2B-1={2 * GRAM_SET_BOUND - 1}")
        if self.jaccard_den + self.containment_num < 2 * self.containment_den:
            raise ValueError("c1 union plus intersection must be at least twice containment_den")
        if self.jaccard_den + self.containment_num - self.containment_den > GRAM_SET_BOUND:
            raise ValueError(f"c1 larger gram set exceeds B={GRAM_SET_BOUND}")
        return self

    @model_validator(mode="after")
    def validate_reference_keys(self) -> "C1Partial":
        if "file" not in self.reference or "row_id" not in self.reference:
            raise ValueError("c1 reference requires file and row_id")
        if self.reference_file != str(self.reference["file"]):
            raise ValueError("reference_file does not match embedded reference")
        if self.locator != str(self.reference["row_id"]):
            raise ValueError("locator does not match embedded reference")
        return self


class BestJPartial(_StrictModel):
    candidate_index: int = Field(ge=0)
    candidate_id: str
    jaccard_num: int = Field(ge=0)
    jaccard_den: int = Field(gt=0)
    reference: dict[str, Any]

    @model_validator(mode="after")
    def validate_wire_domain(self) -> "BestJPartial":
        if self.jaccard_num > self.jaccard_den:
            raise ValueError("jaccard_num must not exceed jaccard_den")
        if self.jaccard_num < 1:
            raise ValueError("bestj intersection must be at least 1")
        if self.jaccard_den > 2 * GRAM_SET_BOUND - 1:
            raise ValueError(f"jaccard_den exceeds 2B-1={2 * GRAM_SET_BOUND - 1}")
        if self.jaccard_den + self.jaccard_num > 2 * GRAM_SET_BOUND:
            raise ValueError(f"bestj union plus intersection exceeds 2B={2 * GRAM_SET_BOUND}")
        return self

    @model_validator(mode="after")
    def validate_reference_keys(self) -> "BestJPartial":
        if "row_id" not in self.reference:
            raise ValueError("bestj reference requires row_id")
        return self


class CellPayload(_StrictModel):
    c1_partial: list[C1Partial]
    bestj_partial: dict[str, list[BestJPartial]]

    @model_validator(mode="after")
    def validate_canonical_arrays(self) -> "CellPayload":
        c1_keys = [(row.candidate_id, row.reference_file) for row in self.c1_partial]
        if c1_keys != sorted(c1_keys):
            raise ValueError("c1_partial must be strictly sorted")
        if len(c1_keys) != len(set(c1_keys)):
            raise ValueError("duplicate c1 key")
        for label, rows in self.bestj_partial.items():
            indices = [row.candidate_index for row in rows]
            if indices != sorted(indices):
                raise ValueError(f"bestj_partial[{label!r}] must be strictly sorted")
            if len(indices) != len(set(indices)):
                raise ValueError(f"duplicate bestj candidate_index for label {label!r}")
        return self


class DuplicateProjectedIdentity(ValueError):
    pass


def _reference_digest(record: Mapping[str, Any]) -> str:
    return canonical_digest(normalize_json(dict(record)))


def _projected(record: Mapping[str, Any], namespace: Literal["c1", "bestj"]) -> tuple[str, str]:
    locator = str(record["row_id"])
    return (str(record.get("file", "")), locator) if namespace == "bestj" else (str(record["file"]), locator)


def _check_identity(left: Mapping[str, Any], right: Mapping[str, Any], namespace: Literal["c1", "bestj"]) -> None:
    if _projected(left, namespace) != _projected(right, namespace):
        return
    if _reference_digest(left) != _reference_digest(right):
        raise DuplicateProjectedIdentity(
            f"duplicate projected {namespace} identity with differing records: "
            f"{json.dumps(normalize_json(dict(left)), sort_keys=True)} != "
            f"{json.dumps(normalize_json(dict(right)), sort_keys=True)}"
        )


def _record_identity(
    identities: dict[tuple[str, str], tuple[str, dict[str, Any]]],
    record: Mapping[str, Any],
    namespace: Literal["c1", "bestj"],
) -> None:
    normalized = normalize_json(dict(record))
    key = _projected(normalized, namespace)
    digest = _reference_digest(normalized)
    old = identities.get(key)
    if old is not None and old[0] != digest:
        raise DuplicateProjectedIdentity(
            f"duplicate projected {namespace} identity with differing records: "
            f"{json.dumps(old[1], sort_keys=True)} != {json.dumps(normalized, sort_keys=True)}"
        )
    identities.setdefault(key, (digest, normalized))


def winner_compare(
    new_num: int,
    new_den: int,
    new_reference: Mapping[str, Any],
    old_num: int,
    old_den: int,
    old_reference: Mapping[str, Any],
    namespace: Literal["c1", "bestj"],
) -> int:
    """Return positive when new wins, using the one legacy binary64 order."""
    _check_identity(new_reference, old_reference, namespace)
    new_score, old_score = new_num / new_den, old_num / old_den
    if new_score != old_score:
        return 1 if new_score > old_score else -1
    new_key, old_key = _projected(new_reference, namespace), _projected(old_reference, namespace)
    if new_key == old_key:
        return 0
    return 1 if new_key < old_key else -1


# Both classes deliberately resolve this exact object; Gate 2 asserts identity.
SHARED_COMPARATOR = winner_compare


def _bounded(grams: set[str], *, where: str) -> set[str]:
    if len(grams) > GRAM_SET_BOUND:
        raise ValueError(f"{where} gram-set cardinality {len(grams)} exceeds B={GRAM_SET_BOUND}")
    return grams


def enforce_gram_bound(cardinality: int, *, where: str) -> None:
    """Set-level bound seam used by the infeasible 2**25 fixture check."""
    if cardinality > GRAM_SET_BOUND:
        raise ValueError(f"{where} gram-set cardinality {cardinality} exceeds B={GRAM_SET_BOUND}")


class PhaseACellExecutor:
    comparator = staticmethod(SHARED_COMPARATOR)

    def __init__(
        self,
        candidates: Sequence[Mapping[str, Any]],
        references: Mapping[str, Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Mapping[str, Any]]]],
        normalized_5grams: Callable[[str], set[str]],
    ) -> None:
        require_binary64()
        self._candidates = [normalize_json(dict(row)) for row in candidates]
        self._references: dict[str, list[dict[str, Any]] | dict[str, list[dict[str, Any]]]] = {}
        for cell_id, value in references.items():
            if isinstance(value, Mapping):
                self._references[cell_id] = {
                    name: [normalize_json(dict(row)) for row in rows]
                    for name, rows in value.items()
                }
            else:
                self._references[cell_id] = [normalize_json(dict(row)) for row in value]
        self._normalizer = normalized_5grams
        self._candidate_grams = [
            _bounded(normalized_5grams(str(row["text"])), where=f"candidate {i}")
            for i, row in enumerate(self._candidates)
        ]
        self._inverted: dict[str, list[int]] = defaultdict(list)
        for index, grams in enumerate(self._candidate_grams):
            for gram in grams:
                self._inverted[gram].append(index)

    def execute(self, cell: CellSpec) -> Mapping[str, Any]:
        require_binary64()
        params = CellParams.model_validate(cell.params)
        if digest_ordered_records(self._candidates) != params.candidate_universe_digest:
            raise ValueError("candidate_universe_digest does not match injected candidates")
        injected = self._references[cell.cell_id]
        if isinstance(injected, list):
            if len(params.chunk) != 1:
                raise ValueError("flat reference injection requires exactly one chunk descriptor")
            chunks = {params.chunk[0].name: injected}
        else:
            chunks = injected
        references: list[dict[str, Any]] = []
        for descriptor in params.chunk:
            if descriptor.name not in chunks:
                raise ValueError(f"missing injected reference chunk {descriptor.name!r}")
            rows = chunks[descriptor.name]
            digest = digest_ordered_records(rows)
            if len(rows) != descriptor.row_count or digest != descriptor.content_digest:
                raise ValueError(f"reference chunk integrity mismatch for {descriptor.name!r}")
            if descriptor.ordered_record_digest is not None and descriptor.ordered_record_digest != digest:
                raise ValueError(f"row-span ordered_record_digest mismatch for {descriptor.name!r}")
            references.extend(rows)
        c1: dict[tuple[str, str], C1Partial] = {}
        bestj: dict[int, BestJPartial] = {}
        c1_identities: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
        bestj_identities: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
        for reference in references:
            if params.contributes_c1:
                _record_identity(c1_identities, reference, "c1")
            _record_identity(bestj_identities, reference, "bestj")
            grams = _bounded(self._normalizer(str(reference["text"])), where=f"reference {reference.get('row_id')}")
            counts: Counter[int] = Counter()
            for gram in grams:
                counts.update(self._inverted.get(gram, ()))
            for ci, intersection in counts.items():
                candidate = self._candidates[ci]
                cg = self._candidate_grams[ci]
                minimum = min(len(cg), len(grams))
                union = len(cg) + len(grams) - intersection
                if not minimum or not union:
                    continue
                locator = str(reference["row_id"])
                if params.contributes_c1 and intersection / minimum >= .3:
                    key = (str(candidate["row_id"]), str(reference["file"]))
                    entry = C1Partial(
                        candidate_id=key[0], reference_file=key[1], locator=locator,
                        containment_num=intersection, containment_den=minimum,
                        jaccard_num=intersection, jaccard_den=union, reference=dict(reference),
                    )
                    old = c1.get(key)
                    if old is None or self.comparator(intersection, minimum, reference, old.containment_num, old.containment_den, old.reference, "c1") > 0:
                        c1[key] = entry
                entryj = BestJPartial(
                    candidate_index=ci, candidate_id=str(candidate["row_id"]),
                    jaccard_num=intersection, jaccard_den=union, reference=dict(reference),
                )
                oldj = bestj.get(ci)
                if oldj is None or self.comparator(intersection, union, reference, oldj.jaccard_num, oldj.jaccard_den, oldj.reference, "bestj") > 0:
                    bestj[ci] = entryj
        payload = CellPayload(
            c1_partial=sorted(c1.values(), key=lambda x: (x.candidate_id, x.reference_file)),
            bestj_partial={params.label: sorted(bestj.values(), key=lambda x: x.candidate_index)},
        )
        return payload.model_dump(mode="json")


class PhaseAResultMerger:
    comparator = staticmethod(SHARED_COMPARATOR)

    def __init__(self, candidates: Sequence[Mapping[str, Any]]) -> None:
        require_binary64()
        self._candidates = [normalize_json(dict(row)) for row in candidates]
        self._candidate_universe_digest = digest_ordered_records(self._candidates)
        self._candidate_ids = [str(row["row_id"]) for row in self._candidates]
        if len(self._candidate_ids) != len(set(self._candidate_ids)):
            raise ValueError("candidate universe contains duplicate row_id values")
        self._candidate_id_set = set(self._candidate_ids)

    def merge(self, results: Sequence[CellResult]) -> Mapping[str, Any]:
        require_binary64()
        c1: dict[tuple[str, str], C1Partial] = {}
        bestj: dict[str, dict[int, BestJPartial]] = defaultdict(dict)
        c1_identities: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
        bestj_identities: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
        candidate_ids: dict[int, str] = {}
        for result in results:
            payload = CellPayload.model_validate(result.payload)
            for entry in payload.c1_partial:
                if entry.candidate_id not in self._candidate_id_set:
                    raise ValueError(
                        f"c1 candidate_id {entry.candidate_id!r} is not present in the pinned candidate universe"
                    )
                _record_identity(c1_identities, entry.reference, "c1")
                key = (entry.candidate_id, entry.reference_file)
                old = c1.get(key)
                if old is None or self.comparator(entry.containment_num, entry.containment_den, entry.reference, old.containment_num, old.containment_den, old.reference, "c1") > 0:
                    c1[key] = entry
            for label, entries in payload.bestj_partial.items():
                for entry in entries:
                    if entry.candidate_index >= len(self._candidate_ids):
                        raise ValueError(
                            f"candidate_index {entry.candidate_index} is out of range for pinned candidate universe "
                            f"{self._candidate_universe_digest}"
                        )
                    pinned_candidate_id = self._candidate_ids[entry.candidate_index]
                    if entry.candidate_id != pinned_candidate_id:
                        raise ValueError(
                            f"candidate_index {entry.candidate_index} maps to candidate_id "
                            f"{entry.candidate_id!r}, not pinned {pinned_candidate_id!r}"
                        )
                    _record_identity(bestj_identities, entry.reference, "bestj")
                    old_candidate_id = candidate_ids.setdefault(entry.candidate_index, entry.candidate_id)
                    if old_candidate_id != entry.candidate_id:
                        raise ValueError(
                            f"candidate_index {entry.candidate_index} maps to conflicting candidate_id values: "
                            f"{old_candidate_id!r} != {entry.candidate_id!r}"
                        )
                    old = bestj[label].get(entry.candidate_index)
                    if old is None or self.comparator(entry.jaccard_num, entry.jaccard_den, entry.reference, old.jaccard_num, old.jaccard_den, old.reference, "bestj") > 0:
                        bestj[label][entry.candidate_index] = entry
        return CellPayload(
            c1_partial=sorted(c1.values(), key=lambda x: (x.candidate_id, x.reference_file)),
            bestj_partial={label: sorted(rows.values(), key=lambda x: x.candidate_index) for label, rows in sorted(bestj.items())},
        ).model_dump(mode="json")


def digest_ordered_records(records: Sequence[Mapping[str, Any]]) -> str:
    return canonical_digest([normalize_json(dict(row)) for row in records])


def code_revision(paths: Sequence[str], contents: Sequence[bytes]) -> str:
    if len(paths) != len(contents) or not paths:
        raise ValueError("code revision manifest and contents must be non-empty and aligned")
    manifest = [{"path": p, "sha256": hashlib.sha256(c).hexdigest()} for p, c in zip(paths, contents)]
    return canonical_digest(manifest)
