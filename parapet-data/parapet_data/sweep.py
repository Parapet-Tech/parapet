"""Resumable sweep execution.

A sweep is a set of independent cells executed once each, where the work is long
enough that losing it to an interruption is expensive. This module owns the
durability concern only: which cells still need running, and how a completed
cell is recorded so that a re-run does not repeat it.

Three seams are injected, because they are what differs between callers:

- ``SweepPlan``     enumerates the cells
- ``CellExecutor``  performs one cell
- ``ResultMerger``  folds completed cells into an aggregate

Integrity contract
------------------
One representation. Params, payloads and aggregates are recursively validated as
native JSON and stored as canonical text. Types that merely resemble JSON
(tuples, sets, ``Path``) are rejected rather than coerced, because coercion makes
identity non-injective: ``(1, 2)`` and ``[1, 2]`` encode to the same bytes, so a
value could round-trip through storage and come back a different shape.

Immutable inputs. ``CellSpec.params`` and ``CellResult.payload`` return a fresh
copy on every access, so nothing a caller holds can mutate what was hashed. The
digest and the data it describes cannot disagree.

Identity is ``(cell_id, input_digest, generation)``, never ``cell_id`` alone. A
stored result is served only when all three match AND its payload still hashes to
its recorded ``payload_digest``, so a tampered or truncated payload is recomputed
rather than trusted. The input digest is enforced by the store itself: a result
recorded under different params never leaves ``load``, no matter what else
matches.

Generations make ``force`` crash-safe. A forced refresh bumps a durable
generation counter before discarding anything, so results from before the refresh
are rejected on identity even if the discard loop dies partway through. A missing
generation record over existing results or completion state is corruption and
stops the run; recreating generation 0 there would revive exactly what a force
discarded, and could hand a later force an already-used generation number.

Storage keys are derived, never raw: a ``cell_id`` cannot escape the output root
or collide with another cell's directory.

A run holds an exclusive lock on the output root. Two runners cannot interleave
writes into the same sweep.

Completion is all-or-nothing. The receipt is removed before the aggregate and
written after it, so a receipt never refers to an aggregate that is missing or
stale. The receipt binds the plan, every cell identity, every result digest, the
aggregate digest, and the generation it was completed under, and
:func:`verify_completion` recomputes all of it against the store's CURRENT
generation, so a receipt left standing across a later forced refresh reads as
stale rather than complete.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "AGGREGATE_FILENAME",
    "CellExecutor",
    "CellRef",
    "CellResult",
    "CellSpec",
    "Clock",
    "DuplicateCellError",
    "FileRunLock",
    "FileSystemResultStore",
    "GENERATION_FILENAME",
    "LOCK_FILENAME",
    "NotJsonError",
    "NullRunLock",
    "RECEIPT_FILENAME",
    "RESULT_FILENAME",
    "require_json_object",
    "ResultMerger",
    "ResultStore",
    "RunLock",
    "SweepLockedError",
    "SweepOutcome",
    "SweepPlan",
    "SweepReceipt",
    "SweepRunner",
    "SystemClock",
    "canonical_digest",
    "canonical_encode",
    "new_sweep_runner",
    "normalize_json",
    "storage_key",
    "verify_completion",
]

RESULT_FILENAME = "result.json"
AGGREGATE_FILENAME = "aggregate.json"
RECEIPT_FILENAME = "receipt.json"
GENERATION_FILENAME = "generation.json"
LOCK_FILENAME = ".sweep.lock"

_SAFE_SLUG = re.compile(r"[^A-Za-z0-9._-]")
_MAX_SLUG = 64


class DuplicateCellError(ValueError):
    """A plan enumerated the same cell_id more than once."""


class SweepLockedError(RuntimeError):
    """Another run holds the lock on this output root."""


class NotJsonError(ValueError):
    """A value is not native JSON and was not silently coerced."""


# ---------------------------------------------------------------------------
# One JSON representation
# ---------------------------------------------------------------------------


def normalize_json(value: Any, *, path: str = "$") -> Any:
    """Recursively validate that ``value`` is native JSON, returning a deep copy.

    Rejects rather than coerces. ``json.dumps`` maps ``tuple`` and ``list`` onto
    identical bytes, so accepting a tuple would make two distinct in-memory
    values share one digest, and a payload stored as a tuple would come back from
    disk as a list. Both break identity, so neither is allowed in.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise NotJsonError(f"{path}: non-finite float is not valid JSON")
        return value
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise NotJsonError(
                    f"{path}: JSON object keys must be str, got {type(key).__name__}"
                )
            out[key] = normalize_json(item, path=f"{path}.{key}")
        return out
    if isinstance(value, list):
        return [normalize_json(v, path=f"{path}[{i}]") for i, v in enumerate(value)]
    raise NotJsonError(
        f"{path}: {type(value).__name__} is not native JSON; "
        "convert it explicitly rather than relying on coercion"
    )


def require_json_object(value: Any, *, what: str) -> dict[str, Any]:
    """Validate that ``value`` is a JSON object, defaulting only when absent.

    ``value or {}`` would fold ``[]``, ``0``, ``False`` and ``""`` into the empty
    object, so four distinct inputs would share the empty-object digest. Absence
    is the only thing that may default.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise NotJsonError(
            f"{what} must be a JSON object, got {type(value).__name__}"
        )
    return normalize_json(dict(value))


def canonical_encode(value: Any) -> str:
    """Canonical JSON text for a validated value."""
    return json.dumps(
        normalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_digest(value: Any) -> str:
    """SHA-256 over the canonical JSON encoding."""
    return hashlib.sha256(canonical_encode(value).encode("utf-8")).hexdigest()


def storage_key(cell_id: str) -> str:
    """Map a cell_id to a filesystem-safe, collision-free directory name.

    Cell ids are caller data and routinely contain path separators (Phase A's
    natural ids are corpus file paths such as ``human_value/100poison.jsonl``).
    Using one raw would nest directories, collide, or escape the output root
    entirely via ``..``. The slug keeps the name legible for debugging; the hash
    suffix carries the identity.
    """
    slug = _SAFE_SLUG.sub("_", cell_id)[:_MAX_SLUG].strip("._-") or "cell"
    return f"{slug}-{hashlib.sha256(cell_id.encode('utf-8')).hexdigest()[:16]}"


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


class CellSpec(BaseModel):
    """One unit of resumable work.

    ``params`` must carry everything that affects the cell's result. Anything
    omitted is, by construction, invisible to the staleness check.

    Params are held as canonical text, not as a live object, so there is no
    mutable state behind the digest. Reading ``.params`` builds a fresh copy.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cell_id: str = Field(min_length=1)
    params_json: str = "{}"

    @model_validator(mode="before")
    @classmethod
    def _canonicalize(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "params" in data:
            params = require_json_object(data.pop("params"), what="params")
            data["params_json"] = canonical_encode(params)
        elif "params_json" in data:
            # Re-canonicalize so a hand-built spec cannot smuggle in a variant
            # encoding that would digest differently for the same value.
            data["params_json"] = canonical_encode(
                require_json_object(json.loads(data["params_json"]), what="params")
            )
        return data

    @property
    def params(self) -> dict[str, Any]:
        """A fresh copy. Mutating it cannot affect this spec's identity."""
        return json.loads(self.params_json)

    @property
    def input_digest(self) -> str:
        return hashlib.sha256(self.params_json.encode("utf-8")).hexdigest()

    @property
    def storage_key(self) -> str:
        return storage_key(self.cell_id)


class CellResult(BaseModel):
    """A completed cell. Written once, never mutated."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cell_id: str = Field(min_length=1)
    input_digest: str = Field(min_length=1)
    generation: int = 0
    payload_json: str = "{}"
    completed_at: str = ""

    @model_validator(mode="before")
    @classmethod
    def _canonicalize(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "payload" in data:
            payload = require_json_object(data.pop("payload"), what="payload")
            data["payload_json"] = canonical_encode(payload)
        elif "payload_json" in data:
            data["payload_json"] = canonical_encode(
                require_json_object(json.loads(data["payload_json"]), what="payload")
            )
        return data

    @property
    def payload(self) -> dict[str, Any]:
        """A fresh copy, in the same JSON shape it will have after a round trip."""
        return json.loads(self.payload_json)

    @property
    def payload_digest(self) -> str:
        return hashlib.sha256(self.payload_json.encode("utf-8")).hexdigest()


class CellRef(BaseModel):
    """A cell as named in a receipt, bound to the result it produced.

    Strict with unknown fields forbidden: this is an integrity record, and a
    field that coerces or rides along unvalidated is a field a forgery can use.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    cell_id: str = Field(min_length=1)
    input_digest: str = Field(min_length=1)
    payload_digest: str = Field(min_length=1)


class SweepReceipt(BaseModel):
    """Binds a completed sweep to the exact cells and results behind its aggregate.

    Present only when the sweep completed. ``verify_completion`` recomputes every
    digest in it against what is actually on disk, so the aggregate can be
    checked rather than trusted.

    Every integrity field is REQUIRED. A defaulted ``complete`` or ``generation``
    would let a receipt missing the field validate as a complete, generation-0
    sweep, which is exactly the shape a truncating editor produces.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    complete: bool
    generation: int = Field(ge=0)
    plan_digest: str = Field(min_length=1)
    aggregate_digest: str = Field(min_length=1)
    cells: list[CellRef]
    completed_at: str = ""


class SweepOutcome(BaseModel):
    """What a run did, in terms a caller can assert against."""

    executed: list[str] = Field(default_factory=list)
    resumed: list[str] = Field(default_factory=list)
    stale: list[str] = Field(default_factory=list)
    aggregate: dict[str, Any] = Field(default_factory=dict)
    receipt: SweepReceipt | None = None


class SweepPlan(Protocol):
    """Enumerate seam."""

    def cells(self) -> Sequence[CellSpec]: ...


class CellExecutor(Protocol):
    """Execute seam. Returns the cell's payload, or raises to fail the cell."""

    def execute(self, cell: CellSpec) -> Mapping[str, Any]: ...


class ResultMerger(Protocol):
    """Merge seam. Folds completed cells into an aggregate."""

    def merge(self, results: Sequence[CellResult]) -> Mapping[str, Any]: ...


class ResultStore(Protocol):
    """Durable per-cell storage."""

    def generation(self) -> int: ...

    def bump_generation(self) -> int: ...

    def load(self, cell: CellSpec, generation: int) -> CellResult | None: ...

    def has_result(self, cell: CellSpec) -> bool: ...

    def save(self, cell: CellSpec, result: CellResult) -> None: ...

    def discard(self, cell: CellSpec) -> None: ...

    def clear_completion(self) -> None: ...

    def save_completion(
        self, aggregate: Mapping[str, Any], receipt: SweepReceipt
    ) -> None: ...


class RunLock(Protocol):
    """Single-writer enforcement for one output root."""

    def hold(self) -> Any: ...


class Clock(Protocol):
    def now_iso(self) -> str: ...


# ---------------------------------------------------------------------------
# Default implementations
# ---------------------------------------------------------------------------


class SystemClock:
    """Wall-clock timestamps. Injected so tests stay deterministic."""

    def now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()


class NullRunLock:
    """No-op lock, for tests and single-process callers that opt out."""

    @contextmanager
    def hold(self) -> Iterator[None]:
        yield


class FileRunLock:
    """Exclusive lock file in the output root.

    Created with ``O_EXCL`` so acquisition is atomic. A surviving lock from a
    killed process is reported rather than silently stolen: deciding whether the
    prior run is really dead is the operator's call, not this module's.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    @contextmanager
    def hold(self) -> Iterator[None]:
        path = self._root / LOCK_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            raise SweepLockedError(
                f"sweep already locked by {path}; remove it if the prior run is dead"
            ) from exc
        try:
            os.write(fd, str(os.getpid()).encode("utf-8"))
            os.close(fd)
            yield
        finally:
            path.unlink(missing_ok=True)


def _fsync_dir(path: Path) -> None:
    """Durably record a directory entry change (create, rename or unlink)."""
    if not path.exists():
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _unlink_durably(path: Path) -> None:
    """Unlink and fsync the parent, so an invalidation cannot be undone by a crash."""
    existed = path.exists()
    path.unlink(missing_ok=True)
    if existed:
        _fsync_dir(path.parent)


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON so the file is either absent or complete.

    The parent directory is fsynced after the rename. Without that the rename
    itself can be lost on power failure even though the file contents were
    durable, which would resurrect an older result under a name we believe is
    current.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


class FileSystemResultStore:
    """One directory per cell, holding an immutable ``result.json``."""

    def __init__(self, root: Path) -> None:
        self._root = root

    # -- generation ------------------------------------------------------

    def generation(self) -> int:
        """Read the current generation, failing CLOSED on corruption.

        Returning 0 for an unreadable file would revive the initial generation,
        which is exactly the state a forced refresh has just moved away from: any
        pre-force result would load again. Corruption must stop the run.
        """
        path = self._root / GENERATION_FILENAME
        if not path.exists():
            raise ValueError(
                f"{GENERATION_FILENAME} is missing; call ensure_generation() first"
            )
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"{GENERATION_FILENAME} is unreadable: {exc}") from exc
        if not isinstance(raw, dict) or "generation" not in raw:
            raise ValueError(f"{GENERATION_FILENAME} is malformed")
        value = raw["generation"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"{GENERATION_FILENAME} generation must be a non-negative int"
            )
        return value

    def ensure_generation(self) -> int:
        """Materialize generation 0 only on a genuinely fresh root.

        Missing and initial must stay distinct once state exists. If results or
        completion state are present but the generation record is not, the record
        was lost, not never written: recreating 0 would let pre-force results
        load again, and would let a subsequent force re-issue a generation number
        that old results already carry. That is corruption and it stops the run.
        """
        path = self._root / GENERATION_FILENAME
        if not path.exists():
            if self._has_prior_state():
                raise ValueError(
                    f"{GENERATION_FILENAME} is missing but stored results or "
                    "completion state exist; refusing to recreate generation 0 "
                    "over prior state"
                )
            _write_atomic(path, {"generation": 0})
            return 0
        return self.generation()

    def bump_generation(self) -> int:
        # Goes through ensure_generation, so a missing record over existing
        # state refuses here too rather than minting an already-used number.
        nxt = self.ensure_generation() + 1
        _write_atomic(self._root / GENERATION_FILENAME, {"generation": nxt})
        return nxt

    def _has_prior_state(self) -> bool:
        """Whether this root holds anything a generation record should govern."""
        if not self._root.exists():
            return False
        if (self._root / RECEIPT_FILENAME).exists():
            return True
        if (self._root / AGGREGATE_FILENAME).exists():
            return True
        return any(
            child.is_dir() and (child / RESULT_FILENAME).exists()
            for child in self._root.iterdir()
        )

    # -- results ---------------------------------------------------------

    def load(self, cell: CellSpec, generation: int) -> CellResult | None:
        path = self._result_path(cell)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return None
            recorded_digest = raw.pop("payload_digest", None)
            result = CellResult.model_validate(raw)
        except (json.JSONDecodeError, TypeError, ValueError, OSError):
            # Unreadable or malformed results are treated as absent so the cell
            # is recomputed. Reusing a result we cannot parse would be worse.
            return None
        if not isinstance(recorded_digest, str) or not recorded_digest:
            # Every result this module writes carries a digest. One without is
            # not a legacy record, it is a record nothing vouches for.
            return None
        if result.cell_id != cell.cell_id:
            # Storage-key collision or a hand-edited tree. Refuse to serve it.
            return None
        if result.input_digest != cell.input_digest:
            # Computed from different params. This is its own identity field,
            # distinct from the payload digest: the payload can hash perfectly
            # and still be the answer to a different question.
            return None
        if result.generation != generation:
            # Produced before a forced refresh.
            return None
        if recorded_digest != result.payload_digest:
            # The payload was altered after it was written. Recompute rather than
            # publish a value nothing vouches for.
            return None
        return result

    def has_result(self, cell: CellSpec) -> bool:
        """Whether ANY stored record exists for this cell, servable or not."""
        return self._result_path(cell).exists()

    def save(self, cell: CellSpec, result: CellResult) -> None:
        record = result.model_dump()
        record["payload_digest"] = result.payload_digest
        _write_atomic(self._result_path(cell), record)

    def discard(self, cell: CellSpec) -> None:
        _unlink_durably(self._result_path(cell))

    # -- completion ------------------------------------------------------

    def clear_completion(self) -> None:
        # Receipt first: it is the completion marker, so it must never outlive
        # the aggregate it describes.
        _unlink_durably(self._root / RECEIPT_FILENAME)
        _unlink_durably(self._root / AGGREGATE_FILENAME)

    def save_completion(
        self, aggregate: Mapping[str, Any], receipt: SweepReceipt
    ) -> None:
        # Aggregate first, receipt last, mirroring clear_completion.
        _write_atomic(self._root / AGGREGATE_FILENAME, dict(aggregate))
        _write_atomic(self._root / RECEIPT_FILENAME, receipt.model_dump())

    def _result_path(self, cell: CellSpec) -> Path:
        return self._root / cell.storage_key / RESULT_FILENAME


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_completion(
    root: Path, *, plan: SweepPlan, merger: ResultMerger
) -> dict[str, Any]:
    """Rederive a completed sweep from stored results and the expected plan.

    Returns the reconstructed aggregate. Raises ``ValueError`` on any mismatch.

    This deliberately takes the plan and the merger rather than working from the
    receipt alone. Hashing the stored aggregate and comparing it to the digest
    stored beside it is circular: a producer that changes both stays consistent,
    and the receipt cannot rederive input digests without knowing what was
    supposed to run. The aggregate is only evidence if it can be rebuilt from the
    cell results by re-merging them.
    """
    receipt_path = root / RECEIPT_FILENAME
    aggregate_path = root / AGGREGATE_FILENAME
    if not receipt_path.exists():
        raise ValueError("no receipt: sweep did not complete")
    if not aggregate_path.exists():
        raise ValueError("receipt present but aggregate missing")

    receipt = SweepReceipt.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    if not receipt.complete:
        raise ValueError(
            "receipt is marked incomplete; its presence alone proves nothing"
        )

    store = FileSystemResultStore(root)
    # The store's CURRENT generation, read fail-closed, not the receipt's word
    # for it. A receipt left standing across a later forced refresh carries the
    # old generation and would otherwise validate a superseded completion.
    current_generation = store.generation()
    if receipt.generation != current_generation:
        raise ValueError(
            f"receipt generation {receipt.generation} is not the store's "
            f"current generation {current_generation}: superseded completion"
        )

    expected = list(plan.cells())
    SweepRunner._reject_duplicates(expected)
    if SweepRunner._plan_digest(expected) != receipt.plan_digest:
        raise ValueError("plan digest mismatch: receipt describes a different sweep")
    if [c.cell_id for c in expected] != [r.cell_id for r in receipt.cells]:
        raise ValueError("receipt cells do not match the expected plan")

    results: list[CellResult] = []
    for cell, ref in zip(expected, receipt.cells):
        if cell.input_digest != ref.input_digest:
            raise ValueError(f"input digest mismatch for {cell.cell_id}")
        result = store.load(cell, receipt.generation)
        if result is None:
            raise ValueError(
                f"receipt names a result that is missing, malformed, tampered, "
                f"from different inputs or of the wrong generation: {cell.cell_id}"
            )
        if result.input_digest != cell.input_digest:
            # The store already refuses this; checked again here because the
            # verifier's job is to trust nothing, including the store.
            raise ValueError(f"stored result input digest mismatch for {cell.cell_id}")
        if result.payload_digest != ref.payload_digest:
            raise ValueError(f"result digest mismatch for {cell.cell_id}")
        results.append(result)

    rebuilt = require_json_object(merger.merge(results), what="merged aggregate")
    stored = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if canonical_digest(rebuilt) != receipt.aggregate_digest:
        raise ValueError("aggregate digest does not match the re-merged results")
    if canonical_encode(stored) != canonical_encode(rebuilt):
        raise ValueError("stored aggregate does not match the re-merged results")
    return rebuilt


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class SweepRunner:
    """Executes a plan, skipping cells already recorded complete.

    Constructor assigns dependencies only. All I/O happens in :meth:`run`.
    """

    def __init__(
        self,
        *,
        plan: SweepPlan,
        executor: CellExecutor,
        merger: ResultMerger,
        store: ResultStore,
        clock: Clock,
        lock: RunLock,
    ) -> None:
        self._plan = plan
        self._executor = executor
        self._merger = merger
        self._store = store
        self._clock = clock
        self._lock = lock

    def run(self, *, force: bool = False) -> SweepOutcome:
        """Run every cell that is not already recorded complete.

        Raises whatever a cell executor raises. The failing cell is left with no
        stored result, and no aggregate or receipt is written, so an incomplete
        sweep is visibly incomplete rather than stale-but-plausible.
        """
        cells = list(self._plan.cells())
        self._reject_duplicates(cells)

        with self._lock.hold():
            # Establish the generation FIRST. On a corrupt store (missing or
            # unreadable generation over existing state) this raises before
            # anything is mutated, so the evidence of what went wrong is intact.
            if force:
                # Bump the generation BEFORE discarding. The bump is a single
                # atomic write, so a crash anywhere in the discard loop still
                # leaves every pre-force result unusable: they carry the old
                # generation and will not load. A crash between the bump and
                # clear_completion leaves the old receipt standing, but it now
                # names a non-current generation, so verify_completion rejects
                # it as superseded.
                generation = self._store.bump_generation()
            else:
                generation = self._store.ensure_generation()

            # Any prior completion describes a sweep that is about to change.
            # Drop it before any result state changes, so a failure cannot
            # leave it standing as if current.
            self._store.clear_completion()

            if force:
                for cell in cells:
                    self._store.discard(cell)

            executed: list[str] = []
            resumed: list[str] = []
            stale: list[str] = []
            results: list[CellResult] = []

            for cell in cells:
                existing = self._store.load(cell, generation)

                if existing is not None:
                    # load enforces full identity, so a served result already
                    # matches this cell's inputs and the current generation.
                    resumed.append(cell.cell_id)
                    results.append(existing)
                    continue

                if self._store.has_result(cell):
                    # A record exists but is not servable under the current
                    # identity: changed params, an old generation, tamper, or a
                    # malformed file. All of them are dead weight to remove.
                    stale.append(cell.cell_id)
                    self._store.discard(cell)

                payload = require_json_object(
                    self._executor.execute(cell), what="executor payload"
                )
                result = CellResult(
                    cell_id=cell.cell_id,
                    input_digest=cell.input_digest,
                    generation=generation,
                    payload=payload,
                    completed_at=self._clock.now_iso(),
                )
                self._store.save(cell, result)
                executed.append(cell.cell_id)
                results.append(result)

            aggregate = require_json_object(
                self._merger.merge(results), what="merged aggregate"
            )
            by_id = {r.cell_id: r for r in results}
            receipt = SweepReceipt(
                complete=True,
                generation=generation,
                plan_digest=self._plan_digest(cells),
                aggregate_digest=canonical_digest(aggregate),
                cells=[
                    CellRef(
                        cell_id=c.cell_id,
                        input_digest=c.input_digest,
                        payload_digest=by_id[c.cell_id].payload_digest,
                    )
                    for c in cells
                ],
                completed_at=self._clock.now_iso(),
            )
            self._store.save_completion(aggregate, receipt)

            return SweepOutcome(
                executed=executed,
                resumed=resumed,
                stale=stale,
                aggregate=aggregate,
                receipt=receipt,
            )

    @staticmethod
    def _reject_duplicates(cells: Sequence[CellSpec]) -> None:
        seen: set[str] = set()
        for cell in cells:
            if cell.cell_id in seen:
                raise DuplicateCellError(
                    f"plan enumerated cell_id more than once: {cell.cell_id!r}"
                )
            seen.add(cell.cell_id)

    @staticmethod
    def _plan_digest(cells: Sequence[CellSpec]) -> str:
        return canonical_digest([[c.cell_id, c.input_digest] for c in cells])


def new_sweep_runner(
    *,
    plan: SweepPlan,
    executor: CellExecutor,
    merger: ResultMerger,
    output_root: Path,
    clock: Clock | None = None,
    lock: RunLock | None = None,
) -> SweepRunner:
    """Build a runner backed by the filesystem store and an exclusive file lock."""
    return SweepRunner(
        plan=plan,
        executor=executor,
        merger=merger,
        store=FileSystemResultStore(output_root),
        clock=clock or SystemClock(),
        lock=lock if lock is not None else FileRunLock(output_root),
    )
