"""Behavior tests for the resumable sweep executor.

The load-bearing tests are the ones that prove durability rather than assert it:

- ``test_sigkilled_sweep_resumes_to_the_same_result_as_an_uninterrupted_run``
  kills a real subprocess mid-cell, so the filesystem crash window is exercised
  rather than simulated with an in-process exception.
- ``test_changed_params_recompute_instead_of_reusing_a_stale_result`` closes the
  trap where identity keyed on cell name alone returns results computed under
  old parameters.
- ``test_failed_rerun_leaves_no_aggregate`` closes the trap where a failed run
  leaves a complete-looking aggregate inconsistent with the stored cells.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from parapet_data.sweep import (
    AGGREGATE_FILENAME,
    GENERATION_FILENAME,
    LOCK_FILENAME,
    RECEIPT_FILENAME,
    RESULT_FILENAME,
    CellResult,
    CellSpec,
    DuplicateCellError,
    FileRunLock,
    NotJsonError,
    FileSystemResultStore,
    NullRunLock,
    SweepLockedError,
    SweepRunner,
    canonical_digest,
    new_sweep_runner,
    normalize_json,
    storage_key,
    verify_completion,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class ListPlan:
    def __init__(self, cells: Sequence[CellSpec]) -> None:
        self._cells = list(cells)

    def cells(self) -> Sequence[CellSpec]:
        return self._cells


class RecordingExecutor:
    """Records which cells it ran, and can be told to fail specific cells."""

    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self.fail_on = fail_on or set()

    def execute(self, cell: CellSpec) -> Mapping[str, object]:
        self.calls.append(cell.cell_id)
        if cell.cell_id in self.fail_on:
            raise RuntimeError(f"cell failed: {cell.cell_id}")
        return {"value": cell.params.get("value"), "seen": cell.cell_id}


class SortedMerger:
    """Deterministic aggregate so two runs are directly comparable."""

    def merge(self, results: Sequence[CellResult]) -> Mapping[str, object]:
        rows = sorted(
            ({"cell_id": r.cell_id, "payload": r.payload} for r in results),
            key=lambda row: row["cell_id"],
        )
        return {"count": len(rows), "rows": rows}


class FixedClock:
    def now_iso(self) -> str:
        return "2026-01-01T00:00:00+00:00"


def build_runner(
    cells: Sequence[CellSpec],
    root: Path,
    executor: RecordingExecutor | None = None,
) -> tuple[SweepRunner, RecordingExecutor]:
    exec_double = executor or RecordingExecutor()
    runner = new_sweep_runner(
        plan=ListPlan(cells),
        executor=exec_double,
        merger=SortedMerger(),
        output_root=root,
        clock=FixedClock(),
        lock=NullRunLock(),
    )
    return runner, exec_double


def three_cells(value: int = 1) -> list[CellSpec]:
    return [
        CellSpec(cell_id="a", params={"value": value}),
        CellSpec(cell_id="b", params={"value": value}),
        CellSpec(cell_id="c", params={"value": value}),
    ]


# ---------------------------------------------------------------------------
# Resumption, proven against a real process kill
# ---------------------------------------------------------------------------


KILL_SCRIPT = textwrap.dedent(
    """
    import os, signal, sys
    from pathlib import Path
    from parapet_data.sweep import CellSpec, new_sweep_runner

    root = Path(sys.argv[1])
    kill_on = sys.argv[2]

    class Plan:
        def cells(self):
            return [CellSpec(cell_id=c, params={"value": 1}) for c in ("a", "b", "c")]

    class Executor:
        def execute(self, cell):
            if cell.cell_id == kill_on:
                # Hard kill: no unwinding, no finally blocks, no flush.
                os.kill(os.getpid(), signal.SIGKILL)
            return {"value": cell.params.get("value"), "seen": cell.cell_id}

    class Merger:
        def merge(self, results):
            rows = sorted(
                ({"cell_id": r.cell_id, "payload": r.payload} for r in results),
                key=lambda row: row["cell_id"],
            )
            return {"count": len(rows), "rows": rows}

    class Clock:
        def now_iso(self):
            return "2026-01-01T00:00:00+00:00"

    new_sweep_runner(
        plan=Plan(), executor=Executor(), merger=Merger(),
        output_root=root, clock=Clock(),
    ).run()
    """
)


def test_sigkilled_sweep_resumes_to_the_same_result_as_an_uninterrupted_run(
    tmp_path: Path,
) -> None:
    """The proof, against a real crash window.

    A SIGKILL gives no unwinding and no ``finally``, so this exercises what an
    in-process exception cannot: partially written files, a surviving lock, and
    whatever the filesystem actually committed.
    """
    clean_root = tmp_path / "clean"
    clean_runner, _ = build_runner(three_cells(), clean_root)
    clean_aggregate = clean_runner.run().aggregate

    killed_root = tmp_path / "killed"
    script = tmp_path / "kill_run.py"
    script.write_text(KILL_SCRIPT, encoding="utf-8")

    env = {**os.environ, "PYTHONPATH": str(PACKAGE_ROOT)}
    proc = subprocess.run(
        [sys.executable, str(script), str(killed_root), "b"],
        capture_output=True,
        cwd=PACKAGE_ROOT,
        env=env,
    )
    assert proc.returncode == -9, (
        f"expected SIGKILL, got {proc.returncode}: {proc.stderr.decode()}"
    )

    store = FileSystemResultStore(killed_root)
    gen = store.generation()
    cells = three_cells()
    assert store.load(cells[0], gen) is not None, "work before the kill must survive"
    assert store.load(cells[1], gen) is None, "the killed cell must leave no result"
    assert store.load(cells[2], gen) is None

    # No completion marker: an interrupted sweep is visibly incomplete.
    assert not (killed_root / AGGREGATE_FILENAME).exists()
    assert not (killed_root / RECEIPT_FILENAME).exists()

    # The killed process could not release its lock. Recovery is explicit.
    with pytest.raises(SweepLockedError):
        new_sweep_runner(
            plan=ListPlan(three_cells()),
            executor=RecordingExecutor(),
            merger=SortedMerger(),
            output_root=killed_root,
            clock=FixedClock(),
        ).run()

    (killed_root / LOCK_FILENAME).unlink()

    healthy = RecordingExecutor()
    runner, _ = build_runner(three_cells(), killed_root, executor=healthy)
    outcome = runner.run()

    assert healthy.calls == ["b", "c"], "completed work must not be repeated"
    assert outcome.resumed == ["a"]
    assert outcome.aggregate == clean_aggregate


def test_interrupted_sweep_resumes_after_an_in_process_failure(
    tmp_path: Path,
) -> None:
    clean_root = tmp_path / "clean"
    clean_runner, _ = build_runner(three_cells(), clean_root)
    clean_aggregate = clean_runner.run().aggregate

    resumed_root = tmp_path / "resumed"
    failing = RecordingExecutor(fail_on={"b"})
    runner, _ = build_runner(three_cells(), resumed_root, executor=failing)
    with pytest.raises(RuntimeError, match="cell failed: b"):
        runner.run()

    healthy = RecordingExecutor()
    runner2, _ = build_runner(three_cells(), resumed_root, executor=healthy)
    outcome = runner2.run()

    assert healthy.calls == ["b", "c"]
    assert outcome.aggregate == clean_aggregate


def test_completed_cells_are_not_re_executed(tmp_path: Path) -> None:
    cells = three_cells()
    runner, executor = build_runner(cells, tmp_path)
    runner.run()
    assert executor.calls == ["a", "b", "c"]

    runner2, executor2 = build_runner(cells, tmp_path)
    outcome = runner2.run()

    assert executor2.calls == []
    assert outcome.resumed == ["a", "b", "c"]
    assert outcome.executed == []


# ---------------------------------------------------------------------------
# Completion integrity
# ---------------------------------------------------------------------------


def test_failed_rerun_leaves_no_aggregate(tmp_path: Path) -> None:
    """A failed re-run must not leave the previous aggregate standing.

    The prior aggregate was folded from the old inputs. If a changed-input run
    dies partway, some cells on disk are new and some are old, and leaving the
    old aggregate in place presents that mixture as a finished result.
    """
    runner, _ = build_runner(three_cells(value=1), tmp_path)
    runner.run()
    assert (tmp_path / AGGREGATE_FILENAME).exists()

    failing = RecordingExecutor(fail_on={"b"})
    runner2, _ = build_runner(three_cells(value=2), tmp_path, executor=failing)
    with pytest.raises(RuntimeError):
        runner2.run()

    assert not (tmp_path / AGGREGATE_FILENAME).exists()
    assert not (tmp_path / RECEIPT_FILENAME).exists()


def test_receipt_binds_the_aggregate_to_the_cells_it_covers(tmp_path: Path) -> None:
    cells = three_cells(value=5)
    runner, _ = build_runner(cells, tmp_path)
    outcome = runner.run()

    receipt = json.loads((tmp_path / RECEIPT_FILENAME).read_text(encoding="utf-8"))
    aggregate = json.loads((tmp_path / AGGREGATE_FILENAME).read_text(encoding="utf-8"))

    assert receipt["complete"] is True
    assert receipt["aggregate_digest"] == canonical_digest(aggregate)
    assert [c["cell_id"] for c in receipt["cells"]] == ["a", "b", "c"]
    assert all(
        c["input_digest"] == cells[i].input_digest
        for i, c in enumerate(receipt["cells"])
    )
    assert outcome.receipt is not None
    assert outcome.receipt.aggregate_digest == canonical_digest(outcome.aggregate)


def test_plan_digest_changes_when_inputs_change(tmp_path: Path) -> None:
    runner, _ = build_runner(three_cells(value=1), tmp_path / "one")
    first = runner.run().receipt
    runner2, _ = build_runner(three_cells(value=2), tmp_path / "two")
    second = runner2.run().receipt

    assert first is not None and second is not None
    assert first.plan_digest != second.plan_digest


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------


def test_changed_params_recompute_instead_of_reusing_a_stale_result(
    tmp_path: Path,
) -> None:
    runner, _ = build_runner(three_cells(value=1), tmp_path)
    first = runner.run()
    assert first.aggregate["rows"][0]["payload"]["value"] == 1

    runner2, executor2 = build_runner(three_cells(value=999), tmp_path)
    second = runner2.run()

    assert executor2.calls == ["a", "b", "c"], "changed params must recompute"
    assert second.stale == ["a", "b", "c"]
    assert second.resumed == []
    assert second.aggregate["rows"][0]["payload"]["value"] == 999


def test_unchanged_params_are_not_treated_as_stale(tmp_path: Path) -> None:
    runner, _ = build_runner(three_cells(value=7), tmp_path)
    runner.run()

    runner2, executor2 = build_runner(three_cells(value=7), tmp_path)
    outcome = runner2.run()

    assert outcome.stale == []
    assert executor2.calls == []


def test_param_order_does_not_affect_identity() -> None:
    assert (
        CellSpec(cell_id="a", params={"x": 1, "y": 2}).input_digest
        == CellSpec(cell_id="a", params={"y": 2, "x": 1}).input_digest
    )


def test_digest_changes_when_a_param_changes() -> None:
    assert (
        CellSpec(cell_id="a", params={"x": 1}).input_digest
        != CellSpec(cell_id="a", params={"x": 2}).input_digest
    )


# ---------------------------------------------------------------------------
# Strict input typing
# ---------------------------------------------------------------------------


def test_non_json_params_are_rejected_rather_than_stringified() -> None:
    """Coercion would collide distinct inputs onto one digest."""
    with pytest.raises(ValueError, match="not native JSON"):
        CellSpec(cell_id="a", params={"path": Path("/a")})


def test_nan_params_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        CellSpec(cell_id="a", params={"x": float("nan")})


def test_params_are_deeply_immutable_not_merely_pinned() -> None:
    """Mutating what ``.params`` returns must not change the spec at all.

    Pinning only the digest is not enough and is actively worse: the executed
    params and the digest describing them would disagree, so a result would be
    cached under a digest that does not describe it. ``.params`` returns a fresh
    copy, so there is no shared state to mutate.
    """
    cell = CellSpec(cell_id="a", params={"top": 1, "nested": {"value": 1}})
    pinned = cell.input_digest

    grabbed = cell.params
    grabbed["top"] = 99
    grabbed["nested"]["value"] = 99

    assert cell.params == {"top": 1, "nested": {"value": 1}}
    assert cell.input_digest == pinned


def test_caller_mutation_of_the_source_dict_does_not_affect_the_cell() -> None:
    source = {"x": 1, "nested": {"y": 2}}
    cell = CellSpec(cell_id="a", params=source)
    pinned = cell.input_digest

    source["x"] = 99
    source["nested"]["y"] = 99

    assert cell.input_digest == pinned
    assert cell.params == {"x": 1, "nested": {"y": 2}}


# ---------------------------------------------------------------------------
# Storage keys
# ---------------------------------------------------------------------------


def test_path_like_cell_ids_stay_inside_the_output_root(tmp_path: Path) -> None:
    """Cell ids are caller data and routinely look like paths."""
    hostile = [
        CellSpec(cell_id="../../escape", params={"value": 1}),
        CellSpec(cell_id="human_value/100poison.jsonl", params={"value": 1}),
        CellSpec(cell_id="/absolute/path", params={"value": 1}),
    ]
    runner, _ = build_runner(hostile, tmp_path)
    runner.run()

    written = [p for p in tmp_path.rglob(RESULT_FILENAME)]
    assert len(written) == 3
    for path in written:
        assert tmp_path.resolve() in path.resolve().parents
        # Exactly one directory level below the root.
        assert path.resolve().parent.parent == tmp_path.resolve()


def test_distinct_cell_ids_do_not_share_a_storage_key() -> None:
    # These collapse to the same slug; the hash suffix must separate them.
    assert storage_key("a/b") != storage_key("a_b")
    assert storage_key("../x") != storage_key("__x")


def test_path_like_cell_ids_resume_correctly(tmp_path: Path) -> None:
    cells = [CellSpec(cell_id="human_value/100poison.jsonl", params={"value": 1})]
    runner, _ = build_runner(cells, tmp_path)
    runner.run()

    runner2, executor2 = build_runner(cells, tmp_path)
    outcome = runner2.run()

    assert executor2.calls == []
    assert outcome.resumed == ["human_value/100poison.jsonl"]


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------


def test_duplicate_cell_ids_are_rejected(tmp_path: Path) -> None:
    """Duplicates would overwrite each other and never converge."""
    cells = [
        CellSpec(cell_id="a", params={"value": 1}),
        CellSpec(cell_id="a", params={"value": 2}),
    ]
    runner, executor = build_runner(cells, tmp_path)

    with pytest.raises(DuplicateCellError, match="'a'"):
        runner.run()
    assert executor.calls == [], "rejection must happen before any work"


# ---------------------------------------------------------------------------
# force
# ---------------------------------------------------------------------------


def test_failed_force_refresh_does_not_leave_reusable_stale_results(
    tmp_path: Path,
) -> None:
    """A forced refresh that dies must not leave old results a later run resumes.

    Otherwise the refresh is silently partial: the operator asked for everything
    to be recomputed and a subsequent ordinary run quietly serves pre-refresh
    values for the cells the crash never reached.
    """
    runner, _ = build_runner(three_cells(value=1), tmp_path)
    runner.run()

    failing = RecordingExecutor(fail_on={"a"})
    runner2, _ = build_runner(three_cells(value=1), tmp_path, executor=failing)
    with pytest.raises(RuntimeError):
        runner2.run(force=True)

    # Nothing from before the forced refresh may survive.
    store = FileSystemResultStore(tmp_path)
    for cell in three_cells(value=1):
        assert store.load(cell, store.generation()) is None

    healthy = RecordingExecutor()
    runner3, _ = build_runner(three_cells(value=1), tmp_path, executor=healthy)
    outcome = runner3.run()
    assert healthy.calls == ["a", "b", "c"]
    assert outcome.resumed == []


def test_force_re_executes_completed_cells(tmp_path: Path) -> None:
    runner, _ = build_runner(three_cells(), tmp_path)
    runner.run()

    runner2, executor2 = build_runner(three_cells(), tmp_path)
    outcome = runner2.run(force=True)

    assert executor2.calls == ["a", "b", "c"]
    assert outcome.resumed == []
    assert outcome.stale == []


# ---------------------------------------------------------------------------
# Single writer
# ---------------------------------------------------------------------------


def test_second_runner_cannot_enter_a_locked_sweep(tmp_path: Path) -> None:
    lock = FileRunLock(tmp_path)
    with lock.hold():
        runner = new_sweep_runner(
            plan=ListPlan(three_cells()),
            executor=RecordingExecutor(),
            merger=SortedMerger(),
            output_root=tmp_path,
            clock=FixedClock(),
        )
        with pytest.raises(SweepLockedError):
            runner.run()


def test_lock_is_released_after_a_successful_run(tmp_path: Path) -> None:
    runner = new_sweep_runner(
        plan=ListPlan(three_cells()),
        executor=RecordingExecutor(),
        merger=SortedMerger(),
        output_root=tmp_path,
        clock=FixedClock(),
    )
    runner.run()
    assert not (tmp_path / LOCK_FILENAME).exists()


def test_lock_is_released_after_a_failed_run(tmp_path: Path) -> None:
    runner = new_sweep_runner(
        plan=ListPlan(three_cells()),
        executor=RecordingExecutor(fail_on={"b"}),
        merger=SortedMerger(),
        output_root=tmp_path,
        clock=FixedClock(),
    )
    with pytest.raises(RuntimeError):
        runner.run()
    assert not (tmp_path / LOCK_FILENAME).exists()


# ---------------------------------------------------------------------------
# Durability of the stored result
# ---------------------------------------------------------------------------


def test_malformed_result_file_is_treated_as_absent(tmp_path: Path) -> None:
    cells = three_cells()
    runner, _ = build_runner(cells, tmp_path)
    runner.run()

    corrupt = tmp_path / cells[1].storage_key / RESULT_FILENAME
    corrupt.write_text('{"cell_id": "b", "inp', encoding="utf-8")

    runner2, executor2 = build_runner(three_cells(), tmp_path)
    outcome = runner2.run()

    assert executor2.calls == ["b"]
    assert outcome.executed == ["b"]


def test_result_stored_under_a_mismatched_cell_id_is_refused(tmp_path: Path) -> None:
    cells = three_cells()
    runner, _ = build_runner(cells, tmp_path)
    runner.run()

    path = tmp_path / cells[1].storage_key / RESULT_FILENAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["cell_id"] = "somebody-else"
    path.write_text(json.dumps(raw), encoding="utf-8")

    runner2, executor2 = build_runner(three_cells(), tmp_path)
    runner2.run()
    assert executor2.calls == ["b"]


def test_save_leaves_no_temporary_files(tmp_path: Path) -> None:
    runner, _ = build_runner(three_cells(), tmp_path)
    runner.run()
    assert list(tmp_path.rglob("*.tmp")) == []


def test_stored_result_records_the_digest_it_was_computed_under(
    tmp_path: Path,
) -> None:
    cells = [CellSpec(cell_id="a", params={"value": 3})]
    runner, _ = build_runner(cells, tmp_path)
    runner.run()

    path = tmp_path / cells[0].storage_key / RESULT_FILENAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["input_digest"] == canonical_digest({"value": 3})


def test_aggregate_is_written(tmp_path: Path) -> None:
    runner, _ = build_runner(three_cells(), tmp_path)
    outcome = runner.run()

    written = json.loads((tmp_path / AGGREGATE_FILENAME).read_text(encoding="utf-8"))
    assert written == outcome.aggregate
    assert written["count"] == 3


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


def test_empty_plan_produces_an_empty_aggregate(tmp_path: Path) -> None:
    runner, executor = build_runner([], tmp_path)
    outcome = runner.run()

    assert executor.calls == []
    assert outcome.executed == []
    assert outcome.aggregate == {"count": 0, "rows": []}


def test_store_does_not_touch_the_filesystem_on_construction(tmp_path: Path) -> None:
    """Constructors assign dependencies only. No I/O, per the house rule."""
    root = tmp_path / "not-created-yet"
    FileSystemResultStore(root)
    FileRunLock(root)
    assert not root.exists()


# ---------------------------------------------------------------------------
# One JSON representation (identity must be injective)
# ---------------------------------------------------------------------------


def test_tuple_and_list_params_do_not_share_an_identity() -> None:
    """json maps tuple and list onto identical bytes, so tuples are refused."""
    with pytest.raises(ValueError, match="not native JSON"):
        CellSpec(cell_id="a", params={"x": (1, 2)})

    assert (
        CellSpec(cell_id="a", params={"x": [1, 2]}).input_digest
        != CellSpec(cell_id="a", params={"x": [1, 2, 3]}).input_digest
    )


def test_non_json_types_are_refused_everywhere_they_could_enter() -> None:
    with pytest.raises(NotJsonError):
        normalize_json({"x": Path("/a")})
    with pytest.raises(NotJsonError):
        normalize_json({"x": {1, 2}})
    with pytest.raises(NotJsonError):
        normalize_json({1: "int key"})


def test_executor_payload_shape_survives_a_resume_unchanged(tmp_path: Path) -> None:
    """A payload must not change shape between first run and resume.

    Previously a tuple payload merged as a tuple on the first run and as a list
    after a round trip, so an identical plan produced two different aggregates
    with zero execution.
    """

    class TuplePayloadExecutor:
        def execute(self, cell: CellSpec) -> Mapping[str, object]:
            return {"pair": (1, 2)}

    runner = new_sweep_runner(
        plan=ListPlan([CellSpec(cell_id="a", params={"v": 1})]),
        executor=TuplePayloadExecutor(),
        merger=SortedMerger(),
        output_root=tmp_path,
        clock=FixedClock(),
        lock=NullRunLock(),
    )
    with pytest.raises(NotJsonError, match="not native JSON"):
        runner.run()


def test_identical_plan_produces_an_identical_aggregate_on_resume(
    tmp_path: Path,
) -> None:
    class ListPayloadExecutor:
        def execute(self, cell: CellSpec) -> Mapping[str, object]:
            return {"pair": [1, 2], "kind": "list"}

    def build() -> SweepRunner:
        return new_sweep_runner(
            plan=ListPlan([CellSpec(cell_id="a", params={"v": 1})]),
            executor=ListPayloadExecutor(),
            merger=SortedMerger(),
            output_root=tmp_path,
            clock=FixedClock(),
            lock=NullRunLock(),
        )

    first = build().run()
    second = build().run()

    assert second.executed == []
    assert second.aggregate == first.aggregate


# ---------------------------------------------------------------------------
# Result integrity
# ---------------------------------------------------------------------------


def test_tampered_payload_is_recomputed_not_published(tmp_path: Path) -> None:
    """A syntactically valid edit to a stored payload must not be trusted.

    Binding only ``(cell_id, input_digest)`` let an edited payload ride through:
    nothing executed, the cell reported as resumed, and the altered value was
    published in the aggregate.
    """
    cells = [CellSpec(cell_id="a", params={"value": 1})]
    runner, _ = build_runner(cells, tmp_path)
    first = runner.run()
    assert first.aggregate["rows"][0]["payload"]["value"] == 1

    path = tmp_path / cells[0].storage_key / RESULT_FILENAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["payload_json"] = json.dumps({"value": 777, "seen": "a"}, sort_keys=True)
    path.write_text(json.dumps(raw), encoding="utf-8")

    runner2, executor2 = build_runner(cells, tmp_path)
    second = runner2.run()

    assert executor2.calls == ["a"], "tampered payload must be recomputed"
    assert second.aggregate["rows"][0]["payload"]["value"] == 1


def test_receipt_binds_result_digests(tmp_path: Path) -> None:
    cells = three_cells(value=4)
    runner, _ = build_runner(cells, tmp_path)
    outcome = runner.run()

    assert outcome.receipt is not None
    refs = {r.cell_id: r.payload_digest for r in outcome.receipt.cells}
    store = FileSystemResultStore(tmp_path)
    gen = store.generation()
    for cell in cells:
        stored = store.load(cell, gen)
        assert stored is not None
        assert refs[cell.cell_id] == stored.payload_digest


def test_verify_completion_accepts_an_untouched_sweep(tmp_path: Path) -> None:
    runner, _ = build_runner(three_cells(), tmp_path)
    runner.run()
    verify_completion(tmp_path, plan=ListPlan(three_cells()), merger=SortedMerger())


def test_verify_completion_rejects_a_tampered_payload(tmp_path: Path) -> None:
    cells = three_cells()
    runner, _ = build_runner(cells, tmp_path)
    runner.run()

    path = tmp_path / cells[1].storage_key / RESULT_FILENAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["payload_json"] = json.dumps({"value": 42})
    raw["payload_digest"] = _sha(raw["payload_json"])
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="missing, malformed"):
        verify_completion(tmp_path, plan=ListPlan(three_cells()), merger=SortedMerger())


def test_verify_completion_rejects_a_tampered_aggregate(tmp_path: Path) -> None:
    runner, _ = build_runner(three_cells(), tmp_path)
    runner.run()

    path = tmp_path / AGGREGATE_FILENAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["count"] = 99
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the re-merged"):
        verify_completion(tmp_path, plan=ListPlan(three_cells()), merger=SortedMerger())


def test_verify_completion_rejects_an_incomplete_sweep(tmp_path: Path) -> None:
    failing = RecordingExecutor(fail_on={"b"})
    runner, _ = build_runner(three_cells(), tmp_path, executor=failing)
    with pytest.raises(RuntimeError):
        runner.run()

    with pytest.raises(ValueError, match="did not complete"):
        verify_completion(tmp_path, plan=ListPlan(three_cells()), merger=SortedMerger())


# ---------------------------------------------------------------------------
# Result input identity: the digest of the QUESTION, not just the answer
# ---------------------------------------------------------------------------


def test_store_refuses_a_result_whose_input_digest_disagrees(tmp_path: Path) -> None:
    """The store itself enforces input identity; the runner is not the last line."""
    cells = [CellSpec(cell_id="a", params={"value": 1})]
    runner, _ = build_runner(cells, tmp_path)
    runner.run()

    path = tmp_path / cells[0].storage_key / RESULT_FILENAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["input_digest"] = "0" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")

    store = FileSystemResultStore(tmp_path)
    assert store.load(cells[0], store.generation()) is None


def test_verify_completion_rejects_a_result_with_a_substituted_input_digest(
    tmp_path: Path,
) -> None:
    """Routing's fourth-review probe, byte for byte.

    Change ONLY the stored result's ``input_digest``; payload, payload_digest,
    receipt and aggregate all stay untouched and mutually consistent. The result
    is the answer to a different question, and must not validate.
    """
    cells = three_cells(value=1)
    runner, _ = build_runner(cells, tmp_path)
    runner.run()

    path = tmp_path / cells[0].storage_key / RESULT_FILENAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["input_digest"] = "0" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="different inputs|input digest"):
        _verify(tmp_path)


def test_verify_completion_rejects_a_result_substituted_from_other_inputs(
    tmp_path: Path,
) -> None:
    """The full substitution attack: an honest-looking payload from elsewhere.

    Take a valid result computed under DIFFERENT params, drop it into this
    sweep's cell, and recompute the receipt's payload digest and the aggregate
    exactly as an honest producer would. Every digest is self-consistent; only
    the input identity betrays it.
    """
    victim_cells = [CellSpec(cell_id="a", params={"value": 1})]
    runner, _ = build_runner(victim_cells, tmp_path / "victim")
    runner.run()

    donor_cells = [CellSpec(cell_id="a", params={"value": 2})]
    runner2, _ = build_runner(donor_cells, tmp_path / "donor")
    runner2.run()

    victim_root = tmp_path / "victim"
    donor_result = (
        tmp_path / "donor" / donor_cells[0].storage_key / RESULT_FILENAME
    ).read_text(encoding="utf-8")
    (victim_root / victim_cells[0].storage_key / RESULT_FILENAME).write_text(
        donor_result, encoding="utf-8"
    )

    donor_raw = json.loads(donor_result)
    forged_payload_digest = donor_raw["payload_digest"]

    rec_path = victim_root / RECEIPT_FILENAME
    receipt = json.loads(rec_path.read_text(encoding="utf-8"))
    receipt["cells"][0]["payload_digest"] = forged_payload_digest

    forged_rows = [
        {"cell_id": "a", "payload": json.loads(donor_raw["payload_json"])}
    ]
    forged_aggregate = {"count": 1, "rows": forged_rows}
    (victim_root / AGGREGATE_FILENAME).write_text(
        json.dumps(forged_aggregate), encoding="utf-8"
    )
    receipt["aggregate_digest"] = canonical_digest(forged_aggregate)
    rec_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError):
        verify_completion(
            victim_root, plan=ListPlan(victim_cells), merger=SortedMerger()
        )


# ---------------------------------------------------------------------------
# Completion is judged against the CURRENT generation
# ---------------------------------------------------------------------------


def test_verify_completion_rejects_a_receipt_from_a_superseded_generation(
    tmp_path: Path,
) -> None:
    """Routing's probe: bump the generation, leave the old completion standing.

    This is also the crash window between a force's generation bump and its
    clear_completion: the old receipt survives, but it names a generation that
    is no longer current, and must read as superseded rather than complete.
    """
    runner, _ = build_runner(three_cells(), tmp_path)
    runner.run()
    _verify(tmp_path)  # sanity: valid before the bump

    FileSystemResultStore(tmp_path).bump_generation()

    with pytest.raises(ValueError, match="superseded|current generation"):
        _verify(tmp_path)


def test_verify_completion_fails_closed_when_the_generation_record_is_missing(
    tmp_path: Path,
) -> None:
    runner, _ = build_runner(three_cells(), tmp_path)
    runner.run()

    (tmp_path / GENERATION_FILENAME).unlink()

    with pytest.raises(ValueError, match="missing"):
        _verify(tmp_path)


# ---------------------------------------------------------------------------
# A missing generation record over existing state is corruption, not a fresh start
# ---------------------------------------------------------------------------


def test_missing_generation_over_existing_results_refuses_to_run(
    tmp_path: Path,
) -> None:
    """Routing's probe: deleting generation.json must not quietly resume.

    The refusal must also happen BEFORE anything is mutated: the receipt and
    aggregate are evidence of what the store held, and a corruption check that
    destroys evidence first is a corruption check that runs second.
    """
    runner, _ = build_runner(three_cells(), tmp_path)
    runner.run()

    (tmp_path / GENERATION_FILENAME).unlink()

    store = FileSystemResultStore(tmp_path)
    with pytest.raises(ValueError, match="refusing to recreate"):
        store.ensure_generation()

    runner2, executor2 = build_runner(three_cells(), tmp_path)
    with pytest.raises(ValueError, match="refusing to recreate"):
        runner2.run()

    assert executor2.calls == [], "no work may run on a corrupt store"
    assert (tmp_path / RECEIPT_FILENAME).exists(), "evidence must be preserved"
    assert (tmp_path / AGGREGATE_FILENAME).exists(), "evidence must be preserved"


def test_missing_generation_with_only_completion_state_refuses_to_run(
    tmp_path: Path,
) -> None:
    """Receipt or aggregate alone is prior state too, even with no cell results."""
    runner, _ = build_runner(three_cells(), tmp_path)
    runner.run()

    (tmp_path / GENERATION_FILENAME).unlink()
    for cell in three_cells():
        (tmp_path / cell.storage_key / RESULT_FILENAME).unlink()

    with pytest.raises(ValueError, match="refusing to recreate"):
        FileSystemResultStore(tmp_path).ensure_generation()


def test_force_cannot_reuse_a_generation_number_after_record_loss(
    tmp_path: Path,
) -> None:
    """Routing's probe: completed force at gen 1, delete the record, force again.

    bump_generation used to recreate 0 and return 1, so gen-1 survivors of a
    crashed discard would remain valid under the 'new' generation. It must
    refuse instead.
    """
    runner, _ = build_runner(three_cells(), tmp_path)
    runner.run()
    runner2, _ = build_runner(three_cells(), tmp_path)
    runner2.run(force=True)
    assert FileSystemResultStore(tmp_path).generation() == 1

    (tmp_path / GENERATION_FILENAME).unlink()

    with pytest.raises(ValueError, match="refusing to recreate"):
        FileSystemResultStore(tmp_path).bump_generation()

    runner3, executor3 = build_runner(three_cells(), tmp_path)
    with pytest.raises(ValueError, match="refusing to recreate"):
        runner3.run(force=True)
    assert executor3.calls == []


def test_fresh_root_still_materializes_generation_zero(tmp_path: Path) -> None:
    """The corruption check must not break genuinely fresh starts."""
    store = FileSystemResultStore(tmp_path / "fresh")
    assert store.ensure_generation() == 0
    assert store.generation() == 0


# ---------------------------------------------------------------------------
# The receipt's own claims are validated, not just its presence
# ---------------------------------------------------------------------------


def test_verify_completion_rejects_a_receipt_marked_incomplete(
    tmp_path: Path,
) -> None:
    """complete=false is a receipt saying 'do not trust me'. Believe it."""
    runner, _ = build_runner(three_cells(), tmp_path)
    runner.run()

    rec_path = tmp_path / RECEIPT_FILENAME
    receipt = json.loads(rec_path.read_text(encoding="utf-8"))
    receipt["complete"] = False
    rec_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete"):
        _verify(tmp_path)


@pytest.mark.parametrize("field", ["complete", "generation", "plan_digest", "cells"])
def test_receipt_missing_an_integrity_field_is_refused(
    tmp_path: Path, field: str
) -> None:
    """A truncated receipt must not validate via defaults."""
    runner, _ = build_runner(three_cells(), tmp_path)
    runner.run()

    rec_path = tmp_path / RECEIPT_FILENAME
    receipt = json.loads(rec_path.read_text(encoding="utf-8"))
    del receipt[field]
    rec_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError):
        _verify(tmp_path)


def test_receipt_with_unknown_fields_is_refused(tmp_path: Path) -> None:
    runner, _ = build_runner(three_cells(), tmp_path)
    runner.run()

    rec_path = tmp_path / RECEIPT_FILENAME
    receipt = json.loads(rec_path.read_text(encoding="utf-8"))
    receipt["rider"] = "unvalidated"
    rec_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError):
        _verify(tmp_path)


def test_cell_ref_with_unknown_fields_is_refused(tmp_path: Path) -> None:
    runner, _ = build_runner(three_cells(), tmp_path)
    runner.run()

    rec_path = tmp_path / RECEIPT_FILENAME
    receipt = json.loads(rec_path.read_text(encoding="utf-8"))
    receipt["cells"][0]["rider"] = "unvalidated"
    rec_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError):
        _verify(tmp_path)


# ---------------------------------------------------------------------------
# Stale accounting survives the stricter store
# ---------------------------------------------------------------------------


def test_unservable_result_is_discarded_and_reported_stale(tmp_path: Path) -> None:
    """A record that exists but cannot be served is reported, not silently eaten."""
    cells = [CellSpec(cell_id="a", params={"value": 1})]
    runner, _ = build_runner(cells, tmp_path)
    runner.run()

    path = tmp_path / cells[0].storage_key / RESULT_FILENAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["input_digest"] = "0" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")

    runner2, executor2 = build_runner(cells, tmp_path)
    outcome = runner2.run()

    assert executor2.calls == ["a"]
    assert outcome.stale == ["a"]
    assert outcome.resumed == []


# ---------------------------------------------------------------------------
# force is crash-safe across the discard loop
# ---------------------------------------------------------------------------


DISCARD_KILL_SCRIPT = textwrap.dedent(
    """
    import os, signal, sys
    from pathlib import Path
    import parapet_data.sweep as sweep
    from parapet_data.sweep import CellSpec, new_sweep_runner

    root = Path(sys.argv[1])

    class Plan:
        def cells(self):
            return [CellSpec(cell_id=c, params={"value": 1}) for c in ("a", "b", "c")]

    class Executor:
        def execute(self, cell):
            return {"value": cell.params.get("value"), "seen": cell.cell_id}

    class Merger:
        def merge(self, results):
            return {"count": len(results)}

    class Clock:
        def now_iso(self):
            return "2026-01-01T00:00:00+00:00"

    # Die partway through the discard loop, after the generation bump.
    store = sweep.FileSystemResultStore(root)
    original = sweep.FileSystemResultStore.discard
    state = {"n": 0}

    def dying_discard(self, cell):
        state["n"] += 1
        if state["n"] == 2:
            os.kill(os.getpid(), signal.SIGKILL)
        return original(self, cell)

    sweep.FileSystemResultStore.discard = dying_discard

    new_sweep_runner(
        plan=Plan(), executor=Executor(), merger=Merger(),
        output_root=root, clock=Clock(),
    ).run(force=True)
    """
)


def test_force_refresh_killed_mid_discard_invalidates_every_prior_result(
    tmp_path: Path,
) -> None:
    """The generation bump makes a forced refresh atomic in effect.

    A crash inside the discard loop leaves some results deleted and some intact.
    Those survivors are pre-refresh values, and an ordinary run afterwards must
    not serve them, or the operator's request to recompute everything is
    silently honoured for only part of the sweep.
    """
    runner, _ = build_runner(three_cells(value=1), tmp_path)
    runner.run()

    script = tmp_path / "discard_kill.py"
    script.write_text(DISCARD_KILL_SCRIPT, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(PACKAGE_ROOT)}
    proc = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        capture_output=True,
        cwd=PACKAGE_ROOT,
        env=env,
    )
    assert proc.returncode == -9, (
        f"expected SIGKILL, got {proc.returncode}: {proc.stderr.decode()}"
    )

    (tmp_path / LOCK_FILENAME).unlink()

    healthy = RecordingExecutor()
    runner2, _ = build_runner(three_cells(value=1), tmp_path, executor=healthy)
    outcome = runner2.run()

    assert outcome.resumed == [], "no pre-force result may survive the refresh"
    assert healthy.calls == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Completion marker ordering
# ---------------------------------------------------------------------------


def test_receipt_never_outlives_the_aggregate_it_describes(tmp_path: Path) -> None:
    """Clearing must drop the receipt first.

    If the aggregate goes first, an interruption between the two unlinks leaves a
    receipt pointing at nothing, which reads as a completed sweep.
    """
    runner, _ = build_runner(three_cells(), tmp_path)
    runner.run()

    store = FileSystemResultStore(tmp_path)
    order: list[str] = []
    real_unlink = Path.unlink

    def watched(self: Path, *args: object, **kwargs: object) -> None:
        order.append(self.name)
        return real_unlink(self, *args, **kwargs)

    Path.unlink = watched  # type: ignore[method-assign]
    try:
        store.clear_completion()
    finally:
        Path.unlink = real_unlink  # type: ignore[method-assign]

    assert order.index(RECEIPT_FILENAME) < order.index(AGGREGATE_FILENAME)


WRITE_KILL_SCRIPT = textwrap.dedent(
    """
    import os, signal, sys
    from pathlib import Path
    import parapet_data.sweep as sweep
    from parapet_data.sweep import CellSpec, new_sweep_runner

    root = Path(sys.argv[1])

    class Plan:
        def cells(self):
            return [CellSpec(cell_id=c, params={"value": 1}) for c in ("a", "b", "c")]

    class Executor:
        def execute(self, cell):
            return {"value": cell.params.get("value"), "seen": cell.cell_id}

    class Merger:
        def merge(self, results):
            return {"count": len(results)}

    class Clock:
        def now_iso(self):
            return "2026-01-01T00:00:00+00:00"

    # Die inside the atomic write itself, between the temp write and the rename.
    real_replace = os.replace
    state = {"n": 0}

    def dying_replace(src, dst):
        state["n"] += 1
        if state["n"] == 2:
            os.kill(os.getpid(), signal.SIGKILL)
        return real_replace(src, dst)

    os.replace = dying_replace

    new_sweep_runner(
        plan=Plan(), executor=Executor(), merger=Merger(),
        output_root=root, clock=Clock(),
    ).run()
    """
)


def test_kill_inside_the_atomic_write_leaves_no_half_written_result(
    tmp_path: Path,
) -> None:
    """Exercise the write crash window, not just a crash in the executor."""
    script = tmp_path / "write_kill.py"
    script.write_text(WRITE_KILL_SCRIPT, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(PACKAGE_ROOT)}
    proc = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        capture_output=True,
        cwd=PACKAGE_ROOT,
        env=env,
    )
    assert proc.returncode == -9, (
        f"expected SIGKILL, got {proc.returncode}: {proc.stderr.decode()}"
    )

    (tmp_path / LOCK_FILENAME).unlink()

    # Whatever landed must be loadable or absent, never partial.
    store = FileSystemResultStore(tmp_path)
    gen = store.generation()
    for cell in three_cells(value=1):
        store.load(cell, gen)  # must not raise

    healthy = RecordingExecutor()
    runner, _ = build_runner(three_cells(value=1), tmp_path, executor=healthy)
    outcome = runner.run()
    verify_completion(tmp_path, plan=ListPlan(three_cells()), merger=SortedMerger())
    assert outcome.aggregate["count"] == 3


# ---------------------------------------------------------------------------
# The validator must rederive, not self-compare
# ---------------------------------------------------------------------------


def _verify(tmp_path: Path, cells: Sequence[CellSpec] | None = None) -> object:
    return verify_completion(
        tmp_path,
        plan=ListPlan(list(cells) if cells is not None else three_cells()),
        merger=SortedMerger(),
    )


def test_validator_rejects_a_forged_aggregate_with_a_consistent_receipt(
    tmp_path: Path,
) -> None:
    """The circularity test.

    Hashing the stored aggregate and comparing it to the digest stored beside it
    proves nothing: a producer that edits both stays internally consistent. Only
    re-merging the stored cell results can catch it.
    """
    runner, _ = build_runner(three_cells(value=1), tmp_path)
    runner.run()

    agg_path = tmp_path / AGGREGATE_FILENAME
    forged = json.loads(agg_path.read_text(encoding="utf-8"))
    forged["count"] = 999
    agg_path.write_text(json.dumps(forged), encoding="utf-8")

    # Update the receipt exactly as an honest producer would, leaving the cell
    # results untouched.
    rec_path = tmp_path / RECEIPT_FILENAME
    receipt = json.loads(rec_path.read_text(encoding="utf-8"))
    receipt["aggregate_digest"] = canonical_digest(forged)
    rec_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="re-merged"):
        _verify(tmp_path)


def test_validator_rejects_a_tampered_plan_digest(tmp_path: Path) -> None:
    runner, _ = build_runner(three_cells(), tmp_path)
    runner.run()

    rec_path = tmp_path / RECEIPT_FILENAME
    receipt = json.loads(rec_path.read_text(encoding="utf-8"))
    receipt["plan_digest"] = "0" * 64
    rec_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="plan digest mismatch"):
        _verify(tmp_path)


def test_validator_rejects_a_sweep_run_under_a_different_plan(
    tmp_path: Path,
) -> None:
    """Verifying against the wrong expected plan must fail, not pass quietly."""
    runner, _ = build_runner(three_cells(value=1), tmp_path)
    runner.run()

    with pytest.raises(ValueError, match="plan digest mismatch"):
        _verify(tmp_path, cells=three_cells(value=2))


def test_validator_returns_the_rebuilt_aggregate(tmp_path: Path) -> None:
    runner, _ = build_runner(three_cells(), tmp_path)
    outcome = runner.run()
    assert _verify(tmp_path) == outcome.aggregate


# ---------------------------------------------------------------------------
# Result digests are mandatory, not optional
# ---------------------------------------------------------------------------


def test_result_without_a_payload_digest_is_refused(tmp_path: Path) -> None:
    """Deleting the digest must not turn tamper-detection off.

    A record with no digest is not a legacy record. It is a record nothing
    vouches for, and this module has never written one.
    """
    cells = [CellSpec(cell_id="a", params={"value": 1})]
    runner, _ = build_runner(cells, tmp_path)
    runner.run()

    path = tmp_path / cells[0].storage_key / RESULT_FILENAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("payload_digest")
    raw["payload_json"] = json.dumps({"value": 777, "seen": "a"}, sort_keys=True)
    path.write_text(json.dumps(raw), encoding="utf-8")

    runner2, executor2 = build_runner(cells, tmp_path)
    outcome = runner2.run()

    assert executor2.calls == ["a"]
    assert outcome.aggregate["rows"][0]["payload"]["value"] == 1


def test_result_that_is_not_a_json_object_is_treated_as_absent(
    tmp_path: Path,
) -> None:
    """Documented behavior is 'treated as absent', so it must not raise."""
    cells = three_cells()
    runner, _ = build_runner(cells, tmp_path)
    runner.run()

    (tmp_path / cells[1].storage_key / RESULT_FILENAME).write_text(
        "[]", encoding="utf-8"
    )

    runner2, executor2 = build_runner(three_cells(), tmp_path)
    outcome = runner2.run()
    assert executor2.calls == ["b"]
    assert outcome.executed == ["b"]


# ---------------------------------------------------------------------------
# Generation fails closed
# ---------------------------------------------------------------------------


def test_corrupt_generation_fails_closed_rather_than_reviving_generation_zero(
    tmp_path: Path,
) -> None:
    """Failing open to 0 would resurrect exactly what a force refresh discarded."""
    runner, _ = build_runner(three_cells(), tmp_path)
    runner.run()

    store = FileSystemResultStore(tmp_path)
    store.bump_generation()
    (tmp_path / GENERATION_FILENAME).write_text("{ not json", encoding="utf-8")

    with pytest.raises(ValueError, match="unreadable"):
        store.generation()

    runner2, _ = build_runner(three_cells(), tmp_path)
    with pytest.raises(ValueError):
        runner2.run()


@pytest.mark.parametrize(
    "content",
    ['{"generation": "1"}', '{"generation": -1}', '{"generation": true}', "{}", "[]"],
)
def test_malformed_generation_values_are_refused(tmp_path: Path, content: str) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / GENERATION_FILENAME).write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        FileSystemResultStore(tmp_path).generation()


def test_generation_is_materialized_so_missing_and_initial_are_distinct(
    tmp_path: Path,
) -> None:
    store = FileSystemResultStore(tmp_path)
    with pytest.raises(ValueError, match="missing"):
        store.generation()

    runner, _ = build_runner(three_cells(), tmp_path)
    runner.run()
    assert (tmp_path / GENERATION_FILENAME).exists()
    assert store.generation() == 0


# ---------------------------------------------------------------------------
# Falsy non-objects are not silently the empty object
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("falsy", [[], 0, False, ""])
def test_falsy_non_object_params_are_refused_not_coerced(falsy: object) -> None:
    """``params or {}`` would give four distinct inputs one shared digest."""
    with pytest.raises(ValueError, match="must be a JSON object"):
        CellSpec(cell_id="a", params=falsy)


def test_absent_params_default_to_the_empty_object() -> None:
    assert CellSpec(cell_id="a").params == {}
    assert CellSpec(cell_id="a", params=None).params == {}


@pytest.mark.parametrize("falsy", [[], 0, False, ""])
def test_falsy_non_object_payloads_are_refused(tmp_path: Path, falsy: object) -> None:
    class FalsyExecutor:
        def execute(self, cell: CellSpec) -> Mapping[str, object]:
            return falsy  # type: ignore[return-value]

    runner = new_sweep_runner(
        plan=ListPlan([CellSpec(cell_id="a", params={"v": 1})]),
        executor=FalsyExecutor(),
        merger=SortedMerger(),
        output_root=tmp_path,
        clock=FixedClock(),
        lock=NullRunLock(),
    )
    with pytest.raises((ValueError, TypeError)):
        runner.run()
