from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError

import parapet_data.phase_a_cells as phase_a_cells_module
from parapet_data.phase_a_cells import (
    GRAM_SET_BOUND,
    RUNTIME_REV,
    CellPayload,
    ChunkRecord,
    DuplicateProjectedIdentity,
    PhaseACellExecutor,
    PhaseAResultMerger,
    enforce_gram_bound,
    require_binary64,
    winner_compare,
)
from parapet_data.sweep import CellResult, CellSpec, FileSystemResultStore, canonical_digest

RQ_PATH = Path("/_redacted_/dev/DefenseSector/parapet-l1-coverage-ru-paraphrase/implement/zh_specialist_bootstrap/production_corpus/source_qualification/run_qualification.py")
spec = importlib.util.spec_from_file_location("phase_a_real_rq", RQ_PATH)
assert spec and spec.loader
rq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rq)
normalized_5grams = rq.normalized_5grams


def ref(row_id: str, text: str, file: str | None = "same.jsonl", **extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "text": text, "file": file, "row_id": row_id, "source": "fixture",
        "role": "user", "earmark_source": "fixture", "metadata": {},
    }
    if file is None:
        del row["file"]
    row.update(extra)
    return row


CANDIDATES = [
    {"row_id": "c0", "text": "abcdefghijklmn", "source": "fixture"},
    {"row_id": "c1", "text": "mnopqrstuvwx", "source": "fixture"},
]


def params(records: list[dict[str, object]], label: str = "R1", contributes: bool = True) -> dict[str, object]:
    manifest = [
        {"path": "phase_a_cells.py", "sha256": "1" * 64},
        {"path": "run_qualification.py", "sha256": "2" * 64},
    ]
    return {
        "tier": "r1", "label": label, "contributes_c1": contributes,
        "chunk": [{"name": "fixture", "row_count": len(records), "content_digest": canonical_digest(records),
                   "parent_file_digest": None, "start_row": None, "end_row": None,
                   "ordered_record_digest": None}],
        "candidate_universe_digest": canonical_digest(CANDIDATES),
        "code_rev": canonical_digest(manifest), "code_rev_manifest": manifest,
        "plan_rev": "fixture-plan", "runtime_rev": RUNTIME_REV,
    }


def cell(cell_id: str, records: list[dict[str, object]], *, label: str = "R1", contributes: bool = True) -> CellSpec:
    return CellSpec(cell_id=cell_id, params=params(records, label, contributes))


def result(c: CellSpec, payload: dict[str, object]) -> CellResult:
    return CellResult(cell_id=c.cell_id, input_digest=c.input_digest, generation=0,
                      payload=payload, completed_at="2026-08-22T00:00:00Z")


def execute(reference_cells: dict[str, list[dict[str, object]]], specs: list[CellSpec]) -> list[CellResult]:
    executor = PhaseACellExecutor(CANDIDATES, reference_cells, normalized_5grams)
    return [result(c, dict(executor.execute(c))) for c in specs]


def merger() -> PhaseAResultMerger:
    return PhaseAResultMerger(CANDIDATES)


def direct_legacy(refs: list[dict[str, object]], label: str = "R1") -> dict[str, object]:
    """Independent fixture fold matching run_phase_a_benign.py:95-111."""
    from collections import Counter, defaultdict
    require_binary64()
    gs = [normalized_5grams(str(c["text"])) for c in CANDIDATES]
    inv: dict[str, list[int]] = defaultdict(list)
    for ci, grams in enumerate(gs):
        for gram in grams:
            inv[gram].append(ci)
    c1: dict[tuple[str, str], dict[str, object]] = {}
    best: dict[int, tuple[int, int, dict[str, object]]] = {}
    for r in refs:
        rg = normalized_5grams(str(r["text"])); counts: Counter[int] = Counter()
        for gram in rg: counts.update(inv.get(gram, ()))
        for ci, inter in counts.items():
            union = len(gs[ci]) + len(rg) - inter; minimum = min(len(gs[ci]), len(rg))
            loc = str(r["row_id"]); con = inter / minimum; jac = inter / union
            key = (str(CANDIDATES[ci]["row_id"]), str(r["file"])); old = c1.get(key)
            if con >= .3 and (old is None or con > int(old["containment_num"]) / int(old["containment_den"]) or
              (con == int(old["containment_num"]) / int(old["containment_den"]) and (str(r["file"]), loc) < (str(old["reference_file"]), str(old["locator"])))):
                c1[key] = {"candidate_id": key[0], "reference_file": key[1], "locator": loc,
                           "containment_num": inter, "containment_den": minimum,
                           "jaccard_num": inter, "jaccard_den": union, "reference": r}
            oldj = best.get(ci)
            if oldj is None or jac > oldj[0] / oldj[1] or (jac == oldj[0] / oldj[1] and
              (str(r.get("file", "")), loc) < (str(oldj[2].get("file", "")), str(oldj[2]["row_id"]))):
                best[ci] = (inter, union, r)
    return {"c1_partial": [c1[k] for k in sorted(c1)],
            "bestj_partial": {label: [{"candidate_index": i, "candidate_id": str(CANDIDATES[i]["row_id"]),
                                        "jaccard_num": v[0], "jaccard_den": v[1], "reference": v[2]}
                                       for i, v in sorted(best.items())]}}


def first_difference(left: object, right: object, path: str = "$") -> str | None:
    if type(left) is not type(right): return path
    if isinstance(left, dict):
        if set(left) != set(right): return path
        for key in sorted(left):
            found = first_difference(left[key], right[key], f"{path}.{key}")
            if found: return found
    elif isinstance(left, list):
        if len(left) != len(right): return path
        for i, (a, b) in enumerate(zip(left, right)):
            found = first_difference(a, b, f"{path}[{i}]")
            if found: return found
    elif left != right: return path
    return None


def test_gate0_real_normalizer_legacy_parity_and_partitions_a_b_c() -> None:
    # A/C/E plus cross-cell merge: the equal-containment winner is locator a,
    # even though its same-pair Jaccard is lower than locator z.
    refs = [ref("z", "abcdefghijklmnZZZZ"), ref("a", "abcdefghijklmnYYYYYYYY")]
    cells = {"left": [refs[0]], "right": [refs[1]]}
    specs = [cell("left", cells["left"]), cell("right", cells["right"])]
    merged = dict(merger().merge(execute(cells, specs)))
    legacy = direct_legacy(refs)
    assert first_difference(merged, legacy) is None, f"first differing key: {first_difference(merged, legacy)}"
    reversed_merge = dict(merger().merge(list(reversed(execute(cells, specs)))))
    one_cell = dict(merger().merge(execute({"all": refs}, [cell("all", refs)])))
    assert canonical_digest(merged) == canonical_digest(reversed_merge) == canonical_digest(one_cell)
    assert merged["c1_partial"][0]["locator"] == "a"


def test_cross_tier_c1_and_xc_shared_label_topologies() -> None:
    # B: same global c1 key across r1/r2e. F/G: shared XC label, file missing
    # ordering, and all candidates indexed while both corpora act as references.
    cross = {"r1": [ref("late", "abcdefghij")], "r2": [ref("early", "abcdefghijklmn")]}
    merged = merger().merge(execute(cross, [cell("r1", cross["r1"]), cell("r2", cross["r2"], label="R2")]))
    assert merged["c1_partial"][0]["locator"] == "early"
    xc = {"olcc": [ref("ol", "mnopqrstuv", None)], "coig": [ref("co", "mnopqrstuvwx", "coig.jsonl")]}
    merged_xc = merger().merge(execute(xc, [cell("olcc", xc["olcc"], label="XC", contributes=False), cell("coig", xc["coig"], label="XC", contributes=False)]))
    assert set(merged_xc["bestj_partial"]) == {"XC"}
    assert {r["candidate_id"] for r in merged_xc["bestj_partial"]["XC"]} == {"c1"}
    assert merged_xc["c1_partial"] == []


def test_admission_exact_point_and_below_gate_non_perturbation() -> None:
    assert 3 / 10 == .3 and 3 / 10 >= .3 and not (3 / 10 > .3)
    # Comparator/admission sufficient-statistic oracle for case E.
    admitted = [x for x in [(3, 10), (2, 10)] if x[0] / x[1] >= .3]
    mutated = [x for x in [(3, 10), (2, 10)] if x[0] / x[1] > .3]
    assert admitted == [(3, 10)] and mutated == []
    assert max([(4, 10)] + admitted, key=lambda x: x[0] / x[1]) == (4, 10)


def test_admission_exact_point_is_retained_through_real_fold() -> None:
    # c0 and this reference each have ten real normalized grams, with exactly
    # abcde/bcdef/cdefg in common.  This record is the sole c1 candidate for
    # its projected key, so changing the production >= .3 gate to > .3 drops it.
    exact = ref("exact-three-of-ten", "abcdefgQRSTUVW", "exact.jsonl")
    assert len(normalized_5grams(str(CANDIDATES[0]["text"]))) == 10
    assert len(normalized_5grams(str(exact["text"]))) == 10
    assert len(normalized_5grams(str(CANDIDATES[0]["text"])) & normalized_5grams(str(exact["text"]))) == 3
    merged = merger().merge(execute({"exact": [exact]}, [cell("exact", [exact])]))
    exact_key_rows = [row for row in merged["c1_partial"]
                      if (row["candidate_id"], row["reference_file"]) == ("c0", "exact.jsonl")]
    assert [(row["locator"], row["containment_num"], row["containment_den"])
            for row in exact_key_rows] == [("exact-three-of-ten", 3, 10)]


@pytest.mark.parametrize("namespace,left,right", [
    ("c1", ref("same", "abcdefghij"), ref("same", "abcdefghik")),
    ("bestj", ref("same", "abcdefghij"), ref("same", "abcdefghik")),
    ("bestj", ref("same", "abcdefghij", None), ref("same", "abcdefghik", None)),
])
def test_duplicate_projected_identity_fails_closed_with_both_records(namespace: str, left: dict[str, object], right: dict[str, object]) -> None:
    with pytest.raises(DuplicateProjectedIdentity) as caught:
        winner_compare(1, 2, left, 1, 2, right, namespace)  # type: ignore[arg-type]
    assert "abcdefghij" in str(caught.value) and "abcdefghik" in str(caught.value)


def test_identical_digest_duplicate_keeps_first() -> None:
    record = ref("same", "abcdefghij")
    assert winner_compare(1, 2, record, 1, 2, dict(record), "c1") == 0


def test_gate2_mutations_fire_and_shared_comparator_is_single(monkeypatch: pytest.MonkeyPatch) -> None:
    low, high = ref("a", "x"), ref("z", "x")
    assert winner_compare(1, 2, low, 1, 2, high, "c1") > 0
    assert PhaseACellExecutor.comparator is PhaseAResultMerger.comparator is winner_compare
    rows = {"low": [ref("a", "abcdefghijklmn")], "high": [ref("z", "abcdefghijklmn")]}
    specs = [cell("low", rows["low"]), cell("high", rows["high"])]
    partials = execute(rows, specs)
    baseline = merger().merge(partials)

    def flipped(*args: object) -> int:
        verdict = winner_compare(*args)  # type: ignore[arg-type]
        new_ref, old_ref = args[2], args[5]
        if args[0] / args[1] == args[3] / args[4] and new_ref != old_ref:  # type: ignore[operator]
            return -verdict
        return verdict

    monkeypatch.setattr(PhaseAResultMerger, "comparator", staticmethod(flipped))
    assert canonical_digest(merger().merge(partials)) != canonical_digest(baseline)
    # Fork mutation must fail the object-identity gate.
    fork = lambda *args: winner_compare(*args)
    assert fork is not PhaseACellExecutor.comparator


def test_gate3_strict_wire_and_store_round_trip(tmp_path: Path) -> None:
    rows = [ref("r", "abcdefghijklmn")]
    c = cell("one", rows); payload = dict(PhaseACellExecutor(CANDIDATES, {"one": rows}, normalized_5grams).execute(c))
    CellPayload.model_validate(payload)
    with pytest.raises(ValidationError):
        CellPayload.model_validate({**payload, "unknown": 1})
    original = result(c, payload)
    store = FileSystemResultStore(tmp_path); store.ensure_generation(); store.save(c, original)
    loaded = store.load(c, 0)
    assert loaded is not None and loaded.payload_json == original.payload_json
    assert canonical_digest(merger().merge([loaded])) == canonical_digest(payload)


def test_gate4_runtime_contract_and_bound_branch_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    assert require_binary64() == RUNTIME_REV["float_contract"]
    enforce_gram_bound(GRAM_SET_BOUND, where="synthetic set")
    with pytest.raises(ValueError, match="exceeds B"):
        enforce_gram_bound(GRAM_SET_BOUND + 1, where="synthetic set")

    monkeypatch.setattr(phase_a_cells_module, "GRAM_SET_BOUND", 9)
    with pytest.raises(ValueError, match="candidate 0 gram-set cardinality 10 exceeds B=9"):
        PhaseACellExecutor(CANDIDATES, {}, normalized_5grams)

    monkeypatch.setattr(phase_a_cells_module, "GRAM_SET_BOUND", GRAM_SET_BOUND)
    reference = ref("over-bound", "abcdefghijklmn", "bound.jsonl")
    executor = PhaseACellExecutor(CANDIDATES, {"bound": [reference]}, normalized_5grams)
    bound_cell = cell("bound", [reference])
    monkeypatch.setattr(phase_a_cells_module, "GRAM_SET_BOUND", 9)
    with pytest.raises(ValueError, match="reference over-bound gram-set cardinality 10 exceeds B=9"):
        executor.execute(bound_cell)


def test_wire_arrays_are_sorted() -> None:
    refs = {"one": [ref("z", "abcdefghijklmn", "z"), ref("a", "mnopqrstuvwx", "a")]}
    payload = PhaseACellExecutor(CANDIDATES, refs, normalized_5grams).execute(cell("one", refs["one"]))
    assert payload["c1_partial"] == sorted(payload["c1_partial"], key=lambda x: (x["candidate_id"], x["reference_file"]))
    assert payload["bestj_partial"]["R1"] == sorted(payload["bestj_partial"]["R1"], key=lambda x: x["candidate_index"])


def test_wire_redundant_keys_are_bound_to_embedded_reference() -> None:
    rows = [ref("r", "abcdefghijklmn")]
    payload = dict(PhaseACellExecutor(CANDIDATES, {"one": rows}, normalized_5grams).execute(cell("one", rows)))
    c1 = dict(payload["c1_partial"][0])
    for field, value in (("reference_file", "wrong"), ("locator", "wrong")):
        with pytest.raises(ValidationError, match=field):
            CellPayload.model_validate({**payload, "c1_partial": [{**c1, field: value}]})


def test_wire_rejects_duplicate_and_unsorted_arrays() -> None:
    rows = [ref("a", "abcdefghijklmn", "a"), ref("z", "mnopqrstuvwx", "z")]
    payload = dict(PhaseACellExecutor(CANDIDATES, {"one": rows}, normalized_5grams).execute(cell("one", rows)))
    c1 = payload["c1_partial"]
    bestj = payload["bestj_partial"]["R1"]
    assert len(c1) == len(bestj) == 2
    with pytest.raises(ValidationError, match="sorted"):
        CellPayload.model_validate({**payload, "c1_partial": list(reversed(c1))})
    with pytest.raises(ValidationError, match="duplicate c1"):
        CellPayload.model_validate({**payload, "c1_partial": [c1[0], c1[0]]})
    with pytest.raises(ValidationError, match="sorted"):
        CellPayload.model_validate({**payload, "bestj_partial": {"R1": list(reversed(bestj))}})
    with pytest.raises(ValidationError, match="duplicate bestj"):
        CellPayload.model_validate({**payload, "bestj_partial": {"R1": [bestj[0], bestj[0]]}})


def test_adversarial_valid_payloads_merge_permutation_equally() -> None:
    rows = {"a": [ref("a", "abcdefghijklmn")], "b": [ref("b", "abcdefghijklmn")]}
    specs = [cell(name, records) for name, records in rows.items()]
    partials = execute(rows, specs)
    forward = merger().merge(partials)
    backward = merger().merge(list(reversed(partials)))
    assert canonical_digest(forward) == canonical_digest(backward)


def test_executor_detects_namespace_duplicates_even_when_not_competing() -> None:
    records = [ref("same", "abcdefghijklmn"), ref("same", "mnopqrstuvwx")]
    with pytest.raises(DuplicateProjectedIdentity, match="duplicate projected bestj"):
        execute({"one": records}, [cell("one", records, contributes=False)])
    with pytest.raises(DuplicateProjectedIdentity, match="duplicate projected c1"):
        execute({"one": records}, [cell("one", records, contributes=True)])


def test_merger_detects_cross_cell_duplicate_emitted_references() -> None:
    left_ref, right_ref = ref("same", "abcdefghijklmn"), ref("same", "mnopqrstuvwx")
    left = {"c1_partial": [{"candidate_id": "c0", "reference_file": "same.jsonl", "locator": "same",
                            "containment_num": 1, "containment_den": 2, "jaccard_num": 1, "jaccard_den": 2,
                            "reference": left_ref}], "bestj_partial": {}}
    right = {"c1_partial": [{"candidate_id": "c1", "reference_file": "same.jsonl", "locator": "same",
                             "containment_num": 1, "containment_den": 2, "jaccard_num": 1, "jaccard_den": 2,
                             "reference": right_ref}], "bestj_partial": {}}
    cells = [cell("left", [left_ref]), cell("right", [right_ref])]
    with pytest.raises(DuplicateProjectedIdentity, match="duplicate projected c1"):
        merger().merge([result(cells[0], left), result(cells[1], right)])


@pytest.mark.parametrize("change,match", [
    ({"start_row": 0}, "all present"),
    ({"start_row": -1, "end_row": 1, "parent_file_digest": "p", "ordered_record_digest": "o"}, "0 <="),
    ({"start_row": 2, "end_row": 2, "parent_file_digest": "p", "ordered_record_digest": "o"}, "0 <="),
    ({"start_row": 0, "end_row": 2, "parent_file_digest": "p", "ordered_record_digest": "o"}, "row_count"),
])
def test_chunk_record_rejects_invalid_row_spans(change: dict[str, object], match: str) -> None:
    base = {"name": "x", "row_count": 1, "content_digest": "d", "parent_file_digest": None,
            "start_row": None, "end_row": None, "ordered_record_digest": None}
    with pytest.raises(ValidationError, match=match):
        ChunkRecord.model_validate({**base, **change})


def test_chunk_record_accepts_coherent_row_span() -> None:
    ChunkRecord.model_validate({"name": "x", "row_count": 2, "content_digest": "d", "parent_file_digest": "p",
                                "start_row": 3, "end_row": 5, "ordered_record_digest": "o"})


@pytest.mark.parametrize("mutation,match", [
    ("candidate", "candidate_universe_digest"),
    ("row_count", "chunk integrity"),
    ("content_digest", "chunk integrity"),
    ("runtime", "runtime_rev"),
])
def test_executor_rejects_integrity_mutations(mutation: str, match: str) -> None:
    rows = [ref("r", "abcdefghijklmn")]
    p = params(rows)
    if mutation == "candidate": p["candidate_universe_digest"] = "wrong"
    elif mutation == "row_count": p["chunk"][0]["row_count"] = 2  # type: ignore[index]
    elif mutation == "content_digest": p["chunk"][0]["content_digest"] = "wrong"  # type: ignore[index]
    else: p["runtime_rev"] = {**RUNTIME_REV, "unicode_version": "wrong"}
    spec = CellSpec(cell_id="one", params=p)
    executor = PhaseACellExecutor(CANDIDATES, {"one": rows}, normalized_5grams)
    with pytest.raises((ValueError, ValidationError), match=match):
        executor.execute(spec)


def test_merger_sorts_two_labels_and_rejects_candidate_mapping_conflict() -> None:
    r = ref("r", "abcdefghijklmn")
    def payload(label: str, candidate_id: str) -> dict[str, object]:
        return {"c1_partial": [], "bestj_partial": {label: [{"candidate_index": 0, "candidate_id": candidate_id,
                "jaccard_num": 1, "jaccard_den": 2, "reference": r}]}}
    a, z = cell("a", [r], label="A"), cell("z", [r], label="Z")
    merged = merger().merge([result(z, payload("Z", "c0")), result(a, payload("A", "c0"))])
    assert list(merged["bestj_partial"]) == ["A", "Z"]
    with pytest.raises(ValueError, match="not pinned"):
        merger().merge([result(a, payload("A", "c0")), result(z, payload("Z", "wrong"))])


def test_merger_rejects_single_payload_false_candidate_mapping() -> None:
    r = ref("r", "abcdefghijklmn")
    c = cell("one", [r])
    payload = {"c1_partial": [], "bestj_partial": {"R1": [{
        "candidate_index": 0, "candidate_id": "WRONG-ID",
        "jaccard_num": 1, "jaccard_den": 2, "reference": r,
    }]}}
    with pytest.raises(ValueError, match="not pinned 'c0'"):
        merger().merge([result(c, payload)])


def test_merger_rejects_consistently_wrong_multi_payload_mapping() -> None:
    r = ref("r", "abcdefghijklmn")
    left, right = cell("left", [r], label="A"), cell("right", [r], label="B")
    def payload(label: str) -> dict[str, object]:
        return {"c1_partial": [], "bestj_partial": {label: [{
            "candidate_index": 0, "candidate_id": "WRONG-ID",
            "jaccard_num": 1, "jaccard_den": 2, "reference": r,
        }]}}
    with pytest.raises(ValueError, match="not pinned 'c0'"):
        merger().merge([result(left, payload("A")), result(right, payload("B"))])


def test_merger_rejects_out_of_range_candidate_index() -> None:
    r = ref("r", "abcdefghijklmn")
    c = cell("one", [r])
    payload = {"c1_partial": [], "bestj_partial": {"R1": [{
        "candidate_index": len(CANDIDATES), "candidate_id": "c2",
        "jaccard_num": 1, "jaccard_den": 2, "reference": r,
    }]}}
    with pytest.raises(ValueError, match="out of range"):
        merger().merge([result(c, payload)])


def test_merger_rejects_unknown_c1_candidate_id() -> None:
    r = ref("r", "abcdefghijklmn")
    c = cell("one", [r])
    payload = {"c1_partial": [{
        "candidate_id": "WRONG-ID", "reference_file": "same.jsonl", "locator": "r",
        "containment_num": 1, "containment_den": 2,
        "jaccard_num": 1, "jaccard_den": 2, "reference": r,
    }], "bestj_partial": {}}
    with pytest.raises(ValueError, match="not present in the pinned candidate universe"):
        merger().merge([result(c, payload)])


@pytest.mark.parametrize("namespace,missing", [("bestj", "row_id"), ("c1", "row_id"), ("c1", "file")])
def test_malformed_embedded_reference_is_rejected(namespace: str, missing: str) -> None:
    reference = ref("r", "abcdefghijklmn")
    del reference[missing]
    if namespace == "c1":
        payload = {"c1_partial": [{"candidate_id": "c0", "reference_file": "same.jsonl", "locator": "r",
                   "containment_num": 1, "containment_den": 2, "jaccard_num": 1, "jaccard_den": 2,
                   "reference": reference}], "bestj_partial": {}}
    else:
        payload = {"c1_partial": [], "bestj_partial": {"R1": [{"candidate_index": 0, "candidate_id": "c0",
                   "jaccard_num": 1, "jaccard_den": 2, "reference": reference}]}}
    with pytest.raises(ValidationError, match=missing):
        CellPayload.model_validate(payload)
