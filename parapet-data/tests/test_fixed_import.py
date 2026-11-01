"""Behavior checks for exact pinned imports; all data is synthetic."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

from parapet_data.fixed_import import FixedImportSpec, compose_fixed


def pin(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def setup(tmp_path):
    rows = [{"id": str(i), "text": f"  synthetic {i}\n", "label": "benign" if i % 2 else "malicious", "family": str(i // 3), "metadata": {"i": i}} for i in range(60)]
    source = tmp_path / "rows.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    selection = tmp_path / "arms.json"
    selection.write_text(json.dumps({"arm": {"rows": [r["id"] for r in rows]}}))
    spec = dict(mode="fixed_import", name="synthetic", inputs=[dict(name="rows", path=str(source), sha256=pin(source), text_column="text", id_column="id", label_column="label", group_column="family")], selection=dict(path=str(selection), sha256=pin(selection), arm="arm"))
    return spec, rows


@pytest.mark.parametrize("failure", ["hash", "selection_hash", "missing", "duplicate_selection", "duplicate_within", "duplicate_across", "label", "counts"])
def test_rejections(setup, tmp_path, failure):
    spec, rows = setup
    src = spec["inputs"][0]
    if failure == "hash":
        src["sha256"] = "0" * 64
    elif failure == "selection_hash":
        spec["selection"]["sha256"] = "0" * 64
    elif failure in ("missing", "duplicate_selection"):
        p = Path(spec["selection"]["path"])
        a = json.loads(p.read_text())
        a["arm"]["rows"].append("absent" if failure == "missing" else "0")
        p.write_text(json.dumps(a)); spec["selection"]["sha256"] = pin(p)
    elif failure == "duplicate_within":
        p = Path(src["path"])
        with p.open("a") as f:
            f.write(json.dumps(rows[0]) + "\n")
        src["sha256"] = pin(p)
    elif failure == "duplicate_across":
        spec["selection"] = None
        spec["inputs"].append(dict(src, name="other"))
    elif failure == "label":
        src["fixed_label"] = "benign"
    else:
        spec["expected_counts"] = {"benign": 31}
    with pytest.raises(ValueError):
        compose_fixed(FixedImportSpec.model_validate(spec), tmp_path / "out", tmp_path)
    assert not (tmp_path / "out").exists()


def test_lossless_deterministic_grouped_and_runner_roundtrip(setup, tmp_path):
    spec, rows = setup
    spec["selection"] = None
    spec["splits"] = {"train": .6, "val": .2, "holdout": .2}
    spec = FixedImportSpec.model_validate(spec)
    first, second = tmp_path / "a", tmp_path / "b"
    manifest = compose_fixed(spec, first, tmp_path)
    compose_fixed(spec, second, tmp_path)
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {p.name: p.read_bytes() for p in second.iterdir()}
    actual = [json.loads(l) for l in (first / "curated.jsonl").read_text().splitlines()]
    assert actual == [dict(r, content=r["text"]) for r in rows]
    groups = {}
    for split in manifest.splits:
        for r in map(json.loads, (first / f"{split}.jsonl").read_text().splitlines()):
            assert groups.setdefault(r["family"], split) == split
    assert len(set(groups.values())) == 3
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "parapet-runner"))
    from parapet_runner.runner import ManifestSplitResolver
    from parapet_runner.manifest import CurationManifest
    receipt = CurationManifest.model_validate_json((first / "manifest.json").read_text())
    resolved = ManifestSplitResolver(root_dir=first).resolve(receipt)
    assert resolved.train_path == first / "train.jsonl"


def test_all_train_and_own_groups(setup, tmp_path):
    spec, rows = setup
    spec["inputs"][0].pop("group_column")
    manifest = compose_fixed(FixedImportSpec.model_validate(spec), tmp_path / "out", tmp_path)
    assert manifest.splits["train"].sample_count == len(rows)
    assert manifest.splits["val"].sample_count == manifest.splits["holdout"].sample_count == 0


@pytest.mark.parametrize("missing", ["absent", "null"])
def test_mixed_missing_groups_use_distinct_row_identities(setup, tmp_path, missing):
    spec, rows = setup
    for row in rows[20:]:
        if missing == "absent":
            row.pop("family")
        else:
            row["family"] = None
    source = Path(spec["inputs"][0]["path"])
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    spec["inputs"][0]["sha256"] = pin(source)
    spec["selection"] = None
    spec["splits"] = {"train": .6, "val": .2, "holdout": .2}
    out = tmp_path / "out"
    manifest = compose_fixed(FixedImportSpec.model_validate(spec), out, tmp_path)
    grouped, ungrouped = {}, set()
    actual = []
    for split in manifest.splits:
        for row in map(json.loads, (out / f"{split}.jsonl").read_text().splitlines()):
            actual.append(row["id"])
            if row.get("family") is None:
                ungrouped.add(split)
            else:
                assert grouped.setdefault(row["family"], split) == split
    assert sorted(actual) == sorted(r["id"] for r in rows)
    assert len(ungrouped) == 3
    assert len(set(grouped.values())) > 1


def test_splitlines_boundaries_lossless_runner_roundtrip(setup, tmp_path):
    spec, rows = setup
    boundaries = "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"
    for row in rows:
        row["text"] = f"left{boundaries}right"
        row["metadata"] = {"nested": [f"before{boundaries}after"]}
    source = Path(spec["inputs"][0]["path"])
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    spec["inputs"][0]["sha256"] = pin(source)
    spec["selection"] = None
    spec["splits"] = {"train": .6, "val": .2, "holdout": .2}
    out = tmp_path / "out"
    compose_fixed(FixedImportSpec.model_validate(spec), out, tmp_path)
    expected = [dict(r, content=r["text"]) for r in rows]
    assert list(map(json.loads, (out / "curated.jsonl").read_text().splitlines())) == expected
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "parapet-runner"))
    from parapet_runner.runner import ManifestSplitResolver
    from parapet_runner.manifest import CurationManifest
    receipt = CurationManifest.model_validate_json((out / "manifest.json").read_text())
    resolved = ManifestSplitResolver(root_dir=out).resolve(receipt)
    assert resolved.train_path == out / "train.jsonl"
    actual = []
    for split in receipt.splits:
        actual.extend(map(json.loads, (out / f"{split}.jsonl").read_text().splitlines()))
    assert sorted(actual, key=lambda r: r["id"]) == sorted(expected, key=lambda r: r["id"])


def test_zero_total_selection_rejected_without_artifacts(setup, tmp_path):
    spec, _ = setup
    selection = Path(spec["selection"]["path"])
    selection.write_text(json.dumps({"arm": {"rows": []}}))
    spec["selection"]["sha256"] = pin(selection)
    assert "expected_counts" not in spec
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="selected population must be nonempty"):
        compose_fixed(FixedImportSpec.model_validate(spec), out, tmp_path)
    assert not out.exists()


def test_empty_individual_selection_part_allowed(setup, tmp_path):
    spec, rows = setup
    source = tmp_path / "empty.jsonl"
    source.write_text("")
    spec["inputs"].append(dict(spec["inputs"][0], name="empty", path=str(source), sha256=pin(source)))
    selection = Path(spec["selection"]["path"])
    selection.write_text(json.dumps({"arm": {"rows": [r["id"] for r in rows], "empty": []}}))
    spec["selection"]["sha256"] = pin(selection)
    manifest = compose_fixed(FixedImportSpec.model_validate(spec), tmp_path / "out", tmp_path)
    assert manifest.total_samples == len(rows)


def test_weights_stratified_groups_and_frozen_ids(setup, tmp_path):
    spec, rows = setup
    source = Path(spec["inputs"][0]["path"])
    for row in rows:
        row["weight"] = .25 if row["label"] == "malicious" else 2
        row["stratum"] = str(int(row["family"]) % 2)
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    spec["inputs"][0].update(sha256=pin(source), weight_column="weight", stratify_column="stratum")
    spec["selection"] = None
    spec["splits"] = {"train": .8, "val": .2, "holdout": 0}
    out = tmp_path / "out"
    compose_fixed(FixedImportSpec.model_validate(spec), out, tmp_path)
    receipt = json.loads((out / "manifest.json").read_text())
    assert receipt["fixed_import"]["inputs"][0]["weight_column"] == "weight"
    assert receipt["split_ids"]["sha256"] == pin(out / "split_ids.json")
    ids = json.loads((out / "split_ids.json").read_text())
    assert sorted(ids["train"] + ids["val"]) == sorted(r["id"] for r in rows)
    groups = {}
    for split, expected_groups in [("train", 8), ("val", 2)]:
        entries = list(map(json.loads, (out / f"{split}.jsonl").read_text().splitlines()))
        for r in entries:
            assert r["sample_weight"] == r["weight"]
            assert groups.setdefault(r["family"], split) == split
        for stratum in ("0", "1"):
            assert len({r["family"] for r in entries if r["stratum"] == stratum}) == expected_groups
    second = tmp_path / "second"
    compose_fixed(FixedImportSpec.model_validate(spec), second, tmp_path)
    assert (second / "split_ids.json").read_bytes() == (out / "split_ids.json").read_bytes()


@pytest.mark.parametrize("weight", [0, -1, True, "1", float("nan"), float("inf")])
def test_invalid_sample_weights_rejected_before_writes(setup, tmp_path, weight):
    spec, rows = setup
    for row in rows:
        row["weight"] = weight
    source = Path(spec["inputs"][0]["path"])
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    spec["inputs"][0].update(sha256=pin(source), weight_column="weight")
    with pytest.raises(ValueError, match="weight"):
        compose_fixed(FixedImportSpec.model_validate(spec), tmp_path / "out", tmp_path)
    assert not (tmp_path / "out").exists()


def test_nested_arm_keeps_full_population_calibration(tmp_path):
    # Five groups: full population trains groups 1-4, calibrates group 5.
    rows = [dict(row_id=str(i), text=f"synthetic {i}", label="benign", group=str(i), stratum="same") for i in range(1, 6)]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assignment = tmp_path / "assignment.json"
    assignment.write_text(json.dumps({str(i): "calibration" if i == 5 else "train" for i in range(1, 6)}))
    arms = tmp_path / "arms.json"
    arms.write_text(json.dumps({"full": {"rows": [str(i) for i in range(1, 6)]}, "nested": {"rows": ["4", "5"]}, "excluded": {"rows": ["4"]}}))
    common = dict(mode="fixed_import", name="nested", inputs=[dict(name="rows", path=source,
        sha256=pin(source), text_column="text", id_column="row_id", label_column="label",
        group_column="group", stratify_column="stratum")],
        assignment=dict(path=assignment, sha256=pin(assignment)), splits=dict(train=.8, val=.2, holdout=0))
    for arm in ("full", "nested", "excluded"):
        spec = FixedImportSpec(**common, selection=dict(path=arms, sha256=pin(arms), arm=arm))
        out = tmp_path / arm
        compose_fixed(spec, out, tmp_path)
        ids = json.loads((out / "split_ids.json").read_text())
        assert ids["val"] == ["5"]
        assert ids["train"] == (["1", "2", "3", "4"] if arm == "full" else ["4"])
        assert all(r["id"] == r["row_id"] for r in map(json.loads, (out / "curated.jsonl").read_text().splitlines()))


@pytest.mark.parametrize("mutation", ["deleted", "changed", "swapped"])
def test_frozen_split_weights_cannot_drift(setup, tmp_path, mutation):
    from parapet_runner.runner import ManifestSplitResolver
    from parapet_runner.manifest import CurationManifest
    spec, rows = setup
    source = Path(spec["inputs"][0]["path"])
    for i, row in enumerate(rows):
        row["sample_weight"] = i + .125
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    spec["inputs"][0].update(sha256=pin(source), weight_column="sample_weight")
    out = tmp_path / "out"
    compose_fixed(FixedImportSpec.model_validate(spec), out, tmp_path)
    receipt = CurationManifest.model_validate_json((out / "manifest.json").read_text())
    assert receipt.split_ids is not None
    resolver = ManifestSplitResolver(root_dir=out)
    resolver.resolve(receipt)
    train = out / "train.jsonl"
    entries = list(map(json.loads, train.read_text().splitlines()))
    if mutation == "deleted":
        del entries[0]["sample_weight"]
    elif mutation == "changed":
        entries[0]["sample_weight"] = 99
    else:
        entries[0]["sample_weight"], entries[1]["sample_weight"] = entries[1]["sample_weight"], entries[0]["sample_weight"]
    train.write_text("".join(json.dumps(r) + "\n" for r in entries))
    with pytest.raises(ValueError, match="byte hash"):
        resolver.resolve(receipt)


def test_canonical_id_conflict_refused(setup, tmp_path):
    spec, rows = setup
    for row in rows:
        row["row_id"] = "different-" + row["id"]
    source = Path(spec["inputs"][0]["path"])
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    spec["selection"] = None
    spec["inputs"][0].update(sha256=pin(source), id_column="row_id")
    with pytest.raises(ValueError, match="id column conflicts"):
        compose_fixed(FixedImportSpec.model_validate(spec), tmp_path / "out", tmp_path)


def test_arm_calibration_requires_pinned_population_assignment(setup, tmp_path):
    spec, _ = setup
    spec["splits"] = dict(train=.8, val=.2, holdout=0.)
    with pytest.raises(ValueError, match="full-population assignment"):
        compose_fixed(FixedImportSpec.model_validate(spec), tmp_path / "out", tmp_path)
