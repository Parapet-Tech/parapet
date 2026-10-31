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
