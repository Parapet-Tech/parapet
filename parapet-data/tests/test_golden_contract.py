from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from parapet_data.golden_contract import build_golden_contract, compare_contract


@pytest.fixture()
def tmp_dir() -> Path:
    root = Path(__file__).resolve().parent / ".tmp_temp"
    root.mkdir(exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="golden_contract_", dir=root))
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _write_curated_dir(tmp_path: Path, name: str, entries: list[dict], semantic_hash: str) -> Path:
    curated_dir = Path(tempfile.mkdtemp(prefix=f"{name}_", dir=tmp_path))
    (curated_dir / "curated.yaml").write_text(yaml.safe_dump(entries, sort_keys=False), encoding="utf-8")
    (curated_dir / "manifest.json").write_text(
        json.dumps(
            {
                "spec_name": "mirror_test",
                "spec_version": "0.0.1",
                "spec_hash": "spec123",
                "seed": 42,
                "semantic_hash": semantic_hash,
                "output_hash": f"out-{semantic_hash}",
                "source_hashes": {"en_attacks_merged": "src-hash"},
                "source_metadata": {},
            }
        ),
        encoding="utf-8",
    )
    return curated_dir


def test_build_golden_contract_counts_sources_and_languages(tmp_dir: Path) -> None:
    curated_dir = _write_curated_dir(
        tmp_dir,
        "baseline",
        [
            {"content": "a", "label": "malicious", "language": "EN", "reason": "instruction_override", "source": "en_attacks_merged"},
            {"content": "b", "label": "malicious", "language": "RU", "reason": "instruction_override", "source": "ru_attacks_corpus"},
            {"content": "c", "label": "benign", "language": "EN", "reason": "background", "source": "en_benign_curated"},
        ],
        semantic_hash="sem-baseline",
    )
    runner_root = tmp_dir / "runs"
    run_dir = runner_root / "sample_run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "runtime": {
                    "git_sha": "abc",
                    "trainer_script_hash": "trainer-1",
                    "parapet_eval_hash": "eval-1",
                },
                "curation": {"semantic_hash": "sem-baseline", "spec_hash": "spec123"},
            }
        ),
        encoding="utf-8",
    )

    contract = build_golden_contract(curated_dir, runner_root=runner_root, project_root=tmp_dir)

    assert contract["totals"]["samples"] == 3
    assert contract["totals"]["attack_samples"] == 2
    assert contract["distributions"]["language"]["EN"]["count"] == 2
    assert contract["distributions"]["source_by_label"]["malicious"]["en_attacks_merged"]["count"] == 1
    assert contract["matching_runs"][0]["trainer_script_hash"] == "trainer-1"


def test_compare_contract_flags_source_and_language_drift(tmp_dir: Path) -> None:
    baseline_dir = _write_curated_dir(
        tmp_dir,
        "baseline",
        [
            {"content": "a", "label": "malicious", "language": "EN", "reason": "instruction_override", "source": "en_attacks_merged"},
            {"content": "b", "label": "malicious", "language": "RU", "reason": "instruction_override", "source": "ru_attacks_corpus"},
            {"content": "c", "label": "benign", "language": "EN", "reason": "background", "source": "en_benign_curated"},
            {"content": "d", "label": "benign", "language": "RU", "reason": "background", "source": "ru_benign_xquad"},
        ],
        semantic_hash="sem-baseline",
    )
    observed_dir = _write_curated_dir(
        tmp_dir,
        "observed",
        [
            {"content": "a", "label": "malicious", "language": "EN", "reason": "instruction_override", "source": "en_attacks_merged"},
            {"content": "b", "label": "malicious", "language": "EN", "reason": "instruction_override", "source": "en_attacks_merged"},
            {"content": "c", "label": "benign", "language": "EN", "reason": "background", "source": "en_benign_curated"},
            {"content": "d", "label": "benign", "language": "EN", "reason": "background", "source": "en_benign_curated"},
        ],
        semantic_hash="sem-observed",
    )

    baseline = build_golden_contract(baseline_dir)
    observed = build_golden_contract(observed_dir)

    violations = compare_contract(
        baseline,
        observed,
        max_source_share_delta=0.20,
        max_language_share_delta=0.20,
        min_monitored_source_share=0.20,
    )

    assert any("source share drift [malicious]: en_attacks_merged" in v for v in violations)
    assert any("language share drift: RU" in v for v in violations)
