from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest
import yaml

from parapet_runner.baseline import CommandResult
from parapet_runner.config import TrainConfig
from parapet_runner.manifest import CurationManifest
from parapet_runner.runner import (
    ManifestSplitResolver,
    ParapetEvalEvaluator,
    ParapetEvalPG2BaselineProvider,
    ResolvedSplits,
    TrainScriptTrainer,
)


def _new_output_dir(case_name: str) -> Path:
    output_dir = Path("tests/.tmp_outputs") / case_name
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _hash_content(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _build_manifest(
    *,
    train_path: Path,
    val_path: Path,
    holdout_path: Path,
    train_hashes: list[str],
    val_hashes: list[str],
    holdout_hashes: list[str],
) -> CurationManifest:
    return CurationManifest(
        spec_name="mirror_v1",
        spec_version="1.0.0",
        spec_hash="spec_hash",
        seed=42,
        timestamp="2026-03-01T00:00:00Z",
        source_hashes={"source_a": "hash_a"},
        output_path=Path("data/curated/curated.jsonl"),
        output_hash="out_hash",
        semantic_hash="semantic_hash",
        total_samples=6,
        attack_samples=3,
        benign_samples=3,
        splits={
            "train": {
                "name": "train",
                "sample_count": len(train_hashes),
                "content_hashes": train_hashes,
                "artifact_path": train_path,
            },
            "val": {
                "name": "val",
                "sample_count": len(val_hashes),
                "content_hashes": val_hashes,
                "artifact_path": val_path,
            },
            "holdout": {
                "name": "holdout",
                "sample_count": len(holdout_hashes),
                "content_hashes": holdout_hashes,
                "artifact_path": holdout_path,
            },
        },
        cell_fills={
            "instruction_override__EN_benign": {
                "target": 3,
                "actual": 3,
                "backfilled": 0,
                "backfill_sources": [],
            }
        },
        gaps=[],
        cross_contamination_dropped=0,
    )


class RecordingExecutor:
    def __init__(self, *, returncode: int = 0, payload: dict | None = None) -> None:
        self.calls: list[tuple[list[str], Path | None]] = []
        self.returncode = returncode
        self.payload = payload or {}

    def run(self, args, *, cwd=None):  # type: ignore[no-untyped-def]
        args_list = [str(a) for a in args]
        self.calls.append((args_list, cwd))

        if "--out" in args_list:
            out_path = Path(args_list[args_list.index("--out") + 1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("// generated model\n", encoding="utf-8")

        if "--output" in args_list:
            output_path = Path(args_list[args_list.index("--output") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(self.payload), encoding="utf-8")

        return CommandResult(
            args=tuple(args_list),
            returncode=self.returncode,
            stdout="",
            stderr="",
        )


def test_manifest_split_resolver_resolves_and_validates_hashes() -> None:
    output_dir = _new_output_dir("resolver_ok")
    train_rows = [
        {"id": "t1", "label": "malicious", "content": "attack a"},
        {"id": "t2", "label": "benign", "content": "benign a"},
    ]
    val_rows = [
        {"id": "v1", "label": "malicious", "content": "attack b"},
    ]
    holdout_rows = [
        {"id": "h1", "label": "benign", "content": "benign b"},
    ]

    train_rel = Path("data/curated/train.jsonl")
    val_rel = Path("data/curated/val.jsonl")
    holdout_rel = Path("data/curated/holdout.jsonl")
    _write_jsonl(output_dir / train_rel, train_rows)
    _write_jsonl(output_dir / val_rel, val_rows)
    _write_jsonl(output_dir / holdout_rel, holdout_rows)

    manifest = _build_manifest(
        train_path=train_rel,
        val_path=val_rel,
        holdout_path=holdout_rel,
        train_hashes=[_hash_content(row["content"]) for row in train_rows],
        val_hashes=[_hash_content(row["content"]) for row in val_rows],
        holdout_hashes=[_hash_content(row["content"]) for row in holdout_rows],
    )

    resolved = ManifestSplitResolver(root_dir=output_dir).resolve(manifest)
    assert resolved.train_path == (output_dir / train_rel).resolve()
    assert resolved.val_path == (output_dir / val_rel).resolve()
    assert resolved.holdout_path == (output_dir / holdout_rel).resolve()
    assert resolved.holdout_source == "holdout"


def test_manifest_split_resolver_rejects_hash_mismatch() -> None:
    output_dir = _new_output_dir("resolver_mismatch")
    train_rel = Path("data/curated/train.jsonl")
    val_rel = Path("data/curated/val.jsonl")
    holdout_rel = Path("data/curated/holdout.jsonl")
    _write_jsonl(output_dir / train_rel, [{"id": "x", "label": "malicious", "content": "attack"}])
    _write_jsonl(output_dir / val_rel, [{"id": "y", "label": "benign", "content": "benign"}])
    _write_jsonl(output_dir / holdout_rel, [{"id": "z", "label": "benign", "content": "holdout"}])

    manifest = _build_manifest(
        train_path=train_rel,
        val_path=val_rel,
        holdout_path=holdout_rel,
        train_hashes=["not-the-right-hash"],
        val_hashes=[_hash_content("benign")],
        holdout_hashes=[_hash_content("holdout")],
    )

    with pytest.raises(ValueError, match="Content hash mismatch"):
        ManifestSplitResolver(root_dir=output_dir).resolve(manifest)


def test_train_script_trainer_wires_arguments_and_returns_model_artifact() -> None:
    output_dir = _new_output_dir("trainer")
    train_split = output_dir / "train.yaml"
    train_split.write_text(
        yaml.safe_dump(
            [
                {"id": "a", "label": "malicious", "description": "atk", "content": "attack case"},
                {"id": "b", "label": "benign", "description": "ben", "content": "benign case"},
            ],
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    executor = RecordingExecutor(returncode=0)
    trainer = TrainScriptTrainer(
        trainer_script=Path("scripts/train_l1_specialist.py"),
        workspace_root=Path("C:/tmp/workspace"),
        command_executor=executor,
        python_bin="python",
    )
    config = TrainConfig(mode="iteration", cv_folds=0, max_features=15_000)
    artifact = trainer.train(train_split=train_split, config=config, output_dir=output_dir)

    assert artifact.exists()
    assert len(executor.calls) == 1
    args, cwd = executor.calls[0]
    assert args[0] == "python"
    assert args[1] == "-u"
    assert Path(args[2]).name == "train_l1_specialist.py"
    assert "--attack-files" in args
    assert "--benign-files" in args
    assert "--out" in args
    assert cwd == Path("C:/tmp/workspace")


def test_parapet_eval_evaluator_uses_threshold_and_parses_metrics() -> None:
    output_dir = _new_output_dir("evaluator")
    split_path = output_dir / "val.jsonl"
    _write_jsonl(
        split_path,
        [
            {"id": "v1", "label": "malicious", "description": "atk", "content": "attack one"},
            {"id": "v2", "label": "benign", "description": "ben", "content": "benign one"},
        ],
    )
    model_artifact = output_dir / "weights.rs"
    model_artifact.write_text("// model\n", encoding="utf-8")

    eval_config = output_dir / "eval_config.yaml"
    eval_config.write_text(
        yaml.safe_dump({"layers": {"L1": {"mode": "block", "threshold": 0.0}}}, sort_keys=False),
        encoding="utf-8",
    )

    payload = {
        "layers": [
            {
                "layer": "l1",
                "tp": 8,
                "fp": 2,
                "fn_count": 1,
                "tn": 9,
                "precision": 0.8,
                "recall": 0.8889,
                "f1": 0.8421,
                "total": 20,
            }
        ]
    }
    executor = RecordingExecutor(returncode=1, payload=payload)
    evaluator = ParapetEvalEvaluator(
        parapet_eval_bin=Path("parapet-eval.exe"),
        eval_config=eval_config,
        workspace_root=Path("C:/tmp/workspace"),
        command_executor=executor,
    )

    result = evaluator.evaluate(
        model_artifact=model_artifact,
        split_path=split_path,
        threshold=0.25,
        split_name="val",
        output_dir=output_dir,
    )

    assert result.f1 == pytest.approx(0.8421)
    assert result.false_positives == 2
    assert result.false_negatives == 1
    assert result.threshold == pytest.approx(0.25)
    assert result.holdout_size == 20

    args, _ = executor.calls[0]
    assert "--layer" in args
    assert args[args.index("--layer") + 1] == "l1"
    config_path = Path(args[args.index("--config") + 1])
    applied = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert applied["layers"]["L1"]["threshold"] == pytest.approx(0.25)


def test_pg2_baseline_provider_runs_l2a_remap_on_holdout() -> None:
    output_dir = _new_output_dir("baseline_provider")
    holdout_path = output_dir / "holdout.jsonl"
    _write_jsonl(
        holdout_path,
        [
            {"id": "h1", "label": "malicious", "description": "atk", "content": "attack"},
            {"id": "h2", "label": "benign", "description": "ben", "content": "benign"},
        ],
    )
    payload = {
        "layers": [
            {
                "layer": "l2a",
                "tp": 7,
                "fp": 1,
                "fn_count": 2,
                "tn": 10,
                "precision": 0.875,
                "recall": 0.7778,
                "f1": 0.8235,
                "total": 20,
            }
        ]
    }
    executor = RecordingExecutor(returncode=1, payload=payload)
    provider = ParapetEvalPG2BaselineProvider(
        parapet_eval_bin=Path("parapet-eval.exe"),
        eval_config=Path("eval_config_l2a_only.yaml"),
        workspace_root=Path("C:/tmp/workspace"),
        command_executor=executor,
    )

    holdout = ResolvedSplits(
        train_path=Path("train.jsonl"),
        val_path=Path("val.jsonl"),
        holdout_path=holdout_path,
        holdout_source="holdout",
        dataset_dir=output_dir,
    )
    result = provider.run(
        holdout=holdout,
        train_config=TrainConfig(mode="iteration", cv_folds=0, max_features=15_000),
        output_dir=output_dir,
    )

    assert "pg2" in result.results
    pg2 = result.results["pg2"]
    assert pg2.f1 == pytest.approx(0.8235)
    assert pg2.false_positives == 1
    assert pg2.false_negatives == 2

    args, _ = executor.calls[0]
    assert "--remap-layer" in args
    assert args[args.index("--remap-layer") + 1] == "l2a"
    assert "--layer" in args
    assert args[args.index("--layer") + 1] == "l1"
