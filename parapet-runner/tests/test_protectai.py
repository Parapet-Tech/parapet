from __future__ import annotations

import json
from pathlib import Path
import shutil

import yaml

from parapet_runner.config import ThresholdPolicy, TrainConfig
from parapet_runner.manifest import EvalResult
from parapet_runner.protectai import ProtectAIRecipeSource, materialize_protectai_recipe
from parapet_runner.runner import ProtectAIBaselineProvider, ResolvedSplits


def _write_yaml(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(rows, sort_keys=False), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _new_output_dir(case_name: str) -> Path:
    output_dir = Path("tests/.tmp_outputs") / case_name
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def test_materialize_size_matched_is_deterministic_and_exact_size() -> None:
    root = _new_output_dir("protectai_materialize")
    data_root = root / "schema_eval"
    attack_rows = [
        {"id": f"a{i}", "label": "malicious", "content": f"attack sample {i}"}
        for i in range(10)
    ]
    benign_rows = [
        {"id": f"b{i}", "label": "benign", "content": f"benign sample {i}"}
        for i in range(10)
    ]
    holdout_rows = [
        {"id": "h1", "label": "malicious", "content": "holdout attack"},
        {"id": "h2", "label": "benign", "content": "holdout benign"},
    ]
    _write_yaml(data_root / "malicious/attacks.yaml", attack_rows)
    _write_yaml(data_root / "benign/benign.yaml", benign_rows)
    _write_yaml(data_root / "holdout.yaml", holdout_rows)

    sources = (
        ProtectAIRecipeSource(Path("malicious/attacks.yaml"), "malicious"),
        ProtectAIRecipeSource(Path("benign/benign.yaml"), "benign"),
    )

    first = materialize_protectai_recipe(
        data_root=data_root,
        output_dir=root / "out1",
        holdout_path=data_root / "holdout.yaml",
        baseline_family="protectai_size_matched",
        seed=42,
        target_size=12,
        recipe_sources=sources,
    )
    second = materialize_protectai_recipe(
        data_root=data_root,
        output_dir=root / "out2",
        holdout_path=data_root / "holdout.yaml",
        baseline_family="protectai_size_matched",
        seed=42,
        target_size=12,
        recipe_sources=sources,
    )

    assert first.baseline_recipe_hash == second.baseline_recipe_hash
    assert first.baseline_data_hash == second.baseline_data_hash
    assert first.baseline_data_size == 12
    assert len(_read_jsonl(first.train_path)) + len(_read_jsonl(first.val_path)) == 12


def test_materialize_escapes_json_line_separators_in_jsonl() -> None:
    root = _new_output_dir("protectai_jsonl_escape")
    data_root = root / "schema_eval"
    attack_rows = [
        {"id": "a1", "label": "malicious", "content": "Name: Skynet\u2028\u2028Model: hostile"},
        {"id": "a2", "label": "malicious", "content": "attack sample plain"},
    ]
    benign_rows = [
        {"id": "b1", "label": "benign", "content": "benign sample one"},
        {"id": "b2", "label": "benign", "content": "benign sample two"},
    ]
    holdout_rows = [
        {"id": "h1", "label": "malicious", "content": "holdout attack"},
        {"id": "h2", "label": "benign", "content": "holdout benign"},
    ]
    _write_yaml(data_root / "malicious/attacks.yaml", attack_rows)
    _write_yaml(data_root / "benign/benign.yaml", benign_rows)
    _write_yaml(data_root / "holdout.yaml", holdout_rows)

    recipe = materialize_protectai_recipe(
        data_root=data_root,
        output_dir=root / "out",
        holdout_path=data_root / "holdout.yaml",
        baseline_family="protectai_size_matched",
        seed=42,
        target_size=4,
        recipe_sources=(
            ProtectAIRecipeSource(Path("malicious/attacks.yaml"), "malicious"),
            ProtectAIRecipeSource(Path("benign/benign.yaml"), "benign"),
        ),
    )

    # Must parse as JSONL without unterminated-string failures.
    parsed = _read_jsonl(recipe.train_path) + _read_jsonl(recipe.val_path)
    assert len(parsed) == 4
    assert any("\u2028" in str(row.get("content", "")) for row in parsed)


class _FakeTrainer:
    def __init__(self) -> None:
        self.train_calls: list[Path] = []

    def train(self, *, train_split: Path, config: TrainConfig, output_dir: Path) -> Path:  # noqa: ARG002
        self.train_calls.append(train_split)
        model_path = output_dir / "model" / "weights.rs"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("// fake model\n", encoding="utf-8")
        return model_path


class _FakeEvaluator:
    def __init__(self) -> None:
        self.eval_calls: list[Path] = []

    def evaluate(
        self,
        *,
        model_artifact: Path,  # noqa: ARG002
        split_path: Path,
        threshold: float,
        split_name: str,  # noqa: ARG002
        output_dir: Path,  # noqa: ARG002
    ) -> EvalResult:
        self.eval_calls.append(split_path)
        return EvalResult(
            f1=0.75,
            precision=0.74,
            recall=0.76,
            false_positives=3,
            false_negatives=4,
            threshold=threshold,
            holdout_size=20,
        )


def test_protectai_baseline_provider_runs_training_and_eval() -> None:
    root = _new_output_dir("protectai_provider")
    data_root = root / "schema_eval"
    _write_yaml(
        data_root / "malicious/attacks.yaml",
        [{"id": f"a{i}", "label": "malicious", "content": f"attack sample {i}"} for i in range(8)],
    )
    _write_yaml(
        data_root / "benign/benign.yaml",
        [{"id": f"b{i}", "label": "benign", "content": f"benign sample {i}"} for i in range(8)],
    )
    _write_yaml(
        data_root / "holdout.yaml",
        [
            {"id": "h1", "label": "malicious", "content": "holdout attack"},
            {"id": "h2", "label": "benign", "content": "holdout benign"},
        ],
    )

    trainer = _FakeTrainer()
    evaluator = _FakeEvaluator()
    provider = ProtectAIBaselineProvider(
        data_root=data_root,
        trainer=trainer,
        evaluator=evaluator,
        baseline_family="protectai_size_matched",
        target_size=10,
        seed=7,
        recipe_sources=(
            ProtectAIRecipeSource(Path("malicious/attacks.yaml"), "malicious"),
            ProtectAIRecipeSource(Path("benign/benign.yaml"), "benign"),
        ),
    )

    run = provider.run(
        holdout=ResolvedSplits(
            train_path=Path("train.jsonl"),
            val_path=Path("val.jsonl"),
            holdout_path=data_root / "holdout.yaml",
            holdout_source="holdout",
            dataset_dir=data_root,
            content_hashes=["h1", "h2", "h3"],
        ),
        train_config=TrainConfig(
            mode="iteration",
            cv_folds=0,
            max_features=15_000,
            threshold_policy=ThresholdPolicy.FIXED,
            threshold_value=-0.5,
        ),
        output_dir=root / "run_out",
    )

    assert "protectai_size_matched" in run.results
    assert run.baseline_family == "protectai_size_matched"
    assert run.baseline_recipe_hash is not None
    assert run.baseline_data_hash is not None
    assert run.baseline_data_size == 10
    assert len(trainer.train_calls) == 1
    assert len(evaluator.eval_calls) == 1
