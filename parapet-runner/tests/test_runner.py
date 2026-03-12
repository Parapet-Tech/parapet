from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from parapet_runner.config import ThresholdPolicy, TrainConfig
from parapet_runner.manifest import CurationManifest, EvalResult, RuntimeIdentity
from parapet_runner.baseline import CommandResult
from parapet_runner.runner import (
    BaselineRun,
    ExperimentDependencies,
    ExperimentRunner,
    F1GridSearchCalibrator,
    OutputHashVerifier,
    RandomBaselineProvider,
    ResolvedSplits,
    _build_parser,
    _install_weights_and_rebuild,
    assert_no_leakage,
)


class FakeSplitResolver:
    def resolve(self, curation: CurationManifest) -> ResolvedSplits:  # noqa: ARG002
        return ResolvedSplits(
            train_path=Path("train.jsonl"),
            val_path=Path("val.jsonl"),
            holdout_path=Path("holdout.jsonl"),
            holdout_source="holdout",
            dataset_dir=Path("."),
            content_hashes=["h2", "h1"],
            per_cell_counts={"roleplay_jailbreak__EN_benign": {"target": 100, "actual": 98, "backfilled": 2}},
        )


class FakeTrainer:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def train(self, *, train_split: Path, config: TrainConfig, output_dir: Path) -> Path:  # noqa: ARG002
        self._calls.append(f"train:{train_split}")
        return output_dir / "weights.rs"


class FakeEvaluator:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def evaluate(
        self,
        *,
        model_artifact: Path,  # noqa: ARG002
        split_path: Path,  # noqa: ARG002
        threshold: float,
        split_name: str,
        output_dir: Path,  # noqa: ARG002
    ) -> EvalResult:
        self._calls.append(f"eval:{split_name}:{threshold}")
        if split_name == "val":
            f1 = 1.0 - abs(threshold - 0.2)
            return EvalResult(
                f1=f1,
                precision=max(0.0, min(1.0, f1 - 0.01)),
                recall=max(0.0, min(1.0, f1 - 0.02)),
                false_positives=1,
                false_negatives=1,
                threshold=threshold,
                holdout_size=50,
            )
        return EvalResult(
            f1=0.82,
            precision=0.84,
            recall=0.80,
            false_positives=5,
            false_negatives=7,
            threshold=threshold,
            holdout_size=100,
        )


class FakeBaseline:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def run(
        self,
        *,
        holdout: ResolvedSplits,  # noqa: ARG002
        train_config: TrainConfig,  # noqa: ARG002
        output_dir: Path,  # noqa: ARG002
    ) -> BaselineRun:
        self._calls.append("baseline")
        return BaselineRun(
            results={
                "pg2": EvalResult(
                    f1=0.78,
                    precision=0.79,
                    recall=0.77,
                    false_positives=8,
                    false_negatives=10,
                    threshold=0.0,
                    holdout_size=100,
                )
            }
        )


class FakeErrorAnalyzer:
    def write(
        self,
        *,
        eval_result: EvalResult,  # noqa: ARG002
        baseline_results: dict[str, EvalResult],  # noqa: ARG002
        baseline_deltas: dict[str, dict[str, float]],  # noqa: ARG002
        output_dir: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "errors.yaml"
        path.write_text("errors: []\n", encoding="utf-8")
        return path


class FakeRuntimeIdentityProvider:
    def collect(self) -> RuntimeIdentity:
        return RuntimeIdentity(
            git_sha="abc123",
            trainer_script_hash="trainer",
            parapet_eval_hash="eval",
            pg2_model_id="pg2-22m",
            eval_config_hash="config",
            env_hash="env",
        )


def _make_manifest() -> CurationManifest:
    return CurationManifest(
        spec_name="mirror_v1",
        spec_version="1.0.0",
        spec_hash="spec-hash",
        seed=42,
        timestamp="2026-03-01T00:00:00Z",
        source_hashes={"src_a": "hash_a"},
        output_path=Path("curated.jsonl"),
        output_hash="deadbeef",
        semantic_hash="semantic",
        total_samples=200,
        attack_samples=100,
        benign_samples=100,
        splits={
            "train": {
                "name": "train",
                "sample_count": 160,
                "content_hashes": ["h1", "h2"],
                "artifact_path": Path("train.jsonl"),
            },
            "val": {
                "name": "val",
                "sample_count": 20,
                "content_hashes": ["h3"],
                "artifact_path": Path("val.jsonl"),
            },
            "holdout": {
                "name": "holdout",
                "sample_count": 20,
                "content_hashes": ["h4"],
                "artifact_path": Path("holdout.jsonl"),
            },
        },
        cell_fills={
            "roleplay_jailbreak__EN_benign": {
                "target": 100,
                "actual": 98,
                "backfilled": 4,
                "backfill_sources": ["roleplay_any_language"],
            }
        },
        gaps=[],
        cross_contamination_dropped=0,
    )


def _new_output_dir(case_name: str) -> Path:
    output_dir = Path("tests/.tmp_outputs") / case_name
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def test_runner_calibrates_on_val_then_evaluates_holdout() -> None:
    calls: list[str] = []
    deps = ExperimentDependencies(
        split_resolver=FakeSplitResolver(),
        trainer=FakeTrainer(calls),
        evaluator=FakeEvaluator(calls),
        threshold_calibrator=F1GridSearchCalibrator(thresholds=[-0.5, 0.2, 0.8]),
        baseline_provider=FakeBaseline(calls),
        error_analyzer=FakeErrorAnalyzer(),
        runtime_identity_provider=FakeRuntimeIdentityProvider(),
    )
    runner = ExperimentRunner(deps)
    config = TrainConfig(
        mode="iteration",
        cv_folds=0,
        max_features=15_000,
        threshold_policy=ThresholdPolicy.CALIBRATE_F1,
    )
    run_manifest = runner.run_experiment(
        curation_manifest=_make_manifest(),
        train_config=config,
        output_dir=_new_output_dir("calibrated"),
        run_id="run_001",
    )

    assert run_manifest.run_id == "run_001"
    assert run_manifest.eval_result.threshold == 0.2
    assert run_manifest.delta is not None
    assert run_manifest.delta["f1_delta"] == pytest.approx(0.04)
    assert "pg2" in run_manifest.baseline_results
    assert "pg2" in run_manifest.baseline_deltas
    assert run_manifest.semantic_parity_hash is not None
    assert calls == [
        "train:train.jsonl",
        "eval:val:-0.5",
        "eval:val:0.2",
        "eval:val:0.8",
        "eval:holdout:0.2",
        "baseline",
    ]


def test_runner_uses_fixed_threshold_without_calibration() -> None:
    calls: list[str] = []
    deps = ExperimentDependencies(
        split_resolver=FakeSplitResolver(),
        trainer=FakeTrainer(calls),
        evaluator=FakeEvaluator(calls),
        threshold_calibrator=None,
        baseline_provider=FakeBaseline(calls),
        error_analyzer=FakeErrorAnalyzer(),
        runtime_identity_provider=FakeRuntimeIdentityProvider(),
    )
    runner = ExperimentRunner(deps)
    config = TrainConfig(
        mode="iteration",
        cv_folds=0,
        max_features=15_000,
        threshold_policy=ThresholdPolicy.FIXED,
        threshold_value=-0.7,
    )

    run_manifest = runner.run_experiment(
        curation_manifest=_make_manifest(),
        train_config=config,
        output_dir=_new_output_dir("fixed"),
        run_id="run_002",
    )

    assert run_manifest.eval_result.threshold == -0.7
    assert run_manifest.delta is not None
    assert run_manifest.baseline_results["pg2"].f1 == pytest.approx(0.78)
    assert calls == [
        "train:train.jsonl",
        "eval:holdout:-0.7",
        "baseline",
    ]


def test_output_hash_verifier_rejects_mismatch() -> None:
    output_dir = _new_output_dir("verifier")
    artifact = output_dir / "curated.jsonl"
    artifact.write_text("hello\n", encoding="utf-8")

    manifest = _make_manifest().model_copy(
        update={
            "output_path": artifact,
            "output_hash": "not-a-real-hash",
        }
    )
    verifier = OutputHashVerifier()
    with pytest.raises(ValueError, match="hash mismatch"):
        verifier.verify(manifest)


def test_assert_no_leakage_passes_on_disjoint_sets() -> None:
    train = {"a", "b", "c"}
    eval_set = {"d", "e", "f"}
    assert_no_leakage(train, eval_set, context="test")  # should not raise


def test_assert_no_leakage_raises_on_overlap() -> None:
    train = {"a", "b", "c"}
    eval_set = {"b", "c", "d"}
    with pytest.raises(ValueError, match="Data leakage detected"):
        assert_no_leakage(train, eval_set, context="test")


def test_assert_no_leakage_reports_overlap_count() -> None:
    train = {"a", "b", "c", "d", "e"}
    eval_set = {"b", "d"}
    with pytest.raises(ValueError, match=r"2.*of.*5.*train samples.*40\.0%"):
        assert_no_leakage(train, eval_set)


class _RecordingExecutor:
    """Captures all commands run via the executor."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def run(self, args, *, cwd=None):  # noqa: ARG002
        self.commands.append(tuple(str(a) for a in args))
        return CommandResult(args=tuple(str(a) for a in args), returncode=0, stdout="", stderr="")


def test_install_weights_copies_and_rebuilds() -> None:
    base = _new_output_dir("recompile_copy")
    rust_crate = base / "parapet"
    layers_dir = rust_crate / "src" / "layers"
    layers_dir.mkdir(parents=True)

    weights = base / "model" / "l1_weights_generalist.rs"
    weights.parent.mkdir(parents=True)
    weights.write_text("pub const BIAS: f64 = -0.5;\n", encoding="utf-8")

    executor = _RecordingExecutor()
    target = "l1_weights_test.rs"

    h1 = _install_weights_and_rebuild(
        model_artifact=weights,
        rust_crate_dir=rust_crate,
        weights_install_target=target,
        last_installed_hash=None,
        executor=executor,
    )

    installed = layers_dir / target
    assert installed.exists()
    assert installed.read_text(encoding="utf-8") == "pub const BIAS: f64 = -0.5;\n"
    assert len(executor.commands) == 1
    assert "cargo" in executor.commands[0][0]
    assert "--features" in executor.commands[0]
    assert "eval" in executor.commands[0]


def test_install_weights_skips_rebuild_when_hash_matches() -> None:
    base = _new_output_dir("recompile_skip")
    rust_crate = base / "parapet"
    (rust_crate / "src" / "layers").mkdir(parents=True)

    weights = base / "model" / "l1_weights_generalist.rs"
    weights.parent.mkdir(parents=True)
    weights.write_text("pub const BIAS: f64 = -0.5;\n", encoding="utf-8")

    executor = _RecordingExecutor()

    h1 = _install_weights_and_rebuild(
        model_artifact=weights,
        rust_crate_dir=rust_crate,
        weights_install_target="test.rs",
        last_installed_hash=None,
        executor=executor,
    )
    assert len(executor.commands) == 1

    executor.commands.clear()
    h2 = _install_weights_and_rebuild(
        model_artifact=weights,
        rust_crate_dir=rust_crate,
        weights_install_target="test.rs",
        last_installed_hash=h1,
        executor=executor,
    )

    assert h2 == h1
    assert len(executor.commands) == 0  # no rebuild


def test_install_weights_rebuilds_on_new_weights() -> None:
    base = _new_output_dir("recompile_new")
    rust_crate = base / "parapet"
    layers_dir = rust_crate / "src" / "layers"
    layers_dir.mkdir(parents=True)

    weights = base / "model" / "l1_weights_generalist.rs"
    weights.parent.mkdir(parents=True)
    weights.write_text("pub const BIAS: f64 = -0.5;\n", encoding="utf-8")

    executor = _RecordingExecutor()
    target = "test.rs"

    h1 = _install_weights_and_rebuild(
        model_artifact=weights,
        rust_crate_dir=rust_crate,
        weights_install_target=target,
        last_installed_hash=None,
        executor=executor,
    )

    # Change weights
    weights.write_text("pub const BIAS: f64 = -1.0;\n", encoding="utf-8")
    executor.commands.clear()

    h2 = _install_weights_and_rebuild(
        model_artifact=weights,
        rust_crate_dir=rust_crate,
        weights_install_target=target,
        last_installed_hash=h1,
        executor=executor,
    )

    assert h2 != h1
    assert len(executor.commands) == 1  # rebuilt
    assert (layers_dir / target).read_text(encoding="utf-8") == "pub const BIAS: f64 = -1.0;\n"


def test_install_weights_raises_on_cargo_failure() -> None:
    base = _new_output_dir("recompile_fail")
    rust_crate = base / "parapet"
    (rust_crate / "src" / "layers").mkdir(parents=True)

    weights = base / "model" / "l1_weights_generalist.rs"
    weights.parent.mkdir(parents=True)
    weights.write_text("broken", encoding="utf-8")

    class FailingExecutor:
        def run(self, args, *, cwd=None):  # noqa: ARG002
            return CommandResult(
                args=tuple(str(a) for a in args),
                returncode=101,
                stdout="",
                stderr="error: could not compile",
            )

    with pytest.raises(RuntimeError, match="Cargo build failed"):
        _install_weights_and_rebuild(
            model_artifact=weights,
            rust_crate_dir=rust_crate,
            weights_install_target="test.rs",
            last_installed_hash=None,
            executor=FailingExecutor(),
        )


def test_runner_cli_defaults_install_into_runtime_l1_weights_module() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "run",
            "--curation-manifest",
            "manifest.json",
            "--train-config",
            "config.yaml",
            "--output-dir",
            "runs/out",
        ]
    )
    assert args.weights_install_target == "l1_weights.rs"


def test_install_weights_can_rebuild_with_eval_and_l2a() -> None:
    base = _new_output_dir("recompile_l2a")
    rust_crate = base / "parapet"
    layers_dir = rust_crate / "src" / "layers"
    layers_dir.mkdir(parents=True)

    weights = base / "model" / "l1_weights_generalist.rs"
    weights.parent.mkdir(parents=True)
    weights.write_text("pub const BIAS: f64 = -0.5;\n", encoding="utf-8")

    executor = _RecordingExecutor()

    _install_weights_and_rebuild(
        model_artifact=weights,
        rust_crate_dir=rust_crate,
        weights_install_target="test.rs",
        last_installed_hash=None,
        executor=executor,
        build_features=("eval", "l2a"),
    )

    assert len(executor.commands) == 1
    assert "--features" in executor.commands[0]
    assert "eval,l2a" in executor.commands[0]
    assert (layers_dir / "test.rs").exists()


def _write_pool(path: Path, entries: list[dict]) -> None:
    """Write a YAML pool file for testing."""
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(entries, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def test_random_baseline_provider_trains_and_evaluates() -> None:
    base = _new_output_dir("random_baseline")

    # Create attack pool (20 entries)
    atk_entries = [
        {"content": f"attack payload {i}", "label": "malicious", "id": f"atk-{i}"}
        for i in range(20)
    ]
    atk_pool = base / "pools" / "attacks.yaml"
    _write_pool(atk_pool, atk_entries)

    # Create benign pool (20 entries)
    ben_entries = [
        {"content": f"benign text {i}", "label": "benign", "id": f"ben-{i}"}
        for i in range(20)
    ]
    ben_pool = base / "pools" / "benign.yaml"
    _write_pool(ben_pool, ben_entries)

    calls: list[str] = []
    provider = RandomBaselineProvider(
        attack_pool=atk_pool,
        benign_pool=ben_pool,
        trainer=FakeTrainer(calls),
        evaluator=FakeEvaluator(calls),
        target_size=20,  # 10 attack + 10 benign
        seed=42,
    )

    holdout = ResolvedSplits(
        train_path=Path("train.yaml"),
        val_path=Path("val.yaml"),
        holdout_path=Path("holdout.yaml"),
        holdout_source="holdout",
        dataset_dir=Path("."),
        content_hashes=["unrelated_hash"],
        per_cell_counts={},
    )

    config = TrainConfig(
        mode="iteration",
        cv_folds=0,
        max_features=15_000,
        threshold_policy=ThresholdPolicy.FIXED,
        threshold_value=-0.5,
    )

    result = provider.run(
        holdout=holdout,
        train_config=config,
        output_dir=base / "output",
    )

    assert "random" in result.results
    assert result.results["random"].f1 > 0
    # Should have trained and evaluated
    assert any("train:" in c for c in calls)
    assert any("eval:holdout" in c for c in calls)

    # Recipe manifest should exist
    manifest_path = base / "output" / "_baseline_random" / "data" / "recipe_manifest.json"
    assert manifest_path.exists()
