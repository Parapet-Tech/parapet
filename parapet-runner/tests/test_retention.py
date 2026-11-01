"""Hand-checked specialist retention policy and boundary behavior."""
import hashlib
import json
from pathlib import Path

import pytest

from parapet_runner.config import TrainConfig
from parapet_runner.retention import constrained_retention, calibrate_pinned_scores
from parapet_runner.runner import _install_weights_and_rebuild, RuntimeIdentityCollector


def row(label, g, s):
    return dict(label=label, generalist_score=g, specialist_score=s)


def test_relief_subject_to_retention_and_lowest_tie():
    rows = [row("malicious", 1, 2), row("malicious", 3, -5),
            row("benign", 1, 0), row("benign", 1, 3), row("benign", -1, 1)]
    result = constrained_retention(rows, retention_floor=1)
    assert result["threshold"] == 1  # t=1 and t=2 give one relief; choose 1
    assert result["relief_count"] == 1
    assert result["total"]["retention"] == 1
    assert result["total"]["wilson95"] == pytest.approx([.3423802275, 1])
    assert result["deciding_band"]["total"] == 1
    assert result["deciding_band"]["wilson95"] == pytest.approx([.2065493144, 1])


def test_total_and_band_retention_differ_and_boundaries_are_inclusive():
    rows = [row("malicious", 0, 0), row("malicious", 2, 2), row("malicious", 3, -9),
            row("malicious", -1, -10), row("benign", 0, 0), row("benign", 3, -10)]
    result = constrained_retention(rows, retention_floor=2/3)
    assert result["threshold"] == 2
    assert result["total"]["retention"] == 2/3
    assert result["deciding_band"]["retention"] == .5
    assert result["relief_count"] == 1


@pytest.mark.parametrize("rows", [[], [row("benign", 1, 2)], [row("malicious", -1, 2)]])
def test_no_feasible_threshold(rows):
    with pytest.raises(ValueError, match="no feasible"):
        constrained_retention(rows)


@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_invalid_scores(score):
    with pytest.raises(ValueError, match="finite"):
        constrained_retention([row("malicious", 0, score)])


def test_empty_deciding_band_reports_undefined_interval():
    result = constrained_retention([row("malicious", 3, 0)])
    assert result["deciding_band"] == dict(kept=0, total=0, retention=None, wilson95=None)


def test_pinned_scores_require_exact_validation_population(tmp_path):
    scores = tmp_path / "scores.jsonl"
    scores.write_text(json.dumps(dict(row("malicious", 0, 1), id="a")) + "\n")
    config = TrainConfig(threshold_policy="constrained_retention", calibration_scores=scores,
                         calibration_scores_sha256=hashlib.sha256(scores.read_bytes()).hexdigest())
    assert calibrate_pinned_scores(config, [dict(id="a", label="malicious")])["threshold"] == 1
    with pytest.raises(ValueError, match="match validation"):
        calibrate_pinned_scores(config, [dict(id="b", label="malicious")])
    scores.write_text("[]")
    with pytest.raises(ValueError, match="sha256"):
        calibrate_pinned_scores(config, [])


def test_no_cap_and_min_df_are_expressible():
    config = TrainConfig(max_features=None, min_df=2)
    args = config.to_train_script_args(attack_files=[Path("a")], benign_files=[Path("b")],
                                      out_path=Path("out"), holdout_path=Path("holdout"))
    assert args[args.index("--max-features") + 1] == "none"
    assert args[args.index("--min-df") + 1] == "2"


def test_specialist_cannot_overwrite_shipped_weights(tmp_path):
    layers = tmp_path / "src" / "layers"
    layers.mkdir(parents=True)
    shipped = layers / "l1_weights.rs"
    shipped.write_text("generalist")
    specialist = tmp_path / "l1_weights_example.rs"
    specialist.write_text("specialist")
    class Executor:
        def run(self, args, **kwargs):
            return type("Result", (), {"returncode": 0})()
    kwargs = dict(model_artifact=specialist, rust_crate_dir=tmp_path, last_installed_hash=None,
                  executor=Executor(), build_features=["eval"])
    with pytest.raises(ValueError, match="separate"):
        _install_weights_and_rebuild(weights_install_target="l1_weights.rs", **kwargs)
    _install_weights_and_rebuild(weights_install_target="l1_weights_example.rs", **kwargs)
    assert shipped.read_text() == "generalist"
    assert (layers / "l1_weights_example.rs").read_text() == "specialist"


def test_environment_freeze_uses_configured_python(tmp_path):
    calls = []
    class Executor:
        def run(self, args, **kwargs):
            calls.append(args)
            return type("Result", (), {"returncode": 0, "stdout": "synthetic==1", "stderr": ""})()
    collector = RuntimeIdentityCollector(workspace_root=tmp_path, trainer_script=Path("trainer"),
        parapet_eval_binary=Path("eval"), pg2_model_id="example", eval_config=Path("config"),
        command_executor=Executor(), python_bin="configured-python")
    collector._environment_hash()
    assert calls == [["configured-python", "-m", "pip", "freeze", "--all"]]


def test_runner_preserves_weights_and_uses_frozen_split(tmp_path):
    import yaml
    from parapet_runner.runner import TrainScriptTrainer
    split = tmp_path / "train.jsonl"
    split.write_text("\n".join(json.dumps(dict(id=str(i), content=f"synthetic {i}",
        label=label, sample_weight=weight)) for i, (label, weight) in enumerate([("malicious", .25), ("benign", 2)])))
    calls = []
    class Executor:
        def run(self, args, **kwargs):
            calls.append(args)
            artifact = Path(args[args.index("--out") + 1])
            artifact.parent.mkdir(parents=True)
            artifact.write_text("weights")
            return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    trainer = TrainScriptTrainer(trainer_script=Path("trainer"), workspace_root=tmp_path,
        command_executor=Executor(), python_bin="configured-python")
    trainer.train(train_split=split, config=TrainConfig(), output_dir=tmp_path)
    args = calls[0]
    assert args[0] == "configured-python"
    assert "--pre-split" in args
    attacks = yaml.safe_load(Path(args[args.index("--attack-files") + 1]).read_text())
    benign = yaml.safe_load(Path(args[args.index("--benign-files") + 1]).read_text())
    assert attacks[0]["sample_weight"] == .25
    assert benign[0]["sample_weight"] == 2
