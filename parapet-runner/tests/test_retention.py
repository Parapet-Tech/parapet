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


def calibration_fixture(tmp_path):
    from parapet_runner.retention import weights_identity
    generalist = tmp_path / "generalist.rs"
    generalist.write_text("// Trained: yesterday\nfrozen table")
    candidate = tmp_path / "candidate.rs"
    candidate.write_text('pub const BIAS: f64 = 1_f64;\n    " a " => 1_f64,\n')
    scores = tmp_path / "scores.jsonl"
    identity = weights_identity(generalist)
    score_row = dict(id="a", label="malicious", generalist_score=0,
        generalist_weights_sha256=identity, content_sha256=hashlib.sha256(b"a").hexdigest())
    scores.write_text(json.dumps(score_row) + "\n")
    config = TrainConfig(specialist="example", threshold_policy="constrained_retention", calibration_scores=scores,
        calibration_scores_sha256=hashlib.sha256(scores.read_bytes()).hexdigest(),
        generalist_weights=generalist, generalist_weights_sha256=identity)
    return config, candidate, [dict(id="a", content="a", label="malicious")]


def test_candidate_scores_come_from_actual_model(tmp_path):
    config, model, rows = calibration_fixture(tmp_path)
    assert calibrate_pinned_scores(config, rows, model_artifact=model)["threshold"] == 2
    model.write_text('pub const BIAS: f64 = 4_f64;\n    " a " => 1_f64,\n')
    assert calibrate_pinned_scores(config, rows, model_artifact=model)["threshold"] == 5


@pytest.mark.parametrize("failure", ["id", "content", "generalist", "score_identity", "external_specialist", "bytes"])
def test_pinned_scores_reject_mismatches(tmp_path, failure):
    config, model, rows = calibration_fixture(tmp_path)
    if failure == "id":
        rows[0]["id"] = "b"
    elif failure == "content":
        rows[0]["content"] = "changed"
    elif failure == "generalist":
        config.generalist_weights.write_text("wrong table")
    else:
        score = json.loads(config.calibration_scores.read_text())
        if failure == "score_identity":
            score["generalist_weights_sha256"] = "0" * 64
        elif failure == "external_specialist":
            score["specialist_score"] = 99
        config.calibration_scores.write_text(json.dumps(score) + " ")
        if failure != "bytes":
            config.calibration_scores_sha256 = hashlib.sha256(config.calibration_scores.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        calibrate_pinned_scores(config, rows, model_artifact=model)


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
                  executor=Executor(), build_features=["eval"], model_role="example")
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


@pytest.mark.parametrize("pre_split", [False, True])
def test_runner_preserves_weights_and_uses_frozen_split(tmp_path, pre_split):
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
    trainer.train(train_split=split, config=TrainConfig(pre_split=pre_split), output_dir=tmp_path)
    args = calls[0]
    assert args[0] == "configured-python"
    assert ("--pre-split" in args) == pre_split
    attacks = yaml.safe_load(Path(args[args.index("--attack-files") + 1]).read_text())
    benign = yaml.safe_load(Path(args[args.index("--benign-files") + 1]).read_text())
    assert attacks[0]["sample_weight"] == .25
    assert benign[0]["sample_weight"] == 2


@pytest.mark.parametrize("alias", ["symlink", "case", "hardlink"])
def test_install_refuses_generalist_aliases_before_copy_or_build(tmp_path, alias):
    import os
    layers = tmp_path / "src" / "layers"
    layers.mkdir(parents=True)
    protected = layers / "l1_weights.rs"
    protected.write_text("frozen")
    target = layers / ("L1_WEIGHTS.RS" if alias == "case" else "alias.rs")
    if alias == "symlink":
        target.symlink_to(protected)
    elif alias == "hardlink":
        os.link(protected, target)
    model = tmp_path / "arbitrary.rs"
    model.write_text("candidate")
    class Executor:
        def run(self, *args, **kwargs):
            pytest.fail("must refuse before build")
    with pytest.raises(ValueError):
        _install_weights_and_rebuild(model_artifact=model, rust_crate_dir=tmp_path,
            weights_install_target=target.name, last_installed_hash=None,
            executor=Executor(), model_role="example")
    assert protected.read_text() == "frozen"


def test_threshold_yaml_roundtrip_targets_rust_specialist_fields(tmp_path):
    import yaml
    from parapet_runner.runner import _set_l1_threshold
    policy, _, _ = calibration_fixture(tmp_path)
    source = {"layers": {"L1": {"mode": "block", "threshold": 0.,
        "generalist_solo_threshold": 3., "specialists": {"example": {"threshold": 0.}}}}}
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump(_set_l1_threshold(source, 2., policy=policy)))
    # These are precisely RawL1Config/RawL1SpecialistConfig's YAML fields.
    l1 = yaml.safe_load(path.read_text())["layers"]["L1"]
    assert l1["threshold"] == 0.
    assert l1["generalist_solo_threshold"] == 3.
    assert l1["specialists"] == {"example": {"threshold": 2.}}
    assert not (1. >= l1["specialists"]["example"]["threshold"])
    assert source["layers"]["L1"]["specialists"]["example"]["threshold"] == 0.
    for field, value in [("threshold", -1.), ("generalist_solo_threshold", 4.)]:
        changed = yaml.safe_load(yaml.safe_dump(source))
        changed["layers"]["L1"][field] = value
        with pytest.raises(ValueError, match="disagree"):
            _set_l1_threshold(changed, 2., policy=policy)
    source["layers"]["L1"]["specialists"]["other"] = {"threshold": 0.}
    with pytest.raises(ValueError, match="exactly"):
        _set_l1_threshold(source, 2., policy=policy)


def test_evaluator_writes_selected_specialist_threshold(tmp_path):
    import yaml
    from parapet_runner.runner import ParapetEvalEvaluator
    config, model, rows = calibration_fixture(tmp_path)
    split = tmp_path / "holdout.jsonl"
    split.write_text(json.dumps(rows[0]) + "\n")
    source = tmp_path / "eval.yaml"
    source.write_text(yaml.safe_dump({"layers": {"L1": {"mode": "block", "threshold": 0.,
        "generalist_solo_threshold": 3., "specialists": {"example": {"threshold": 0.}}}}}))
    class Executor:
        def run(self, args, **kwargs):
            generated = yaml.safe_load(Path(args[args.index("--config") + 1]).read_text())
            l1 = generated["layers"]["L1"]
            assert l1["threshold"] == 0.
            assert l1["generalist_solo_threshold"] == 3.
            assert l1["specialists"]["example"]["threshold"] == 2.
            Path(args[args.index("--output") + 1]).write_text(json.dumps({"layers": [dict(
                layer="l1", tp=0, fp=0, fn_count=1, tn=0, precision=0., recall=0., f1=0., total=1)]}))
            return type("Result", (), {"returncode": 0})()
    evaluator = ParapetEvalEvaluator(parapet_eval_bin=Path("eval"), eval_config=source,
        workspace_root=tmp_path, command_executor=Executor(), model_role="example")
    result = evaluator.evaluate(model_artifact=model, split_path=split, threshold=2.,
        split_name="holdout", output_dir=tmp_path / "run", calibration_policy=config)
    assert result.threshold == 2.
