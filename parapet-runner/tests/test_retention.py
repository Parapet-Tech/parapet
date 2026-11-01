"""Hand-checked specialist retention policy and boundary behavior."""
import hashlib
import json
from pathlib import Path

import pytest

from parapet_runner.config import TrainConfig
from parapet_runner.retention import constrained_retention, calibrate_runtime, score_candidate
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
    identity = weights_identity(generalist)
    config = TrainConfig(specialist="pilot", threshold_policy="constrained_retention",
        generalist_weights=generalist, generalist_weights_sha256=identity)
    return config, candidate, [dict(id="a", content="a", label="malicious")]


def test_runtime_overrides_surrogate_at_float_boundary(tmp_path):
    config, model, rows = calibration_fixture(tmp_path)
    rows[0]["content"] = "ab"
    model.write_text('pub const BIAS: f64 = 0.30000000_f64;\n    " ab" => 0.20000000_f64,\n    "ab " => 0.10000000_f64,\n')
    proposed = score_candidate(model, rows, config)[0]
    assert proposed == 0.6000000000000001
    calls = []
    def runtime(t, generalist_only=False):
        calls.append((t, generalist_only))
        return {"a": t <= (1. if generalist_only else proposed - 1e-16)}
    with pytest.raises(ValueError, match="no feasible runtime"):
        calibrate_runtime(config, rows, model_artifact=model, runtime=runtime)
    assert (proposed, False) in calls


def test_empty_map_and_runtime_tie_search(tmp_path):
    config, model, rows = calibration_fixture(tmp_path)
    model.write_text("pub const BIAS: f64 = 1_f64;\npub static WEIGHTS: phf::Map<&'static str, f64> = phf::phf_map! {};\n")
    assert score_candidate(model, rows, config) == [1.]
    def runtime(t, generalist_only=False):
        return {"a": t <= 1.}
    result = calibrate_runtime(config, rows, model_artifact=model, runtime=runtime)
    assert result["threshold"] == 1.
    assert result["total"]["retention"] == 1.
    assert result["deciding_band"]["total"] == 1
    assert len(result["runtime_evaluations"]) == 1


def test_runtime_search_maximizes_relief_and_uses_lowest_tie(tmp_path):
    config, model, rows = calibration_fixture(tmp_path)
    rows += [dict(id="b", content="b", label="benign"), dict(id="c", content="c", label="benign")]
    model.write_text('pub const BIAS: f64 = 0_f64;\n    " a " => 3_f64,\n    " b " => 1_f64,\n    " c " => 2_f64,\n')
    def runtime(t, generalist_only=False):
        if generalist_only:
            return {r["id"]: t < 3 for r in rows}
        return {"a": t <= 3, "b": t <= 1, "c": True}
    result = calibrate_runtime(config, rows, model_artifact=model, runtime=runtime)
    assert result["threshold"] == 2
    assert result["relief_count"] == 1
    assert result["total"]["retention"] == 1
    assert {r["threshold"] for r in result["runtime_evaluations"]} == {1., 2., 3.}


def test_runtime_rejects_generalist_drift(tmp_path):
    config, model, rows = calibration_fixture(tmp_path)
    config.generalist_weights.write_text("changed")
    with pytest.raises(ValueError, match="identity"):
        calibrate_runtime(config, rows, model_artifact=model,
                          runtime=lambda *args, **kwargs: pytest.fail("must refuse before measurement"))


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
                  executor=Executor(), build_features=["eval"], model_role="pilot")
    with pytest.raises(ValueError, match="separate"):
        _install_weights_and_rebuild(weights_install_target="l1_weights.rs", **kwargs)
    _install_weights_and_rebuild(weights_install_target="l1_weights_example.rs", **kwargs)
    assert shipped.read_text() == "generalist"
    assert (layers / "l1_weights_example.rs").read_text() == "specialist"


def test_environment_freeze_uses_configured_python(tmp_path):
    calls = []
    binary = tmp_path / "custom-target" / "parapet-eval"
    binary.parent.mkdir()
    binary.write_bytes(b"built evaluator")
    class Executor:
        def run(self, args, **kwargs):
            calls.append(args)
            return type("Result", (), {"returncode": 0, "stdout": "synthetic==1", "stderr": ""})()
    collector = RuntimeIdentityCollector(workspace_root=tmp_path, trainer_script=Path("trainer"),
        parapet_eval_binary=Path("eval"), pg2_model_id="pilot", eval_config=Path("config"),
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
    binary = tmp_path / "custom-target" / "parapet-eval"
    binary.parent.mkdir()
    binary.write_bytes(b"built evaluator")
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
            executor=Executor(), model_role="pilot")
    assert protected.read_text() == "frozen"


def test_threshold_yaml_roundtrip_targets_rust_specialist_fields(tmp_path):
    import yaml
    from parapet_runner.runner import _set_l1_threshold
    policy, _, _ = calibration_fixture(tmp_path)
    source = {"layers": {"L1": {"mode": "block", "threshold": 0.,
        "generalist_solo_threshold": 3., "specialists": {"generalist": {"threshold": 0.}, "pilot": {"threshold": 0.}}}}}
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump(_set_l1_threshold(source, 2., policy=policy)))
    # These are precisely RawL1Config/RawL1SpecialistConfig's YAML fields.
    l1 = yaml.safe_load(path.read_text())["layers"]["L1"]
    assert l1["threshold"] == 0.
    assert l1["generalist_solo_threshold"] == 3.
    assert l1["specialists"] == {"generalist": {"threshold": 0.}, "pilot": {"threshold": 2.}}
    assert not (1. >= l1["specialists"]["pilot"]["threshold"])
    assert source["layers"]["L1"]["specialists"]["pilot"]["threshold"] == 0.
    for field, value in [("generalist_solo_threshold", 4.)]:
        changed = yaml.safe_load(yaml.safe_dump(source))
        changed["layers"]["L1"][field] = value
        with pytest.raises(ValueError, match="disagree"):
            _set_l1_threshold(changed, 2., policy=policy)
    source["layers"]["L1"]["specialists"]["generalist"]["threshold"] = -1.
    with pytest.raises(ValueError, match="disagree"):
        _set_l1_threshold(source, 2., policy=policy)
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
        "generalist_solo_threshold": 3., "specialists": {"generalist": {"threshold": 0.}, "pilot": {"threshold": 0.}}}}}))
    class Executor:
        def run(self, args, **kwargs):
            generated = yaml.safe_load(Path(args[args.index("--config") + 1]).read_text())
            l1 = generated["layers"]["L1"]
            assert l1["threshold"] == 0.
            assert l1["generalist_solo_threshold"] == 3.
            assert l1["specialists"]["pilot"]["threshold"] == 2.
            Path(args[args.index("--output") + 1]).write_text(json.dumps({"layers": [dict(
                layer="l1", tp=0, fp=0, fn_count=1, tn=0, precision=0., recall=0., f1=0., total=1)]}))
            return type("Result", (), {"returncode": 0})()
    evaluator = ParapetEvalEvaluator(parapet_eval_bin=Path("eval"), eval_config=source,
        workspace_root=tmp_path, command_executor=Executor(), model_role="pilot")
    result = evaluator.evaluate(model_artifact=model, split_path=split, threshold=2.,
        split_name="holdout", output_dir=tmp_path / "run", calibration_policy=config)
    assert result.threshold == 2.


@pytest.mark.parametrize("scenario", ["happy", "mismatch", "changed", "no_report", "old_dirs", "stale"])
def test_runtime_adapter_installs_pair_and_measures_generalist_runs(tmp_path, scenario):
    import yaml
    from parapet_runner.retention import weights_identity
    from parapet_runner.runner import ParapetEvalEvaluator
    layers = tmp_path / "crate" / "src" / "layers"
    layers.mkdir(parents=True)
    generalist = layers / "l1_weights.rs"
    generalist.write_text("frozen")
    config = TrainConfig(specialist="pilot", threshold_policy="constrained_retention",
        generalist_weights=generalist, generalist_weights_sha256=weights_identity(generalist))
    model = tmp_path / "candidate.rs"
    model.write_text("pub const BIAS: f64 = 0_f64;\npub static WEIGHTS: phf::Map<&'static str, f64> = phf::phf_map! {};\n")
    split = tmp_path / "val.jsonl"
    split.write_text(json.dumps(dict(id="a", label="malicious", content="<b>ab</b>")) + "\n")
    source = tmp_path / "source.yaml"
    source.write_text(yaml.safe_dump({"layers": {"L0": {"mode": "sanitize"}, "L1": {
        "mode": "block", "specialists": {"generalist": {"threshold": 0.}, "pilot": {"threshold": 9.}},
        "generalist_solo_threshold": 3.}}}))
    calls = []
    binary = tmp_path / "custom-target" / "parapet-eval"
    binary.parent.mkdir()
    binary.write_bytes(b"built evaluator")
    class Executor:
        def run(self, args, **kwargs):
            if args[0] == "cargo":
                assert (layers / "l1_weights_pilot.rs").read_bytes() == model.read_bytes()
                calls.append("build")
                assert "--message-format=json" in args
                return type("Result", (), {"returncode": 0, "stdout": json.dumps({
                    "reason": "compiler-artifact", "target": {"name": "parapet-eval", "kind": ["bin"]},
                    "executable": str(binary)})})()
            else:
                assert Path(args[0]) == binary.resolve()
                output = Path(args[args.index("--output") + 1])
                if scenario in ("no_report", "stale"):
                    return type("Result", (), {"returncode": 1, "args": args, "stdout": "", "stderr": ""})()
                if scenario == "old_dirs":
                    for name in ("_eval_aaa", "_eval_zzz"):
                        stale = output.parent.parent / name
                        stale.mkdir()
                        (stale / "eval.json").write_text("{}")
                payload = yaml.safe_load(Path(args[args.index("--config") + 1]).read_text())
                assert payload["layers"]["L0"] == {"mode": "sanitize"}
                l1 = payload["layers"]["L1"]
                names = set(l1["specialists"])
                calls.append(names)
                if names == {"generalist"}:
                    threshold = l1["specialists"]["generalist"]["threshold"]
                    assert l1["generalist_solo_threshold"] is None
                    assert l1["min_agree"] == 1
                    blocked = threshold == 0
                else:
                    assert names == {"generalist", "pilot"}
                    assert l1["specialists"]["generalist"]["threshold"] == 0
                    assert l1["specialists"]["pilot"]["threshold"] == 0
                    blocked = True
                Path(args[args.index("--output") + 1]).write_text(json.dumps({
                    "results": [{"case_id": "a", "actual": "blocked" if blocked else "allowed"}],
                    "layers": [dict(layer="l1", tp=int(blocked), fp=0, fn_count=int(not blocked), tn=0,
                                    precision=1., recall=float(blocked), f1=float(blocked), total=1)]}))
            if scenario == "changed":
                binary.write_bytes(b"replaced after first invocation")
            return type("Result", (), {"returncode": 1})()
    evaluator = ParapetEvalEvaluator(parapet_eval_bin=(tmp_path / "wrong" if scenario == "mismatch" else binary), eval_config=source,
        workspace_root=tmp_path, command_executor=Executor(), rust_crate_dir=tmp_path / "crate",
        weights_install_target="l1_weights_pilot.rs", model_role="specialist")
    if scenario == "stale":
        stale = tmp_path / "run" / "_calibration" / "1_generalist_0.0" / "_eval_calibration_0p0" / "eval.json"
        stale.parent.mkdir(parents=True)
        stale.write_text(json.dumps({"results": [{"case_id": "a", "actual": "blocked"}],
            "layers": [dict(layer="l1", tp=1, fp=0, fn_count=0, tn=0,
                precision=1., recall=1., f1=1., total=1)]}))
        with pytest.raises(FileExistsError):
            evaluator.calibrate(model_artifact=model, split_path=split, config=config, output_dir=tmp_path / "run")
        assert calls == ["build"]
        return
    if scenario in ("mismatch", "changed", "no_report"):
        error = FileNotFoundError if scenario == "no_report" else ValueError
        pattern = {"mismatch": "does not match Cargo", "changed": "artifact changed", "no_report": "did not produce"}[scenario]
        with pytest.raises(error, match=pattern):
            evaluator.calibrate(model_artifact=model, split_path=split, config=config, output_dir=tmp_path / "run")
        return
    result = evaluator.calibrate(model_artifact=model, split_path=split, config=config, output_dir=tmp_path / "run")
    assert result["parapet_eval_sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
    assert result["total"]["retention"] == 1
    assert result["deciding_band"]["total"] == 1
    assert calls == ["build", {"generalist"}, {"generalist"}, {"generalist", "pilot"}]
    evaluator.evaluate(model_artifact=model, split_path=split, threshold=0., split_name="holdout",
        output_dir=tmp_path / "holdout-run", calibration_policy=config)
    assert calls.count("build") == 1
    generalist.write_text("drift")
    with pytest.raises(ValueError, match="generalist.*(identity|pinned)"):
        evaluator.calibrate(model_artifact=model, split_path=split, config=config, output_dir=tmp_path / "other")

@pytest.mark.parametrize("mode", [None, "shadow"])
def test_calibration_refuses_nonblocking_engine_config(tmp_path, mode):
    from parapet_runner.runner import _set_l1_threshold
    policy, _, _ = calibration_fixture(tmp_path)
    l1 = {"generalist_solo_threshold": 3., "specialists": {
        "generalist": {"threshold": 0.}, "pilot": {"threshold": 0.}}}
    if mode is not None:
        l1["mode"] = mode
    with pytest.raises(ValueError, match="block mode"):
        _set_l1_threshold({"layers": {"L1": l1}}, 1., policy=policy)


@pytest.mark.parametrize("returncode", [0, 1])
def test_evaluator_refuses_existing_report_before_command(tmp_path, returncode):
    from parapet_runner.runner import ParapetEvalEvaluator
    config, model, rows = calibration_fixture(tmp_path)
    split = tmp_path / "split.jsonl"
    split.write_text(json.dumps(rows[0]) + "\n")
    source = tmp_path / "source.yaml"
    source.write_text("layers: {L1: {mode: block}}")
    stale = tmp_path / "run" / "_eval_calibration_0p0" / "eval.json"
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps({"results": [{"case_id": rows[0]["id"], "actual": "blocked"}],
        "layers": [dict(layer="l1", tp=1, fp=0, fn_count=0, tn=0, precision=1., recall=1., f1=1., total=1)]}))
    class Executor:
        def run(self, args, **kwargs):
            return type("Result", (), {"returncode": returncode, "args": args, "stdout": "", "stderr": ""})()
    evaluator = ParapetEvalEvaluator(parapet_eval_bin=Path("eval"), eval_config=source,
        workspace_root=tmp_path, command_executor=Executor())
    with pytest.raises(FileExistsError):
        evaluator.evaluate(model_artifact=model, split_path=split, threshold=0.,
            split_name="calibration", output_dir=tmp_path / "run")
