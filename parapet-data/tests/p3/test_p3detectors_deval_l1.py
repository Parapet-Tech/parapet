"""Contract tests for the L1 D_eval member.

Uses a FAKE parapet-eval script that emits CSV, so tests never need the real Rust
binary. Covers the happy path, calibration, and every fail-closed branch (column
drift, subprocess failure, missing row, bad value), plus batch-only behavior.
"""
from parapet_data.p3.detectors.deval_l1 import DEvalL1, L1_SIGMOID_A, L1_SIGMOID_B, calibrate
from parapet_data.p3.detectors.interface import FAMILY_LINEAR


def _fake_binary(tmp_path, csv_body="", exit_code=0):
    """Write an executable fake parapet-eval that writes csv_body to --output."""
    script = tmp_path / "fake_parapet_eval.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "args = sys.argv\n"
        f"exit_code = {exit_code}\n"
        "if exit_code != 0:\n"
        "    sys.stderr.write('simulated l1 failure\\n')\n"
        "    sys.exit(exit_code)\n"
        "out = args[args.index('--output') + 1]\n"
        "with open(out, 'w') as f:\n"
        f"    f.write({csv_body!r})\n"
        "sys.exit(0)\n"
    )
    script.chmod(0o755)
    return str(script)


def _l1(tmp_path, binary):
    return DEvalL1(
        engine_root=str(tmp_path), binary_path=binary,
        config_path=str(tmp_path / "cfg.yaml"), engine_sha="testsha123",
    )


HAPPY = (
    "case_id,label,source,generalist,roleplay_jailbreak\n"
    "ev0,unknown,p3_events,2.091532,-0.1\n"
    "ev1,unknown,p3_events,-0.532872,-0.1\n"
)


def test_happy_batch(tmp_path):
    l1 = _l1(tmp_path, _fake_binary(tmp_path, HAPPY))
    res = l1.score_batch(["attacky text", "benign text"])
    assert len(res) == 2
    for r in res:
        assert r.error is None and 0.0 <= r.score <= 1.0
        assert r.family == FAMILY_LINEAR and r.detector_id == "deval_l1_generalist"
        assert r.model_id == "parapet-l1-generalist"
        assert r.engine_sha == "testsha123"
        assert r.binary_path.endswith("fake_parapet_eval.py")
    assert abs(res[0].score - calibrate(2.091532)) < 1e-9
    assert abs(res[1].score - calibrate(-0.532872)) < 1e-9
    assert "generalist raw margin" in res[0].rationale


def test_calibration_constants_named_and_correct():
    assert L1_SIGMOID_A == 0.6 and L1_SIGMOID_B == 0.0
    assert abs(calibrate(0.0) - 0.5) < 1e-9          # midpoint at raw 0
    assert calibrate(5.0) > 0.9 and calibrate(-5.0) < 0.1  # monotone, saturating


def test_column_drift_fails_closed(tmp_path):
    drift = "case_id,label,source,roleplay_jailbreak\nev0,unknown,p3_events,0.5\n"
    l1 = _l1(tmp_path, _fake_binary(tmp_path, drift))
    res = l1.score_batch(["x"])
    assert res[0].score is None  # NOT a silent 0.0
    assert res[0].error.startswith("l1_column_drift") and "generalist" in res[0].error
    assert res[0].engine_sha == "testsha123"  # provenance on error results too


def test_subprocess_failure_fails_closed(tmp_path):
    l1 = _l1(tmp_path, _fake_binary(tmp_path, exit_code=3))
    res = l1.score_batch(["x", "y"])
    assert all(r.score is None and r.error.startswith("l1_subprocess_failed") for r in res)


def test_missing_row_fails_closed(tmp_path):
    only0 = "case_id,label,source,generalist\nev0,unknown,p3_events,1.0\n"
    l1 = _l1(tmp_path, _fake_binary(tmp_path, only0))
    res = l1.score_batch(["a", "b"])  # ev0 present, ev1 absent
    assert res[0].error is None and res[0].score is not None
    assert res[1].score is None and res[1].error == "l1_missing_row"


def test_bad_value_fails_closed(tmp_path):
    bad = "case_id,label,source,generalist\nev0,unknown,p3_events,n/a\n"
    l1 = _l1(tmp_path, _fake_binary(tmp_path, bad))
    res = l1.score_batch(["x"])
    assert res[0].score is None and res[0].error == "l1_bad_value"


def test_empty_text_skipped_binary_still_runs(tmp_path):
    body = "case_id,label,source,generalist\nev1,unknown,p3_events,0.8\n"
    l1 = _l1(tmp_path, _fake_binary(tmp_path, body))
    res = l1.score_batch(["   ", "real"])
    assert res[0].score is None and res[0].error == "empty_event_text"
    assert res[1].error is None and abs(res[1].score - calibrate(0.8)) < 1e-9


def test_all_empty_skips_binary(tmp_path):
    calls = []
    l1 = DEvalL1(engine_root=str(tmp_path), engine_sha="s", run_fn=lambda ds, src, out: calls.append(1))
    res = l1.score_batch(["", "  "])
    assert all(r.error == "empty_event_text" for r in res)
    assert calls == []  # binary never invoked


def test_batch_only_single_invocation(tmp_path):
    calls = []

    def run_fn(ds_dir, source, out_csv):
        calls.append((ds_dir, source, out_csv))
        with open(out_csv, "w") as f:
            f.write("case_id,label,source,generalist\n"
                    "ev0,u,s,0.1\nev1,u,s,0.2\nev2,u,s,0.3\n")

    l1 = DEvalL1(engine_root=str(tmp_path), engine_sha="s", run_fn=run_fn)
    res = l1.score_batch(["a", "b", "c"])
    assert len(calls) == 1  # ONE process for three events, not three
    assert [round(r.score, 6) for r in res] == [round(calibrate(v), 6) for v in (0.1, 0.2, 0.3)]


def test_score_single_routes_through_batch(tmp_path):
    body = "case_id,label,source,generalist\nev0,u,s,1.5\n"
    l1 = _l1(tmp_path, _fake_binary(tmp_path, body))
    r = l1.score("hello")
    assert r.error is None and abs(r.score - calibrate(1.5)) < 1e-9
