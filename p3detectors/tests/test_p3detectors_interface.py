"""Tests for the detector result schema and helpers."""
from p3detectors.interface import (
    FAMILY_GENERATIVE_MLX,
    DetectorResult,
    EventContext,
    clamp_unit,
)


def test_detector_result_to_dict_shape():
    r = DetectorResult(score=0.5, family=FAMILY_GENERATIVE_MLX, model_id="m",
                       detector_id="d", rationale="x")
    d = r.to_dict()
    assert d == {
        "score": 0.5, "family": "generative-mlx", "model_id": "m",
        "detector_id": "d", "rationale": "x", "error": None,
    }


def test_detector_result_error_path():
    r = DetectorResult(score=None, family=FAMILY_GENERATIVE_MLX, model_id="m",
                       detector_id="d", error="boom")
    d = r.to_dict()
    assert d["score"] is None and d["error"] == "boom" and d["rationale"] is None


def test_clamp_unit():
    assert clamp_unit(-0.2) == 0.0
    assert clamp_unit(1.5) == 1.0
    assert clamp_unit(0.3) == 0.3


def test_event_context_all_optional():
    c = EventContext()
    assert c.function is None and c.provenance_tag is None
    assert c.position is None and c.prior_text is None
