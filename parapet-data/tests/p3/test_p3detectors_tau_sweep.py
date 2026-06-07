import json

import pytest

from parapet_data.p3.detectors.interface import (
    FAMILY_EMBEDDING,
    FAMILY_ENSEMBLE,
    FAMILY_LINEAR,
    DetectorResult,
)
from parapet_data.p3.detectors.tau_sweep import (
    CalibrationEvent,
    event_from_record,
    load_events_jsonl,
    score_calibration_events,
    summarize_scores,
    write_jsonl,
)


class FakeEnsemble:
    family = FAMILY_ENSEMBLE
    detector_id = "deval_ensemble_max"
    model_id = "max(fake_l1,fake_embed)"

    def score_batch_with_members(self, texts, contexts=None):
        l1 = []
        embed = []
        aggregate = []
        for text in texts:
            if text == "bad":
                l1.append(self._r(None, FAMILY_LINEAR, "fake_l1", error="l1_bad"))
                embed.append(self._r(0.1, FAMILY_EMBEDDING, "fake_embed"))
                aggregate.append(self._r(None, FAMILY_ENSEMBLE, self.detector_id,
                                         error="member_error: fake_l1=l1_bad"))
                continue
            l1_score = 0.2 if "benign" in text else 0.7
            embed_score = 0.1 if "benign" in text else 0.9
            l1.append(self._r(l1_score, FAMILY_LINEAR, "fake_l1"))
            embed.append(self._r(embed_score, FAMILY_EMBEDDING, "fake_embed"))
            aggregate.append(self._r(max(l1_score, embed_score), FAMILY_ENSEMBLE,
                                     self.detector_id, rationale="max"))
        return aggregate, {"fake_l1": l1, "fake_embed": embed}

    def _r(self, score, family, detector_id, *, error=None, rationale=None):
        return DetectorResult(
            score=score,
            family=family,
            model_id=detector_id,
            detector_id=detector_id,
            error=error,
            rationale=rationale,
        )


def test_event_from_record_parses_context():
    ev = event_from_record({
        "event_id": "e1",
        "class_label": "benign_other",
        "trajectory_id": "t1",
        "event_text": "benign text",
        "context": {"function": "send_email", "position": 3},
    }, line_no=7)
    assert ev.event_id == "e1"
    assert ev.context.function == "send_email"
    assert ev.context.position == 3


def test_event_from_record_rejects_missing_fields():
    with pytest.raises(ValueError, match="event_text"):
        event_from_record({"class_label": "x"}, line_no=1)
    with pytest.raises(ValueError, match="class_label"):
        event_from_record({"event_text": "x"}, line_no=2)


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps({"class_label": "benign_other", "event_text": "benign one"}) + "\n"
    )
    events = load_events_jsonl(str(path))
    assert len(events) == 1
    assert events[0].event_id == "ev1"


def test_score_calibration_events_preserves_members():
    events = [
        CalibrationEvent("e1", "benign_other", "benign text"),
        CalibrationEvent("e2", "naive_attack", "attack text"),
        CalibrationEvent("e3", "benign_mention", "bad"),
    ]
    scored = score_calibration_events(events, FakeEnsemble())
    assert [r["score"] for r in scored] == [0.2, 0.9, None]
    assert scored[1]["members"]["fake_l1"]["score"] == 0.7
    assert scored[1]["members"]["fake_embed"]["score"] == 0.9
    assert scored[2]["error"] == "member_error: fake_l1=l1_bad"


def test_write_jsonl(tmp_path):
    out = tmp_path / "scores.jsonl"
    write_jsonl(str(out), [{"b": 2, "a": 1}])
    assert out.read_text() == '{"a": 1, "b": 2}\n'


def test_summarize_scores_per_class_histograms_and_sweep():
    scored = [
        {"class_label": "benign_other", "score": 0.1},
        {"class_label": "benign_other", "score": 0.3},
        {"class_label": "benign_other", "score": None, "error": "member_error"},
        {"class_label": "naive_attack", "score": 0.8},
        {"class_label": "naive_attack", "score": 0.9},
    ]
    summary = summarize_scores(scored, thresholds=[0.25, 0.75], bins=4)
    assert summary["classes"] == ["benign_other", "naive_attack"]
    benign = summary["histograms"]["benign_other"]
    assert benign["n_events"] == 3
    assert benign["n_scored"] == 2
    assert benign["n_errors"] == 1
    assert benign["stats"]["median"] == 0.2
    assert [b["count"] for b in benign["bins"]] == [1, 1, 0, 0]
    assert summary["sweep"]["benign_other"][0]["below_tau"] == 1
    assert summary["sweep"]["benign_other"][0]["at_or_above_tau"] == 1
    assert summary["sweep"]["naive_attack"][1]["below_tau"] == 0
