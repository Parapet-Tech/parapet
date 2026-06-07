"""Contract tests for the D_eval conservative-max ensemble + cross-family assertion.

Uses fake member detectors (no models, no network), so tests run anywhere. Covers the
conservative max, fail-closed-on-any-member-error, the all-empty case, distinct-family
construction, the cross-family assertion, and per-member raw access.
"""
import pytest

from parapet_data.p3.detectors.ensemble import (
    DEvalEnsemble,
    assert_cross_family,
    cross_family_ok,
)
from parapet_data.p3.detectors.interface import (
    FAMILY_EMBEDDING,
    FAMILY_ENSEMBLE,
    FAMILY_GENERATIVE_MLX,
    FAMILY_LINEAR,
    DetectorResult,
)


class FakeDetector:
    """A member that returns preset scores per text, or a fixed error for every text."""

    def __init__(self, family, detector_id, scores=None, error=None):
        self.family = family
        self.detector_id = detector_id
        self.model_id = detector_id
        self._scores = scores or {}
        self._error = error

    def score(self, text, context=None):
        return self.score_batch([text], [context])[0]

    def score_batch(self, texts, contexts=None):
        out = []
        for t in texts:
            if not t or not str(t).strip():
                out.append(self._r(None, error="empty_event_text"))
            elif self._error:
                out.append(self._r(None, error=self._error))
            else:
                s = self._scores.get(t, 0.0)
                out.append(self._r(s, rationale=f"s={s}"))
        return out

    def _r(self, score, rationale=None, error=None):
        return DetectorResult(
            score=score, family=self.family, model_id=self.model_id,
            detector_id=self.detector_id, rationale=rationale, error=error,
        )


def _l1(scores=None, error=None):
    return FakeDetector(FAMILY_LINEAR, "deval_l1_generalist", scores=scores, error=error)


def _embed(scores=None, error=None):
    return FakeDetector(FAMILY_EMBEDDING, "deval_embed_cos", scores=scores, error=error)


# ---- conservative max ----

def test_conservative_max_picks_highest_member():
    ens = DEvalEnsemble([_l1({"x": 0.3}), _embed({"x": 0.7})])
    r = ens.score("x")
    assert r.error is None
    assert r.score == 0.7  # max over members
    assert r.family == FAMILY_ENSEMBLE and r.detector_id == "deval_ensemble_max"
    assert "from deval_embed_cos" in r.rationale
    assert "deval_l1_generalist=0.3000" in r.rationale and "deval_embed_cos=0.7000" in r.rationale


def test_max_low_only_when_all_members_low():
    # the slow-burn case: every member sub-threshold -> aggregate sub-threshold
    ens = DEvalEnsemble([_l1({"x": 0.10}), _embed({"x": 0.18})])
    r = ens.score("x")
    assert r.error is None and r.score == 0.18


def test_model_id_lists_members():
    ens = DEvalEnsemble([_l1(), _embed()])
    assert ens.model_id == "max(deval_l1_generalist,deval_embed_cos)"


def test_score_single_routes_through_batch():
    ens = DEvalEnsemble([_l1({"a": 0.2}), _embed({"a": 0.9})])
    assert ens.score("a").score == 0.9


# ---- fail-closed ----

def test_any_member_error_fails_event_closed():
    # embedding backend down on a real event: do NOT silently max over L1 alone, since
    # that could understate the signal and falsely call the event slow-burn.
    ens = DEvalEnsemble([_l1({"x": 0.3}), _embed(error="embed_failed: RuntimeError")])
    r = ens.score("x")
    assert r.score is None
    assert r.error.startswith("member_error:")
    assert "deval_embed_cos=embed_failed: RuntimeError" in r.error
    # the healthy member must NOT leak through as the score
    assert "deval_l1_generalist" not in r.error


def test_all_members_empty_reported_as_empty():
    ens = DEvalEnsemble([_l1(), _embed()])
    r = ens.score("   ")
    assert r.score is None and r.error == "empty_event_text"


def test_mixed_batch_scores_per_row():
    ens = DEvalEnsemble([_l1({"good": 0.4, "other": 0.5}), _embed({"good": 0.6})])
    res = ens.score_batch(["good", "", "other"])
    assert res[0].error is None and res[0].score == 0.6      # max(0.4, 0.6)
    assert res[1].error == "empty_event_text"                # empty
    assert res[2].error is None and res[2].score == 0.5      # max(0.5, 0.0 embed default)


# ---- construction guards ----

def test_requires_at_least_one_member():
    with pytest.raises(ValueError):
        DEvalEnsemble([])


def test_members_must_be_distinct_families():
    with pytest.raises(ValueError, match="cross-family"):
        DEvalEnsemble([_l1(), _l1()])  # two linear members is not a cross-family ensemble


# ---- cross-family assertion (D_gen vs D_eval members) ----

def test_cross_family_ok_when_disjoint():
    dgen = FakeDetector(FAMILY_GENERATIVE_MLX, "dgen_mlx_judge")
    ok, msg = cross_family_ok(dgen, [_l1(), _embed()])
    assert ok and "disjoint" in msg
    assert_cross_family(dgen, [_l1(), _embed()])  # does not raise


def test_cross_family_violation_when_dgen_shares_family():
    # a generative D_gen and a generative D_eval member would share a blind spot
    dgen = FakeDetector(FAMILY_GENERATIVE_MLX, "dgen_mlx_judge")
    judge_member = FakeDetector(FAMILY_GENERATIVE_MLX, "deval_judge")
    ok, msg = cross_family_ok(dgen, [_l1(), judge_member])
    assert not ok and "violation" in msg
    with pytest.raises(ValueError, match="violation"):
        assert_cross_family(dgen, [_l1(), judge_member])


# ---- raw per-member access (for the tau_event sweep) ----

def test_score_members_returns_aligned_raw_results():
    ens = DEvalEnsemble([_l1({"a": 0.2, "b": 0.3}), _embed({"a": 0.8, "b": 0.1})])
    raw = ens.score_members(["a", "b"])
    assert set(raw) == {"deval_l1_generalist", "deval_embed_cos"}
    assert [r.score for r in raw["deval_l1_generalist"]] == [0.2, 0.3]
    assert [r.score for r in raw["deval_embed_cos"]] == [0.8, 0.1]
