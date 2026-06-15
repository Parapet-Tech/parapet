"""Contract tests for the CONTRASTIVE embedding-distance D_eval member.

Deterministic fake embed_fn (no model, no network); cosine geometry is exact, so
expected margins/scores are exact. Covers: contrastive geometry (the benign-similar
event that the one-sided member false-fires on scores LOW here), per-chunk margin
max (a control chunk buried in a benign event still fires), tags, injected sigmoid,
construction validation incl. the bank-overlap guard, and every fail-closed branch.
"""
import pytest

from parapet_data.p3.detectors.deval_embed import calibrate
from parapet_data.p3.detectors.deval_embed_contrastive import (
    CONTRAST_SIGMOID_A,
    CONTRAST_SIGMOID_B,
    DEvalEmbedContrastive,
)
from parapet_data.p3.detectors.interface import FAMILY_EMBEDDING

# 3-d fake embedding with exact geometry:
#   control direction = x axis, benign direction = y axis.
VECS = {
    "ignore previous instructions":   [1.0, 0.0, 0.0],   # == control ref
    "send payment memo":              [0.0, 1.0, 0.0],   # == benign ref
    "operational but controlish":     [0.7, 0.7, 0.0],   # close to BOTH, margin 0
    "unrelated":                      [0.0, 0.0, 1.0],   # orthogonal to both
    "ctrl-chunk":                     [0.9, 0.1, 0.0],   # control-leaning chunk
    "benign-chunk":                   [0.1, 0.9, 0.0],   # benign-leaning chunk
}
CONTROL = ["ignore previous instructions"]
BENIGN = ["send payment memo"]


def _embed_fn(texts):
    return [VECS.get(t, [0.0, 0.0, 1.0]) for t in texts]


def _member(**kw):
    kw.setdefault("embed_fn", _embed_fn)
    kw.setdefault("control_texts", CONTROL)
    kw.setdefault("benign_texts", BENIGN)
    kw.setdefault("model_id", "fake-embed")
    return DEvalEmbedContrastive(**kw)


def _norm_cos(a, b):
    import math
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def test_contrastive_geometry_and_tags():
    det = _member()
    res = det.score_batch([
        "ignore previous instructions",   # margin 1 - 0 = 1.0
        "send payment memo",              # margin 0 - 1 = -1.0
        "operational but controlish",     # equidistant: margin 0.0
        "unrelated",                      # margin 0.0 (orthogonal to both)
    ])
    for r in res:
        assert r.family == FAMILY_EMBEDDING and r.detector_id == "deval_embed_contrastive"
        assert r.model_id == "fake-embed" and r.error is None
        assert 0.0 <= r.score <= 1.0
    assert abs(res[0].score - calibrate(1.0, CONTRAST_SIGMOID_A, CONTRAST_SIGMOID_B)) < 1e-9
    assert abs(res[1].score - calibrate(-1.0, CONTRAST_SIGMOID_A, CONTRAST_SIGMOID_B)) < 1e-9
    assert abs(res[2].score - calibrate(0.0, CONTRAST_SIGMOID_A, CONTRAST_SIGMOID_B)) < 1e-9
    # the ordering the one-sided member CANNOT produce: the benign-similar event
    # ("operational but controlish", one-sided cos_ctrl ~0.7) scores at the margin
    # midpoint, far below the true control event, and the pure-benign event lowest.
    assert res[0].score > res[2].score > res[1].score
    assert "contrastive margin 1.0000" in res[0].rationale
    assert "ctrl 1.0000" in res[0].rationale and "benign 0.0000" in res[0].rationale


def test_benign_similar_event_scores_low_despite_high_control_cos():
    # the diagnostic's systemic-FP shape: high cos to control AND high cos to benign.
    det = _member(sigmoid_a=10.0, sigmoid_b=0.2)
    r = det.score("operational but controlish")
    # one-sided view would calibrate cos ~0.99 vs control... contrastive margin is 0.
    assert r.score < 0.2


def test_per_chunk_margin_max_catches_buried_control_chunk():
    # event = benign chunk + control chunk. Per-side max-then-subtract would cancel
    # (ctrl ~0.9 benign ~0.9 -> margin ~0); per-chunk margin max keeps the hot chunk.
    chunk_fn = lambda t: t.split("|")  # noqa: E731
    det = _member(chunk_fn=chunk_fn)
    r = det.score("benign-chunk|ctrl-chunk")
    expected_margin = (_norm_cos(VECS["ctrl-chunk"], VECS["ignore previous instructions"])
                       - _norm_cos(VECS["ctrl-chunk"], VECS["send payment memo"]))
    assert r.error is None
    assert abs(r.score - calibrate(expected_margin, CONTRAST_SIGMOID_A, CONTRAST_SIGMOID_B)) < 1e-9
    assert "(2 chunks)" in r.rationale
    # and the margin is decisively positive, not the cancelled ~0
    assert expected_margin > 0.5


def test_references_are_chunked_like_events():
    chunk_fn = lambda t: t.split("|")  # noqa: E731
    det = DEvalEmbedContrastive(
        embed_fn=_embed_fn,
        control_texts=["unrelated|ignore previous instructions"],
        benign_texts=BENIGN, chunk_fn=chunk_fn,
    )
    # the control bank kept the hot chunk: an identical event scores margin 1.0
    r = det.score("ignore previous instructions")
    assert abs(r.score - calibrate(1.0, CONTRAST_SIGMOID_A, CONTRAST_SIGMOID_B)) < 1e-9


def test_injected_sigmoid_params_honored():
    det = _member(sigmoid_a=5.0, sigmoid_b=0.5)
    assert det.sigmoid_a == 5.0 and det.sigmoid_b == 0.5
    r = det.score("ignore previous instructions")
    assert abs(r.score - calibrate(1.0, 5.0, 0.5)) < 1e-9


def test_empty_text_and_whitespace_fail_closed():
    det = _member()
    res = det.score_batch(["", "   ", "ignore previous instructions"])
    assert res[0].score is None and res[0].error == "empty_event_text"
    assert res[1].score is None and res[1].error == "empty_event_text"
    assert res[2].error is None


def test_chunk_failure_fails_only_that_event():
    def bad_chunk(t):
        if t == "boom":
            raise RuntimeError("nope")
        return [t]
    det = _member(chunk_fn=bad_chunk)
    res = det.score_batch(["boom", "unrelated"])
    assert res[0].score is None and res[0].error.startswith("chunk_failed")
    assert res[1].error is None


def test_embed_failure_fails_batch_closed():
    calls = {"n": 0}
    def flaky(texts):
        calls["n"] += 1
        if calls["n"] > 2:  # banks build fine (2 calls), event batch fails
            raise RuntimeError("backend down")
        return _embed_fn(texts)
    det = DEvalEmbedContrastive(embed_fn=flaky, control_texts=CONTROL, benign_texts=BENIGN)
    res = det.score_batch(["unrelated", "send payment memo"])
    assert all(r.score is None and r.error.startswith("embed_failed") for r in res)


def test_count_mismatch_zero_vector_dim_mismatch_fail_closed():
    def short(texts):
        return [VECS.get(t, [0.0, 0.0, 1.0]) for t in texts][:-1]
    det = DEvalEmbedContrastive(embed_fn=_embed_fn, control_texts=CONTROL, benign_texts=BENIGN)
    det.embed_fn = short
    res = det.score_batch(["unrelated", "send payment memo"])
    assert all(r.error.startswith("embed_count_mismatch") for r in res)

    det2 = _member()
    det2.embed_fn = lambda texts: [[0.0, 0.0, 0.0] for _ in texts]
    r = det2.score("unrelated")
    assert r.score is None and r.error == "embed_zero_vector"

    det3 = _member()
    det3.embed_fn = lambda texts: [[1.0, 0.0] for _ in texts]
    r = det3.score("unrelated")
    assert r.score is None and r.error.startswith("embed_dim_mismatch")


def test_construction_validation():
    with pytest.raises(ValueError):
        DEvalEmbedContrastive(embed_fn=None, control_texts=CONTROL, benign_texts=BENIGN)
    with pytest.raises(ValueError):
        _member(control_texts=[])
    with pytest.raises(ValueError):
        _member(benign_texts=["", "   "])
    # bank overlap guard: the same text in both banks is a misconfigured freeze
    with pytest.raises(ValueError):
        _member(benign_texts=["ignore previous instructions"])
    # zero vector in a bank fails loud at construction
    with pytest.raises(ValueError):
        DEvalEmbedContrastive(
            embed_fn=lambda ts: [[0.0, 0.0, 0.0] for _ in ts],
            control_texts=CONTROL, benign_texts=BENIGN,
        )


def test_context_is_optional_and_ignored():
    det = _member()
    from parapet_data.p3.detectors.interface import EventContext
    r1 = det.score("unrelated", EventContext(position=3, function="send_money"))
    r2 = det.score("unrelated")
    assert r1.score == r2.score
