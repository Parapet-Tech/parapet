"""D_eval member: embedding-distance retrieval detector (embedding family).

Scores a per-event surface_signal as the calibrated max cosine similarity of the
event-span embedding to a reference set of known control-plane payloads (mined from
carrier injections + the MHJ control-plane minority + handcrafted; see
detector_ensemble_spec.md section 3). Intuition: an event that reads like a known
control-plane move sits close in embedding space to the reference payloads; a benign
event sits far. This is the retrieval-family foil in the cross-family D_eval ensemble.

Backend-agnostic: the embedding model is INJECTED as embed_fn(texts) -> list[vector]
(mirrors deval_l1.run_fn and mlx_judge.complete_fn). This keeps the member testable
with a deterministic fake embedder and lets the concrete MLX-native backend be chosen
separately.

Chunk-and-max-risk (detector_ensemble_spec.md section 3): the encoder has a hard token
window (bge-small: 512), so an event span longer than the window is split into
overlapping chunks via an injected chunk_fn and scored as the MAX cosine over its
chunks, NOT silently truncated (truncation drops end-of-span signal, exactly where a
slow-burn control-plane move can sit in a long tool output). The default chunk_fn is
identity (one chunk == whole text); the token-aware splitter lives in the backend
(mlx_embed_backend.make_mlx_chunk_fn) where the tokenizer is. References are chunked
the same way, so a long reference payload is not truncated either.

Scouting note (2026-06-06): there is no clean lightweight MLX-native embedding package.
mlx_lm has no embeddings; mlx-embeddings is torch-free but pulls mlx-vlm (+ datasets,
opencv, audio, fastapi), which AGENTS.md says to defer; sentence-transformers pulls
torch (deferred fallback). The clean route is a hand-rolled mlx encoder (mlx +
safetensors + the present transformers tokenizer, no new deps). The backend is an
author decision, so this module never imports it: callers pass embed_fn.

FAILS CLOSED: an embed_fn error, a wrong return count, a zero vector, or a dim mismatch
yields error DetectorResults (score=None), never a silent 0.0.

Calibration: cosine similarity in [-1,1] -> surface_signal in [0,1] via the in-tree
sigmoid (constants below). Provisional; final values come from the tau_event
calibration sweep (detector_ensemble_spec.md section 6).
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

from parapet_data.p3.detectors.interface import (
    FAMILY_EMBEDDING,
    DetectorResult,
    EventContext,
    clamp_unit,
)

# Calibrated on the tau_event calibration batch (section 6); provisional defaults.
# P = 1 / (1 + exp(-A * (max_cos - B))). B ~ the benign/attack cosine crossover,
# A the sharpness. Update both once the calibration sweep picks them.
EMBED_SIGMOID_A = 12.0
EMBED_SIGMOID_B = 0.5

DETECTOR_ID = "deval_embed_cos"
DEFAULT_MODEL_ID = "mlx-embed-unset"

# embed_fn maps a sequence of texts to a same-length sequence of equal-length vectors.
EmbedFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]

# chunk_fn maps one text to its overlapping sub-spans. The default is identity (one
# chunk == the whole text); a token-aware chunk_fn (mlx_embed_backend.make_mlx_chunk_fn)
# splits spans longer than the encoder window so signal at the end of a long span is not
# truncated away (detector_ensemble_spec.md section 3, chunk-and-max-risk).
ChunkFn = Callable[[str], Sequence[str]]


def _identity_chunk(text: str) -> list:
    return [text]


def calibrate(max_cos: float, a: float = EMBED_SIGMOID_A, b: float = EMBED_SIGMOID_B) -> float:
    """Max cosine similarity in [-1,1] -> [0,1] via the sigmoid P = 1/(1+exp(-a*(cos-b))).

    a/b default to the provisional module constants but are overridable: the tau_event
    calibration sweep FITS them from the per-class raw-cosine distribution (the provisional
    defaults saturate for high-baseline encoders like bge-small and destroy member
    comparability, so a fit is required before any tau LOCK; detector_ensemble_spec.md s3).
    """
    return clamp_unit(1.0 / (1.0 + math.exp(-a * (max_cos - b))))


def _l2_normalize(vec: Sequence[float]) -> Optional[list]:
    """Return the unit-norm vector as a list of floats, or None for a zero/empty vector."""
    try:
        floats = [float(x) for x in vec]
    except (TypeError, ValueError):
        return None
    if not floats:
        return None
    norm = math.sqrt(sum(x * x for x in floats))
    if norm == 0.0:
        return None
    return [x / norm for x in floats]


class DEvalEmbed:
    family = FAMILY_EMBEDDING
    detector_id = DETECTOR_ID

    def __init__(
        self,
        *,
        embed_fn: EmbedFn,
        reference_texts: Sequence[str],
        model_id: str = DEFAULT_MODEL_ID,
        chunk_fn: Optional[ChunkFn] = None,
        sigmoid_a: float = EMBED_SIGMOID_A,
        sigmoid_b: float = EMBED_SIGMOID_B,
    ):
        if embed_fn is None:
            raise ValueError("embed_fn is required")
        refs = [t for t in (reference_texts or []) if t and str(t).strip()]
        if not refs:
            raise ValueError("reference_texts must be non-empty (control-plane payload set)")
        self.embed_fn = embed_fn
        self.chunk_fn = chunk_fn if chunk_fn is not None else _identity_chunk
        self.model_id = model_id
        self.sigmoid_a = sigmoid_a
        self.sigmoid_b = sigmoid_b
        # Chunk each reference the same way events are chunked, so a long reference
        # payload is not silently truncated either; each chunk becomes an independent
        # reference vector. A broken chunk_fn here is a misconfiguration: let it raise.
        ref_chunks: list = []
        for t in refs:
            ref_chunks.extend(self._chunks_of(t))
        if not ref_chunks:
            raise ValueError("reference_texts produced no non-empty chunks")
        # Embed + normalize the reference set ONCE at construction. A backend failure
        # here is a misconfiguration, not a per-event error: fail loud.
        ref_vecs = list(embed_fn(ref_chunks))
        if len(ref_vecs) != len(ref_chunks):
            raise ValueError(
                f"embed_fn returned {len(ref_vecs)} vectors for {len(ref_chunks)} reference chunks"
            )
        normed = []
        for v in ref_vecs:
            nv = _l2_normalize(v)
            if nv is None:
                raise ValueError("embed_fn produced an empty or zero reference vector")
            normed.append(nv)
        self._ref = normed
        self._dim = len(normed[0])
        if any(len(v) != self._dim for v in normed):
            raise ValueError("reference embeddings have inconsistent dimensions")

    def _chunks_of(self, text: str) -> list:
        """Apply chunk_fn and drop empty/blank chunks. May raise if chunk_fn raises;
        callers decide whether that is a construction error or a per-event fail-close."""
        return [c for c in self.chunk_fn(text) if c and str(c).strip()]

    def _result(self, score, error, rationale=None) -> DetectorResult:
        return DetectorResult(
            score=score, family=self.family, model_id=self.model_id,
            detector_id=self.detector_id, rationale=rationale, error=error,
        )

    def _max_cos(self, qn: list) -> float:
        """Max cosine similarity of a unit-norm query to the unit-norm reference set."""
        best = -1.0
        for r in self._ref:
            dot = 0.0
            for a, b in zip(qn, r):
                dot += a * b
            if dot > best:
                best = dot
        return best

    def score(self, event_text: str, context: Optional[EventContext] = None) -> DetectorResult:
        return self.score_batch([event_text], [context])[0]

    def score_batch(self, texts: list, contexts=None) -> list:
        n = len(texts)
        results: list = [None] * n
        # Chunk each non-empty event; embed ALL chunks in one backend call, then take
        # the max cosine over each event's chunks. Empty texts never reach the backend.
        chunk_texts: list = []
        owner: list = []  # event index for each chunk in chunk_texts
        for i, t in enumerate(texts):
            if not t or not str(t).strip():
                results[i] = self._result(None, "empty_event_text")
                continue
            try:
                chunks = self._chunks_of(t)
            except Exception as exc:  # chunk_fn is arbitrary; fail this event closed
                results[i] = self._result(None, f"chunk_failed: {type(exc).__name__}")
                continue
            if not chunks:
                results[i] = self._result(None, "empty_event_text")
                continue
            for c in chunks:
                chunk_texts.append(c)
                owner.append(i)
        if not chunk_texts:
            return results

        try:
            vecs = list(self.embed_fn(chunk_texts))
        except Exception as exc:  # backend is arbitrary; fail closed, do not crash the run
            return self._fill(results, sorted(set(owner)), f"embed_failed: {type(exc).__name__}")

        if len(vecs) != len(chunk_texts):
            return self._fill(
                results, sorted(set(owner)),
                f"embed_count_mismatch: got {len(vecs)} for {len(chunk_texts)}",
            )

        # Group chunk positions by event, then aggregate by MAX cosine. A single bad
        # chunk (zero / dim-mismatched vector) fails the whole event closed, never a
        # silent drop that could discard the chunk carrying the signal.
        by_event: dict = {}
        for pos, i in enumerate(owner):
            by_event.setdefault(i, []).append(pos)

        for i, positions in by_event.items():
            best = -1.0
            err = None
            for pos in positions:
                qn = _l2_normalize(vecs[pos])
                if qn is None:
                    err = "embed_zero_vector"
                    break
                if len(qn) != self._dim:
                    err = f"embed_dim_mismatch: {len(qn)} != ref {self._dim}"
                    break
                c = self._max_cos(qn)
                if c > best:
                    best = c
            if err:
                results[i] = self._result(None, err)
                continue
            rationale = f"max cosine {best:.4f} over {len(self._ref)} refs"
            if len(positions) > 1:
                rationale += f" ({len(positions)} chunks)"
            results[i] = self._result(
                calibrate(best, self.sigmoid_a, self.sigmoid_b), None, rationale=rationale,
            )
        return results

    def _fill(self, results: list, idxs: list, error: str) -> list:
        for i in idxs:
            results[i] = self._result(None, error)
        return results
