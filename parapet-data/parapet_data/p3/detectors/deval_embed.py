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


def calibrate(max_cos: float) -> float:
    """Max cosine similarity in [-1,1] -> [0,1] via the in-tree sigmoid (constants above)."""
    return clamp_unit(1.0 / (1.0 + math.exp(-EMBED_SIGMOID_A * (max_cos - EMBED_SIGMOID_B))))


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
    ):
        if embed_fn is None:
            raise ValueError("embed_fn is required")
        refs = [t for t in (reference_texts or []) if t and str(t).strip()]
        if not refs:
            raise ValueError("reference_texts must be non-empty (control-plane payload set)")
        self.embed_fn = embed_fn
        self.model_id = model_id
        # Embed + normalize the reference set ONCE at construction. A backend failure
        # here is a misconfiguration, not a per-event error: fail loud.
        ref_vecs = list(embed_fn(refs))
        if len(ref_vecs) != len(refs):
            raise ValueError(
                f"embed_fn returned {len(ref_vecs)} vectors for {len(refs)} reference texts"
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
        # Empty texts never reach the backend; mirror the L1/judge empty handling.
        send_idx = []
        for i, t in enumerate(texts):
            if not t or not str(t).strip():
                results[i] = self._result(None, "empty_event_text")
            else:
                send_idx.append(i)
        if not send_idx:
            return results

        try:
            vecs = list(self.embed_fn([texts[i] for i in send_idx]))
        except Exception as exc:  # backend is arbitrary; fail closed, do not crash the run
            return self._fill(results, send_idx, f"embed_failed: {type(exc).__name__}")

        if len(vecs) != len(send_idx):
            return self._fill(
                results, send_idx,
                f"embed_count_mismatch: got {len(vecs)} for {len(send_idx)}",
            )

        for pos, i in enumerate(send_idx):
            qn = _l2_normalize(vecs[pos])
            if qn is None:
                results[i] = self._result(None, "embed_zero_vector")
            elif len(qn) != self._dim:
                results[i] = self._result(
                    None, f"embed_dim_mismatch: {len(qn)} != ref {self._dim}",
                )
            else:
                max_cos = self._max_cos(qn)
                results[i] = self._result(
                    calibrate(max_cos), None,
                    rationale=f"max cosine {max_cos:.4f} over {len(self._ref)} refs",
                )
        return results

    def _fill(self, results: list, idxs: list, error: str) -> list:
        for i in idxs:
            results[i] = self._result(None, error)
        return results
