"""Hand-rolled MLX-native encoder backend for the embedding-distance D_eval member.

Provides a concrete `embed_fn` for `deval_embed.DEvalEmbed` by running a small
BGE/E5-style BERT encoder in MLX. No torch, no sentence-transformers, no mlx-vlm:
just `mlx` + `safetensors` (via mlx's loader) + the already-present `transformers`
tokenizer. This is the author-chosen clean backend (see detector_ensemble_spec.md
section 7 scouting note).

Design / guardrails:
- mlx and transformers are imported LAZILY (inside the class), so importing this module
  never requires them. The pooling/normalization math is pure numpy and unit-testable
  with no mlx, no weights, no network.
- An explicit local model path (a directory) is REQUIRED. A bare repo id is only
  fetched when allow_download=True; otherwise a clear error is raised. Nothing is ever
  silently downloaded.
- Tokenizer loads local_files_only unless allow_download=True.

Scope: BERT-convention encoders (positions 0..L-1, learned position embeddings),
which covers bge-small-en-v1.5 and e5-small-v2. XLM-RoBERTa-based encoders (e.g.
multilingual-e5) use a position offset and are out of scope for this adapter; point it
at a BERT-based encoder.
"""
from __future__ import annotations

import json
import math
import os
from typing import Callable, Optional, Sequence

import numpy as np

POOL_MEAN = "mean"
POOL_CLS = "cls"

# Standard HF BERT weight-key template (no prefix). _resolve_prefix() also tries "bert.".
_K = {
    "word": "embeddings.word_embeddings.weight",
    "pos": "embeddings.position_embeddings.weight",
    "tok_type": "embeddings.token_type_embeddings.weight",
    "emb_ln_w": "embeddings.LayerNorm.weight",
    "emb_ln_b": "embeddings.LayerNorm.bias",
}


def _layer_keys(prefix: str, i: int) -> dict:
    p = f"{prefix}encoder.layer.{i}."
    return {
        "q_w": p + "attention.self.query.weight", "q_b": p + "attention.self.query.bias",
        "k_w": p + "attention.self.key.weight", "k_b": p + "attention.self.key.bias",
        "v_w": p + "attention.self.value.weight", "v_b": p + "attention.self.value.bias",
        "ao_w": p + "attention.output.dense.weight", "ao_b": p + "attention.output.dense.bias",
        "aln_w": p + "attention.output.LayerNorm.weight", "aln_b": p + "attention.output.LayerNorm.bias",
        "im_w": p + "intermediate.dense.weight", "im_b": p + "intermediate.dense.bias",
        "out_w": p + "output.dense.weight", "out_b": p + "output.dense.bias",
        "oln_w": p + "output.LayerNorm.weight", "oln_b": p + "output.LayerNorm.bias",
    }


# ---- pure numpy helpers (no mlx; the bug-prone bits, unit-tested directly) ----

def mean_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mask-aware mean over tokens. last_hidden: (B,L,H), mask: (B,L) -> (B,H)."""
    mask = attention_mask.astype(last_hidden.dtype)[:, :, None]  # (B,L,1)
    summed = (last_hidden * mask).sum(axis=1)                    # (B,H)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)               # (B,1)
    return summed / counts


def cls_pool(last_hidden: np.ndarray) -> np.ndarray:
    """First-token ([CLS]) pooling. last_hidden: (B,L,H) -> (B,H)."""
    return last_hidden[:, 0, :]


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize. Zero rows stay zero (caller fails closed downstream)."""
    norms = np.sqrt((mat * mat).sum(axis=1, keepdims=True))
    norms = np.where(norms == 0.0, 1.0, norms)
    return mat / norms


def split_windows(ids: Sequence[int], window: int, step: int) -> list:
    """Split a token-id list into overlapping windows of <= `window` ids, advancing by
    `step` each time (so the overlap between adjacent windows is `window - step`).

    Returns ``[list(ids)]`` unchanged when the input already fits in one window. The
    final window always reaches the end of `ids`, so tail tokens are never dropped (the
    truncation failure mode this exists to fix). Pure (no tokenizer / mlx): this is the
    chunking math, unit-tested directly.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if step <= 0:
        raise ValueError("step must be positive")
    ids = list(ids)
    if len(ids) <= window:
        return [ids]
    out: list = []
    start = 0
    while start < len(ids):
        out.append(ids[start:start + window])
        if start + window >= len(ids):
            break
        start += step
    return out


class MLXEncoder:
    """A small BERT-convention encoder run in MLX. Loads config + safetensors weights."""

    def __init__(
        self,
        model_path: str,
        *,
        pooling: str = POOL_MEAN,
        allow_download: bool = False,
        max_length: int = 512,
    ):
        if pooling not in (POOL_MEAN, POOL_CLS):
            raise ValueError(f"pooling must be '{POOL_MEAN}' or '{POOL_CLS}', got {pooling!r}")
        self.pooling = pooling
        self.allow_download = allow_download
        self.max_length = max_length
        self.model_dir = self._resolve_model_dir(model_path, allow_download)

        cfg = json.loads(self._read(os.path.join(self.model_dir, "config.json")))
        self.hidden = int(cfg["hidden_size"])
        self.n_layers = int(cfg["num_hidden_layers"])
        self.n_heads = int(cfg["num_attention_heads"])
        self.head_dim = self.hidden // self.n_heads
        self.eps = float(cfg.get("layer_norm_eps", 1e-12))
        if self.hidden % self.n_heads != 0:
            raise ValueError("hidden_size not divisible by num_attention_heads")

        self._mx = self._import_mx()
        self._w = self._load_weights()
        self._tokenizer = None  # lazy: embed_token_batch never needs it

    # -- resolution / IO --

    @staticmethod
    def _resolve_model_dir(model_path: str, allow_download: bool) -> str:
        if os.path.isdir(model_path):
            return model_path
        if not allow_download:
            raise ValueError(
                f"model_path {model_path!r} is not a local directory and allow_download=False. "
                "Pass a local encoder dir, or set allow_download=True to fetch a repo id."
            )
        from huggingface_hub import snapshot_download  # lazy
        return snapshot_download(model_path)

    @staticmethod
    def _read(path: str) -> str:
        with open(path) as fh:
            return fh.read()

    @staticmethod
    def _import_mx():
        import mlx.core as mx  # lazy: module imports fine without mlx
        return mx

    def _safetensors_path(self) -> str:
        for name in ("model.safetensors", "pytorch_model.safetensors"):
            cand = os.path.join(self.model_dir, name)
            if os.path.isfile(cand):
                return cand
        raise FileNotFoundError(f"no model.safetensors in {self.model_dir}")

    def _resolve_prefix(self, keys) -> str:
        if _K["word"] in keys:
            return ""
        if "bert." + _K["word"] in keys:
            return "bert."
        raise KeyError(
            f"could not find {_K['word']} (with or without 'bert.' prefix) in weights; "
            "is this a BERT-convention encoder?"
        )

    def _load_weights(self) -> dict:
        raw = self._mx.load(self._safetensors_path())  # dict[str, mx.array]
        prefix = self._resolve_prefix(raw)
        w = {k: raw[prefix + v] for k, v in _K.items()}
        w["layers"] = [
            {k: raw[name] for k, name in _layer_keys(prefix, i).items()}
            for i in range(self.n_layers)
        ]
        return w

    # -- mlx forward --

    def _ln(self, x, weight, bias):
        mx = self._mx
        mu = x.mean(axis=-1, keepdims=True)
        var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
        return (x - mu) / mx.sqrt(var + self.eps) * weight + bias

    def _linear(self, x, w, b):
        return x @ w.T + b  # HF linear weight is (out,in); y = x W^T + b

    def _gelu(self, x):
        return 0.5 * x * (1.0 + self._mx.erf(x / math.sqrt(2.0)))

    def _attention(self, x, lw, add_mask):
        mx = self._mx
        b, l, _ = x.shape
        h, d = self.n_heads, self.head_dim

        def heads(t):
            return t.reshape(b, l, h, d).transpose(0, 2, 1, 3)  # (B,H,L,d)

        q = heads(self._linear(x, lw["q_w"], lw["q_b"]))
        k = heads(self._linear(x, lw["k_w"], lw["k_b"]))
        v = heads(self._linear(x, lw["v_w"], lw["v_b"]))
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(d)
        scores = scores + add_mask  # (B,1,1,L) additive, -inf on pad
        probs = mx.softmax(scores, axis=-1)
        ctx = (probs @ v).transpose(0, 2, 1, 3).reshape(b, l, h * d)
        attn = self._linear(ctx, lw["ao_w"], lw["ao_b"])
        return self._ln(x + attn, lw["aln_w"], lw["aln_b"])

    def _layer(self, x, lw, add_mask):
        x = self._attention(x, lw, add_mask)
        h = self._gelu(self._linear(x, lw["im_w"], lw["im_b"]))
        out = self._linear(h, lw["out_w"], lw["out_b"])
        return self._ln(x + out, lw["oln_w"], lw["oln_b"])

    def forward(self, input_ids: np.ndarray, attention_mask: np.ndarray):
        """Run the encoder. Returns last_hidden_state as a numpy array (B,L,H)."""
        mx = self._mx
        w = self._w
        ids = mx.array(np.asarray(input_ids, dtype=np.int32))
        b, l = ids.shape
        pos = mx.arange(l).reshape(1, l)
        emb = w["word"][ids] + w["pos"][pos] + w["tok_type"][mx.zeros((1, l), dtype=mx.int32)]
        x = self._ln(emb, w["emb_ln_w"], w["emb_ln_b"])
        # additive mask: 0 where attend, large-negative where pad -> (B,1,1,L)
        m = mx.array(np.asarray(attention_mask, dtype=np.float32))
        add_mask = (1.0 - m).reshape(b, 1, 1, l) * -1e9
        for lw in w["layers"]:
            x = self._layer(x, lw, add_mask)
        mx.eval(x)
        return np.array(x)

    # -- public embedding API --

    def embed_token_batch(self, input_ids, attention_mask=None) -> list:
        """Forward + pool + L2-normalize pre-tokenized ids. Returns list[list[float]]."""
        ids = np.asarray(input_ids, dtype=np.int64)
        if ids.ndim != 2:
            raise ValueError("input_ids must be 2-d (batch, seq)")
        mask = (np.ones_like(ids) if attention_mask is None
                else np.asarray(attention_mask)).astype(np.int64)
        hidden = self.forward(ids, mask)
        pooled = mean_pool(hidden, mask) if self.pooling == POOL_MEAN else cls_pool(hidden)
        return l2_normalize_rows(pooled).astype(np.float64).tolist()

    def _tok(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer  # lazy
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir, local_files_only=not self.allow_download
            )
        return self._tokenizer

    def embed(self, texts: Sequence[str]) -> list:
        enc = self._tok()(
            list(texts), padding=True, truncation=True,
            max_length=self.max_length, return_tensors=None,
        )
        return self.embed_token_batch(enc["input_ids"], enc["attention_mask"])


def make_mlx_embed_fn(
    model_path: str,
    *,
    pooling: str = POOL_MEAN,
    allow_download: bool = False,
    max_length: int = 512,
    batch_size: int = 32,
) -> Callable[[Sequence[str]], list]:
    """Build an embed_fn(texts) -> list[vector] for DEvalEmbed, backed by an MLX encoder.

    model_path: a LOCAL encoder directory (config.json + model.safetensors + tokenizer),
    or a repo id only if allow_download=True. Nothing is downloaded otherwise.
    """
    enc = MLXEncoder(model_path, pooling=pooling, allow_download=allow_download,
                     max_length=max_length)

    def embed_fn(texts: Sequence[str]) -> list:
        out: list = []
        items = list(texts)
        for i in range(0, len(items), batch_size):
            out.extend(enc.embed(items[i:i + batch_size]))
        return out

    return embed_fn


def make_mlx_chunk_fn(
    model_path: str,
    *,
    max_length: int = 512,
    overlap: int = 64,
    allow_download: bool = False,
) -> Callable[[str], list]:
    """Build a chunk_fn(text) -> list[str] for DEvalEmbed using the encoder's tokenizer.

    Long event spans are split into overlapping token windows so the embedding member
    can take a max over chunks instead of letting the encoder silently truncate at
    max_length (detector_ensemble_spec.md section 3: chunk-and-max-risk, parity with
    L1). A span that already fits the window is handed back as the single original
    string (no detokenization round-trip). Two special-token slots ([CLS]/[SEP]) are
    reserved, since embed() re-adds them when it encodes each returned chunk.

    model_path: a LOCAL tokenizer dir, or a repo id only when allow_download=True;
    nothing is fetched otherwise. Tokenizer-only (no encoder weights loaded here).
    """
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    window = max(1, max_length - 2)  # reserve [CLS]/[SEP] re-added by embed()
    if overlap >= window:
        raise ValueError(f"overlap {overlap} must be < content window {window}")
    step = window - overlap

    from transformers import AutoTokenizer  # lazy: module imports without transformers
    tok = AutoTokenizer.from_pretrained(
        model_path, local_files_only=not allow_download
    )

    def chunk_fn(text: str) -> list:
        ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
        windows = split_windows(ids, window, step)
        if len(windows) == 1:
            return [text]  # fits in one window: keep the original string untouched
        return [tok.decode(w, skip_special_tokens=True) for w in windows]

    return chunk_fn
