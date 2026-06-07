"""Tests for the hand-rolled MLX encoder backend.

Two layers:
- Pure numpy pooling/normalization helpers (need numpy, not mlx): mask-aware mean,
  cls, L2 normalize. These are the bug-prone bits.
- A tiny SYNTHETIC-WEIGHTS encoder (needs mlx): random weights saved to safetensors in
  a tmp dir, loaded through MLXEncoder, asserting forward shape, unit-norm output, and
  mean-vs-cls pooling difference. No weights are ever downloaded.

Imports are deferred behind importorskip so the light `uv run --with-editable
parapet-data` env (no numpy/mlx) skips this file cleanly instead of erroring.
"""
import json
import os

import pytest


# ---- pure numpy helpers (no mlx) ----

def test_mean_pool_is_mask_aware():
    np = pytest.importorskip("numpy")
    from parapet_data.p3.detectors.mlx_embed_backend import mean_pool
    hidden = np.array([[[1.0, 1.0], [3.0, 3.0], [9.0, 9.0]]])  # (1,3,2)
    mask = np.array([[1, 1, 0]])                               # last token padded
    pooled = mean_pool(hidden, mask)
    assert pooled.shape == (1, 2)
    assert np.allclose(pooled, [[2.0, 2.0]])  # mean of first two only, pad ignored


def test_cls_pool_takes_first_token():
    np = pytest.importorskip("numpy")
    from parapet_data.p3.detectors.mlx_embed_backend import cls_pool
    hidden = np.array([[[7.0, 8.0], [1.0, 1.0]]])  # (1,2,2)
    assert np.allclose(cls_pool(hidden), [[7.0, 8.0]])


def test_l2_normalize_rows_unit_norm_and_zero_safe():
    np = pytest.importorskip("numpy")
    from parapet_data.p3.detectors.mlx_embed_backend import l2_normalize_rows
    mat = np.array([[3.0, 4.0], [0.0, 0.0]])
    out = l2_normalize_rows(mat)
    assert np.allclose(np.sqrt((out[0] ** 2).sum()), 1.0)  # 3-4-5 -> unit
    assert np.allclose(out[1], [0.0, 0.0])                 # zero row stays zero, no nan


# ---- token-window chunking math (split_windows: pure, needs numpy only for import) ----

def test_split_windows_no_split_when_fits():
    pytest.importorskip("numpy")
    from parapet_data.p3.detectors.mlx_embed_backend import split_windows
    assert split_windows([1, 2, 3], window=5, step=2) == [[1, 2, 3]]
    assert split_windows([1, 2, 3, 4, 5], window=5, step=2) == [[1, 2, 3, 4, 5]]  # exact fit


def test_split_windows_overlapping():
    pytest.importorskip("numpy")
    from parapet_data.p3.detectors.mlx_embed_backend import split_windows
    # 7 ids, window 4, step 3 (overlap 1): [0:4], [3:7]
    assert split_windows([0, 1, 2, 3, 4, 5, 6], window=4, step=3) == [[0, 1, 2, 3], [3, 4, 5, 6]]


def test_split_windows_covers_tail():
    pytest.importorskip("numpy")
    from parapet_data.p3.detectors.mlx_embed_backend import split_windows
    # 6 ids, window 4, step 4 (no overlap): [0:4], [4:6] -- the tail is covered, not dropped
    assert split_windows([0, 1, 2, 3, 4, 5], window=4, step=4) == [[0, 1, 2, 3], [4, 5]]


def test_split_windows_validates():
    pytest.importorskip("numpy")
    from parapet_data.p3.detectors.mlx_embed_backend import split_windows
    with pytest.raises(ValueError):
        split_windows([1, 2], window=0, step=1)
    with pytest.raises(ValueError):
        split_windows([1, 2], window=2, step=0)


# ---- model-path gate (needs numpy for module import, not mlx) ----

def test_requires_local_dir_or_download():
    pytest.importorskip("numpy")
    from parapet_data.p3.detectors.mlx_embed_backend import MLXEncoder, make_mlx_embed_fn
    with pytest.raises(ValueError, match="allow_download"):
        MLXEncoder("/no/such/encoder/dir", allow_download=False)
    with pytest.raises(ValueError, match="allow_download"):
        make_mlx_embed_fn("some/repo-id", allow_download=False)


# ---- synthetic-weights encoder (needs mlx; no downloads) ----

def _write_synthetic_model(mx, np, model_dir, *, hidden=8, layers=2, heads=2,
                           inter=16, vocab=20, max_pos=16, type_vocab=2):
    cfg = {
        "hidden_size": hidden, "num_hidden_layers": layers,
        "num_attention_heads": heads, "intermediate_size": inter,
        "vocab_size": vocab, "max_position_embeddings": max_pos,
        "type_vocab_size": type_vocab, "layer_norm_eps": 1e-12, "hidden_act": "gelu",
    }
    with open(os.path.join(model_dir, "config.json"), "w") as fh:
        json.dump(cfg, fh)

    rng = np.random.default_rng(0)

    def rnd(*shape):
        return mx.array((rng.standard_normal(shape) * 0.02).astype(np.float32))

    def ln_w(n):  # LayerNorm: ones weight, zero bias (stable, still exercises the path)
        return mx.array(np.ones(n, dtype=np.float32))

    def ln_b(n):
        return mx.array(np.zeros(n, dtype=np.float32))

    w = {
        "embeddings.word_embeddings.weight": rnd(vocab, hidden),
        "embeddings.position_embeddings.weight": rnd(max_pos, hidden),
        "embeddings.token_type_embeddings.weight": rnd(type_vocab, hidden),
        "embeddings.LayerNorm.weight": ln_w(hidden),
        "embeddings.LayerNorm.bias": ln_b(hidden),
    }
    for i in range(layers):
        p = f"encoder.layer.{i}."
        w.update({
            p + "attention.self.query.weight": rnd(hidden, hidden),
            p + "attention.self.query.bias": ln_b(hidden),
            p + "attention.self.key.weight": rnd(hidden, hidden),
            p + "attention.self.key.bias": ln_b(hidden),
            p + "attention.self.value.weight": rnd(hidden, hidden),
            p + "attention.self.value.bias": ln_b(hidden),
            p + "attention.output.dense.weight": rnd(hidden, hidden),
            p + "attention.output.dense.bias": ln_b(hidden),
            p + "attention.output.LayerNorm.weight": ln_w(hidden),
            p + "attention.output.LayerNorm.bias": ln_b(hidden),
            p + "intermediate.dense.weight": rnd(inter, hidden),
            p + "intermediate.dense.bias": ln_b(inter),
            p + "output.dense.weight": rnd(hidden, inter),
            p + "output.dense.bias": ln_b(hidden),
            p + "output.LayerNorm.weight": ln_w(hidden),
            p + "output.LayerNorm.bias": ln_b(hidden),
        })
    mx.save_safetensors(os.path.join(model_dir, "model.safetensors"), w)
    return hidden


def test_synthetic_encoder_forward_shape_and_pooling(tmp_path):
    mx = pytest.importorskip("mlx.core")
    np = pytest.importorskip("numpy")
    from parapet_data.p3.detectors.mlx_embed_backend import MLXEncoder

    model_dir = str(tmp_path)
    hidden = _write_synthetic_model(mx, np, model_dir)

    ids = [[2, 5, 7, 3, 0], [4, 6, 1, 0, 0]]
    mask = [[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]

    enc_mean = MLXEncoder(model_dir, pooling="mean")
    hid = enc_mean.forward(np.array(ids), np.array(mask))
    assert hid.shape == (2, 5, hidden)  # (B, L, H)

    vecs = enc_mean.embed_token_batch(ids, mask)
    assert len(vecs) == 2 and all(len(v) == hidden for v in vecs)
    for v in vecs:
        assert abs(sum(x * x for x in v) - 1.0) < 1e-5  # L2-normalized output

    enc_cls = MLXEncoder(model_dir, pooling="cls")
    vecs_cls = enc_cls.embed_token_batch(ids, mask)
    # mean and cls pooling must produce different embeddings for the same input
    assert any(abs(a - b) > 1e-4 for a, b in zip(vecs[0], vecs_cls[0]))


def test_synthetic_encoder_masking_changes_output(tmp_path):
    mx = pytest.importorskip("mlx.core")
    np = pytest.importorskip("numpy")
    from parapet_data.p3.detectors.mlx_embed_backend import MLXEncoder

    model_dir = str(tmp_path)
    _write_synthetic_model(mx, np, model_dir)
    enc = MLXEncoder(model_dir, pooling="mean")

    ids = [[2, 5, 7, 3, 9]]
    full = enc.embed_token_batch(ids, [[1, 1, 1, 1, 1]])
    masked = enc.embed_token_batch(ids, [[1, 1, 1, 0, 0]])
    # dropping real tokens from the mask must change the pooled embedding
    assert any(abs(a - b) > 1e-4 for a, b in zip(full[0], masked[0]))
