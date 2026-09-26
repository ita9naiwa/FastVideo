# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the VSA-H3 backend: tiling geometry, mask policy, and
end-to-end equivalence against dense SDPA through a token-level mask
reference. The same reference doubles as the GPU kernel parity oracle."""

import contextlib
import math

import pytest
import torch
import torch.nn.functional as F

from fastvideo.attention.backends.video_sparse_attn_h3 import (_TILE_ELEMS, MiniMaxH3VSAImpl,
                                                               MiniMaxH3VSAMetadataBuilder, _build_block_mask,
                                                               _pool_tiles, _validate_h3_tile_geometry,
                                                               token_tile_and_valid)

_720P = dict(raw_latent_shape=(30, 44, 80), patch_size=(1, 2, 2), prefix_segments=(512, 1760, 400))
_TINY = dict(raw_latent_shape=(8, 8, 12), patch_size=(1, 2, 2), prefix_segments=(7, 5, 3))
# (4,4,4) coverage: dit grid (9, 10, 13) is ragged in all three dims
# (t: 4+4+1, h: 4+4+2, w: 4+4+4+1) and every prefix segment leaves a
# partial tail tile at 64 (70 -> 64+6, 5 -> 5, 130 -> 64+64+2).
_TINY64 = dict(raw_latent_shape=(9, 20, 26), patch_size=(1, 2, 2), prefix_segments=(70, 5, 130))
# production-shape request: 768x1344, 124 frames -> latents (37, 48, 84),
# patch (1,2,2) -> token grid (37, 24, 42); text 300 + audio 414 rows.
_PROD = dict(raw_latent_shape=(37, 48, 84), patch_size=(1, 2, 2), prefix_segments=(300, 0, 414))

_CPU = torch.device("cpu")


def _build(spec, sparsity=0.0, device=_CPU, tile_size=_TILE_ELEMS, **kwargs):
    return MiniMaxH3VSAMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=spec["raw_latent_shape"],
        patch_size=spec["patch_size"],
        VSA_sparsity=sparsity,
        prefix_segments=spec["prefix_segments"],
        device=device,
        tile_size=tile_size,
        **kwargs,
    )


def _impl():
    return MiniMaxH3VSAImpl(num_heads=2, head_size=8, causal=False, softmax_scale=8**-0.5)


def reference_sparse_attention(query, key, value, mask, meta):
    """Token-level oracle: SDPA over the padded tile buffer with the block
    mask expanded to tokens. query/key/value: tiled [B, S_pad, H, D]."""
    token_tile, token_valid = token_tile_and_valid(meta.variable_block_sizes, meta.tile_elems)
    out = torch.empty_like(query)
    for b in range(query.shape[0]):
        for h in range(query.shape[2]):
            allow = mask[b, h][token_tile][:, token_tile] & token_valid[None, :]
            bias = torch.zeros(allow.shape, dtype=query.dtype, device=query.device)
            bias.masked_fill_(~allow, float("-inf"))
            out[b, :, h] = F.scaled_dot_product_attention(
                query[b, :, h][None],
                key[b, :, h][None],
                value[b, :, h][None],
                attn_mask=bias[None],
            )[0]
    return out


def test_geometry_720p():
    meta = _build(_720P)
    seq = meta.total_seq_length
    assert seq == 512 + 1760 + 400 + 26400
    assert meta.num_prefix_tiles == 2 + 7 + 2
    assert meta.num_video_tiles == 8 * 3 * 5
    assert int(meta.variable_block_sizes.sum()) == seq
    # (permutation coverage of [0, seq) is implied by the roundtrip below:
    # untile_combined_index scatters seq distinct rows and recovers all of x)
    # segment purity: no prefix tile straddles a segment boundary
    boundaries = [512, 512 + 1760, 512 + 1760 + 400]
    start = 0
    for size in meta.variable_block_sizes[:meta.num_prefix_tiles].tolist():
        end = start + size
        assert all(not (start < b < end) for b in boundaries), (start, end)
        start = end
    # untile(tile(x)) == x
    x = torch.randn(1, seq, 2, 4)
    buf = _impl().tile(x, meta)
    assert buf.shape[1] == meta.variable_block_sizes.numel() * _TILE_ELEMS
    assert torch.equal(buf[:, meta.untile_combined_index], x)


def test_mask_policy():
    meta = _build(_720P, sparsity=0.9)
    n = meta.num_prefix_tiles + meta.num_video_tiles
    P, V = meta.num_prefix_tiles, meta.num_video_tiles
    k_vid = math.ceil(0.1 * V)
    scores = torch.randn(1, 2, n, n)

    exempt = _build_block_mask(scores, P, V, 0.9, exempt=True)
    assert exempt[:, :, :P].all(), "prefix queries must be dense"
    assert exempt[..., :P].all(), "prefix keys must be visible to every query"
    assert (exempt[:, :, P:, P:].sum(-1) == k_vid).all(), "video rows select exactly k_vid video tiles"

    compete = _build_block_mask(scores, P, V, 0.9, exempt=False)
    assert compete[:, :, :P].all()
    assert (compete[:, :, P:].sum(-1) == min(k_vid + P, n)).all(), "budget-matched top-k"

    dense = _build_block_mask(scores, P, V, 0.0, exempt=True)
    assert dense.all(), "sparsity 0 must select everything"


def test_sparsity_zero_matches_dense_sdpa():
    torch.manual_seed(0)
    meta = _build(_TINY)
    seq = meta.total_seq_length
    q, k, v = (torch.randn(1, seq, 2, 8) for _ in range(3))
    impl = _impl()
    tq, tk, tv = (impl.tile(t, meta).clone() for t in (q, k, v))

    scores = torch.matmul(_pool_tiles(tq, meta.variable_block_sizes),
                          _pool_tiles(tk, meta.variable_block_sizes).transpose(-2, -1))
    mask = _build_block_mask(scores, meta.num_prefix_tiles, meta.num_video_tiles, 0.0, exempt=True)
    sparse_out = impl.postprocess_output(reference_sparse_attention(tq, tk, tv, mask, meta), meta)

    dense_out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    assert torch.allclose(sparse_out, dense_out, atol=1e-5), (sparse_out - dense_out).abs().max()


def test_prefix_queries_stay_dense_at_high_sparsity():
    torch.manual_seed(1)
    meta = _build(_TINY, sparsity=0.75)
    seq = meta.total_seq_length
    prefix_len = sum(_TINY["prefix_segments"])
    q, k, v = (torch.randn(1, seq, 2, 8) for _ in range(3))
    impl = _impl()
    tq, tk, tv = (impl.tile(t, meta).clone() for t in (q, k, v))

    scores = torch.matmul(_pool_tiles(tq, meta.variable_block_sizes),
                          _pool_tiles(tk, meta.variable_block_sizes).transpose(-2, -1))
    for exempt in (True, False):
        mask = _build_block_mask(scores, meta.num_prefix_tiles, meta.num_video_tiles, 0.75, exempt=exempt)
        sparse_out = impl.postprocess_output(reference_sparse_attention(tq, tk, tv, mask, meta), meta)
        dense_out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                                   v.transpose(1, 2)).transpose(1, 2)
        assert torch.allclose(sparse_out[:, :prefix_len], dense_out[:, :prefix_len], atol=1e-5)
        assert not torch.allclose(sparse_out[:, prefix_len:], dense_out[:, prefix_len:], atol=1e-5), \
            "video rows should actually be sparse at 75%"


# ---------------------------------------------------------------------------
# 64-token (4,4,4) tile geometry
# ---------------------------------------------------------------------------


def test_geometry_tile64_ragged_tails():
    """Hand-computed (4,4,4) oracle on a grid ragged in all three dims."""
    meta = _build(_TINY64, tile_size=64)
    assert meta.tile_elems == 64
    t, h, w = 9, 10, 13  # raw latents (9, 20, 26) under patch (1, 2, 2)
    n_t, n_h, n_w = 3, 3, 4
    prefix_len = sum(_TINY64["prefix_segments"])
    seq = prefix_len + t * h * w
    assert meta.total_seq_length == seq
    assert meta.num_prefix_tiles == 2 + 1 + 3
    assert meta.num_video_tiles == n_t * n_h * n_w
    assert int(meta.variable_block_sizes.sum()) == seq
    assert int(meta.variable_block_sizes.max()) <= 64
    assert meta.variable_block_sizes[:meta.num_prefix_tiles].tolist() == [64, 6, 5, 64, 64, 2]

    # per-tile valid sizes: product of the per-dim clamped tails
    expected = torch.tensor([
        min(4, t - 4 * tt) * min(4, h - 4 * hh) * min(4, w - 4 * ww) for tt in range(n_t) for hh in range(n_h)
        for ww in range(n_w)
    ],
                            dtype=torch.long)
    assert torch.equal(meta.variable_block_sizes[meta.num_prefix_tiles:], expected)
    assert int(expected.min()) == 1 * 2 * 1  # the (t,h,w) ragged corner

    # every packed video row lands in the 3D tile its (t,h,w) coordinate says
    idx = meta.untile_combined_index
    row = torch.arange(t * h * w)
    row_t, row_h, row_w = row // (h * w), (row // w) % h, row % w
    expected_tile = meta.num_prefix_tiles + ((row_t // 4) * n_h + row_h // 4) * n_w + row_w // 4
    assert torch.equal(idx[prefix_len:] // 64, expected_tile)
    # and in a non-pad slot of that tile
    assert bool((idx % 64 < meta.variable_block_sizes[idx // 64]).all())

    # untile(tile(x)) == x on the 64-wide padded buffer
    x = torch.randn(1, seq, 2, 4)
    buf = _impl().tile(x, meta)
    assert buf.shape[1] == meta.variable_block_sizes.numel() * 64
    assert torch.equal(buf[:, idx], x)


def test_geometry_tile64_production_shape():
    """Production latents (37, 48, 84): ragged t and w tails at (4,4,4)."""
    meta64 = _build(_PROD, tile_size=64)
    assert meta64.num_prefix_tiles == 5 + 7  # 300 -> 4x64+44, 414 -> 6x64+30
    assert meta64.num_video_tiles == 10 * 6 * 11  # (37, 24, 42) / (4, 4, 4)
    assert meta64.total_seq_length == 300 + 414 + 37 * 24 * 42
    assert int(meta64.variable_block_sizes.sum()) == meta64.total_seq_length
    sizes_vid = meta64.variable_block_sizes[meta64.num_prefix_tiles:]
    assert int(sizes_vid.max()) == 64 and int(sizes_vid.min()) == 1 * 4 * 2  # (t, w) ragged corner

    # same packed sequence under the default 256 geometry, fewer tiles
    meta256 = _build(_PROD)
    assert meta256.tile_elems == _TILE_ELEMS
    assert meta256.num_prefix_tiles == 2 + 2
    assert meta256.num_video_tiles == 10 * 3 * 6
    assert meta256.total_seq_length == meta64.total_seq_length

    x = torch.randn(1, meta64.total_seq_length, 2, 4)
    buf = _impl().tile(x, meta64)
    assert torch.equal(buf[:, meta64.untile_combined_index], x)


def test_sparsity_zero_matches_dense_sdpa_tile64():
    torch.manual_seed(2)
    meta = _build(_TINY64, tile_size=64)
    seq = meta.total_seq_length
    q, k, v = (torch.randn(1, seq, 2, 8) for _ in range(3))
    impl = _impl()
    tq, tk, tv = (impl.tile(t, meta).clone() for t in (q, k, v))

    scores = torch.matmul(_pool_tiles(tq, meta.variable_block_sizes, meta.tile_elems),
                          _pool_tiles(tk, meta.variable_block_sizes, meta.tile_elems).transpose(-2, -1))
    mask = _build_block_mask(scores, meta.num_prefix_tiles, meta.num_video_tiles, 0.0, exempt=True)
    sparse_out = impl.postprocess_output(reference_sparse_attention(tq, tk, tv, mask, meta), meta)

    dense_out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    assert torch.allclose(sparse_out, dense_out, atol=1e-5), (sparse_out - dense_out).abs().max()


def test_geometry_guard_enforces_tile64_bound():
    """A 65-token tile passes the 256 bound but must fail the 64 one."""
    meta = _build(_TINY64, tile_size=64)
    prefix = tuple(s for s in _TINY64["prefix_segments"] if s > 0)
    dit_shape = (9, 10, 13)
    sizes = meta.variable_block_sizes.clone()
    sizes[0] = 65
    with pytest.raises(ValueError, match="tile sizes out of bounds"):
        _validate_h3_tile_geometry(prefix, dit_shape, sizes, meta.untile_combined_index, 64)
    # the untampered tile-64 geometry passes its own bound
    _validate_h3_tile_geometry(prefix, dit_shape, meta.variable_block_sizes, meta.untile_combined_index, 64)


def test_builder_rejects_unknown_tile_size():
    for bad in (0, 128, 512):
        with pytest.raises(ValueError, match="tile_size"):
            _build(_TINY, tile_size=bad)


# Real H3 packed-document grids (token t,h,w after patch (1,2,2)) with (text, vidclip, keyframe, audio)
# prefixes, and the chunk256 video-tile count each must produce (ceil(video / 256)).
_H3_REAL_GRIDS = [((61, 15, 28), (420, 1, 420, 810), 101), ((50, 15, 28), (300, 1, 0, 666), 83),
                  ((91, 10, 18), (500, 1, 180, 1206), 64), ((60, 10, 18), (250, 1, 0, 800), 43),
                  ((38, 15, 28), (350, 1, 420, 500), 63), ((75, 7, 13), (200, 1, 0, 1000), 27),
                  ((61, 15, 28), (380, 1, 420, 810), 101), ((31, 7, 13), (600, 1, 0, 414), 12)]


@pytest.mark.parametrize("merge_prefix", [False, True])
@pytest.mark.parametrize("thw,prefix,n_video", _H3_REAL_GRIDS)
def test_geometry_chunk256_real_grids(thw, prefix, n_video, merge_prefix):
    spec = dict(raw_latent_shape=(thw[0], 2 * thw[1], 2 * thw[2]), patch_size=(1, 2, 2), prefix_segments=prefix)
    meta = _build(spec, tile_layout="chunk256", merge_prefix=merge_prefix)
    cube = _build(spec)
    sizes = meta.variable_block_sizes
    P, video = meta.num_prefix_tiles, math.prod(thw)
    assert meta.tile_layout == "chunk256" and meta.num_video_tiles == n_video
    assert int(sizes.sum()) == meta.total_seq_length == cube.total_seq_length
    vs = sizes[P:].tolist()
    assert vs[:-1] == [256] * (n_video - 1) and vs[-1] == video - 256 * (n_video - 1)
    segments = [x for x in prefix if x > 0]
    if merge_prefix:
        assert math.ceil(sum(segments) / 256) == P and sizes[:P - 1].eq(256).all()
    else:
        assert cube.num_prefix_tiles == P and torch.equal(sizes[:P], cube.variable_block_sizes[:P])
    # same token order as the cube layout, only the tile boundaries move
    x = torch.randn(1, meta.total_seq_length, 1, 2)
    impl = _impl()
    buf, cube_buf = impl.tile(x, meta), impl.tile(x, cube)
    assert torch.equal(buf[:, meta.untile_combined_index], x)
    keep = lambda b, m: b[:, token_tile_and_valid(m.variable_block_sizes, m.tile_elems)[1]]
    assert torch.equal(keep(buf, meta), keep(cube_buf, cube))


@pytest.mark.parametrize("grid,prefix,n_video", _H3_REAL_GRIDS)
def test_pack_tails_policy(monkeypatch, grid, prefix, n_video):
    """Tail packing only when many parents are partial and the residual rows fit the bounded plan."""
    from fastvideo.attention.backends.video_sparse_attn_h3 import _pack_tails_policy
    monkeypatch.delenv("FASTVIDEO_VSA_PACK_TAILS", raising=False)
    canonical = torch.tensor([256, 131] * 75, dtype=torch.int32)  # half the parents partial
    assert _pack_tails_policy(canonical, 256, 0.15) and not _pack_tails_policy(canonical, 64, 0.15)
    assert not _pack_tails_policy(canonical, 256, 0.6)
    assert not _pack_tails_policy(torch.full((150, ), 255, dtype=torch.int32), 256, 0.15)  # residuals overflow
    spec = dict(raw_latent_shape=(grid[0], 2 * grid[1], 2 * grid[2]), patch_size=(1, 2, 2), prefix_segments=prefix)
    chunk = _build(spec, tile_layout="chunk256", merge_prefix=True)
    assert chunk.pack_tails is False  # ~2 partial tiles per document
    assert _build(spec, tile_layout="chunk256", pack_tails=True).pack_tails is True
    monkeypatch.setenv("FASTVIDEO_VSA_PACK_TAILS", "1")
    assert _build(spec, tile_layout="chunk256", merge_prefix=True).pack_tails is True  # explicit env forces
    monkeypatch.setenv("FASTVIDEO_VSA_PACK_TAILS", "0")
    assert _build(spec, pack_tails="auto").pack_tails is False
    other = _build(spec, tile_layout="chunk256", merge_prefix=True, pack_tails=False)
    assert torch.equal(other.variable_block_sizes, chunk.variable_block_sizes)
    assert torch.equal(other.untile_combined_index, chunk.untile_combined_index)


def test_sparsity_zero_matches_dense_sdpa_chunk256():
    torch.manual_seed(0)
    # ragged token grid (9, 10, 13) = 1170 video rows -> 4 full tiles + 146; prefix 300 | 5 | 130 (merged)
    spec = dict(raw_latent_shape=(9, 20, 26), patch_size=(1, 2, 2), prefix_segments=(300, 5, 130))
    for merge_prefix in (False, True):
        meta = _build(spec, tile_layout="chunk256", merge_prefix=merge_prefix)
        seq = meta.total_seq_length
        q, k, v = (torch.randn(1, seq, 2, 8) for _ in range(3))
        impl = _impl()
        tq, tk, tv = (impl.tile(t, meta).clone() for t in (q, k, v))
        scores = torch.matmul(_pool_tiles(tq, meta.variable_block_sizes),
                              _pool_tiles(tk, meta.variable_block_sizes).transpose(-2, -1))
        mask = _build_block_mask(scores, meta.num_prefix_tiles, meta.num_video_tiles, 0.0, exempt=True)
        sparse_out = impl.postprocess_output(reference_sparse_attention(tq, tk, tv, mask, meta), meta)
        dense_out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                                   v.transpose(1, 2)).transpose(1, 2)
        assert torch.allclose(sparse_out, dense_out, atol=1e-5), (sparse_out - dense_out).abs().max()


def test_default_tile_layout_is_cube():
    meta, explicit = _build(_720P), _build(_720P, tile_layout="cube")
    assert meta.tile_layout == explicit.tile_layout == "cube"
    assert torch.equal(meta.variable_block_sizes, explicit.variable_block_sizes)
    assert torch.equal(meta.untile_combined_index, explicit.untile_combined_index)


def test_builder_rejects_bad_tile_layout():
    with pytest.raises(ValueError, match="tile_layout"):
        _build(_TINY, tile_layout="rows")
    with pytest.raises(ValueError, match="chunk256"):
        _build(_TINY64, tile_size=64, tile_layout="chunk256")
    with pytest.raises(ValueError, match="merge_prefix"):
        _build(_TINY, merge_prefix=True)
    with pytest.raises(ValueError, match="merge_prefix"):
        MiniMaxH3VSAMetadataBuilder().build(current_timestep=0,
                                            raw_latent_shape=_TINY["raw_latent_shape"],
                                            patch_size=_TINY["patch_size"],
                                            VSA_sparsity=0.0,
                                            prefix_segments=_TINY["prefix_segments"],
                                            device=_CPU,
                                            exempt=False,
                                            tile_layout="chunk256",
                                            merge_prefix=True)


# ---------------------------------------------------------------------------
# Training untile gradient (vsa_h3_untile_fwd/bwd op pair) vs the generic indexed backward
# ---------------------------------------------------------------------------

# (avg_abs, max_rel) gradient tolerance of fastvideo-kernel/tests/test_vsa256_backward.py
_GRAD_TOL = (1e-3, 0.25)
# s085k32 spec v4 pack-4 docs: token grid (t, h, w), prefixes (text, vidclip, keyframe, audio)
_S085K32_DOCS = [((47, 7, 12), (87, 1, 84, 514)), ((52, 15, 26), (476, 1, 390, 572)),
                 ((37, 12, 31), (224, 1, 0, 402)), ((42, 25, 10), (223, 1, 250, 458))]
_UNTILE_CASES = [(dict(raw_latent_shape=(t, 2 * h, 2 * w), patch_size=(1, 2, 2), prefix_segments=prefix),
                  dict(tile_layout="chunk256", merge_prefix=True), 0.85) for (t, h, w), prefix in _S085K32_DOCS]
_UNTILE_CASES.append((_720P, {}, 0.75))
_UNTILE_IDS = [f"chunk256-{t}x{h}x{w}" for (t, h, w), _ in _S085K32_DOCS] + ["cube-720p"]
_requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
# autograd node of the opaque training untile op (torch.ops.fastvideo_kernel.vsa_h3_untile_fwd)
_UNTILE_OP = "GeneratedBackwardFor_fastvideo_kernel_vsa_h3_untile_fwd_defaultBackward"


@contextlib.contextmanager
def _indexed(meta):
    """The unchanged indexed postprocess_output: drop the untile certificate for the block."""
    state = meta.__dict__.pop("_untile_grad_state", None)
    try:
        yield
    finally:
        if state is not None:
            meta._untile_grad_state = state


def _untile_and_grad(impl, meta, output, grad_out):
    out = impl.postprocess_output(output, meta)
    grad, = torch.autograd.grad(out, output, grad_out)
    return out, grad, type(out.grad_fn).__name__


@_requires_cuda
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("spec,layout,sparsity", _UNTILE_CASES, ids=_UNTILE_IDS)
def test_untile_grad_matches_indexed_exactly(spec, layout, sparsity, batch):
    meta = _build(spec, sparsity, torch.device("cuda"), **layout)
    impl = MiniMaxH3VSAImpl(num_heads=4, head_size=128, causal=False, softmax_scale=128**-0.5)
    padded = meta.variable_block_sizes.numel() * meta.tile_elems
    output = torch.randn(batch, padded, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    # non-contiguous upstream gradient (the op backward makes it contiguous)
    grad_out = torch.randn(batch, meta.total_seq_length, 4, 256, device="cuda", dtype=torch.bfloat16)[..., ::2]
    out, grad, path = _untile_and_grad(impl, meta, output, grad_out)
    with _indexed(meta):
        ref, ref_grad, ref_path = _untile_and_grad(impl, meta, output, grad_out)
    assert (path, ref_path) == (_UNTILE_OP, "IndexBackward0")
    assert torch.equal(out, ref) and torch.equal(grad, ref_grad)
    pad = ~token_tile_and_valid(meta.variable_block_sizes, meta.tile_elems)[1]
    assert int(pad.sum()) > 0 and not grad[:, pad].any(), "pad-row gradients must be exactly zero"
    with torch.no_grad():
        nograd = impl.postprocess_output(output, meta)
    assert nograd.grad_fn is None and torch.equal(nograd, output.detach()[:, meta.untile_combined_index])


def _h3_block_grads(impl, meta, inputs, grad_out):
    q, k, v = (t.detach().clone().requires_grad_(True) for t in inputs)
    tiled = impl.preprocess_qkv(torch.cat([q, k, v], dim=0), meta)
    out = impl.postprocess_output(impl.forward(*tiled.chunk(3, dim=0), None, meta), meta)
    return (out.detach(), *torch.autograd.grad(out, (q, k, v), grad_out)), type(out.grad_fn).__name__


def _grad_metrics(ref, got):  # test_vsa256_backward._metrics form
    diff = (ref.float() - got.float()).abs()
    return float(diff.mean()), float(diff.max() / (ref.float().abs().mean() + 1e-6)), float(diff.max())


@_requires_cuda
@pytest.mark.parametrize("spec,layout,sparsity", _UNTILE_CASES, ids=_UNTILE_IDS)
def test_untile_grad_attention_dkdv_bitwise_dq_in_envelope(monkeypatch, spec, layout, sparsity):
    """Real H3 attention fwd+bwd (FA4 CuTe). O/dK/dV bitwise vs the indexed baseline; dQ is not repeatable
    (FA4 dQ accumulation order), so it must stay within the baseline's own repeat spread + _GRAD_TOL."""
    pytest.importorskip("flash_attn.cute.block_sparsity", reason="optional FA4 CuTe build (flash_attn.cute) not installed")
    monkeypatch.setenv("FASTVIDEO_VSA_CUTEDSL", "1")
    monkeypatch.delenv("FASTVIDEO_VSA_TRITON", raising=False)
    monkeypatch.delenv("FASTVIDEO_KERNEL_VSA_FORCE_TRITON", raising=False)
    heads = 8
    meta = _build(spec, sparsity, torch.device("cuda"), **layout)
    impl = MiniMaxH3VSAImpl(num_heads=heads, head_size=128, causal=False, softmax_scale=128**-0.5,
                            prefix="blocks.0.attn")
    g = torch.Generator(device="cuda").manual_seed(0)
    inputs = [
        torch.randn(1, meta.total_seq_length, heads, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        for _ in range(4)
    ]
    with _indexed(meta):
        (base, base_path), (repeat, _) = (_h3_block_grads(impl, meta, inputs[:3], inputs[3]) for _ in range(2))
    cand, cand_path = _h3_block_grads(impl, meta, inputs[:3], inputs[3])
    assert (cand_path, base_path) == (_UNTILE_OP, "IndexBackward0")
    for name, i in (("O", 0), ("dK", 2), ("dV", 3)):
        assert torch.equal(repeat[i], base[i]), f"baseline {name} is not repeatable"
        assert torch.equal(cand[i], base[i]), f"{name} differs from the indexed baseline"
    spread, delta = _grad_metrics(base[1], repeat[1]), _grad_metrics(base[1], cand[1])
    print(f"[untile-grad {spec['raw_latent_shape']} {layout or 'cube'}] dQ baseline spread "
          f"avg_abs={spread[0]:.3e} max_rel={spread[1]:.3e} max_abs={spread[2]:.3e}; cand-vs-base "
          f"avg_abs={delta[0]:.3e} max_rel={delta[1]:.3e} max_abs={delta[2]:.3e}")
    assert delta[0] <= spread[0] + _GRAD_TOL[0] and delta[1] <= spread[1] + _GRAD_TOL[1], (spread, delta)


def _graph_recorder(graphs):

    def backend(gm, example_inputs):
        graphs.append(gm.code)
        return gm.forward

    return backend


def _untile_case(mutate, text_len, compiled):
    # Own geometry per case: the index tensors are lru-cached per geometry, so a mutation taints later builds.
    meta = _build(dict(_720P, prefix_segments=(text_len, 1760, 400)), 0.75, torch.device("cuda"))
    impl = MiniMaxH3VSAImpl(num_heads=2, head_size=128, causal=False, softmax_scale=128**-0.5)
    output = torch.randn(1, meta.variable_block_sizes.numel() * meta.tile_elems, 2, 128, device="cuda",
                         dtype=torch.bfloat16, requires_grad=True)
    grad_out = torch.randn(1, meta.total_seq_length, 2, 128, device="cuda", dtype=torch.bfloat16)
    mutate(meta)
    graphs = []
    post = impl.postprocess_output
    if compiled:
        torch._dynamo.reset()
        post = torch.compile(impl.postprocess_output, backend=_graph_recorder(graphs), fullgraph=True, dynamic=True)
    out = post(output, meta)
    grad, = torch.autograd.grad(out, output, grad_out)
    with _indexed(meta):
        ref, ref_grad, _ = _untile_and_grad(impl, meta, output, grad_out)
    assert torch.equal(out, ref) and torch.equal(grad, ref_grad)
    assert torch.equal(out, output.detach()[:, meta.untile_combined_index])
    return type(out.grad_fn).__name__, graphs


# explorer-1's guard cases (init/check_untile_v2_explorer1.py): every replaced or mutated index, derived plan tensor, or
# missing certificate must end in IndexBackward0's result.
_MUTATIONS = {
    "untile_replaced": lambda m: setattr(m, "untile_combined_index", m.untile_combined_index.roll(1)),
    "untile_in_place": lambda m: m.untile_combined_index.copy_(m.untile_combined_index.roll(1)),
    "partition_replaced": lambda m: setattr(m, "tile_partition_indices", m.tile_partition_indices.clone()),
    "partition_in_place": lambda m: m.tile_partition_indices.add_(0),
    "nonpad_replaced": lambda m: setattr(m, "non_pad_index", m.non_pad_index.clone()),
    "nonpad_in_place": lambda m: m.non_pad_index.add_(0),
    "source_in_place": lambda m: m._untile_grad_state[2].add_(0),
    "pad_rows_in_place": lambda m: m._untile_grad_state[4].add_(0),
    "missing_certificate": lambda m: delattr(m, "_untile_grad_state"),
}
_MUTATION_NAMES = list(_MUTATIONS)


@_requires_cuda
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@pytest.mark.parametrize("mutation", _MUTATION_NAMES)
def test_untile_grad_untrusted_metadata_falls_back(mutation, compiled):
    # Distinct geometry per (mutation, mode): a mutation taints the cached index tensors of its geometry.
    path, _ = _untile_case(_MUTATIONS[mutation], 520 + 2 * _MUTATION_NAMES.index(mutation) + int(compiled), compiled)
    if not compiled:
        assert path == "IndexBackward0", "untrusted metadata must take the indexed fallback in eager"
    # compiled: identity mismatches trace the indexed path; version mismatches reach the op, whose backward re-checks the
    # recorded versions and computes IndexBackward0's adjoint (asserted bitwise above)


@_requires_cuda
def test_untile_grad_compiled_reaches_the_op():
    path, graphs = _untile_case(lambda m: None, 540, compiled=True)
    assert path == _UNTILE_OP and graphs and all("vsa_h3_untile_fwd" in g for g in graphs), graphs


@_requires_cuda
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_untile_public_entry_h3mh_shape(compiled):
    """h3mh's wrapper shape (conductor 34247, worker-3 34269): per doc, out [1, P, 56, 128] bf16 and
    out.index_select(1, meta.untile_combined_index). The public vsa_h3_untile, called as a free function under the h3mh
    compile config (inductor, dynamic, fullgraph), reaches the op pair and equals index_select + its index_add adjoint."""
    from fastvideo.attention.backends.video_sparse_attn_h3 import vsa_h3_untile
    meta = _build(dict(_720P, prefix_segments=(515, 1760, 400)), 0.85, torch.device("cuda"))
    output = torch.randn(1, meta.variable_block_sizes.numel() * meta.tile_elems, 56, 128, device="cuda",
                         dtype=torch.bfloat16, requires_grad=True)
    grad_out = torch.randn(1, meta.total_seq_length, 56, 128, device="cuda", dtype=torch.bfloat16)
    ref = output.index_select(1, meta.untile_combined_index)
    ref_grad, = torch.autograd.grad(ref, output, grad_out)
    fn = vsa_h3_untile
    if compiled:
        torch._dynamo.reset()
        fn = torch.compile(vsa_h3_untile, backend="inductor", fullgraph=True, dynamic=True)
        torch.autograd.grad(fn(output, meta), output, grad_out)  # compile fwd + bwd outside the profiled call
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        out = fn(output, meta)
        grad, = torch.autograd.grad(out, output, grad_out)
    names = [e.name for e in prof.events()]
    assert names.count("fastvideo_kernel::vsa_h3_untile_fwd") == 1 and names.count("fastvideo_kernel::vsa_h3_untile_bwd") == 1
    assert not any("index_add" in n or "index_put" in n for n in names), [n for n in names if "index" in n]
    assert torch.equal(out, ref.detach()) and torch.equal(grad, ref_grad)


@_requires_cuda
def test_untile_grad_op_stale_versions_use_indexed_adjoint():
    meta = _build(dict(_720P, prefix_segments=(541, 1760, 400)), 0.75, torch.device("cuda"))
    untile, uv, source, sv, pad, pv = meta._untile_grad_state
    partition, nonpad = meta.tile_partition_indices, meta.non_pad_index
    output = torch.randn(2, source.numel(), 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad_out = torch.randn(2, untile.numel(), 2, 128, device="cuda", dtype=torch.bfloat16)
    ref_grad, = torch.autograd.grad(output[:, untile], output, grad_out)
    for slot in range(5):
        versions = [uv, sv, pv, 0, 0]
        versions[slot] += 1  # as if that index changed after the metadata recorded it
        out = torch.ops.fastvideo_kernel.vsa_h3_untile_fwd(output, untile, source, pad, partition, nonpad, versions)
        grad, = torch.autograd.grad(out, output, grad_out)
        assert torch.equal(out, output.detach()[:, untile]) and torch.equal(grad, ref_grad)


@_requires_cuda
def test_untile_grad_mutation_before_backward_raises():
    meta = _build(dict(_720P, prefix_segments=(514, 1760, 400)), 0.75, torch.device("cuda"))
    impl = MiniMaxH3VSAImpl(num_heads=2, head_size=128, causal=False, softmax_scale=128**-0.5)
    output = torch.randn(1, meta.variable_block_sizes.numel() * meta.tile_elems, 2, 128, device="cuda",
                         dtype=torch.bfloat16, requires_grad=True)
    out = impl.postprocess_output(output, meta)
    assert type(out.grad_fn).__name__ == _UNTILE_OP
    meta.untile_combined_index.add_(0)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        out.sum().backward()



def _pinned(output, untile, version=0):
    """What a query-pad-pruning forward returns: a view carrying the untile map its backward trusted."""
    output = output.view_as(output)
    output._vsa_h3_query_pad_untile = (untile, version)
    return output


@_requires_cuda
@pytest.mark.parametrize("case", ["pinned_certified", "metadata_replaced_after_forward", "pinned_uncertified"])
def test_untile_grad_uses_the_pinned_effective_map(case):
    """Composition with the query-pad pin: the effective map is the pinned one; the op runs only when the certificate is
    for exactly that map, and otherwise the fallback indexes the pinned map (never a newer metadata map)."""
    meta = _build(dict(_720P, prefix_segments=(550 + ["pinned_certified", "metadata_replaced_after_forward",
                                                      "pinned_uncertified"].index(case), 1760, 400)), 0.75,
                  torch.device("cuda"))
    impl = MiniMaxH3VSAImpl(num_heads=2, head_size=128, causal=False, softmax_scale=128**-0.5)
    leaf = torch.randn(1, meta.variable_block_sizes.numel() * meta.tile_elems, 2, 128, device="cuda",
                       dtype=torch.bfloat16, requires_grad=True)
    grad_out = torch.randn(1, meta.total_seq_length, 2, 128, device="cuda", dtype=torch.bfloat16)
    pinned_map = meta.untile_combined_index
    if case == "pinned_uncertified":
        pinned_map = pinned_map.roll(1)  # a valid map the builder never certified
    output = _pinned(leaf, pinned_map)
    if case == "metadata_replaced_after_forward":
        meta.untile_combined_index = meta.untile_combined_index.roll(1)
    out = impl.postprocess_output(output, meta)
    grad, = torch.autograd.grad(out, leaf, grad_out)
    ref_grad, = torch.autograd.grad(leaf[:, pinned_map], leaf, grad_out)
    assert torch.equal(out, leaf.detach()[:, pinned_map]) and torch.equal(grad, ref_grad)
    pad = ~token_tile_and_valid(meta.variable_block_sizes, meta.tile_elems)[1]
    assert not grad[:, pad].any(), "the pin requires exact zero dO on padded rows"
    assert type(out.grad_fn).__name__ == (_UNTILE_OP if case != "pinned_uncertified" else "IndexBackward0")


@_requires_cuda
def test_untile_grad_pinned_map_mutated_raises():
    meta = _build(dict(_720P, prefix_segments=(553, 1760, 400)), 0.75, torch.device("cuda"))
    impl = MiniMaxH3VSAImpl(num_heads=2, head_size=128, causal=False, softmax_scale=128**-0.5)
    leaf = torch.randn(1, meta.variable_block_sizes.numel() * meta.tile_elems, 2, 128, device="cuda",
                       dtype=torch.bfloat16, requires_grad=True)
    output = _pinned(leaf, meta.untile_combined_index)
    meta.untile_combined_index.add_(0)
    with pytest.raises(RuntimeError, match="modified in place between forward and postprocess_output"):
        impl.postprocess_output(output, meta)

def test_untile_grad_not_certified_for_inference_tensors():
    with torch.inference_mode():
        meta = _build(dict(_TINY, raw_latent_shape=(8, 8, 14)))  # own geometry: inference tensors stay local
    assert getattr(meta, "_untile_grad_state", None) is None
    assert _build(_TINY)._untile_grad_state[0] is not None


if __name__ == "__main__":
    test_geometry_720p()
    test_mask_policy()
    test_sparsity_zero_matches_dense_sdpa()
    test_prefix_queries_stay_dense_at_high_sparsity()
    test_geometry_tile64_ragged_tails()
    test_geometry_tile64_production_shape()
    test_sparsity_zero_matches_dense_sdpa_tile64()
    test_geometry_guard_enforces_tile64_bound()
    test_builder_rejects_unknown_tile_size()
    print("all VSA-H3 CPU checks passed")


@pytest.mark.parametrize("n_blocks", [20, 40, 60, 80, 100, 120, 160, 19, 33])
def test_compute_topk_is_exact(n_blocks):
    from fastvideo.attention.backends.video_sparse_attn import compute_topk
    assert compute_topk(0.85, n_blocks) == (3 * n_blocks + 19) // 20  # float ceil gave one more at multiples of 20
    assert compute_topk(0.75, n_blocks) == (n_blocks + 3) // 4
    assert compute_topk(0.9, n_blocks) == (n_blocks + 9) // 10
    assert compute_topk(1e-9, n_blocks) == n_blocks
    for k in range(1, n_blocks + 1):  # the per-document midpoint encoding used to pin k through the sparsity
        assert compute_topk(1 - (k - 0.5) / n_blocks, n_blocks) == k
