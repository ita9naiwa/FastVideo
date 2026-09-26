# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the VSA-H3 backend: tiling geometry, mask policy, and
end-to-end equivalence against dense SDPA through a token-level mask
reference. The same reference doubles as the GPU kernel parity oracle."""

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
        MiniMaxH3VSAMetadataBuilder().build(current_timestep=0, raw_latent_shape=_TINY["raw_latent_shape"],
                                            patch_size=_TINY["patch_size"], VSA_sparsity=0.0,
                                            prefix_segments=_TINY["prefix_segments"], device=_CPU, exempt=False,
                                            tile_layout="chunk256", merge_prefix=True)


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
