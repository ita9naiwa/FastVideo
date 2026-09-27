# SPDX-License-Identifier: Apache-2.0
"""VSA-H3 tile 128: cube128 (4,4,8) and chunk128(-merged-prefix) geometry, the FA4 Q128/KV128 training op pair
(tile-parameterized vsa_train_fwd/bwd, ruling 84) against a dense masked reference, and the h3mh compile contract.

Tolerances (declared before the first GPU run):
- vs the fp32 dense masked reference: O (avg_abs < 1e-3, max_rel < 0.2), dQ/dK/dV (avg_abs < 2e-2, max_rel < 0.5) = the
  dense-reference tolerances of fastvideo-kernel/tests/test_vsa128_backward.py for the same kernels.
- vs the library VSA-128 autograd route (_CuteAttentionQ128, same kernels) and compiled vs eager: O, dK, dV bitwise; dQ
  (FP32 atomics) within the campaign _GRAD_TOL (avg_abs < 1e-3, max_rel < 0.25).
"""

import functools
import math

import pytest
import torch
import torch.nn.functional as F

from fastvideo.attention.backends import video_sparse_attn_h3 as h3
from fastvideo.attention.backends.video_sparse_attn_h3 import (MiniMaxH3VSAImpl, _build_block_mask, _pool_tiles,
                                                               _validate_h3_tile_geometry, token_tile_and_valid)
from fastvideo.tests.attention.test_vsa_h3_metadata import (_H3_REAL_GRIDS, _S085K32_DOCS, _build, _impl,
                                                            reference_sparse_attention)

_RAGGED = dict(raw_latent_shape=(9, 20, 26), patch_size=(1, 2, 2), prefix_segments=(70, 5, 130))
_CHUNK = dict(tile_size=128, tile_layout="chunk128")
_DENSE_TOL = {"O": (1e-3, 0.2), "dQ": (2e-2, 0.5), "dK": (2e-2, 0.5), "dV": (2e-2, 0.5)}
_GRAD_TOL = (1e-3, 0.25)
_requires_sm100 = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                                     reason="SM100 GPU required")


def _grid_spec(grid, prefix):
    return dict(raw_latent_shape=(grid[0], 2 * grid[1], 2 * grid[2]), patch_size=(1, 2, 2), prefix_segments=prefix)


def test_tile128_cube_geometry_ragged():
    """Hand-computed (4,4,8) oracle on a grid ragged in all three dims; the partition refines the 256 cube one."""
    meta = _build(_RAGGED, tile_size=128)
    t, h, w = 9, 10, 13
    n_t, n_h, n_w = 3, 3, 2
    prefix_len = 205
    assert meta.tile_elems == 128 and meta.tile_layout == "cube"
    assert meta.variable_block_sizes[:meta.num_prefix_tiles].tolist() == [70, 5, 128, 2]
    expected = torch.tensor([
        min(4, t - 4 * tt) * min(4, h - 4 * hh) * min(8, w - 8 * ww) for tt in range(n_t) for hh in range(n_h)
        for ww in range(n_w)
    ])
    assert meta.num_video_tiles == n_t * n_h * n_w
    assert torch.equal(meta.variable_block_sizes[meta.num_prefix_tiles:], expected)
    idx = meta.untile_combined_index
    row = torch.arange(t * h * w)
    row_t, row_h, row_w = row // (h * w), (row // w) % h, row % w
    tile128 = ((row_t // 4) * n_h + row_h // 4) * n_w + row_w // 8
    assert torch.equal(idx[prefix_len:] // 128 - meta.num_prefix_tiles, tile128)
    meta256 = _build(_RAGGED)
    tile256 = meta256.untile_combined_index[prefix_len:] // 256 - meta256.num_prefix_tiles
    assert torch.equal(tile256, ((row_t // 4) * 2 + row_h // 8) * n_w + row_w // 8)  # 256 tile = h-pair of 128 tiles
    x = torch.randn(1, meta.total_seq_length, 2, 4)
    buf = _impl().tile(x, meta)
    assert buf.shape[1] == meta.variable_block_sizes.numel() * 128
    assert torch.equal(buf[:, idx], x)
    assert not buf[:, ~token_tile_and_valid(meta.variable_block_sizes, 128)[1]].any(), "pad rows must stay zero"


@pytest.mark.parametrize("merge_prefix", [False, True])
@pytest.mark.parametrize("grid,prefix,n_video", _H3_REAL_GRIDS)
def test_tile128_chunk_real_grids(grid, prefix, n_video, merge_prefix):
    meta = _build(_grid_spec(grid, prefix), merge_prefix=merge_prefix, **_CHUNK)
    cube = _build(_grid_spec(grid, prefix), tile_size=128)
    sizes, P, video = meta.variable_block_sizes, meta.num_prefix_tiles, math.prod(grid)
    n128 = math.ceil(video / 128)
    assert meta.tile_layout == "chunk128" and meta.num_video_tiles == n128
    assert sizes[P:-1].eq(128).all() and int(sizes[-1]) == video - 128 * (n128 - 1)
    segments = [x for x in prefix if x > 0]
    if merge_prefix:
        assert math.ceil(sum(segments) / 128) == P and sizes[:P - 1].eq(128).all()
    else:
        assert cube.num_prefix_tiles == P and torch.equal(sizes[:P], cube.variable_block_sizes[:P])
    x = torch.randn(1, meta.total_seq_length, 1, 2)
    impl = _impl()
    buf, cube_buf = impl.tile(x, meta), impl.tile(x, cube)
    assert torch.equal(buf[:, meta.untile_combined_index], x)
    keep = lambda b, m: b[:, token_tile_and_valid(m.variable_block_sizes, m.tile_elems)[1]]
    assert torch.equal(keep(buf, meta), keep(cube_buf, cube)), "same (4,4,8) cube token order, only boundaries move"


@pytest.mark.parametrize("layout", [{}, dict(tile_layout="chunk128"), dict(tile_layout="chunk128", merge_prefix=True)],
                         ids=["cube128", "chunk128", "chunk128-merged-prefix"])
def test_tile128_index_certificates(layout):
    """Builder plans: untile = nonpad[argsort(partition)], inverse source / pad rows exact, versions recorded at 0."""
    meta = _build(_grid_spec(*_S085K32_DOCS[1]), 0.85, tile_size=128, topk_cap=64, **layout)
    partition, nonpad, untile = meta.tile_partition_indices, meta.non_pad_index, meta.untile_combined_index
    padded = meta.variable_block_sizes.numel() * 128
    assert torch.equal(untile, nonpad[torch.argsort(partition)])
    assert meta._tile_index_state == (partition, 0, nonpad, 0)
    state = meta._untile_grad_state
    source, pad_rows = state[2], state[4]
    assert state[0] is untile and source.numel() == padded
    assert torch.equal(source[nonpad], partition)
    valid = token_tile_and_valid(meta.variable_block_sizes, 128)[1]
    assert torch.equal(pad_rows, torch.nonzero(~valid).flatten())
    assert torch.equal(torch.sort(torch.cat([nonpad, pad_rows])).values, torch.arange(padded))


def test_tile128_guard_and_layout_validation():
    meta = _build(_RAGGED, tile_size=128)
    sizes = meta.variable_block_sizes.clone()
    sizes[0] = 129  # passes the 256 bound, must fail the 128 one
    with pytest.raises(ValueError, match="tile sizes out of bounds"):
        _validate_h3_tile_geometry((70, 5, 130), (9, 10, 13), sizes, meta.untile_combined_index, 128)
    _validate_h3_tile_geometry((70, 5, 130), (9, 10, 13), meta.variable_block_sizes, meta.untile_combined_index, 128)
    with pytest.raises(ValueError, match="chunk128"):
        _build(_RAGGED, tile_layout="chunk128")  # tile_size 256
    with pytest.raises(ValueError, match="chunk256"):
        _build(_RAGGED, tile_size=128, tile_layout="chunk256")
    with pytest.raises(ValueError, match="merge_prefix"):
        _build(_RAGGED, tile_size=128, merge_prefix=True)
    with pytest.raises(ValueError, match="tile_layout"):
        h3._h3_tile_geometry((70, ), (9, 10, 13), torch.device("cpu"), (4, 8, 8), "chunk128")
    with pytest.raises(ValueError, match="'chunk' supports"):
        _build(_RAGGED, tile_size=64, tile_layout="chunk")


# Every accepted builder layout name x tile: (name, merge_prefix kwarg) -> canonical (tile_layout, merged).
_LAYOUT_NAMES = [("cube", None, "cube", False), ("cube{T}", None, "cube", False), ("chunk", None, "chunk{T}", False),
                 ("chunk", True, "chunk{T}", True), ("chunk{T}", None, "chunk{T}", False), ("chunk{T}", True, "chunk{T}", True),
                 ("chunk-merged-prefix", None, "chunk{T}", True), ("chunk-merged-prefix", True, "chunk{T}", True),
                 ("chunk{T}-merged-prefix", None, "chunk{T}", True), ("chunk{T}-merged-prefix", True, "chunk{T}", True)]


@pytest.mark.parametrize("tile", [128, 256])
@pytest.mark.parametrize("name,merge,canonical,merged", _LAYOUT_NAMES)
def test_builder_layout_names(tile, name, merge, canonical, merged):
    """Tile-neutral and sized aliases resolve to the SAME cached geometry tensors as the canonical call."""
    kw = {} if merge is None else {"merge_prefix": merge}
    meta = _build(_RAGGED, tile_size=tile, tile_layout=name.format(T=tile), **kw)
    ref = _build(_RAGGED, tile_size=tile, tile_layout=canonical.format(T=tile), merge_prefix=merged)
    assert meta.tile_layout == canonical.format(T=tile)
    assert meta.variable_block_sizes is ref.variable_block_sizes and meta.untile_combined_index is ref.untile_combined_index
    assert (getattr(meta, "_query_pad_state", None) is not None) == (canonical == "cube")


@pytest.mark.parametrize("tile,name,kw,match", [
    (256, "cube128", {}, "requires tile_size=128"), (128, "cube256", {}, "requires tile_size=256"),
    (256, "chunk128", {}, "requires tile_size=128"), (128, "chunk256", {}, "requires tile_size=256"),
    (256, "chunk128-merged-prefix", {}, "requires tile_size=128"), (128, "chunk256-merged-prefix", {}, "requires tile_size=256"),
    (128, "chunk-merged-prefix", {"merge_prefix": False}, "implies merge_prefix=True"),
    (256, "chunk256-merged-prefix", {"merge_prefix": False}, "implies merge_prefix=True"),
    (128, "cube-merged-prefix", {}, "tile_layout must be"), (128, "cube128", {"merge_prefix": True}, "merge_prefix requires"),
    (64, "chunk", {}, "'chunk' supports"), (128, "chunk64", {}, "tile_layout must be"),
    (128, "chunk", {"merge_prefix": True, "exempt": False}, "merge_prefix requires")])
def test_builder_layout_name_errors(tile, name, kw, match):
    with pytest.raises(ValueError, match=match):
        _build(_RAGGED, tile_size=tile, tile_layout=name, **kw)


@pytest.mark.parametrize("cap", [64, 32])
@pytest.mark.parametrize("grid,prefix", _S085K32_DOCS)
def test_tile128_topk_rule(grid, prefix, cap):
    for layout in ({}, dict(tile_layout="chunk128", merge_prefix=True)):
        meta = _build(_grid_spec(grid, prefix), 0.85, tile_size=128, topk_cap=cap, **layout)
        n = meta.num_video_tiles
        assert meta.video_topk == max(1, min((3 * n + 19) // 20, cap, n))


def test_tile128_sparsity_zero_matches_dense_sdpa():
    torch.manual_seed(3)
    for layout in ({}, dict(tile_layout="chunk128"), dict(tile_layout="chunk128", merge_prefix=True)):
        meta = _build(_RAGGED, tile_size=128, **layout)
        q, k, v = (torch.randn(1, meta.total_seq_length, 2, 8) for _ in range(3))
        impl = _impl()
        tq, tk, tv = (impl.tile(t, meta).clone() for t in (q, k, v))
        scores = torch.matmul(_pool_tiles(tq, meta.variable_block_sizes, 128),
                              _pool_tiles(tk, meta.variable_block_sizes, 128).transpose(-2, -1))
        mask = _build_block_mask(scores, meta.num_prefix_tiles, meta.num_video_tiles, 0.0, exempt=True)
        sparse_out = impl.postprocess_output(reference_sparse_attention(tq, tk, tv, mask, meta), meta)
        dense = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        assert torch.allclose(sparse_out, dense, atol=1e-5), (layout, (sparse_out - dense).abs().max())


# ---------------------------------------------------------------------------------------------------------------------
# GPU: FA4 Q128/KV128 training route
# ---------------------------------------------------------------------------------------------------------------------

_HEADS, _DIM = 2, 128
_GPU_CASES = [(_S085K32_DOCS[0], {}), (_S085K32_DOCS[0], dict(tile_layout="chunk128", merge_prefix=True)),
              (_S085K32_DOCS[2], {}), (_S085K32_DOCS[2], dict(tile_layout="chunk128", merge_prefix=True))]
_GPU_IDS = ["cube128-47x7x12", "chunk128mp-47x7x12", "cube128-37x12x31", "chunk128mp-37x12x31"]


@pytest.fixture
def cute(monkeypatch):
    pytest.importorskip("flash_attn.cute.block_sparsity", reason="optional FA4 CuTe build not installed")
    monkeypatch.setenv("FASTVIDEO_VSA_CUTEDSL", "1")
    for name in ("FASTVIDEO_VSA_TRITON", "FASTVIDEO_KERNEL_VSA_FORCE_TRITON", "FASTVIDEO_VSA_VC"):
        monkeypatch.delenv(name, raising=False)


def _metrics(ref, got):
    diff = (ref.float() - got.float()).abs()
    return float(diff.mean()), float(diff.max() / (ref.float().abs().mean() + 1e-6))


def _gpu_meta(case, cap=64):
    (grid, prefix), layout = case
    return _build(_grid_spec(grid, prefix), 0.85, torch.device("cuda"), tile_size=128, topk_cap=cap, **layout)


def _inputs(meta, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(1, meta.total_seq_length, _HEADS, _DIM, device="cuda", dtype=torch.bfloat16, generator=g)
            for _ in range(4)]


def _block(impl, q, k, v, meta):
    x = impl.preprocess_qkv(torch.cat([q, k, v], dim=0), meta)
    return impl.postprocess_output(impl.forward_qkv(x, meta), meta)


def _run(fn, meta, inputs):
    q, k, v = (t.detach().clone().requires_grad_(True) for t in inputs[:3])
    out = fn(q, k, v, meta)
    return (out.detach(), *torch.autograd.grad(out, (q, k, v), inputs[3]))


def _dense_reference(meta, mask, inputs):
    """fp32 masked attention on the packed rows: row i may attend row j iff mask[tile(i), tile(j)]."""
    tile = meta.untile_combined_index // 128
    q, k, v = (t.detach().float().requires_grad_(True) for t in inputs[:3])
    allow = mask[0][:, tile][:, :, tile]  # [H, S, S]
    logits = torch.einsum("shd,thd->hst", q[0], k[0]) / math.sqrt(_DIM)
    out = torch.einsum("hst,thd->shd", torch.softmax(logits.masked_fill(~allow, float("-inf")), -1), v[0])[None]
    return (out.detach(), *torch.autograd.grad(out, (q, k, v), inputs[3].float()))


@_requires_sm100
@pytest.mark.parametrize("case", _GPU_CASES, ids=_GPU_IDS)
def test_tile128_fwd_bwd_matches_dense_reference(cute, monkeypatch, case):
    meta = _gpu_meta(case)
    impl = MiniMaxH3VSAImpl(num_heads=_HEADS, head_size=_DIM, causal=False, softmax_scale=_DIM**-0.5)
    inputs = _inputs(meta)
    maps = []
    build = h3._build_block_mask
    monkeypatch.setattr(h3, "_build_block_mask", lambda *a, **k: maps.append(build(*a, **k)) or maps[-1])
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        got = _run(functools.partial(_block, impl), meta, inputs)
    names = [e.name for e in prof.events()]
    assert "fastvideo_kernel::vsa_train_fwd" in names and "fastvideo_kernel::vsa_train_bwd" in names
    assert len(maps) == 1 and not maps[0].all(), "the case must be sparse"
    ref = _dense_reference(meta, maps[0], inputs)
    for name, r, g in zip(("O", "dQ", "dK", "dV"), ref, got, strict=True):
        avg, rel = _metrics(r, g)
        print(f"[tile128 dense-ref {case[1] or 'cube'}] {name}: avg_abs={avg:.3e} max_rel={rel:.3e}")
        assert torch.isfinite(g).all() and avg < _DENSE_TOL[name][0] and rel < _DENSE_TOL[name][1], (name, avg, rel)


@_requires_sm100
@pytest.mark.parametrize("case", _GPU_CASES[:2], ids=_GPU_IDS[:2])
def test_tile128_op_pair_matches_library_route(cute, monkeypatch, case):
    """Same kernels as the library's _CuteAttentionQ128 autograd route: O/dK/dV bitwise, dQ at FP32-atomic noise."""
    meta = _gpu_meta(case)
    impl = MiniMaxH3VSAImpl(num_heads=_HEADS, head_size=_DIM, causal=False, softmax_scale=_DIM**-0.5)
    inputs = _inputs(meta, seed=1)
    op = _run(functools.partial(_block, impl), meta, inputs)
    monkeypatch.setattr(h3.vsa256_ops, "training_eligible", lambda *a, **k: False)  # -> block_sparse_attn_128_bshd
    sizes32 = meta.variable_block_sizes.to(torch.int32)  # the op bodies cast to int32; give the library the same lists
    monkeypatch.setattr(meta, "variable_block_sizes", sizes32)
    lib = _run(functools.partial(_block, impl), meta, inputs)
    for name, i in (("O", 0), ("dK", 2), ("dV", 3)):
        assert torch.equal(op[i], lib[i]), name
    avg, rel = _metrics(lib[1], op[1])
    assert avg < _GRAD_TOL[0] and rel < _GRAD_TOL[1], (avg, rel)


@_requires_sm100
def test_train_op_tile256_is_the_vsa256_default_route(cute):
    """vsa_train_fwd(tile=256) delegates to vsa256_fwd/bwd with default options: O, dK, dV bitwise; dQ at FP32-atomic noise."""
    torch.manual_seed(5)
    n, heads = 6, 2
    sizes = torch.tensor([256, 200, 256, 131, 256, 17], device="cuda")
    block_map = torch.rand(1, heads, n, n, device="cuda") < 0.5
    block_map[..., 0] = True
    base = [torch.randn(1, n * 256, heads, _DIM, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    ops = torch.ops.fastvideo_kernel
    runs = []
    for fn in (lambda q, k, v: ops.vsa_train_fwd(q, k, v, block_map, sizes, 256),
               lambda q, k, v: ops.vsa256_fwd(q, k, v, block_map, sizes, None, None, 0, 0, False)):
        q, k, v = (t.clone().requires_grad_(True) for t in base[:3])
        out = fn(q, k, v)[0]
        runs.append((out.detach(), *torch.autograd.grad(out, (q, k, v), base[3])))
    for name, i in (("O", 0), ("dK", 2), ("dV", 3)):
        assert torch.equal(runs[0][i], runs[1][i]), name
    avg, rel = _metrics(runs[1][1], runs[0][1])
    assert avg < _GRAD_TOL[0] and rel < _GRAD_TOL[1], (avg, rel)


@_requires_sm100
@pytest.mark.parametrize("tile", [64, 512])
def test_train_ops_reject_unsupported_tile(cute, tile):
    """vsa_train_fwd and vsa_train_bwd both reject tiles other than 128/256 (no silent fall-through to the 128 path)."""
    q = torch.zeros(1, 256, 1, _DIM, device="cuda", dtype=torch.bfloat16)
    sizes = torch.tensor([128, 128], device="cuda")
    block_map = torch.ones(1, 1, 2, 2, device="cuda", dtype=torch.bool)
    lse = torch.zeros(1, 1, 256, device="cuda")
    ops = torch.ops.fastvideo_kernel
    with pytest.raises(ValueError, match="vsa_train_fwd supports tile 128 or 256"):
        ops.vsa_train_fwd(q, q, q, block_map, sizes, tile)
    with pytest.raises(ValueError, match="vsa_train_bwd supports tile 128 or 256"):
        ops.vsa_train_bwd(q, q, q, q, q, lse, block_map, sizes, tile)


@_requires_sm100
def test_tile128_h3mh_compile_contract(cute, monkeypatch):
    """h3mh block compile (inductor, dynamic, fullgraph, fail_on_recompile_limit_hit, use_duck_shape=False, SAC in-region
    MUST_SAVE {vsa256_fwd, vsa_train_fwd, vsa_h3_block_map}): one graph over several doc shapes, no breaks, no recompiles,
    compiled == eager (O/dK/dV bitwise, dQ within _GRAD_TOL, same block maps)."""
    import torch.fx.experimental._config as fx_config
    from torch._dynamo.utils import counters
    from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts
    monkeypatch.setattr(torch._dynamo.config, "fail_on_recompile_limit_hit", True)
    monkeypatch.setattr(fx_config, "use_duck_shape", False)
    ops = torch.ops.fastvideo_kernel
    must_save = {ops.vsa256_fwd.default, ops.vsa_train_fwd.default, ops.vsa_h3_block_map.default}
    impl = MiniMaxH3VSAImpl(num_heads=_HEADS, head_size=_DIM, causal=False, softmax_scale=_DIM**-0.5)
    impl.layer_idx = 0
    policy = lambda ctx, op, *a, **k: CheckpointPolicy.MUST_SAVE if op in must_save else CheckpointPolicy.PREFER_RECOMPUTE
    block = functools.partial(_block, impl)

    def checkpointed(q, k, v, meta):
        return checkpoint(block, q, k, v, meta, use_reentrant=False,
                          context_fn=functools.partial(create_selective_checkpoint_contexts, policy))

    maps = []
    build = h3._build_block_mask
    monkeypatch.setattr(h3, "_build_block_mask", lambda *a, **k: maps.append(build(*a, **k)) or maps[-1])
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(checkpointed, backend="inductor", mode="default", dynamic=True, fullgraph=True)
    layout = dict(tile_layout="chunk128", merge_prefix=True)
    try:
        for i, doc in enumerate(_S085K32_DOCS):
            meta = _gpu_meta((doc, layout))
            inputs = _inputs(meta, seed=10 + i)
            maps.clear()
            c = _run(compiled, meta, inputs)
            e = _run(block, meta, inputs)
            assert len(maps) == 2 and torch.equal(maps[0], maps[1])
            for name, j in (("O", 0), ("dK", 2), ("dV", 3)):
                assert torch.equal(c[j], e[j]), (i, name)
            avg, rel = _metrics(e[1], c[1])
            assert avg < _GRAD_TOL[0] and rel < _GRAD_TOL[1], (i, avg, rel)
        assert counters["stats"]["unique_graphs"] == 1
        assert sum(counters["graph_break"].values()) == 0
    finally:
        torch._dynamo.reset()

