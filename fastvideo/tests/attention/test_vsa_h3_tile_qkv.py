# SPDX-License-Identifier: Apache-2.0
"""preprocess_q_k_v tiles separate q, k, v without a stacked copy: byte-identical to preprocess_qkv(cat([q, k, v]))."""

import os
import re
import types

import pytest
import torch

from fastvideo.attention import layer
from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAImpl, MiniMaxH3VSAMetadataBuilder
from fastvideo.tests.attention.test_vsa_h3_metadata import _S085K32_DOCS
from fastvideo.tests.attention.test_vsa_h3_vc_compiled_route import h3  # noqa: F401 (VC route fixture)

# token grid (37, 24, 42) + text/audio prefix: 38,010 rows; 3 x [1, S, 4, 128] is >= 2**25 elements (eligible eager).
_SPEC = dict(raw_latent_shape=(37, 48, 84), patch_size=(1, 2, 2), prefix_segments=(300, 0, 414))
_LAYOUTS = [dict(), dict(tile_layout="chunk256", merge_prefix=True)]
_CAT = re.compile(r"_cat_|_cat\b|aten\.cat|torch\.cat")  # a cat kernel or call in Inductor output (not "allocate")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _impl():
    return MiniMaxH3VSAImpl(num_heads=4, head_size=128, causal=False, softmax_scale=128**-0.5, prefix="blocks.0.attn")


def _metadata(**layout):
    return MiniMaxH3VSAMetadataBuilder().build(current_timestep=0,
                                               VSA_sparsity=0.75,
                                               device=torch.device("cuda"),
                                               **_SPEC,
                                               **layout)


def _qkv(md, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [
        torch.randn(1, md.total_seq_length, 4, 128, device="cuda", dtype=torch.bfloat16,
                    generator=g).requires_grad_(True) for _ in range(3)
    ]


def _both(impl, md, q, k, v, up):
    reference = impl.preprocess_qkv(torch.cat([q, k, v], dim=0), md)
    ref_grads = torch.autograd.grad(reference, (q, k, v), up)
    actual = impl.preprocess_q_k_v(q, k, v, md)
    assert isinstance(actual, tuple) and len(actual) == 3
    grads = torch.autograd.grad(actual, (q, k, v), up.chunk(3, dim=0))
    return reference, ref_grads, torch.cat(actual, dim=0), grads


@pytest.mark.parametrize("layout", _LAYOUTS, ids=["cube", "chunk256"])
def test_separate_qkv_matches_stacked(layout):
    impl, md = _impl(), _metadata(**layout)
    q, k, v = _qkv(md)
    up = torch.randn(3, md.variable_block_sizes.numel() * 256, 4, 128, device="cuda", dtype=torch.bfloat16)
    reference, ref_grads, actual, grads = _both(impl, md, q, k, v, up)
    assert actual.shape == reference.shape and torch.equal(actual, reference)
    assert all(torch.equal(a, b) for a, b in zip(grads, ref_grads, strict=True))


def test_untrusted_index_falls_back_exactly():
    impl, md = _impl(), _metadata(tile_layout="chunk256", merge_prefix=True)
    q, k, v = _qkv(md, seed=3)
    up = torch.randn(3, md.variable_block_sizes.numel() * 256, 4, 128, device="cuda", dtype=torch.bfloat16)
    state = md._tile_index_state
    md._tile_index_state = (state[0], state[1] - 1, state[2], state[3])  # recorded version no longer matches
    try:
        reference, ref_grads, actual, grads = _both(impl, md, q, k, v, up)
    finally:
        md._tile_index_state = state
    assert torch.equal(actual, reference) and all(torch.equal(a, b) for a, b in zip(grads, ref_grads, strict=True))


def _holder_path(impl, md, x):
    """Today's no-grad route: preprocess_qkv into the builder-owned holder (trusted index state dropped), cloned."""
    state = md.__dict__.pop("_tile_index_state")
    try:
        md.tile_buf_holder.buffer = None
        out = impl.preprocess_qkv(x, md)
        assert out is md.tile_buf_holder.buffer
        return out.clone()
    finally:
        md._tile_index_state = state


def _real_doc(doc, tile):
    (t, h, w), prefix = doc
    return MiniMaxH3VSAMetadataBuilder().build(current_timestep=0,
                                               VSA_sparsity=0.85,
                                               device=torch.device("cuda"),
                                               raw_latent_shape=(t, 2 * h, 2 * w),
                                               patch_size=(1, 2, 2),
                                               prefix_segments=prefix,
                                               tile_size=tile,
                                               tile_layout=f"chunk{tile}",
                                               merge_prefix=True)


@pytest.mark.parametrize("tile", [256, 128])
@pytest.mark.parametrize("doc", _S085K32_DOCS[:2], ids=lambda d: "x".join(map(str, d[0])))
def test_nograd_matches_holder_path_real_docs(doc, tile):
    """s085k32 real-shape docs: no-grad preprocess_q_k_v and preprocess_qkv/tile (eager and compiled) equal today's
    cat + holder route bitwise, never return the holder, and build no autograd state (inputs require grad)."""
    impl, md = _impl(), _real_doc(doc, tile)
    q, k, v = _qkv(md, seed=21)
    qkv = torch.cat([q, k, v], dim=0).detach()
    expected = _holder_path(impl, md, qkv)
    with torch.no_grad():
        routes = {
            "separate": lambda a, b, c: torch.cat(impl.preprocess_q_k_v(a, b, c, md), dim=0),
            "stacked": lambda a, b, c: impl.preprocess_qkv(torch.cat([a, b, c], dim=0), md),
        }
        for name, fn in routes.items():
            torch._dynamo.reset()
            for mode, run in (("eager", fn), ("compiled", torch.compile(fn, fullgraph=True, dynamic=True))):
                out = run(q, k, v)
                assert torch.equal(out, expected), (name, mode)
                assert not out.requires_grad and out.grad_fn is None, (name, mode)
                holder = md.tile_buf_holder.buffer  # compiled tile-128 tile() owns its output since op-family-v2
                assert holder is None or out.data_ptr() != holder.data_ptr(), (name, mode)
    torch._dynamo.reset()


@pytest.mark.parametrize("tile", [256, 128])
def test_nograd_outputs_are_invocation_owned(tile):
    """Two consecutive no-grad calls: the second never overwrites the first result (preprocess_q_k_v and tile)."""
    impl, md = _impl(), _real_doc(_S085K32_DOCS[0], tile)
    a, b = ([t.detach() for t in _qkv(md, seed=s)] for s in (31, 32))
    with torch.no_grad():
        first = impl.preprocess_q_k_v(*a, md)
        snapshot = [t.clone() for t in first]
        second = impl.preprocess_q_k_v(*b, md)
        assert all(torch.equal(x, y) for x, y in zip(first, snapshot, strict=True))
        assert not any(torch.equal(x, y) for x, y in zip(first, second, strict=True))
        t1 = impl.tile(a[0], md)
        s1 = t1.clone()
        impl.tile(b[0], md)
        assert torch.equal(t1, s1)


def test_nograd_ineligible_calls_take_the_stacked_route():
    """Untrusted metadata and non-BF16 no-grad calls keep cat + preprocess_qkv (the holder)."""
    impl, md = _impl(), _metadata()
    q, k, v = (t.detach() for t in _qkv(md, seed=5))
    with torch.no_grad():
        state = md.__dict__.pop("_tile_index_state")
        try:
            out = impl.preprocess_q_k_v(q, k, v, md)
            assert out[0].data_ptr() == md.tile_buf_holder.buffer.data_ptr()
        finally:
            md._tile_index_state = state
        out = impl.preprocess_q_k_v(q.float(), k.float(), v.float(), md)
        assert out[0].data_ptr() == md.tile_buf_holder.buffer.data_ptr()


def test_permute_ops_opcheck():
    """Schema / fake / autograd-registration / AOT checks of the tile gather ops the no-grad routes now call."""
    md = _real_doc(_S085K32_DOCS[0], 256)
    partition, _, nonpad, _ = md._tile_index_state
    padded = md.variable_block_sizes.numel() * 256
    q, k, v = (t.detach()[:, :, :1] for t in _qkv(md, seed=41))
    idx = (partition, nonpad, md.untile_combined_index, padded, 0, 0)
    for grad in (False, True):
        args = [x.clone().requires_grad_(grad) for x in (q, k, v)]
        torch.library.opcheck(torch.ops.fastvideo_kernel.vsa_tile_permute_qkv_fwd, (*args, *idx))
        torch.library.opcheck(torch.ops.fastvideo_kernel.vsa_tile_permute_fwd, (args[0], *idx))


def test_compiled_fullgraph_single_graph():
    impl, md = _impl(), _metadata(tile_layout="chunk256", merge_prefix=True)
    q, k, v = _qkv(md, seed=7)
    torch._dynamo.reset()
    counts = torch._dynamo.utils.counters
    counts.clear()
    fn = torch.compile(lambda a, b, c: impl.preprocess_q_k_v(a, b, c, md), fullgraph=True, dynamic=True)
    out = fn(q, k, v)
    assert torch.equal(torch.cat(out, dim=0), impl.preprocess_qkv(torch.cat([q, k, v], dim=0), md))
    fn(*_qkv(md, seed=8))  # same shapes: no recompile
    assert counts["stats"]["unique_graphs"] == 1


class _RecordingImpl:
    """Stand-in backend that records which preprocess/forward route the layer takes."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name not in ("preprocess_q_k_v", "preprocess_qkv", "forward", "forward_qkv", "postprocess_output"):
            raise AttributeError(name)

        def record(*args, tile_sums=None):
            self.calls.append(name)
            if name == "preprocess_q_k_v":
                return (*args[:3], None) if tile_sums else args[:3]
            return args[0][:1] if name == "forward_qkv" else args[0]

        return record


@pytest.mark.parametrize("vsa", [False, True], ids=["dense", "vsa"])
@pytest.mark.parametrize("sp,gate", [(1, False), (2, False), (1, True)], ids=["sp1", "sp2", "gate"])
def test_route_selection(monkeypatch, vsa, sp, gate):
    """SP=1 (with or without a gate) tiles q, k, v (and the gate) separately; SP>1 keeps the stacked route."""
    if gate and not vsa:
        pytest.skip("gate_compress is VSA-only")
    monkeypatch.setattr(layer, "get_sp_world_size", lambda: sp)
    monkeypatch.setattr(layer, "get_sp_parallel_rank", lambda: 0)
    monkeypatch.setattr(layer, "get_forward_context", lambda: types.SimpleNamespace(attn_metadata=None))
    monkeypatch.setattr(layer, "sequence_model_parallel_all_to_all_4D", lambda x, **_: x)
    impl = _RecordingImpl()
    this = types.SimpleNamespace(attn_impl=impl, _compile_forward_enabled=True)
    q, k, v = (torch.randn(1, 8, 2, 4) for _ in range(3))
    if vsa:
        layer.DistributedAttention_VSA.forward(this, q, k, v, 8, gate_compress=torch.randn_like(q) if gate else None)
    else:
        layer.DistributedAttention.forward(this, q, k, v, 8)
    separate = sp == 1
    assert ("preprocess_q_k_v" in impl.calls) == separate
    assert ("preprocess_qkv" in impl.calls) == (not separate or gate)  # a separate gate: single-operand tile
    assert ("forward_qkv" in impl.calls) == (vsa and not separate and not gate)  # VSA stacked no-gate: fused-QKV input


@pytest.mark.parametrize("layout",
                         [*_LAYOUTS, dict(tile_size=128, tile_layout="chunk128", merge_prefix=True)],
                         ids=["cube", "chunk256", "chunk128"])
def test_compiled_nograd_separate_route_has_no_cat(monkeypatch, layout):
    """Compiled tile-256/128 inference (h3mh-style regional-compile no-grad route, from worker-3's form-(c) test):
    preprocess_q_k_v gathers q, k, v separately inside the compiled region, so the captured graph has no cat and
    Inductor emits no pointwise cat kernel; one graph, no recompile, output bitwise equal to the eager stacked block.
    Tile 128 feeds the separate gather to the op-family tile-parameterised no-grad op (vsa_nograd_fwd, tile 128)."""
    import torch.fx.experimental._config as fx_config
    from torch._dynamo.testing import CompileCounterWithBackend
    from torch._inductor.utils import run_and_get_code
    monkeypatch.setenv("FASTVIDEO_VSA_CUTEDSL", "1")
    monkeypatch.delenv("FASTVIDEO_VSA_VC", raising=False)
    monkeypatch.setattr(torch._dynamo.config, "fail_on_recompile_limit_hit", True)
    monkeypatch.setattr(fx_config, "use_duck_shape", False)
    impl, md = _impl(), _metadata(**layout)
    assert impl.prepare_for_regional_compile(torch.device("cuda")) is None
    assert impl._regional_compile_nograd_route == "bf16"

    def separate(q, k, v):
        tq, tk, tv = impl.preprocess_q_k_v(q, k, v, md)
        return impl.postprocess_output(impl.forward(tq, tk, tv, None, md), md)

    def stacked(q, k, v):
        tq, tk, tv = impl.preprocess_qkv(torch.cat([q, k, v], dim=0), md).chunk(3, dim=0)
        return impl.postprocess_output(impl.forward(tq, tk, tv, None, md), md)

    def inductor_code(fn, *x):
        # Generated wrapper + kernel source, compiled fresh (no FX-graph cache replay). The CUDA profiler is not used:
        # after earlier suite tests it can record no kernels at all, which made worker-3's kernel-name checks vacuous.
        torch._dynamo.reset()
        with torch._inductor.config.patch(fx_graph_cache=False):
            out, code = run_and_get_code(fn, *x)
        return out, "\n".join(code)

    def has_cat(gm):
        return any(n.target in (torch.cat, torch.ops.aten.cat.default) for n in gm.graph.nodes)

    q, k, v = (t.detach() for t in _qkv(md, seed=11))
    q2, k2, v2 = (t.detach() for t in _qkv(md, seed=12))
    with torch.no_grad():
        eager = stacked(q, k, v)
        eager2 = stacked(q2, k2, v2)
        # Control: the stacked block concatenates in the captured graph and in Inductor's generated code.
        control = CompileCounterWithBackend("inductor")
        out, code = inductor_code(torch.compile(stacked, backend=control, dynamic=True, fullgraph=True), q, k, v)
        assert torch.equal(out, eager) and has_cat([gm for gm in control.graphs if len(gm.graph.nodes)][0])
        assert _CAT.search(code), "control: the stacked compiled no-grad block lowers the q/k/v cat"

        counters = torch._dynamo.utils.counters
        counters.clear()
        cnt = CompileCounterWithBackend("inductor")
        sep = torch.compile(separate, backend=cnt, dynamic=True, fullgraph=True)
        out, code = inductor_code(sep, q, k, v)
        assert torch.equal(out, eager)
        assert "vsa_tile_permute_qkv_fwd" in code and not _CAT.search(code)
        assert md.tile_elems == 256 or "vsa_nograd_fwd" in code
        with torch._dynamo.config.patch(error_on_recompile=True):
            assert torch.equal(sep(q2, k2, v2), eager2)
        graphs = [gm for gm in cnt.graphs if len(gm.graph.nodes)]  # Dynamo may also hand the backend an empty graph
        assert counters["stats"]["unique_graphs"] == 1 and len(graphs) == 1 and not counters["graph_break"]
        assert not has_cat(graphs[0])
    torch._dynamo.reset()


# E2 witness (init/nograd-separate-gather-audit-explorer-2/prequeue/compiled-vc-rebase): full-valid latent (4, 8, 16),
# cube tile (4, 8, 8), no prefix, N = Np = 512 rows and a non-identity permutation P, so a tiled gather reaching the VC
# producer would be mapped twice (P(P(x))) without any length error. The padded s085k32 doc (47x7x12) would instead
# lose the fused producer (tiled length != packed length). Tile 128 (fused VC route since VC-128): the same latent with
# cube tile (4, 4, 8) is 4 full tiles, N = Np = 512, again non-identity; the padded doc runs as cube and chunk128.
_VC_CASES = [("witness", (4, 8, 16), (), "cube", 256), ("padded", (47, 7, 12), (87, 1, 84, 514), "cube", 256),
             ("padded", (47, 7, 12), (87, 1, 84, 514), "chunk", 256), ("witness", (4, 8, 16), (), "cube", 128),
             ("padded", (47, 7, 12), (87, 1, 84, 514), "cube", 128),
             ("padded", (47, 7, 12), (87, 1, 84, 514), "chunk", 128)]


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10
                    or not os.environ.get("FASTVIDEO_VSA_VC_ROOT"),
                    reason="needs SM10x and FASTVIDEO_VSA_VC_ROOT")
@pytest.mark.parametrize("case", _VC_CASES, ids=lambda c: f"{c[0]}-{c[3]}-t{c[4]}")
@torch.no_grad()
def test_vc_route_separate_qkv_feeds_packed_rows(request, monkeypatch, case):
    """FASTVIDEO_VSA_VC=1, prepared route "vc": preprocess_q_k_v never gathers (eager or compiled fullgraph); the fused
    producer runs once per call on the packed q, k, v rows, and the separate block equals the stacked block bitwise."""
    _, adapter, impl, calls, _ = request.getfixturevalue("h3")
    _, (t, hh, w), prefix, layout, tile = case
    layout_kw = dict(tile_layout=f"chunk{tile}", merge_prefix=True) if layout == "chunk" else {}
    md = MiniMaxH3VSAMetadataBuilder().build(current_timestep=0,
                                             raw_latent_shape=(t, 2 * hh, 2 * w),
                                             patch_size=(1, 2, 2),
                                             VSA_sparsity=0.85,
                                             prefix_segments=prefix,
                                             device=torch.device("cuda"),
                                             topk_cap=32 if tile == 256 else 64,
                                             tile_size=tile,
                                             **layout_kw)
    assert md.tile_elems == tile
    if not prefix:
        assert md.total_seq_length == md.variable_block_sizes.numel() * tile == 512
        assert not torch.equal(md.tile_partition_indices.cpu(), torch.arange(512))
    seen = []
    counted = adapter.prepare_vsa_vc_fwd_bshd
    monkeypatch.setattr(adapter, "prepare_vsa_vc_fwd_bshd",
                        lambda *a: seen.append([x.clone() for x in a[:3]]) or counted(*a))
    for name in ("vsa_tile_permute_qkv_fwd", "vsa_tile_permute_fwd"):
        monkeypatch.setattr(torch.ops.fastvideo_kernel, name, None)  # any gather on this route raises
    g = torch.Generator(device="cuda").manual_seed(5)
    q, k, v = (torch.randn(1, md.total_seq_length, 4, 128, device="cuda", dtype=torch.bfloat16, generator=g)
               for _ in range(3))

    def separate(a, b, c):
        return impl.postprocess_output(impl.forward(*impl.preprocess_q_k_v(a, b, c, md), None, md), md)

    stacked = impl.postprocess_output(impl.forward_qkv(impl.preprocess_qkv(torch.cat([q, k, v]), md), md), md)
    for mode, fn in (("eager", separate), ("compiled", torch.compile(separate, dynamic=True, fullgraph=True))):
        seen.clear()
        calls.clear()
        assert torch.equal(fn(q, k, v), stacked), mode
        assert len(calls) == len(seen) == 1, mode
        assert all(torch.equal(x, y) for x, y in zip(seen[0], (q, k, v), strict=True)), mode


def _pooled_reference(impl, md, q, k, v, up, up_sums):
    """The tiled q/k/v and their fp32 tile sums as the ungated gather + _pool_tiles's reduction, with their gradients."""
    tiled = impl.preprocess_q_k_v(q, k, v, md)
    n = md.variable_block_sizes.numel()
    sums = torch.stack(
        [x.view(1, n, md.tile_elems, *x.shape[2:]).sum(2, dtype=torch.float32).permute(0, 2, 1, 3) for x in tiled])
    return tiled, sums, torch.autograd.grad((*tiled, sums), (q, k, v), (*up, up_sums))


@pytest.mark.parametrize("tile", [128, 256])
@pytest.mark.parametrize("untrusted", [False, True], ids=["trusted", "untrusted"])
def test_tile_sums_gather_matches_pooling(tile, untrusted):
    """Gated gather: the tiled outputs equal the ungated gather's, the fp32 sums equal _pool_tiles's sum, and the backward
    equals autograd over both (bitwise)."""
    impl, md = _impl(), _metadata(tile_size=tile, tile_layout=f"chunk{tile}", merge_prefix=True)
    q, k, v = _qkv(md, seed=5)
    n = md.variable_block_sizes.numel()
    g = torch.Generator(device="cuda").manual_seed(6)
    up = [torch.randn(1, n * tile, 4, 128, device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(3)]
    up_sums = torch.randn(3, 1, 4, n, 128, device="cuda", generator=g) / tile
    state = md._tile_index_state
    if untrusted:
        md._tile_index_state = (state[0], state[1] - 1, state[2], state[3])
    try:
        tiled, sums, grads = _pooled_reference(impl, md, q, k, v, up, up_sums)
        *actual, actual_sums = impl.preprocess_q_k_v(q, k, v, md, tile_sums=True)
        actual_grads = torch.autograd.grad((*actual, actual_sums), (q, k, v), (*up, up_sums))
    finally:
        md._tile_index_state = state
    assert all(torch.equal(a, b) for a, b in zip(actual, tiled, strict=True))
    assert actual_sums.shape == sums.shape and torch.equal(actual_sums, sums)
    assert all(torch.equal(a, b) for a, b in zip(actual_grads, grads, strict=True))


def test_gated_layer_path_h3mh_compile(monkeypatch):
    """The SP=1 gated layer path (tile sums from the gather) under h3mh's compile config: inductor, dynamic, fullgraph,
    fail_on_recompile_limit_hit, use_duck_shape=False, SAC MUST_SAVE {vsa256_fwd, vsa_train_fwd, vsa_h3_block_map}.
    One graph over two calls, no breaks, compiled close to eager (BF16 coarse chain)."""
    import functools

    import torch.fx.experimental._config as fx_config
    from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts
    monkeypatch.setattr(torch._dynamo.config, "fail_on_recompile_limit_hit", True)
    monkeypatch.setattr(fx_config, "use_duck_shape", False)
    ops = torch.ops.fastvideo_kernel
    must_save = {ops.vsa256_fwd.default, ops.vsa_train_fwd.default, ops.vsa_h3_block_map.default}
    policy = lambda ctx, op, *a, **k: CheckpointPolicy.MUST_SAVE if op in must_save else CheckpointPolicy.PREFER_RECOMPUTE
    impl, md = _impl(), _metadata(tile_size=128, tile_layout="chunk128", merge_prefix=True)
    n = md.total_seq_length

    def block(q, k, v, g):
        return layer._forward_separate_qkv(impl, q, k, v, n, None, md, g)[0]

    def checkpointed(*x):
        return checkpoint(block,
                          *x,
                          use_reentrant=False,
                          context_fn=functools.partial(create_selective_checkpoint_contexts, policy))

    compiled = torch.compile(checkpointed, backend="inductor", mode="default", dynamic=True, fullgraph=True)

    def run(fn, seed):
        q, k, v = _qkv(md, seed=seed)
        gen = torch.Generator(device="cuda").manual_seed(seed + 100)
        g, up = (torch.randn(q.shape, device="cuda", dtype=q.dtype, generator=gen) for _ in range(2))
        leaves = [x.detach().requires_grad_(True) for x in (q, k, v, g)]
        out = fn(*leaves)
        return (out, *torch.autograd.grad(out, leaves, up))

    counters = torch._dynamo.utils.counters
    torch._dynamo.reset()
    try:
        eager = [run(block, seed) for seed in (1, 2)]  # the eager coarse region compiles its own graph: count after it
        counters.clear()
        for seed, e_all in zip((1, 2), eager, strict=True):
            for name, e, c in zip(("out", "dq", "dk", "dv", "dgate"), e_all, run(compiled, seed), strict=True):
                rel = ((c.float() - e.float()).norm() / e.float().norm()).item()
                assert rel < 2e-2, (seed, name, rel)
        assert counters["stats"]["unique_graphs"] == 1
        assert sum(counters["graph_break"].values()) == 0
    finally:
        torch._dynamo.reset()
