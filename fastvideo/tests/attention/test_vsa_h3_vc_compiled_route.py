"""H3 VSA backend, compiled no-grad VC route: the fused VC route runs as three opaque ops (vc_h3_prepare_fused,
vsa_h3_block_map_from_pools, vc_h3_attn_prepared) in eager and compiled mode, at tile 256 and tile 128 (ruling 84).
Eager output must equal the inline fused route (fastvideo 424931e1 at tile 256, the same calls at tile 128) bitwise,
torch.compile(fullgraph=True, dynamic=True) must take the VC route with one graph and equal eager bitwise, a cold Dynamo
compile (no eager warmup, as the loader's regional compile) must work, and CUDA-graph capture of a geometry whose JIT is
not built must raise the op's assertion instead of compiling inside the capture."""
import os
from collections import defaultdict

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10
                                or not os.environ.get("FASTVIDEO_VSA_VC_ROOT"),
                                reason="needs SM10x and FASTVIDEO_VSA_VC_ROOT (VC-enabled FA4 checkout)")

# Real s085k32 spec documents (latent T, H, W; nonzero prefix segments text/vidclip/keyframe/audio): pack 1 doc 0,
# pack 4 doc 0, pack 4 doc 1 of init/h3-real-shape/h3-real-shape-spec-s085k32.json (different tile counts and top-k).
DOCS = [((62, 10, 17), (64, 1, 692)), ((47, 7, 12), (87, 1, 84, 514)), ((52, 15, 26), (476, 1, 390, 572))]
HEADS = 4
COLD = "captured before any eager VC forward"


@pytest.fixture
def h3(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_VSA_VC", "1")
    monkeypatch.setenv("FASTVIDEO_VSA_CUTEDSL", "1")
    monkeypatch.delenv("FASTVIDEO_H3_VSA_PROBE", raising=False)
    monkeypatch.setattr(torch._dynamo.config, "fail_on_recompile_limit_hit", True)
    import torch.fx.experimental._config as fx_config
    monkeypatch.setattr(fx_config, "use_duck_shape", False)
    from fastvideo.attention.backends import video_sparse_attn_h3 as h3
    from fastvideo_kernel import block_sparse_attn_cute_fwd as adapter
    calls = []
    real = adapter.prepare_vsa_vc_fwd_bshd
    monkeypatch.setattr(adapter, "prepare_vsa_vc_fwd_bshd", lambda *a: calls.append(1) or real(*a))
    impl = h3.MiniMaxH3VSAImpl(num_heads=HEADS, head_size=128, causal=False, softmax_scale=128**-0.5)
    impl.layer_idx = 0
    assert impl.prepare_for_regional_compile(torch.device("cuda")) is None
    assert impl._regional_compile_nograd_route == "vc"
    torch._dynamo.reset()
    yield h3, adapter, impl, calls, real
    torch._dynamo.reset()


def _doc(h3, i, layout, seed=0, tile=256, **build_kw):
    (t, hh, w), prefix = DOCS[i]
    layout_kw = dict(tile_layout=f"chunk{tile}", merge_prefix=True) if layout == "chunk" else {}
    kw = dict(VSA_sparsity=0.85, topk_cap=32 if tile == 256 else 64, **layout_kw)
    kw.update(build_kw)
    meta = h3.MiniMaxH3VSAMetadataBuilder().build(current_timestep=0,
                                                  raw_latent_shape=(t, 2 * hh, 2 * w),
                                                  patch_size=(1, 2, 2),
                                                  prefix_segments=prefix,
                                                  device=torch.device("cuda"),
                                                  tile_size=tile,
                                                  **kw)
    g = torch.Generator(device="cuda").manual_seed(100 * i + seed)
    qkv = torch.randn(3, meta.total_seq_length, HEADS, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    return meta, qkv


def _block(impl):
    return lambda qkv, meta: impl.postprocess_output(impl.forward_qkv(impl.preprocess_qkv(qkv, meta), meta), meta)


def _previous_fused_route(h3, adapter, prepare, qkv, meta):
    """The inline fused route before the ops (fastvideo 424931e1 _vc_fused_forward, gate None), as the reference; at
    tile 128 the same calls on 128-token tiles, whose map already has the kernel's KV granularity."""
    from fastvideo_kernel.block_sparse_attn_256 import _expand_mask_and_sizes_256_to_128
    q, k, v = qkv.chunk(3, dim=0)
    sizes, untile, tile = meta.variable_block_sizes, meta.untile_combined_index, meta.tile_elems
    padded = sizes.numel() * tile
    padded_to_original = torch.full((padded, ), -1, dtype=torch.int64, device=q.device)
    padded_to_original[untile] = torch.arange(untile.numel(), dtype=torch.int64, device=q.device)
    p, pools = prepare(q.contiguous(), k.contiguous(), v.contiguous(), padded_to_original, sizes, tile,
                       torch.arange(padded, dtype=torch.int64, device=q.device), padded, 0)
    q_c, k_c = (x.to(q.dtype) for x in pools[:2])  # the original VSA block means (ruling 117c)
    scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / (q.shape[-1]**0.5)
    mask = h3._build_block_mask(scores, meta.num_prefix_tiles, meta.num_video_tiles, meta.VSA_sparsity, meta.exempt,
                                h3._video_topk(meta.VSA_sparsity, meta.num_video_tiles, meta.video_topk_cap))
    kernel_map = _expand_mask_and_sizes_256_to_128(mask, sizes) if tile == 256 else (mask, sizes.to(torch.int32))
    out = adapter.block_sparse_attn_vc_prepared_fwd_bshd(p, *kernel_map)[0]
    return out.to(q.dtype)[:, untile]


@pytest.mark.parametrize("tile", [256, 128])
@pytest.mark.parametrize("layout", ["chunk", "cube"])
@torch.no_grad()
def test_compiled_vc_route_bitwise(h3, layout, tile):
    """Eager ops == previous fused route and fullgraph compiled == eager, bitwise, with the VC producer running once
    per call in both modes; one graph and no recompile across documents with different tile counts and top-k."""
    h3, adapter, impl, calls, real = h3
    block = _block(impl)
    compiled = torch.compile(block, backend="inductor", dynamic=True, fullgraph=True)
    counters = torch._dynamo.utils.counters
    graphs0, breaks0 = counters["stats"]["unique_graphs"], sum(counters["graph_break"].values())
    ks = set()
    for i in range(len(DOCS)):
        meta, qkv = _doc(h3, i, layout, tile=tile)
        ks.add(meta.video_topk)
        calls.clear()
        eager = block(qkv, meta)
        assert len(calls) == 1, "eager call must take the fused VC route"
        assert torch.equal(eager, _previous_fused_route(h3, adapter, real, qkv, meta)), (layout, tile, i)
        out = compiled(qkv, meta)
        # The reference calls the unwrapped producer, so only the two backend calls count.
        assert len(calls) == 2, "compiled call must take the fused VC route"
        assert torch.equal(out, eager), (layout, tile, i)
    assert len(ks) > 1, ks
    assert counters["stats"]["unique_graphs"] - graphs0 == 1
    assert sum(counters["graph_break"].values()) == breaks0


@pytest.mark.parametrize("tile", [256, 128])
@pytest.mark.parametrize("layout", ["chunk", "cube"])
@torch.no_grad()
def test_cold_compile_matches_eager(h3, monkeypatch, layout, tile):
    """A cold process (no eager VC forward yet) compiles with fullgraph: the fakes never check warmth, the first compiled
    call builds the JITs inside the op bodies, and the output equals eager bitwise (loader regional compile, no warmup)."""
    h3, adapter, impl, calls, _ = h3
    monkeypatch.setattr(adapter, "_VC_WARMED", set())
    monkeypatch.setattr(adapter, "_VC_WARMED_ATTN", set())
    block = _block(impl)
    compiled = torch.compile(block, backend="inductor", dynamic=True, fullgraph=True)
    meta, qkv = _doc(h3, 0, layout, tile=tile)
    out = compiled(qkv, meta)
    assert len(calls) == 1, "cold compiled call must take the fused VC route"
    assert adapter._VC_WARMED and adapter._VC_WARMED_ATTN
    assert torch.equal(out, block(qkv, meta))


@torch.no_grad()
def test_vc_ops_cuda_graph(h3, monkeypatch):
    """CUDA-graph capture: an attention geometry never run eagerly raises before any launch; a warmed one captures,
    its producer launches land in the graph, and replays with new inputs equal eager bitwise."""
    h3, adapter, impl, calls, _ = h3
    monkeypatch.setattr(adapter, "_VC_WARMED_ATTN", set())
    block = _block(impl)
    meta_a, qkv_a = _doc(h3, 0, "chunk")
    meta_b, qkv_b = _doc(h3, 2, "chunk")
    assert meta_a.variable_block_sizes.numel() != meta_b.variable_block_sizes.numel()
    block(qkv_a, meta_a)
    torch.cuda.synchronize()
    with pytest.raises(RuntimeError, match=COLD), torch.cuda.graph(torch.cuda.CUDAGraph()):
        block(qkv_b, meta_b)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = block(qkv_a, meta_a)
    for seed in (1, 2):
        qkv_a.copy_(_doc(h3, 0, "chunk", seed)[1])
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, block(qkv_a, meta_a)), seed


@pytest.mark.parametrize("tile", [256, 128])
@torch.no_grad()
def test_vc_warm_keys_match_index_compiles(h3, monkeypatch, tile):
    """The attention warm key tracks the real JIT: with empty warm and Triton caches, every document adds exactly as many
    op keys as classified-index kernel compiles (one per BLOCK bucket), over geometries in different buckets."""
    h3, adapter, impl, _, _ = h3
    from fastvideo_kernel.triton_kernels import index
    kernel = index._classified_map_to_index_kernel
    monkeypatch.setattr(adapter, "_VC_WARMED_ATTN", set())
    monkeypatch.setattr(kernel, "device_caches", defaultdict(kernel.create_binder))
    compiles = []
    real = type(kernel)._do_compile
    monkeypatch.setattr(kernel, "_do_compile", lambda *a, **kw: compiles.append(1) or real(kernel, *a, **kw))
    block = _block(impl)
    for layout in ("chunk", "cube"):
        for i in range(len(DOCS)):
            keys0, compiles0 = len(adapter._VC_WARMED_ATTN), len(compiles)
            block(*_doc(h3, i, layout, tile=tile)[::-1])
            assert len(adapter._VC_WARMED_ATTN) - keys0 == len(compiles) - compiles0, (layout, tile, i)
    assert len(adapter._VC_WARMED_ATTN) == len(compiles) >= 2


@pytest.mark.parametrize("layout", ["chunk", "cube"])
@torch.no_grad()
def test_vc_route_tile128_vs_bf16(h3, monkeypatch, layout):
    """Tile 128: the VC route runs on every document (one producer call each) and stays within FP8 error of the BF16
    no-grad route at tile 128 (the same rel_l2 scale as tile 256); other tiles are rejected by the ops."""
    h3, adapter, impl, calls, _ = h3
    block = _block(impl)
    vc = []
    for i in range(len(DOCS)):
        meta, qkv = _doc(h3, i, layout, tile=128)
        vc.append((meta, qkv, block(qkv, meta)))
        assert len(calls) == i + 1, "tile-128 call must take the fused VC route"
    monkeypatch.setenv("FASTVIDEO_VSA_VC", "0")
    for meta, qkv, out in vc:
        ref = block(qkv, meta).float()
        assert (out.float() - ref).norm() / ref.norm() < 0.1
    assert len(calls) == len(DOCS)
    x = torch.zeros(1, 64, 1, 128, device="cuda", dtype=torch.bfloat16)
    sizes = torch.full((1, ), 64, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="tile must be 128 or 256"):
        torch.ops.fastvideo_kernel.vc_h3_prepare_fused(x, x, x, torch.arange(64, device="cuda"), sizes, 64)


@pytest.mark.parametrize("tile", [256, 128])
@torch.no_grad()
def test_vc_dense_layer_sparsity0_and_gate(h3, tile):
    """VC route with a dense layer, sparsity 0 and a gate: the dense layer (layer_idx in dense_layers eagerly, the
    tensor-valued force_dense under fullgraph capture) equals sparsity 0 bitwise and differs from the sparse layer; a gate
    adds exactly the pooled compression term of the producer's pools (compiled: within BF16 tolerance, since Inductor
    fuses the combine without the intermediate BF16 roundings)."""
    h3, _, impl, calls, _ = h3
    sparse, qkv = _doc(h3, 1, "chunk", tile=tile)
    dense, _ = _doc(h3, 1, "chunk", tile=tile, dense_layers=(impl.layer_idx, ))
    zero, _ = _doc(h3, 1, "chunk", tile=tile, VSA_sparsity=0.0)
    q, k, v = qkv.chunk(3, dim=0)

    def fwd(meta, gate=None):
        return impl.postprocess_output(impl.forward(q, k, v, gate, meta), meta)

    compiled = torch.compile(fwd, backend="inductor", dynamic=True, fullgraph=True)
    out_dense = fwd(dense)
    assert torch.equal(out_dense, fwd(zero)) and not torch.equal(out_dense, fwd(sparse))
    assert torch.equal(compiled(dense), out_dense)
    gate = torch.randn_like(q)
    sizes, untile = sparse.variable_block_sizes, sparse.untile_combined_index
    tiled = impl.forward(q, k, v, None, sparse)
    *_, pq, pk, pv = torch.ops.fastvideo_kernel.vc_h3_prepare_fused(q, k, v, untile, sizes, tile)
    pq, pk, pv = (x.to(q.dtype) for x in (pq, pk, pv))  # the original VSA block means (ruling 117c)
    out_c = torch.softmax(pq @ pk.transpose(-2, -1) / q.shape[-1]**0.5, dim=-1) @ pv  # [B, H, n_tiles, D]
    gate_tiled = torch.zeros_like(tiled)
    gate_tiled[:, untile] = gate
    expected = impl.postprocess_output(
        tiled + out_c.permute(0, 2, 1, 3).to(tiled.dtype).repeat_interleave(tile, dim=1) * gate_tiled, sparse)
    assert torch.equal(fwd(sparse, gate), expected)
    torch.testing.assert_close(compiled(sparse, gate), expected, rtol=1.6e-2, atol=2e-3)  # a few BF16 ulps
    assert len(calls) >= 6, "every call must take the fused VC route"


def test_vc_route_unavailable_reason(monkeypatch):
    """VC=1 without a usable compiled VC route (here head 64) stays eager with a VC-specific reason, not the sm_100a text."""
    monkeypatch.setenv("FASTVIDEO_VSA_VC", "1")
    monkeypatch.setenv("FASTVIDEO_VSA_CUTEDSL", "1")
    from fastvideo.attention.backends import video_sparse_attn_h3 as h3
    impl = h3.MiniMaxH3VSAImpl(num_heads=HEADS, head_size=64, causal=False, softmax_scale=64**-0.5)
    impl.layer_idx = 0
    reason = impl.prepare_for_regional_compile(torch.device("cuda"))
    assert impl._regional_compile_nograd_route is None
    assert reason is not None and reason.startswith("FASTVIDEO_VSA_VC=1"), reason
