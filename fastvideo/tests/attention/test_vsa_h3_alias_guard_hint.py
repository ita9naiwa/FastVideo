# SPDX-License-Identifier: Apache-2.0
"""VSA-H3 passes FA4's alias_guard hint on every route: True for dense-prefix documents, False for prefix-0 documents,
and nothing to a provider that predates the hint."""

import functools
import os

import pytest
import torch

from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAImpl, MiniMaxH3VSAMetadataBuilder

_GRID = dict(raw_latent_shape=(16, 16, 24), patch_size=(1, 2, 2))
_HEADS, _DIM = 2, 128

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                                reason="SM10x CUDA required")


def _meta(prefix_segments):
    return MiniMaxH3VSAMetadataBuilder().build(current_timestep=0, VSA_sparsity=0.5, prefix_segments=prefix_segments,
                                               device=torch.device("cuda"), **_GRID)


def _impl():
    return MiniMaxH3VSAImpl(num_heads=_HEADS, head_size=_DIM, causal=False, softmax_scale=_DIM**-0.5,
                            prefix="blocks.0.attn")


@pytest.fixture
def cute(monkeypatch):
    pytest.importorskip("flash_attn.cute.interface")
    monkeypatch.setenv("FASTVIDEO_VSA_CUTEDSL", "1")
    for name in ("FASTVIDEO_VSA_TRITON", "FASTVIDEO_KERNEL_VSA_FORCE_TRITON", "FASTVIDEO_VSA_VC"):
        monkeypatch.delenv(name, raising=False)
    from flash_attn.cute import interface
    return interface


@pytest.fixture
def hinted(cute):
    import inspect
    if "alias_guard" not in inspect.signature(cute._flash_attn_fwd).parameters:
        pytest.skip("provider predates the alias_guard hint (covered by test_old_provider_*)")
    return cute


def _spy(monkeypatch, interface, old_provider=False):
    """Record the alias_guard kwarg at the FA4 forward entry the adapter calls (flash_attn_func or _flash_attn_fwd).

    A new provider keeps the real signatures (the adapter probes them); an old provider exposes none of the new kwarg.
    Only the outermost call is recorded, since flash_attn_func itself forwards to _flash_attn_fwd.
    """
    from fastvideo_kernel import block_sparse_attn_cute_fwd as adapter
    calls, depth = [], [0]

    def spy(orig):
        def record(*args, **kwargs):
            if depth[0] == 0:
                calls.append(kwargs.get("alias_guard", "absent"))
            depth[0] += 1
            try:
                return orig(*args, **kwargs)
            finally:
                depth[0] -= 1

        wrapped = record if old_provider else functools.wraps(orig)(record)
        wrapped.__dict__.update(getattr(orig, "__dict__", {}))  # e.g. compile_cache
        return wrapped

    for name in ("_flash_attn_fwd", "flash_attn_func"):
        monkeypatch.setattr(interface, name, spy(getattr(interface, name)))
    # The FA4 loader caches the imported functions: reset it now and after the test so no spy outlives it.
    adapter._load_fa4_cute.cache_clear()  # the autouse fixture clears it again after the test
    return calls


@pytest.fixture(autouse=True)
def _clear_fa4_cache():
    from fastvideo_kernel import block_sparse_attn_cute_fwd as adapter
    adapter._load_fa4_cute.cache_clear()
    yield
    adapter._load_fa4_cute.cache_clear()


def _run(meta, grad):
    impl = _impl()
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, meta.total_seq_length, _HEADS, _DIM, device="cuda", dtype=torch.bfloat16,
                           requires_grad=grad) for _ in range(3))
    with torch.set_grad_enabled(grad):
        tq, tk, tv = (impl.preprocess_qkv(t, meta) for t in (q, k, v))
        out = impl.postprocess_output(impl.forward(tq, tk, tv, None, meta), meta)
        if grad:
            out.float().pow(2).sum().backward()
    return out.detach()


@pytest.mark.parametrize("grad", [True, False], ids=["bf16_training", "bf16_nograd"])
def test_bf16_routes_pass_prefix_hint(monkeypatch, hinted, grad):
    calls = _spy(monkeypatch, hinted)
    for prefix, expected in (((64, 32, 16), True), ((), False)):
        meta = _meta(prefix)
        assert (meta.num_prefix_tiles > 0) == expected
        calls.clear()
        _run(meta, grad)
        assert calls and set(calls) == {expected}, (prefix, calls)


def test_vc_fused_route_passes_prefix_hint(monkeypatch, hinted):
    if not os.environ.get("FASTVIDEO_VSA_VC_ROOT"):
        pytest.skip("FASTVIDEO_VSA_VC_ROOT (VC-enabled FA4 checkout) required")
    monkeypatch.setenv("FASTVIDEO_VSA_VC", "1")
    calls = _spy(monkeypatch, hinted)
    for prefix, expected in (((64, 32, 16), True), ((), False)):
        meta = _meta(prefix)
        calls.clear()
        _run(meta, grad=False)
        assert calls and set(calls) == {expected}, (prefix, calls)


@pytest.mark.parametrize("grad", [True, False])
def test_old_provider_gets_no_hint_and_same_output(monkeypatch, cute, grad):
    # Fresh metadata per run: reusing one H3 metadata object for a second training backward fails on the base too.
    reference = _run(_meta((64, 32, 16)), grad)
    calls = _spy(monkeypatch, cute, old_provider=True)
    out = _run(_meta((64, 32, 16)), grad)
    assert calls and set(calls) == {"absent"}, calls
    assert torch.equal(out, reference)  # the hint never changes values, only which CTA takes which tile


def test_compiled_custom_op_carries_hint(monkeypatch, hinted):
    """The compiled training op takes the hint as a CPU tensor input: one graph, no recompile between prefix and prefix-0
    documents, and the kernel still receives True / False."""
    from torch._dynamo.testing import CompileCounter

    from fastvideo_kernel.block_sparse_attn_256 import block_sparse_attn_256_bshd
    calls = _spy(monkeypatch, hinted)
    torch.manual_seed(1)
    n = 4
    q, k, v = (torch.randn(1, n * 256, _HEADS, _DIM, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    block_map = torch.rand(1, _HEADS, n, n, device="cuda") > 0.3
    block_map[..., 0] = True
    sizes = torch.full((n, ), 256, device="cuda", dtype=torch.int32)

    def attend(q, k, v, block_map, sizes, hint):
        return block_sparse_attn_256_bshd(q, k, v, block_map, sizes, alias_guard=hint)[0]

    torch._dynamo.reset()
    counter = CompileCounter()
    compiled = torch.compile(attend, backend=counter, fullgraph=True)
    outs = {}
    for prefix_doc in (True, False, True, False):
        calls.clear()
        leaves = [t.detach().requires_grad_(True) for t in (q, k, v)]
        out = compiled(*leaves, block_map, sizes, torch.tensor(prefix_doc))
        out.float().sum().backward()
        assert set(calls) == {prefix_doc}, (prefix_doc, calls)
        outs.setdefault(prefix_doc, out.detach())
    assert counter.frame_count == 1, counter.frame_count  # one graph, zero recompiles across hint values
    assert torch.equal(outs[True], outs[False])  # the hint never changes values
    assert torch.ops.fastvideo_kernel.vsa256_fwd.default.name() == "fastvideo_kernel::vsa256_fwd"  # SAC op key


def test_metadata_hint_is_host_tensor():
    for prefix, expected in (((64, 32, 16), True), ((), False)):
        hint = _meta(prefix).alias_guard_hint
        assert hint.device.type == "cpu" and hint.dtype == torch.bool and hint.dim() == 0 and bool(hint) == expected


@pytest.mark.parametrize("entry", ["forward", "forward_qkv"])
def test_h3mh_compile_config_mixed_prefix_docs(monkeypatch, hinted, entry):
    """h3mh's compiled training config (as test_vsa256_ops.test_h3_block_h3mh_compile_config: inductor, dynamic=True,
    fullgraph, SAC MUST_SAVE vsa256_fwd + vsa_h3_block_map, metadata as an argument, fail_on_recompile_limit_hit) over
    prefix and prefix-0 documents: the hint reaches the kernel per document; graph count is reported. ``forward_qkv``
    is the gate-free fused-QKV entry: the vsa256 op takes the stacked [3B, S, H, D] input (k = v = None)."""
    from fastvideo_kernel import vsa256_ops
    import torch.fx.experimental._config as fx_config
    from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts
    monkeypatch.setattr(torch._dynamo.config, "fail_on_recompile_limit_hit", True)
    monkeypatch.setattr(fx_config, "use_duck_shape", False)
    calls = _spy(monkeypatch, hinted)
    stacked, chunks = [], vsa256_ops._chunks  # the op body (opaque, runs per call) reports which input form it got
    monkeypatch.setattr(vsa256_ops, "_chunks", lambda q, k, v: (stacked.append(k is None), chunks(q, k, v))[1])
    impl = MiniMaxH3VSAImpl(num_heads=8, head_size=_DIM, causal=False, softmax_scale=_DIM**-0.5)
    impl.layer_idx = 0

    def block(q, k, v, meta):
        x = impl.preprocess_qkv(torch.cat([q, k, v], dim=0), meta)
        if entry == "forward_qkv":
            return impl.postprocess_output(impl.forward_qkv(x, meta), meta)
        q2, k2, v2 = x.chunk(3, dim=0)
        return impl.postprocess_output(impl.forward(q2, k2, v2, None, meta), meta)

    must_save = {torch.ops.fastvideo_kernel.vsa256_fwd.default, torch.ops.fastvideo_kernel.vsa_h3_block_map.default}
    policy = lambda ctx, op, *a, **k: (CheckpointPolicy.MUST_SAVE if op in must_save else CheckpointPolicy.PREFER_RECOMPUTE)
    compiled = torch.compile(lambda q, k, v, meta: checkpoint(block, q, k, v, meta, use_reentrant=False,
                                                              context_fn=functools.partial(
                                                                  create_selective_checkpoint_contexts, policy)),
                             backend="inductor", mode="default", dynamic=True, fullgraph=True)
    geometries = [((42, 14, 24), (175, 1, 170, 402)), ((42, 20, 20), ()), ((37, 16, 56), (250, 1, 0, 300)),
                  ((62, 26, 24), ()), ((47, 32, 30), (175, 1, 170, 402))]
    torch._dynamo.reset()
    counters = torch._dynamo.utils.counters
    graphs0 = counters["stats"]["unique_graphs"]
    for i, (raw, prefix) in enumerate(geometries):
        meta = MiniMaxH3VSAMetadataBuilder().build(current_timestep=0, raw_latent_shape=raw, patch_size=(1, 2, 2),
                                                   VSA_sparsity=0.75, prefix_segments=prefix,
                                                   device=torch.device("cuda"), tile_layout="chunk256", merge_prefix=True)
        torch.manual_seed(100 + i)
        q, k, v = (torch.randn(1, meta.total_seq_length, 8, _DIM, device="cuda", dtype=torch.bfloat16,
                               requires_grad=True) for _ in range(3))
        calls.clear()
        stacked.clear()
        out = compiled(q, k, v, meta)
        torch.autograd.grad(out, (q, k, v), torch.randn_like(out))
        assert calls and set(calls) == {bool(prefix)}, (i, prefix, calls)
        assert stacked and set(stacked) == {entry == "forward_qkv"}, (i, entry, stacked)
    graphs = counters["stats"]["unique_graphs"] - graphs0
    print(f"h3mh compile config ({entry}), mixed prefix/prefix-0 docs: {graphs} graph(s)", flush=True)
    assert graphs == 1, f"{graphs} graphs (recompiles) across prefix and prefix-0 documents"
    torch._dynamo.reset()
