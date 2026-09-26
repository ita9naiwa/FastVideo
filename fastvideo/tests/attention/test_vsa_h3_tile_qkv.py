# SPDX-License-Identifier: Apache-2.0
"""preprocess_q_k_v tiles separate q, k, v without a stacked copy: byte-identical to preprocess_qkv(cat([q, k, v]))."""

import types

import pytest
import torch

from fastvideo.attention import layer
from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAImpl, MiniMaxH3VSAMetadataBuilder

# token grid (37, 24, 42) + text/audio prefix: 38,010 rows; 3 x [1, S, 4, 128] is >= 2**25 elements (eligible eager).
_SPEC = dict(raw_latent_shape=(37, 48, 84), patch_size=(1, 2, 2), prefix_segments=(300, 0, 414))
_LAYOUTS = [dict(), dict(tile_layout="chunk256", merge_prefix=True)]

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


def test_no_grad_and_ineligible_calls_take_the_stacked_route():
    impl, md = _impl(), _metadata()
    q, k, v = (t.detach() for t in _qkv(md, seed=5))
    with torch.no_grad():
        expected = impl.preprocess_qkv(torch.cat([q, k, v], dim=0), md).clone()
        assert torch.equal(torch.cat(impl.preprocess_q_k_v(q, k, v, md), dim=0), expected)


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

        def record(*args):
            self.calls.append(name)
            if name == "preprocess_q_k_v":
                return args[:3]
            return args[0][:1] if name == "forward_qkv" else args[0]

        return record


@pytest.mark.parametrize("vsa", [False, True], ids=["dense", "vsa"])
@pytest.mark.parametrize("sp,gate", [(1, False), (2, False), (1, True)], ids=["sp1", "sp2", "gate"])
def test_route_selection(monkeypatch, vsa, sp, gate):
    """SP=1 without gate/replicated input tiles q, k, v separately; SP>1 or a gate keeps the stacked route."""
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
    separate = sp == 1 and not gate
    assert ("preprocess_q_k_v" in impl.calls) == separate
    assert ("preprocess_qkv" in impl.calls) != separate
    assert ("forward_qkv" in impl.calls) == (vsa and not separate and not gate)  # VSA stacked no-gate: fused-QKV input
